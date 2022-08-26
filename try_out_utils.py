import subprocess

import numpy as np

from db_utils import Database
import os
import shutil
from forward import get_samples
import ek_utlis
import json
import argparse
from tqdm import tqdm
import pandas as pd


def yan81(db, schemes, mul_by):
    histogram_start_df = db.relations[db.predict_rel].copy()  # [db.predict_col.split("@")[0]].to_frame()
    # histogram_start_df.is_copy = False
    histogram_start_df['histogram'] = np.zeros(len(histogram_start_df))
    chosen_starting_tuples = []
    small_df_size = len(histogram_start_df) * mul_by
    while len(chosen_starting_tuples) < small_df_size:
        for scheme in tqdm(schemes):
            if len(scheme.split("-")) == 1:  # scheme len is equal to 0 -> no edges -> no need to run Yan81
                continue
            start_col, start_rel = scheme.split("-")[0].split("@")

            join_cols = [sub.split(">")[0] for sub in scheme.split("-")][1:]
            col_at_rels = [sub.split(">")[-1] for sub in scheme.split("-")]
            for idx in range(len(col_at_rels) - 1, 0, -1):
                # compute joined_df of from_rel JOIN to_rel
                to_rel, from_rel = col_at_rels[idx], col_at_rels[idx - 1]
                from_col, from_rel = from_rel.split("@")
                to_col, to_rel = to_rel.split("@")
                from_df, to_df = db.relations[from_rel], db.relations[to_rel] if idx == (len(col_at_rels) - 1) else left_joined_df
                join_to_col = join_cols[idx-1].split("@")[0]
                # compute the left semi join between from_df and to_df
                semi_table = from_df.merge(to_df, left_on=from_col, right_on=join_to_col, how='inner', suffixes=('', '_to'))
                # joined_df = from_df.merge(to_df, left_on=from_col, right_on=join_to_col, how='inner', suffixes=('_from', ''))  # inner join
                in_both = from_df[from_col].isin(semi_table[from_col])
                left_joined_df = from_df[in_both]

            potential_tuples = []
            for idx, row in histogram_start_df.iterrows():
                if row[start_col] in left_joined_df[start_col].values:
                    # histogram_start_df['histogram'][idx] += 1
                    # histogram_start_df.loc[idx, ('histogram')] += 1
                    potential_tuples.append(idx)
            if len(potential_tuples) > 0:
                added_tuple = int(np.random.choice(potential_tuples, size=1))
                chosen_starting_tuples.append(added_tuple)
                histogram_start_df = histogram_start_df.drop(added_tuple)
            if len(chosen_starting_tuples) >= small_df_size:
                break
    histogram_start_df = histogram_start_df.drop(['histogram'], axis=1)
    return chosen_starting_tuples

def yan81_aux(db, depth, mul_by):
    tuples = [r for _, r, _ in db.iter_rows(db.predict_rel)]
    scheme_tuple_map = list(db.scheme_tuple_map(db.predict_rel, tuples, depth).keys())
    schemes = []
    for scheme in scheme_tuple_map:
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                schemes.append(f"{scheme}>{col_id}")
    return yan81(db, schemes, mul_by)

def tryout_yan81(mul_by, data_name="mondial", num_samples=0, depth=3, sort="loss"):
    exp_name = f"yan81_mul{mul_by}"
    data_path = f'Datasets/{data_name}'
    db = Database.load_csv(data_path)

    # creat a small sample of the data
    small_sample_data_name = f"{data_name}_small_Yan81_EXP_{mul_by}"
    small_sample_data_path = f'Datasets/{small_sample_data_name}'
    os.makedirs(small_sample_data_path, exist_ok=True)
    for fname in os.listdir(data_path):
        if "cols" in fname:
            shutil.copyfile(f'{data_path}/{fname}', f'{small_sample_data_path}/{data_name}_cols')
            break

    # simple copy only first fifth
    for name, df in db.relations.items():
        if name == db.predict_rel:
            # df.sample(n=int(len(df) * mul_by), random_state=0).to_csv(f"{small_sample_data_path}/{db.predict_rel}.csv",index=False)
            df.iloc[yan81_aux(db, depth, mul_by)].to_csv(f"{small_sample_data_path}/{db.predict_rel}.csv", index=False)
        else:
            df.to_csv(f"{small_sample_data_path}/{name}.csv", index=False)

    # TODO: num_samples is usually 500
    num_samples_str = f"--num_samples {num_samples}" if num_samples != 0 else ""
    output = subprocess.check_output(
        f"python forward_split_Loss_per_scheme.py --data_name {small_sample_data_name} --yuval_change {data_name}_{exp_name} --tryout True --depth {depth} {num_samples_str}".split())
    # now the ordered schemes are in "Sorted_schemes\temp_ordered_schemes.txt"
    temp_ordered_schemes_path = os.path.join("Sorted_schemes", f"{data_name}_{exp_name}.txt")
    if os.path.exists(temp_ordered_schemes_path):
        with open(temp_ordered_schemes_path, 'r') as f:
            d = json.load(f)
            print("From yan81 tryout - ordered schemes from lowest loss to highest loss:")
            print(d)
    print(f"\nTime:{str(output.split(b'Time:')[-1])[:-1]}")

def tryout_mul_db(mul_by, data_name="mondial", num_samples=0, depth=3, sort="loss"):
    # for mul_by in [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]:
    exp_name = f"try_out_mul{mul_by}"
    data_path = f'Datasets/{data_name}'
    # load csv
    db = Database.load_csv(data_path)

    # creat a small sample of the data
    small_sample_data_name = f"{data_name}_small_sample_data_EXP_mul{mul_by}"
    small_sample_data_path = f'Datasets/{small_sample_data_name}'
    os.makedirs(small_sample_data_path, exist_ok=True)
    for fname in os.listdir(data_path):
        if "cols" in fname:
            shutil.copyfile(f'{data_path}/{fname}', f'{small_sample_data_path}/{data_name}_cols')
            break

    # simple copy only first fifth
    for name, df in db.relations.items():
        if name == db.predict_rel:
            # pick subset of schemes from the start relation and then all the reachable schemes
            # predict_rel_df = db.relations[db.predict_rel]
            # predict_rel_df[:int(len(predict_rel_df) * mul_by)].to_csv(f"{small_sample_data_path}/{db.predict_rel}.csv", index=False)
            df.sample(n=int(len(df) * mul_by), random_state=0).to_csv(f"{small_sample_data_path}/{db.predict_rel}.csv",
                                                                      index=False)
        else:
            # df[:int(len(df) * mul_by)].to_csv(f"{small_sample_data_path}/{name}.csv", index=False)
            df.to_csv(f"{small_sample_data_path}/{name}.csv", index=False)
    # all FK's are db.arrow_rel_map
    # all_rels = db.relations
    # done_rels = [db.predict_rel]
    #
    # all_schemes = get_samples(db, 3, 500, ek_utlis.ek_sample_fct, yuval_change='')

    pass
    # exit()
    # run scheme finder and get the most beneficial schemes
    pass

    # run forward with the "most beneficial schemes"
    most_beneficial_schemes = []

    # exit()
    # TODO: num_samples is usually 500
    num_samples_str = f"--num_samples {num_samples}" if num_samples != 0 else ""
    output = subprocess.check_output(
        f"python forward_split_Loss_per_scheme.py --data_name {small_sample_data_name} --yuval_change {data_name}_{exp_name} --tryout True --depth {depth} {num_samples_str}".split())
    # now the ordered schemes are in "Sorted_schemes\temp_ordered_schemes.txt"
    temp_ordered_schemes_path = os.path.join("Sorted_schemes", f"{data_name}_{exp_name}.txt")
    if os.path.exists(temp_ordered_schemes_path):
        with open(temp_ordered_schemes_path, 'r') as f:
            d = json.load(f)
            print("From tryout - ordered schemes from lowest loss to highest loss:")
            print(d)

            # if 'scores' in d.keys():
            #     length = f" time:{d['time']}" if 'time' in d.keys() else ""
            #     print(f'{dir:<100}: score:{np.mean(d["scores"]):.4f} std:{np.std(d["scores"]):.4f}{length}')

    pass
    print(f"\nTime:{str(output.split(b'Time:')[-1])[:-1]}")


def stop_n_restart(epoch=1, data_name="mondial", num_samples=0, depth=3, sort="loss"):
    exp_name = f"stop_n_restart{epoch}"
    num_samples_str = f"--num_samples {num_samples}" if num_samples != 0 else ""
    sorting_file = "forward_split_Loss_per_scheme" if sort == "loss" else "forward_scheme_norm"
    output = subprocess.check_output(f"python {sorting_file}.py --data_name {data_name} --yuval_change {data_name}_{exp_name} --tryout True --depth {depth} --stop_n_restart {epoch} {num_samples_str}".split())
    # now the ordered schemes are in "Sorted_schemes\temp_ordered_schemes.txt"
    temp_ordered_schemes_path = os.path.join("Sorted_schemes", f"{data_name}_{exp_name}.txt")
    if os.path.exists(temp_ordered_schemes_path):
        with open(temp_ordered_schemes_path, 'r') as f:
            d = json.load(f)
            print("From tryout - ordered schemes from lowest loss to highest loss:")
            print(d)

    print(f"\nTime:{str(output.split(b'Time:')[-1])[:-1]}")

if __name__ == '__main__':
    # tryout_yan81(0.4, data_name="mondial", num_samples=0, depth=3)
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="mondial", help="dataset name")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the walks")

    parser.add_argument("--mul_by", type=float, default=0.5, help="mul_by number to reduce database")
    parser.add_argument("--method", type=str, default="tryout",
                        help="Partial Training method (tryout or stop_n_restart or yan81)")
    parser.add_argument("--num_samples", type=int, default=0, help="num_samples in tryout")  # 0 is default num_samples

    parser.add_argument("--epoch", type=int, default=1, help="stop_n_restart epoch to stop")
    parser.add_argument("--sorting_method", type=str, default="loss", help="loss or norm")
    args = parser.parse_args()

    if args.method == "tryout":
        tryout_mul_db(mul_by=args.mul_by, data_name=args.data_name, num_samples=args.num_samples, depth=args.depth, sort=args.sorting_method)
    elif args.method == "stop_n_restart":
        stop_n_restart(epoch=args.epoch, data_name=args.data_name, num_samples=args.num_samples, depth=args.depth, sort=args.sorting_method)
    elif args.method == "yan81":
        tryout_yan81(mul_by=args.mul_by, data_name=args.data_name, num_samples=args.num_samples, depth=args.depth, sort=args.sorting_method)
