import numpy as np
import torch
from torch_scatter import scatter_mean
from tqdm import tqdm
import argparse
from db_utils import Database
import ek_utlis
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eps = 1e-15
kernels = {'object': lambda x, y: x == y,
           'int64': lambda x, y: x == y,
           'float64': lambda x, y: np.exp(
               - ((x - y) ** 2) / (2 * ((0.05 * np.maximum(np.abs(x), np.abs(y)) + eps) ** 2)))}

kernels_dict = {}


def get_samples(db, depth, num_samples, sample_fct):
    # -# depth is used here
    tuples = [r for _, r, _ in db.iter_rows(db.predict_rel)]
    scheme_tuple_map = db.scheme_tuple_map(db.predict_rel, tuples, depth)

    # -# scheme ends in a column and not in a table.
    subset_full_schemes = []
    for scheme, tuple_map in tqdm(scheme_tuple_map.items()):
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                subset_full_schemes.append(f"{scheme}>{col_id}")

    samples = {}
    for scheme, tuple_map in tqdm(scheme_tuple_map.items()):
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                full_scheme = f"{scheme}>{col_id}"
                if full_scheme in subset_full_schemes:
                    col_kernel = kernels_dict[col_id] if col_id in kernels_dict.keys() else kernels[db.get_col_type(col_id)]
                    pairs, values = sample_fct(db, col_id, tuple_map, num_samples, col_kernel)
                    samples[full_scheme] = (pairs, values)

    return samples


def samples_to_variance(samples, row_idx, schema_idx):
    #s stack pairs of tuples and map them to integer indices
    row_idx_map = np.vectorize(lambda x: row_idx[x])
    pairs = np.vstack([p for p, _ in samples.values()])
    pairs = torch.tensor(row_idx_map(pairs))

    # stack kernel values
    vals = torch.tensor(np.concatenate([v for _, v in samples.values()], axis=0))

    # stack schemes and map them to integer indices
    scheme = [np.int64([schema_idx[s]] * samples[s][0].shape[0]) for s in samples.keys()]
    scheme = torch.tensor(np.concatenate(scheme, axis=0))

    sample_idx = torch.cat([scheme.view(-1, 1), pairs], dim=1)
    unique_idx, uni_map = torch.unique(sample_idx, dim=0, return_inverse=True)
    tup_pair_ek = scatter_mean(vals, uni_map, dim=0)

    mean_ek = scatter_mean(tup_pair_ek, unique_idx[:, 0], dim=0)
    mean_sq_ek = scatter_mean(tup_pair_ek ** 2, unique_idx[:, 0], dim=0)
    schema_ek_var = mean_sq_ek - (mean_ek ** 2)
    return schema_ek_var


def get_schema_variance(db, depth, num_samples, row_idx):
    # import torch
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device.type)
    # print(f"$3 {time.time()}")
    time_to_ignore = time.time()
    samples = get_samples(db, depth, int(num_samples/10), ek_utlis.ek_sample_fct)
    time_to_ignore = time.time() - time_to_ignore
    # print(f"$4 {time.time()}")
    schema_idx = {s: i for i, s in enumerate(samples.keys())}
    schema_ek_var = samples_to_variance(samples, row_idx, schema_idx)
    schema_ek_var = schema_ek_var.numpy()
    schema_ek_var = {schema: schema_ek_var[i] for schema, i in schema_idx.items()}
    # print(f"$5 {time.time()}")
    return schema_ek_var, time_to_ignore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='genes', help="Name of the data base")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the walks")
    parser.add_argument("--kernel", type=str, default='EK', choices={'EK', 'MMD'}, help="Kernel to use for ForWaRD")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples per start tuple and metapath")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = f'Datasets/{args.data_name}'
    db = Database.load_csv(data_path)

    Y, rows = db.get_labels()
    row_idx = {r: i for i, r in enumerate(rows)}

    schema_ek_var = get_schema_variance(db, args.depth, args.num_samples, row_idx)
    print(schema_ek_var)
