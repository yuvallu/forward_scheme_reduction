import time

import pandas as pd
import numpy as np
import networkx as nx
import glob
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from itertools import chain
import math
from functools import lru_cache

from db_utils import Database


def YIX_conditional_entropy(XY, YIX, XIY, p_Y_dict=None):
    # calc YIX conditional entropy
    H_YIX = 0.0
    for k_x, x_i in XY.items():
        for k_y, xy_i in x_i.items():
            if xy_i != 0:
                H_YIX += xy_i * math.log(YIX[k_x][k_y])
    return -1 * H_YIX

def mutual_information(XY, YIX, XIY, p_Y_dict=None):
    # calc YIX conditional entropy
    H_YIX = 0.0
    for k_x, x_i in XY.items():
        for k_y, xy_i in x_i.items():
            if xy_i != 0:
                H_YIX += xy_i * math.log(YIX[k_x][k_y])
    H_Y = 0.0
    for k_y, y_i in p_Y_dict.items():
        if y_i != 0:
            H_Y -= y_i * math.log(y_i)
    # MI = H_Y - H_YIX
    return H_Y - (-1 * H_YIX)

def mutual_information_optimized(XY, p_X_dict, p_Y_dict):
    MI = 0.0
    for k_x, x_i in XY.items():
        for k_y, xy_i in x_i.items():
            if xy_i != 0:
                MI += xy_i * math.log(xy_i/p_X_dict[k_x]/p_Y_dict[k_y])
    return MI

def XIY_conditional_entropy(XY, YIX, XIY, p_Y_dict=None):
    # calc XIY conditional entropy
    H_XIY = 0.0
    for k_x, x_i in XY.items():
        for k_y, xy_i in x_i.items():
            if xy_i != 0:
                H_XIY += xy_i * math.log(XIY[k_x][k_y])
    return -1 * H_XIY


def normalized_XIY_conditional_entropy(XY, YIX, XIY, p_Y_dict=None):
    # calc normalized XIY conditional entropy
    H_max = 0
    X_size, Y_size = len(XY), len(list(XY.values())[0])

    return XIY_conditional_entropy(XY, YIX, XIY, p_Y_dict) / H_max


def XIY_and_XY_p_Y(XY, YIX, XIY, p_Y_dict=None):
    return {"XIY": YIX.copy(), "XY": XY.copy(), "p_Y_dict": p_Y_dict.copy()}


#######################################################################################################################
# Custom Decorator function
def listToTuple(function):
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result

    return wrapper


@listToTuple
@lru_cache(maxsize=64)
def get_start_to_end_entropy(db, schemes):
    schemes_entropies_dict = {}
    for scheme in tqdm(schemes):
        schemes_entropies_dict[scheme] = get_all_entropies_from_scheme(db=db, scheme=scheme,
                                                                       entropy_func=XIY_and_XY_p_Y)

    ret_dict = {}
    for scheme, entropies in tqdm(schemes_entropies_dict.items()):
        # calc start to end entropy
        # schemes_entropies_dict "=" XIY    ,    YIZ
        XIY = entropies[0]["XIY"]
        XY = entropies[0]["XY"]
        for idx in range(1, len(entropies) - 1):
            YIZ = entropies[idx]["XIY"]
            YZ = entropies[idx]["XY"]
            Z_keys = list(YIZ.values())[0].keys()  # TODO check if correct

            XIZ = {k_x: {} for k_x, x_i in XIY.items()}
            for k_x, x_i in XIY.items():
                for k_z in Z_keys:
                    xIz_sigma = 0
                    for k_y, y_i in YIZ.items():
                        xIz_sigma += x_i[k_y] * y_i[k_z]
                    XIZ[k_x][k_z] = xIz_sigma

            p_Z_dict = entropies[idx]["p_Y_dict"]

            XZ = {k_x: {} for k_x, x_i in XIZ.items()}
            for k_x, x_i in XIZ.items():
                for k_z, z_i in x_i.items():
                    XZ[k_x][k_z] = XIZ[k_x][k_z] * p_Z_dict[k_z]

            XIY = XIZ
            XY = XZ

        # confusing name but add XIZ entropy
        ret_dict[scheme] = YIX_conditional_entropy(XY, XIY, None)

    return ret_dict


#######################################################################################################################

@lru_cache(maxsize=64)
def get_all_entropies_from_scheme(db, scheme, entropy_func):
    list_H_YIX = []
    if len(scheme.split("-")) == 1:  # scheme len is equal to 0 -> no edges -> no entropy calculate
        if db.predict_rel == 'dispat':
            scheme = 'm_id@dispat-m_id@dispat>sex@dispat' if scheme == 'dispat>sex@dispat' else 'm_id@dispat-m_id@dispat>age@dispat'
        elif db.predict_rel == 'molecule':
            # if scheme == 'molecule>ind1@molecule':
            #     scheme = 'molecule_id@molecule-molecule_id@molecule>ind1@molecule'
            # elif scheme == 'molecule>inda@molecule':
            #     scheme = 'molecule_id@molecule-molecule_id@molecule>inda@molecule'
            # elif scheme == 'molecule>logp@molecule':
            #     scheme = 'molecule_id@molecule-molecule_id@molecule>logp@molecule'
            # elif scheme == 'molecule>lumo@molecule':
            #     scheme = 'molecule_id@molecule-molecule_id@molecule>lumo@molecule'
            scheme = f"molecule_id@molecule-molecule_id@{scheme}"
        elif db.predict_rel == 'country':
            scheme = f"Continent@country-Continent@{scheme}"
        else:
            return None
    start_col, start_rel = scheme.split("-")[0].split("@")
    start_col = db.relations[start_rel][start_col]
    p_X_dict = dict(start_col.value_counts() / len(start_col))

    join_cols = [sub.split(">")[0] for sub in scheme.split("-")][1:]
    col_at_rels = [sub.split(">")[-1] for sub in scheme.split("-")]
    for idx in range(len(col_at_rels) - 1):
        # TODO: handle the last edge special case
        # compute joined_df of from_rel JOIN to_rel
        from_rel, to_rel = col_at_rels[idx], col_at_rels[idx + 1]
        from_col, from_rel = from_rel.split("@")
        to_col, to_rel = to_rel.split("@")
        from_df, to_df = db.relations[from_rel], db.relations[to_rel]
        join_to_col = join_cols[idx].split("@")[0]
        joined_df = from_df.merge(to_df, left_on=from_col, right_on=join_to_col, how='inner', suffixes=('_from', ''))
        # del joined_df[from_col+'_from']  # don't have to drop

        # calc Y|X matrix
        p_Y_dict = dict(joined_df[to_col].value_counts() / len(joined_df[to_col]))
        YIX = {}  # init Y|X matrix to zeros
        for x_i in p_X_dict.keys():
            temp = {}
            for y_i in p_Y_dict.keys():
                temp[y_i] = 0
            YIX[x_i] = temp.copy()
        from_col_map, to_col_map = db.get_col_as_map(f"{from_col}@{from_rel}"), db.get_col_as_map(f"{to_col}@{to_rel}")
        for to_tid, from_tid in db.get_arrow_as_map(f"{join_to_col}@{to_rel}", f"{from_col}@{from_rel}").items():
            x_i = from_col_map[from_tid]
            try:
                y_i = to_col_map[to_tid]
            except KeyError as null_val_in_to_rell:  # so there is no error in the graph -> no need to calc entropy
                # print(f"Try---{x_i}  {y_i}")
                i=0
                continue
            if x_i in p_X_dict.keys():
                YIX[x_i][y_i] += 1
            # else:
            #     YIX[x_i][y_i] = 1
        for _, x_i in YIX.items():  # divide by sum to get the probabilities (sum of each row is 1)
            sum_x_i = sum(x_i.values())
            if sum_x_i > 0:
                for k, y_i in x_i.items():
                    x_i[k] = y_i / sum_x_i

        # calc XY
        XY = {}  # XY = X*(Y|X) (each row)
        for k_x, x_i in YIX.items():
            temp, p_x_i = {}, p_X_dict[k_x]
            for k_y, y_i in x_i.items():
                if k_y in temp:
                    print("##### k_y in temp")
                temp[k_y] = y_i * p_x_i
            XY[k_x] = temp.copy()
        # if abs(1 - sum([sum([y_i for k_y, y_i in x_i.items()]) for k_x, x_i in XY.items()])) >= 10e-2:
        #     print(f"assert1- {sum([sum([y_i for k_y, y_i in x_i.items()]) for k_x, x_i in XY.items()])}!=1")
        # #assert abs(1 - sum([y_i for k_y, y_i in x_i.items() for k_x, x_i in XY.items()])) < 10e-2  # sanity check
        #
        # for k_y, p_y in p_Y_dict.items():  # sanity check
        #     yi = 0
        #     for k_x, x_i in XY.items():
        #         yi += x_i[k_y]
        #     if p_y - yi >= 10e-2:
        #         print(f"assert2- p_y={p_y} != {yi}=yi")
        #         #assert p_y - yi < 10e-2

        # calc X|Y matrix
        XIY = {}  # X|Y = Y*(XY) (each column)
        for k_x, x_i in XY.items():
            temp = {}
            for k_y, y_i in x_i.items():
                temp[k_y] = y_i / p_Y_dict[k_y]
            XIY[k_x] = temp.copy()

        # calc entropy
        list_H_YIX.append(entropy_func(XY, YIX, XIY, p_Y_dict))

        # continue to next iteration
        p_X_dict = p_Y_dict
    return list_H_YIX

@lru_cache(maxsize=64)
def get_all_MIs_from_scheme_optimized(db, scheme):
    list_MI = []
    if len(scheme.split("-")) == 1:  # scheme len is equal to 0 -> no edges -> no entropy calculate
        if db.predict_rel == 'dispat':
            scheme = 'm_id@dispat-m_id@dispat>sex@dispat' if scheme == 'dispat>sex@dispat' else 'm_id@dispat-m_id@dispat>age@dispat'
        elif db.predict_rel == 'molecule':
            scheme = f"molecule_id@molecule-molecule_id@{scheme}"
        elif db.predict_rel == 'country':
            scheme = f"Continent@country-Continent@{scheme}"
        else:
            return None
    start_col, start_rel = scheme.split("-")[0].split("@")
    start_col = db.relations[start_rel][start_col]
    p_X_dict = dict(start_col.value_counts() / len(start_col))

    join_cols = [sub.split(">")[0] for sub in scheme.split("-")][1:]
    col_at_rels = [sub.split(">")[-1] for sub in scheme.split("-")]
    for idx in range(len(col_at_rels) - 1):
        # compute joined_df of from_rel JOIN to_rel
        from_rel, to_rel = col_at_rels[idx], col_at_rels[idx + 1]
        from_col, from_rel = from_rel.split("@")
        to_col, to_rel = to_rel.split("@")
        from_df, to_df = db.relations[from_rel], db.relations[to_rel]
        join_to_col = join_cols[idx].split("@")[0]
        joined_df = from_df.merge(to_df, left_on=from_col, right_on=join_to_col, how='inner', suffixes=('_from', ''))
        # del joined_df[from_col+'_from']  # don't have to drop

        # calc Y|X matrix
        p_Y_dict = dict(joined_df[to_col].value_counts() / len(joined_df[to_col]))
        YIX = {}  # init Y|X matrix to zeros
        for x_i in p_X_dict.keys():
            temp = {}
            for y_i in p_Y_dict.keys():
                temp[y_i] = 0
            YIX[x_i] = temp.copy()
        from_col_map, to_col_map = db.get_col_as_map(f"{from_col}@{from_rel}"), db.get_col_as_map(f"{to_col}@{to_rel}")
        for to_tid, from_tid in db.get_arrow_as_map(f"{join_to_col}@{to_rel}", f"{from_col}@{from_rel}").items():
            x_i = from_col_map[from_tid]
            try:
                y_i = to_col_map[to_tid]
            except KeyError as null_val_in_to_rell:  # so there is no error in the graph -> no need to calc entropy
                # print(f"Try---{x_i}  {y_i}")
                continue
            if x_i in p_X_dict.keys():
                YIX[x_i][y_i] += 1
            # else:
            #     YIX[x_i][y_i] = 1
        ###
        XY = {}
        for k_x, x_i in YIX.items():  # divide by sum to get the probabilities (sum of each row is 1)
            sum_x_i = sum(x_i.values())
            temp, p_x_i = {}, p_X_dict[k_x]
            if sum_x_i > 0:
                for k_y, y_i in x_i.items():
                    x_i[k_y] = y_i / sum_x_i
                    temp[k_y] = x_i[k_y] * p_x_i
            XY[k_x] = temp.copy()

        # # calc XY
        # XY = {}  # XY = X*(Y|X) (each row)
        # for k_x, x_i in YIX.items():
        #     temp, p_x_i = {}, p_X_dict[k_x]
        #     for k_y, y_i in x_i.items():
        #         if k_y in temp:
        #             print("##### k_y in temp")
        #         temp[k_y] = y_i * p_x_i
        #     XY[k_x] = temp.copy()
        ###
        # if abs(1 - sum([sum([y_i for k_y, y_i in x_i.items()]) for k_x, x_i in XY.items()])) >= 10e-2:
        #     print(f"assert1- {sum([sum([y_i for k_y, y_i in x_i.items()]) for k_x, x_i in XY.items()])}!=1")
        # #assert abs(1 - sum([y_i for k_y, y_i in x_i.items() for k_x, x_i in XY.items()])) < 10e-2  # sanity check
        #
        # for k_y, p_y in p_Y_dict.items():  # sanity check
        #     yi = 0
        #     for k_x, x_i in XY.items():
        #         yi += x_i[k_y]
        #     if p_y - yi >= 10e-2:
        #         print(f"assert2- p_y={p_y} != {yi}=yi")
        #         #assert p_y - yi < 10e-2
        # # calc X|Y matrix
        # XIY = {}  # X|Y = Y*(XY) (each column)
        # for k_x, x_i in XY.items():
        #     temp = {}
        #     for k_y, y_i in x_i.items():
        #         temp[k_y] = y_i / p_Y_dict[k_y]
        #     XIY[k_x] = temp.copy()

        # calc entropy
        list_MI.append(mutual_information_optimized(XY, p_X_dict, p_Y_dict))

        # continue to next iteration
        p_X_dict = p_Y_dict
    return list_MI

def get_schemes_to_entropies_dict(db, schemes, entropy_func):
    schemes_entropies_dict = {}
    for scheme in tqdm(schemes):
        # schemes_entropies_dict[scheme] = get_all_entropies_from_scheme(db=db, scheme=scheme, entropy_func=YIX_conditional_entropy)
        schemes_entropies_dict[scheme] = get_all_entropies_from_scheme(db=db, scheme=scheme, entropy_func=entropy_func)
    return schemes_entropies_dict

def get_schemes_to_MI_dict_optimized(db, schemes):
    schemes_entropies_dict = {}
    for scheme in tqdm(schemes):
        schemes_entropies_dict[scheme] = get_all_MIs_from_scheme_optimized(db=db, scheme=scheme)
    return schemes_entropies_dict

def sorted_dict_by_max_value_in_list(dct):
    max_dct = {k: max(v) for k, v in dct.items()}
    return {k: v for k, v in sorted(max_dct.items(), key=lambda item: item[1])}

def sorted_dict_by_min_value_in_list(dct):
    max_dct = {k: min(v) for k, v in dct.items()}
    return {k: v for k, v in sorted(max_dct.items(), key=lambda item: item[1])}

if __name__ == '__main__':
    data_path = f'Datasets/mondial'
    db = Database.load_csv(data_path)
    tuples = [r for _, r, _ in db.iter_rows(db.predict_rel)]
    depth = 3
    scheme_tuple_map = list(db.scheme_tuple_map(db.predict_rel, tuples, depth).keys())
    # -# scheme ends in a column and not in a table.
    schemes = []
    for scheme in scheme_tuple_map:
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                schemes.append(f"{scheme}>{col_id}")

    # mutual information
    x=time.time()
    schemes_mutual_inf_dict = get_schemes_to_entropies_dict(db, schemes, mutual_information)
    ordered_schemes1 = sorted_dict_by_min_value_in_list(schemes_mutual_inf_dict)
    print(f"ordered_schemes1-----{time.time()-x}")
    x=time.time()
    schemes_mutual_inf_dict2 = get_schemes_to_MI_dict_optimized(db, schemes)
    ordered_schemes2 = sorted_dict_by_min_value_in_list(schemes_mutual_inf_dict)
    print(f"ordered_schemes2-----{time.time()-x}")
    print(f"ordered_schemes1-----{ordered_schemes1}")
    print(f"ordered_schemes2-----{ordered_schemes2}")
    print(schemes_mutual_inf_dict)
    exit()
    # end mutual information

    # entropy
    # scheme = f"Country@target-Code@country>Code@country-Country@religion>religion>Percentage@religion"
    # Country@target-Code@country>Code@country-Country1@borders>borders>Length@borders
    # Country@target              Code@country                          Length@borders
    schemes_entropies_dict = get_schemes_to_entropies_dict(db, schemes, XIY_conditional_entropy)
    ordered_schemes1 = sorted_dict_by_min_value_in_list(schemes_entropies_dict)
    print(schemes_entropies_dict)
    print([max(entropies) for schem, entropies in schemes_entropies_dict.items()])

    # indexed_sorted_schemes_by_highest_entropy_in_one_of_the_edges:
    indexed_sorted_schemes_by_highest_entropy = {(idx, k): v for idx, (k, v) in enumerate(ordered_schemes.items())}
    print(indexed_sorted_schemes_by_highest_entropy)
    print("END")
    # schemes = schemes[:10]
    # start to end entropy:
    # for i in range(10):
    #     print(i)
    #     start2end_entropies = get_start_to_end_entropy(db, schemes)
    #     ordered_start2end_schemes = {k: v for k, v in sorted(start2end_entropies.items(), key=lambda item: item[1])}
    #     indxed_start2end_schemes = {(idx, k): v for idx, (k, v) in enumerate(ordered_start2end_schemes.items())}
    # print("END")
