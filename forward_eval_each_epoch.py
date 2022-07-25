import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from db_utils import Database
from io_utils import save_dict
import io_utils
import pickle
import mmd_utils
import ek_utlis
import random
import time
from time import strftime
from time import gmtime
import entropy_utils as entropy
import json
from datetime import datetime
import pandas as pd
from functools import reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eps = 1e-15
kernels = {'object': lambda x, y: x == y,
           'int64': lambda x, y: x == y,
           'float64': lambda x, y: np.exp(
               - ((x - y) ** 2) / (2 * ((0.05 * np.maximum(np.abs(x), np.abs(y)) + eps) ** 2)))}

kernels_dict = {}


class Forward(torch.nn.Module):

    def __init__(self, dim, num_schemes, row_idx, scheme_idx):
        super().__init__()

        self.dim = dim
        self.num_relations = num_schemes
        self.row_idx = np.vectorize(lambda x: row_idx[x])
        self.scheme_idx = scheme_idx
        self.num_tuples = len(row_idx)

        self.x = torch.nn.Parameter(torch.Tensor(self.num_tuples, dim))
        torch.nn.init.normal_(self.x, std=np.sqrt(1.0 / dim))
        self.A = torch.nn.Parameter(torch.Tensor(self.num_relations, dim, dim))
        torch.nn.init.normal_(self.A)

    def forward(self, pairs, scheme_idx):
        A_sym = (self.A + torch.transpose(self.A, 2, 1)) / 2
        A = A_sym[scheme_idx]
        # -# converted tensor indices to tensors because it caused an error
        x_v = self.x[pairs[:, 0].long()].view(-1, 1, self.dim)
        x_u = self.x[pairs[:, 1].long()].view(-1, 1, self.dim)
        y = (x_v.matmul(A) * x_u).sum(dim=(1, 2))
        return y

    def loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def get_embedding(self):
        return self.x.cpu().data.numpy()

    def infer(self, old_idx, scheme_idx, y):
        old_idx = old_idx.to(device)
        scheme_idx = scheme_idx.to(device)
        y = y.to(device)

        with torch.no_grad():
            A_sym = (self.A + torch.transpose(self.A, 2, 1)) / 2
            A = A_sym[scheme_idx]
            x_old = self.x[old_idx]
            A_stack = A.matmul(x_old.reshape(-1, self.dim, 1)).view(-1, self.dim)
            A_inv = torch.pinverse(A_stack)
            x = A_inv.matmul(y)

        return x.cpu().detach().numpy()


def train(model, loader, epochs):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(epochs):
        bar = tqdm(desc=f'Epoch {e + 1} Mean Loss: _')
        bar.reset(total=len(loader))

        epoch_losses = []
        for (pairs, vals, scheme) in loader:
            pairs, vals, scheme = pairs.to(device), vals.to(device), scheme.to(device)

            opt.zero_grad()
            y_pred = model(pairs, scheme)  # forward pass
            loss = model.loss(y_pred, vals)
            loss.backward()
            opt.step()

            epoch_losses.append(loss.cpu().detach().numpy())
            bar.set_description(desc=f'Epoch {e + 1} Mean Loss: {epoch_losses[-1]:.4f}')
            bar.update()

            # embedding = model.get_embedding()
            # embedding = {r: embedding[i] for r, i in row_idx.items()}
            # io_utils.save_embedding(model_dir.replace("models", "Embeddings"), embedding)

        bar.close()

    return model


def get_samples(db, depth, num_samples, sample_fct, yuval_change=''):
    # -# depth is used here
    tuples = [r for _, r, _ in db.iter_rows(db.predict_rel)]
    scheme_tuple_map = db.scheme_tuple_map(db.predict_rel, tuples, depth)

    # import json
    # scheme_tuple_map['target'] = {k: list(v) for k, v in scheme_tuple_map['target'].items()}
    # with open("scheme_tuple_map.json", "w") as f:
    #     json.dump(dict(scheme_tuple_map), f)

    # -# cut half of the schemes
    #scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 2+1} #if len(key.split(">")) == 4}
    # random.seed(0)
    # rand_list = random.sample(range(30), 12)
    #scheme_tuple_map = {key: scheme_tuple_map[key] for key in [list(scheme_tuple_map.keys())[idx] for idx in [27, 12, 24, 13, 1, 8, 16, 15, 28, 9, 22, 11]]}
    # try try
    #include_keys = list(scheme_tuple_map.keys())[::2] #take every second key only
    #scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if key in include_keys}
    # try try
    """if yuval_change != '':
        random.seed(0)
        rand_num = yuval_change.split('random')[-1].split('_')[0]
        if rand_num != '' and 'random' in yuval_change:
            for i in range(int(rand_num)):
                rand_list = random.sample(range(11), 8)
            scheme_tuple_map = {key: scheme_tuple_map[key] for key in [list(scheme_tuple_map.keys())[idx] for idx in rand_list]}
    if yuval_change == 'scheme_len_eq_1_2':
        scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) <= 2+1}
    elif yuval_change == 'scheme_len_eq_1_3':
        scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 1+1 or len(key.split(">")) == 3+1}
    elif yuval_change == 'scheme_len_eq_1_4':
        scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 1+1 or len(key.split(">")) == 4+1}
    elif yuval_change == 'scheme_len_eq_1':
        scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 1+1}
    elif yuval_change == 'scheme_len_eq_2':
        scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 2+1}
    elif yuval_change == 'scheme_len_eq_3':
        scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 3+1}
    elif yuval_change == 'scheme_len_eq_4':
        scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 4+1}
    elif 'specific' in yuval_change:
        scheme_tuple_map = {key: scheme_tuple_map[key] for key in [list(scheme_tuple_map.keys())[idx] for idx in [int(yuval_change[8:])]]}
    print(f'scheme_tuple_map: \n{list(scheme_tuple_map.keys())}')
    for l in range(1,6):
        num_schemes = len({key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == l+1})
        print(f"{num_schemes} schemes with {l} length")
    """

    # -# scheme ends in a column and not in a table.
    subset_full_schemes = []
    for scheme, tuple_map in tqdm(scheme_tuple_map.items()):
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                subset_full_schemes.append(f"{scheme}>{col_id}")

    # ------- peek a subset of the full schemes
    if yuval_change == '':
        pass
    elif 'random' in yuval_change:
        random.seed(0)
        rand_num = yuval_change.split('random')[-1].split('_')[0]
        if rand_num != '':
            for i in range(int(rand_num)):
                rand_list = random.sample(range(63), 30)
            subset_full_schemes = [subset_full_schemes[idx] for idx in rand_list]
    elif 'r6andom' in yuval_change:
        random.seed(0)
        rand_num = yuval_change.split('r6andom')[-1].split('_')[0]
        if rand_num != '':
            for i in range(6):
                rand_list = random.sample(range(63), 30)
            subset_full_schemes = [subset_full_schemes[idx] for idx in rand_list]
            for i in range(int(rand_num)):
                rand_list = random.sample(range(30), 20)
            subset_full_schemes = [subset_full_schemes[idx] for idx in rand_list]
    elif 'r6_25andom' in yuval_change:
        random.seed(0)
        rand_num = yuval_change.split('r6_25andom')[-1].split('_')[0]
        if rand_num != '':
            for i in range(6):
                rand_list = random.sample(range(63), 30)
            subset_full_schemes = [subset_full_schemes[idx] for idx in rand_list]
            for i in range(int(rand_num)):
                rand_list = random.sample(range(30), 25)
            subset_full_schemes = [subset_full_schemes[idx] for idx in rand_list]
    elif yuval_change == 'EC_scheme_len_eq_1':
        subset_full_schemes = [scheme for scheme in subset_full_schemes if len(scheme.split("-")) <= 1 + 1]
    elif yuval_change == 'EC_scheme_len_eq_2':
        subset_full_schemes = [scheme for scheme in subset_full_schemes if len(scheme.split("-")) == 2 + 1]
    elif yuval_change == 'EC_scheme_len_eq_3':
        subset_full_schemes = [scheme for scheme in subset_full_schemes if len(scheme.split("-")) == 3+1]
    elif 'r6_manual_reduce' in yuval_change:
        exp_num, rand_list = int(yuval_change.split('r6_manual_reduce')[-1]), []
        random.seed(0)
        for i in range(6):
            rand_list = random.sample(range(63), 30)
        subset_full_schemes = [subset_full_schemes[idx] for idx in rand_list]
        if exp_num == 1:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in remove_from_r6_to_reduce]
        elif exp_num == 2:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[0]]]
        elif exp_num == 3:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[1]]]
        elif exp_num == 4:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[2]]]
        elif exp_num == 5:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[3]]]
        elif exp_num == 6:
            subset_full_schemes = [subset_full_schemes[idx] for idx in random.sample(range(30), 29)]
        elif exp_num == 7:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[1:]]] # 1+2+3
        elif exp_num == 8:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country@province>Country@province-Country@city>city>Latitude@city"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in remove_from_r6_to_reduce]
        elif exp_num == 9:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[1:3]]] # 1+2
        elif exp_num == 10:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[2:]]] # 2+3
        elif exp_num == 11:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[1]]+[remove_from_r6_to_reduce[3]]] # 1+3
        elif exp_num == 12:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country1@borders>borders>Length@borders","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in remove_from_r6_to_reduce] # boreder_len + 1+2+3
        elif exp_num == 13:
            remove_from_r6_to_reduce = ["Country@target-Code@country>Code@country-Country1@borders>borders>Length@borders","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country","Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country","Country@target-Code@country>Code@country-Country@religion>religion>Percentage@religion"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in remove_from_r6_to_reduce] # boreder_len + 1+2+3
        elif exp_num in range(100,110):
            for i in range(100,exp_num):
                random.sample(range(30), 29)
            subset_full_schemes = [subset_full_schemes[idx] for idx in random.sample(range(30), 29)]
    elif 'conditional_entropy_removed_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('conditional_entropy_removed_')[-1])
        ordered_schemes = entropy.sorted_dict_by_max_value_in_list(entropy.get_schemes_to_entropies_dict(db, subset_full_schemes, entropy.XIY_conditional_entropy))
        subset_full_schemes = [scheme for scheme, v in ordered_schemes.items()][:-schemes_to_remove]
    elif 'conditional_entropy_from_low_to_high_removed_and_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('conditional_entropy_from_low_to_high_removed_and_')[-1].split("_")[0])
        rest = [int(num) for num in yuval_change.split('conditional_entropy_from_low_to_high_removed_and_')[-1].split("_")[1:]]
        ordered_schemes = entropy.sorted_dict_by_max_value_in_list(entropy.get_schemes_to_entropies_dict(db, subset_full_schemes, entropy.XIY_conditional_entropy))
        subset_full_schemes = [val for idx, val in enumerate([scheme for scheme, v in ordered_schemes.items()]) if idx not in list(range(schemes_to_remove))+rest]
    elif 'conditional_entropy_from_low_to_high_removed_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('conditional_entropy_from_low_to_high_removed_')[-1])
        ordered_schemes = entropy.sorted_dict_by_max_value_in_list(entropy.get_schemes_to_entropies_dict(db, subset_full_schemes, entropy.XIY_conditional_entropy))
        subset_full_schemes = [scheme for scheme, v in ordered_schemes.items()][schemes_to_remove:]
    elif 'conditional_entropy_start_to_end_high_to_low_removed_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('conditional_entropy_start_to_end_high_to_low_removed_')[-1])
        ordered_schemes = entropy.get_start_to_end_entropy(db, subset_full_schemes)
        ordered_schemes = {k: v for k, v in sorted(ordered_schemes.items(), key=lambda item: item[1])}
        subset_full_schemes = [scheme for scheme, v in ordered_schemes.items()][:-schemes_to_remove]
    elif 'conditional_entropy_start_to_end_low_to_high_removed_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('conditional_entropy_start_to_end_low_to_high_removed_')[-1])
        ordered_schemes = entropy.get_start_to_end_entropy(db, subset_full_schemes)
        ordered_schemes = {k: v for k, v in sorted(ordered_schemes.items(), key=lambda item: item[1])}
        subset_full_schemes = [scheme for scheme, v in ordered_schemes.items()][schemes_to_remove:]
    elif 'specific_entropy_' in yuval_change:
        scheme_to_remain = int(yuval_change.split('specific_entropy_')[-1])
        ordered_schemes = entropy.sorted_dict_by_max_value_in_list(entropy.get_schemes_to_entropies_dict(db, subset_full_schemes, entropy.XIY_conditional_entropy))
        subset_full_schemes = [scheme for scheme, v in ordered_schemes.items()][scheme_to_remain:scheme_to_remain+1]
    elif 'remove_longest_schemes_' in yuval_change:
        scheme_to_remove = int(yuval_change.split('remove_longest_schemes_')[-1])
        ordered_schemes = sorted([(len(scheme.split("-")),scheme) for scheme in subset_full_schemes],reverse=True)
        subset_full_schemes = [scheme for l, scheme in ordered_schemes][scheme_to_remove:]
    elif 'Randomly_42seed_remove_schemes_' in yuval_change:
        scheme_to_remove = int(yuval_change.split('Randomly_42seed_remove_schemes_')[-1])
        ordered_schemes = sorted([(len(scheme.split("-")),scheme) for scheme in subset_full_schemes],reverse=True)
        subset_full_schemes = [scheme for l, scheme in ordered_schemes][scheme_to_remove:]
    elif 'sorted_by_loss_after_1_epoch_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('sorted_by_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [39, 41, 44, 40, 42, 43, 38, 59, 60, 57, 35, 34, 33, 36, 37, 58, 29, 16, 10, 6, 8, 12, 7, 14, 13, 15, 9, 5, 11,
         30, 24, 49, 0, 4, 47, 1, 2, 50, 3, 17, 28, 25, 27, 21, 45, 46, 31, 55, 56, 52, 62, 53, 18, 51, 54, 19, 32, 20,
         61, 23, 48, 26, 22]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_by_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_by_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [39, 41, 44, 40, 42, 43, 38, 59, 60, 57, 35, 34, 33, 36, 37, 58, 29, 16, 10, 6, 8, 12, 7, 14, 13, 15, 9, 5, 11,
         30, 24, 49, 0, 4, 47, 1, 2, 50, 3, 17, 28, 25, 27, 21, 45, 46, 31, 55, 56, 52, 62, 53, 18, 51, 54, 19, 32, 20,
         61, 23, 48, 26, 22][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_by_highest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_by_highest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [7, 61, 62, 36, 48, 11, 41, 40, 54, 51, 37, 56, 53, 39, 55, 52, 46, 42, 10, 43, 44, 9, 8, 38, 3, 6, 45, 47, 2, 1, 50, 49, 4, 0, 28, 34, 30, 24, 5, 33, 27, 25, 26, 31, 35, 29, 32, 13, 15, 14, 16, 12, 58, 57, 17, 59, 60, 23, 20, 19, 22, 18, 21]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_by_lowest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_by_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [7, 61, 62, 36, 48, 11, 41, 40, 54, 51, 37, 56, 53, 39, 55, 52, 46, 42, 10, 43, 44, 9, 8, 38, 3, 6, 45, 47, 2, 1, 50, 49, 4, 0, 28, 34, 30, 24, 5, 33, 27, 25, 26, 31, 35, 29, 32, 13, 15, 14, 16, 12, 58, 57, 17, 59, 60, 23, 20, 19, 22, 18, 21][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]

    elif 'sorted_correct_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 23, 61, 20, 32, 19, 54, 51, 18, 53, 62, 52, 56, 55, 31, 46, 45, 21, 27, 25, 28, 17, 3, 50, 2, 1, 47, 4, 0, 49, 24, 30, 11, 5, 9, 15, 13, 14, 7, 12, 8, 6, 10, 16, 29, 58, 37, 36, 33, 34, 35, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_correct_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 23, 61, 20, 32, 19, 54, 51, 18, 53, 62, 52, 56, 55, 31, 46, 45, 21, 27, 25, 28, 17, 3, 50, 2, 1, 47, 4, 0, 49, 24, 30, 11, 5, 9, 15, 13, 14, 7, 12, 8, 6, 10, 16, 29, 58, 37, 36, 33, 34, 35, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_correct_highest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_highest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [26, 22, 23, 61, 48, 32, 20, 51, 54, 62, 19, 53, 56, 18, 52, 55, 25, 45, 21, 31, 46, 28, 27, 17, 3, 30, 24, 50, 2, 47, 49, 4, 1, 0, 9, 11, 15, 5, 14, 13, 8, 29, 12, 7, 6, 10, 16, 36, 37, 33, 35, 34, 58, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_correct_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [26, 22, 23, 61, 48, 32, 20, 51, 54, 62, 19, 53, 56, 18, 52, 55, 25, 45, 21, 31, 46, 28, 27, 17, 3, 30, 24, 50, 2, 47, 49, 4, 1, 0, 9, 11, 15, 5, 14, 13, 8, 29, 12, 7, 6, 10, 16, 36, 37, 33, 35, 34, 58, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_genes_correct_highest_loss_after_10_epoch' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_correct_highest_loss_after_10_epoch')[-1])
        ordered_schemes_idxs = [6, 0, 28, 16, 22, 7, 10, 8, 5, 20, 3, 30, 18, 29, 25, 17, 13, 27, 15, 2, 12, 21, 24, 9, 31, 1, 19, 11, 23, 4, 26, 14]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_genes_correct_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_correct_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [6, 0, 28, 16, 22, 7, 10, 8, 5, 20, 3, 30, 18, 29, 25, 17, 13, 27, 15, 2, 12, 21, 24, 9, 31, 1, 19, 11, 23, 4, 26, 14][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    #TODO run them:
    elif 'sorted_genes_mean_highest_loss_after_1_epoch' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_mean_highest_loss_after_1_epoch')[-1])
        ordered_schemes_idxs = [0, 7, 6, 10, 22, 8, 20, 5, 18, 17, 29, 30, 28, 16, 3, 25, 2, 15, 27, 13, 21, 9, 12, 24, 1, 4, 31, 19, 23, 11, 14, 26]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_genes_mean_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_mean_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [0, 7, 6, 10, 22, 8, 20, 5, 18, 17, 29, 30, 28, 16, 3, 25, 2, 15, 27, 13, 21, 9, 12, 24, 1, 4, 31, 19, 23, 11, 14, 26][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_genes_mean_highest_loss_after_10_epoch' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_mean_highest_loss_after_10_epoch')[-1])
        ordered_schemes_idxs = [30, 18, 28, 6, 16, 22, 10, 20, 8, 25, 13, 3, 29, 0, 17, 27, 7, 15, 5, 11, 23, 24, 12, 1, 2, 31, 19, 21, 9, 26, 14, 4]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_genes_mean_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_mean_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [30, 18, 28, 6, 16, 22, 10, 20, 8, 25, 13, 3, 29, 0, 17, 27, 7, 15, 5, 11, 23, 24, 12, 1, 2, 31, 19, 21, 9, 26, 14, 4][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]

    elif 'sorted_hepatitis_mean_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_hepatitis_mean_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [7, 0, 6, 8, 9, 5, 1, 12, 3, 4, 2, 13, 10, 11, 14]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_hepatitis_mean_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_hepatitis_mean_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [7, 0, 6, 8, 9, 5, 1, 12, 3, 4, 2, 13, 10, 11, 14][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_hepatitis_mean_highest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_hepatitis_mean_highest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [8, 5, 6, 7, 12, 13, 4, 9, 10, 11, 3, 2, 0, 1, 14]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_hepatitis_mean_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_hepatitis_mean_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [8, 5, 6, 7, 12, 13, 4, 9, 10, 11, 3, 2, 0, 1, 14][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mutagenesis_mean_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mutagenesis_mean_highest_loss_after_1_epoch_')[-1])
        # ordered_schemes_idxs = [1, 12, 0, 13, 7, 11, 8, 4, 9, 5, 14, 3, 2, 6, 10]
        ordered_schemes_idxs = [10, 6, 25, 26, 2, 18, 3, 13, 14, 30, 22, 5, 17, 9, 4, 8, 16, 21, 7, 19, 15, 29, 11, 0,
                                23, 27, 20, 28, 24, 1, 12][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mutagenesis_mean_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mutagenesis_mean_lowest_loss_after_1_epoch_')[-1])
        # ordered_schemes_idxs = [1, 12, 0, 13, 7, 11, 8, 4, 9, 5, 14, 3, 2, 6, 10][::-1]
        ordered_schemes_idxs = [10, 6, 25, 26, 2, 18, 3, 13, 14, 30, 22, 5, 17, 9, 4, 8, 16, 21, 7, 19, 15, 29, 11, 0,
                                23, 27, 20, 28, 24, 1, 12]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mutagenesis_mean_highest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mutagenesis_mean_highest_loss_after_10_epoch_')[-1])
        # ordered_schemes_idxs = [12, 13, 7, 11, 8, 4, 1, 0, 5, 9, 14, 3, 6, 2, 10]
        ordered_schemes_idxs = [25, 2, 13, 10, 6, 18, 26, 3, 14, 30, 22, 23, 11, 24, 1, 0, 12, 9, 17, 5, 28, 4, 8, 21,
                                16, 7, 19, 20, 29, 15, 27][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mutagenesis_mean_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mutagenesis_mean_lowest_loss_after_10_epoch_')[-1])
        # ordered_schemes_idxs = [12, 13, 7, 11, 8, 4, 1, 0, 5, 9, 14, 3, 6, 2, 10][::-1]
        ordered_schemes_idxs = [25, 2, 13, 10, 6, 18, 26, 3, 14, 30, 22, 23, 11, 24, 1, 0, 12, 9, 17, 5, 28, 4, 8, 21,
                                16, 7, 19, 20, 29, 15, 27]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_world_mean_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_world_mean_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [18, 9, 5, 7, 1, 19, 3, 17, 10, 12, 2, 6, 15, 16, 14, 13, 11, 0, 4, 8]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_world_mean_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_world_mean_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [18, 9, 5, 7, 1, 19, 3, 17, 10, 12, 2, 6, 15, 16, 14, 13, 11, 0, 4, 8][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_world_mean_highest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_world_mean_highest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [18, 9, 5, 1, 7, 19, 3, 17, 10, 6, 2, 15, 12, 16, 14, 11, 13, 4, 0, 8]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_world_mean_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_world_mean_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [18, 9, 5, 1, 7, 19, 3, 17, 10, 6, 2, 15, 12, 16, 14, 11, 13, 4, 0, 8][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]

    elif 'sorted_mondial_mean_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mondial_mean_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 38, 30, 23, 61, 20, 37, 32, 36, 33, 19, 34, 54, 35, 51, 18, 53, 62, 52, 56, 55, 25, 31, 29, 46, 45, 5, 28, 27, 11, 21, 9, 15, 3, 13, 17, 14, 50, 24, 58, 7, 8, 2, 12, 1, 47, 6, 4, 0, 49, 10, 16, 60, 57, 59, 43, 42, 40, 44, 41, 39]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mondial_mean_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mondial_mean_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 38, 30, 23, 61, 20, 37, 32, 36, 33, 19, 34, 54, 35, 51, 18, 53, 62, 52, 56, 55, 25, 31, 29, 46, 45, 5, 28, 27, 11, 21, 9, 15, 3, 13, 17, 14, 50, 24, 58, 7, 8, 2, 12, 1, 47, 6, 4, 0, 49, 10, 16, 60, 57, 59, 43, 42, 40, 44, 41, 39][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mean_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mean_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 38, 30, 23, 61, 20, 37, 32, 36, 33, 19, 34, 54, 35, 51, 18, 53, 62, 52, 56, 55, 25, 31, 29, 46, 45, 5, 28, 27, 11, 21, 9, 15, 3, 13, 17, 14, 50, 24, 58, 7, 8, 2, 12, 1, 47, 6, 4, 0, 49, 10, 16, 60, 57, 59, 43, 42, 40, 44, 41, 39]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mean_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mean_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 38, 30, 23, 61, 20, 37, 32, 36, 33, 19, 34, 54, 35, 51, 18, 53, 62, 52, 56, 55, 25, 31, 29, 46, 45, 5, 28, 27, 11, 21, 9, 15, 3, 13, 17, 14, 50, 24, 58, 7, 8, 2, 12, 1, 47, 6, 4, 0, 49, 10, 16, 60, 57, 59, 43, 42, 40, 44, 41, 39][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mean_highest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mean_highest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [26, 22, 23, 61, 48, 30, 32, 20, 54, 51, 62, 19, 53, 25, 56, 36, 37, 52, 18, 55, 33, 35, 38, 34, 29, 45, 21, 31, 46, 9, 11, 15, 5, 28, 27, 24, 14, 8, 13, 17, 7, 12, 6, 3, 10, 16, 50, 2, 47, 4, 49, 1, 0, 58, 60, 57, 43, 59, 42, 40, 44, 41, 39]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mean_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mean_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [26, 22, 23, 61, 48, 30, 32, 20, 54, 51, 62, 19, 53, 25, 56, 36, 37, 52, 18, 55, 33, 35, 38, 34, 29, 45, 21, 31, 46, 9, 11, 15, 5, 28, 27, 24, 14, 8, 13, 17, 7, 12, 6, 3, 10, 16, 50, 2, 47, 4, 49, 1, 0, 58, 60, 57, 43, 59, 42, 40, 44, 41, 39][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]

    # f"low_loss_tryout_{data_name}_try_out_mul{mul_by}"
    elif 'low_loss_tryout_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('low_loss_tryout_')[-1].split("_")[-1])
        ordered_schemes_file_name = "_".join(yuval_change.split('low_loss_tryout_')[-1].split("_")[:-1])
        with open(os.path.join("Sorted_schemes", f"{ordered_schemes_file_name}.txt"), 'r') as f:
            d = json.load(f)
            ordered_schemes_idxs, ordered_schemes_dict = d['Ordered_schemes'], d['Dict']
            #TODO: if the lowest schemes have 0 loss dont remove them
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]

    elif 'sorted_mean_lowest_loss_after_tryout_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_mean_lowest_loss_after_tryout_')[-1].split("_")[-1])
        ordered_schemes_file_name = "_".join(yuval_change.split('sorted_mean_lowest_loss_after_tryout_')[-1].split("_")[:-1])
        with open(os.path.join("Sorted_schemes", f"{ordered_schemes_file_name}.txt"), 'r') as f:
            d = json.load(f)
            ordered_schemes_idxs, ordered_schemes_dict = d['Ordered_schemes'], d['Dict']
            #TODO: if the lowest schemes have 0 loss dont remove them
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_mean_lowest_loss_after_SnR_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('sorted_mean_lowest_loss_after_SnR_')[-1].split("_")[-1])
        ordered_schemes_file_name = "_".join(yuval_change.split('sorted_mean_lowest_loss_after_SnR_')[-1].split("_")[:-1])
        with open(os.path.join("Sorted_schemes", f"{ordered_schemes_file_name}.txt"), 'r') as f:
            d = json.load(f)
            ordered_schemes_idxs, ordered_schemes_dict = d['Ordered_schemes'], d['Dict']
            # TODO: if the lowest schemes have 0 loss dont remove them
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'low_loss_SnR_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('low_loss_SnR_')[-1].split("_")[-1])
        ordered_schemes_file_name = "_".join(yuval_change.split('low_loss_SnR_')[-1].split("_")[:-1])
        with open(os.path.join("Sorted_schemes", f"{ordered_schemes_file_name}.txt"), 'r') as f:
            d = json.load(f)
            ordered_schemes_idxs, ordered_schemes_dict = d['Ordered_schemes'], d['Dict']
            # TODO: if the lowest schemes have 0 loss dont remove them
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_d4_mean_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_d4_mean_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [128, 140, 141, 127, 144, 122, 146, 126, 138, 147, 121, 124, 123, 164, 184, 420, 334, 180, 429, 135, 321, 186, 212, 285, 335, 418, 428, 187, 241, 197, 419, 322, 210, 173, 283, 410, 421, 411, 172, 272, 298, 332, 244, 163, 255, 426, 353, 416, 296, 299, 177, 400, 381, 59, 200, 301, 213, 232, 273, 264, 286, 31, 376, 387, 199, 168, 243, 29, 263, 220, 32, 152, 252, 179, 270, 342, 161, 223, 309, 303, 341, 62, 155, 308, 366, 389, 61, 0, 230, 4, 233, 159, 222, 304, 1, 154, 171, 2, 339, 162, 261, 402, 306, 365, 170, 403, 319, 254, 10, 424, 354, 405, 129, 116, 125, 38, 392, 6, 363, 36, 390, 142, 352, 40, 378, 8, 7, 259, 3, 245, 114, 225, 414, 37, 326, 48, 423, 204, 191, 72, 65, 370, 195, 317, 215, 425, 330, 190, 413, 374, 313, 346, 350, 234, 117, 357, 23, 396, 12, 268, 281, 203, 312, 75, 76, 42, 53, 325, 35, 115, 277, 18, 415, 39, 229, 9, 290, 294, 5, 208, 219, 377, 409, 331, 69, 151, 260, 269, 239, 289, 228, 395, 386, 218, 139, 250, 181, 175, 247, 11, 318, 236, 46, 150, 49, 196, 369, 16, 276, 408, 399, 240, 174, 41, 149, 327, 182, 57, 19, 21, 209, 251, 56, 282, 90, 88, 192, 58, 136, 246, 345, 385, 28, 295, 351, 27, 375, 314, 257, 235, 51, 398, 205, 266, 291, 278, 143, 393, 406, 382, 73, 347, 13, 183, 176, 105, 338, 362, 380, 371, 64, 358, 361, 337, 34, 43, 431, 166, 356, 216, 106, 158, 14, 157, 226, 167, 86, 84, 77, 87, 98, 214, 391, 44, 15, 224, 66, 404, 133, 45, 85, 256, 78, 384, 336, 60, 360, 156, 165, 397, 265, 95, 93, 94, 30, 67, 333, 96, 349, 320, 91, 80, 68, 83, 305, 92, 293, 373, 300, 280, 198, 207, 82, 383, 97, 211, 119, 359, 316, 100, 329, 242, 194, 81, 253, 112, 103, 24, 113, 430, 118, 111, 110, 99, 178, 22, 368, 148, 63, 344, 71, 120, 79, 185, 271, 33, 262, 284, 417, 427, 89, 288, 132, 315, 275, 297, 54, 328, 52, 348, 131, 193, 249, 238, 372, 206, 108, 101, 323, 137, 279, 292, 310, 189, 26, 202, 102, 130, 311, 237, 324, 364, 407, 248, 201, 340, 394, 227, 302, 267, 258, 188, 217, 231, 160, 307, 169, 153, 55, 343, 221, 388, 274, 287, 379, 367, 401, 109, 355, 145, 107, 104, 422, 74, 20, 50, 412, 134, 25, 17, 70, 47][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_d4_mean_lowest_loss_after_1_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_d4_mean_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [128, 140, 141, 127, 144, 122, 146, 126, 138, 147, 121, 124, 123, 164, 184, 420, 334, 180, 429, 135, 321, 186, 212, 285, 335, 418, 428, 187, 241, 197, 419, 322, 210, 173, 283, 410, 421, 411, 172, 272, 298, 332, 244, 163, 255, 426, 353, 416, 296, 299, 177, 400, 381, 59, 200, 301, 213, 232, 273, 264, 286, 31, 376, 387, 199, 168, 243, 29, 263, 220, 32, 152, 252, 179, 270, 342, 161, 223, 309, 303, 341, 62, 155, 308, 366, 389, 61, 0, 230, 4, 233, 159, 222, 304, 1, 154, 171, 2, 339, 162, 261, 402, 306, 365, 170, 403, 319, 254, 10, 424, 354, 405, 129, 116, 125, 38, 392, 6, 363, 36, 390, 142, 352, 40, 378, 8, 7, 259, 3, 245, 114, 225, 414, 37, 326, 48, 423, 204, 191, 72, 65, 370, 195, 317, 215, 425, 330, 190, 413, 374, 313, 346, 350, 234, 117, 357, 23, 396, 12, 268, 281, 203, 312, 75, 76, 42, 53, 325, 35, 115, 277, 18, 415, 39, 229, 9, 290, 294, 5, 208, 219, 377, 409, 331, 69, 151, 260, 269, 239, 289, 228, 395, 386, 218, 139, 250, 181, 175, 247, 11, 318, 236, 46, 150, 49, 196, 369, 16, 276, 408, 399, 240, 174, 41, 149, 327, 182, 57, 19, 21, 209, 251, 56, 282, 90, 88, 192, 58, 136, 246, 345, 385, 28, 295, 351, 27, 375, 314, 257, 235, 51, 398, 205, 266, 291, 278, 143, 393, 406, 382, 73, 347, 13, 183, 176, 105, 338, 362, 380, 371, 64, 358, 361, 337, 34, 43, 431, 166, 356, 216, 106, 158, 14, 157, 226, 167, 86, 84, 77, 87, 98, 214, 391, 44, 15, 224, 66, 404, 133, 45, 85, 256, 78, 384, 336, 60, 360, 156, 165, 397, 265, 95, 93, 94, 30, 67, 333, 96, 349, 320, 91, 80, 68, 83, 305, 92, 293, 373, 300, 280, 198, 207, 82, 383, 97, 211, 119, 359, 316, 100, 329, 242, 194, 81, 253, 112, 103, 24, 113, 430, 118, 111, 110, 99, 178, 22, 368, 148, 63, 344, 71, 120, 79, 185, 271, 33, 262, 284, 417, 427, 89, 288, 132, 315, 275, 297, 54, 328, 52, 348, 131, 193, 249, 238, 372, 206, 108, 101, 323, 137, 279, 292, 310, 189, 26, 202, 102, 130, 311, 237, 324, 364, 407, 248, 201, 340, 394, 227, 302, 267, 258, 188, 217, 231, 160, 307, 169, 153, 55, 343, 221, 388, 274, 287, 379, 367, 401, 109, 355, 145, 107, 104, 422, 74, 20, 50, 412, 134, 25, 17, 70, 47]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]

    # after second epoch when first zeroed = [22, 26, 48, 23, 61, 32, 20, 19, 54, 51, 18, 53, 56, 52, 62, 55, 30, 31, 25, 45, 21, 46, 28, 27, 17, 3, 24, 50, 47,
    #  2, 4, 49, 1, 0, 9, 11, 15, 5, 14, 29, 13, 8, 12, 6, 7, 10, 16, 36, 33, 37, 35, 34, 58, 57, 60, 59, 38, 43, 40, 42,
    #  41, 39, 44]
    elif 'sorted_by_shuffle' in yuval_change:
        schemes_to_remove = int(yuval_change.split('sorted_by_shuffle')[-1].split('_')[-1])
        np.random.seed(int(yuval_change.split('sorted_by_shuffle')[-1].split('_')[0]))
        ordered_schemes_idxs = list(range(len(subset_full_schemes)))
        np.random.shuffle(ordered_schemes_idxs)
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    #elif 'EC_specific' in yuval_change:
    #    scheme_tuple_map = {key: scheme_tuple_map[key] for key in
    #                        [list(scheme_tuple_map.keys())[idx] for idx in [int(yuval_change[8:])]]}
    print(f'subset_full_schemes: \n{subset_full_schemes}')
    # -#

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


def preproc_data(samples, model, batch_size):
    # stack pairs of tuples and map them to integer indices
    pairs = np.vstack([p for p, _ in samples.values()])
    pairs = torch.tensor(model.row_idx(pairs))

    # stack kernel values
    vals = torch.tensor(np.concatenate([v for _, v in samples.values()], axis=0))

    # stack schemes and map them to integer indices
    scheme = [np.int64([model.scheme_idx[s]] * samples[s][0].shape[0]) for s in samples.keys()]
    scheme = torch.tensor(np.concatenate(scheme, axis=0))

    # build torch loader for training
    data = TensorDataset(pairs, vals, scheme)
    # -# changed on vm because it crashed
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)
    return loader

def preproc_data_tryout(samples, model, batch_size):
    # stack pairs of tuples and map them to integer indices
    #pairs = np.vstack([p for p, _ in samples.values()])

    # -# temp change for tryout
    pairs = np.vstack([p for p, v in samples.values() if v is not None])
    pairs = torch.tensor(model.row_idx(pairs))

    # stack kernel values
    vals = torch.tensor(np.concatenate([v for _, v in samples.values() if v is not None], axis=0))

    # stack schemes and map them to integer indices
    scheme = [np.int64([model.scheme_idx[s]] * samples[s][0].shape[0]) for s in samples.keys() if samples[s][0] is not None]
    scheme = torch.tensor(np.concatenate(scheme, axis=0))

    # build torch loader for training
    data = TensorDataset(pairs, vals, scheme)
    # -# changed on vm because it crashed
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)
    return loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='genes', help="Name of the data base")
    parser.add_argument("--dim", type=int, default=100, help="Dimension of the embedding")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the walks")
    parser.add_argument("--kernel", type=str, default='EK', choices={'EK', 'MMD'}, help="Kernel to use for ForWaRD")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples per start tuple and metapath")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size during training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs during training")
    parser.add_argument("--classifier", type=str, default='SVM', choices={'NN', 'SVM'}, help="Downstream Classifier")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--yuval_change", type=str, default='', help="Yuval's experiment description")
    parser.add_argument("--tryout", type=bool, default=False, help="Is tryout experiment")
    parser.add_argument("--pre_time", type=str, default='', help="pre processing time in str format")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    emb_time = 0
    start = time.time()
    if args.pre_time != '':
        pt = datetime.strptime(args.pre_time, '%H:%M:%S')
        start -= (pt.second + pt.minute * 60 + pt.hour * 3600)  # add in seconds


    mesures_df = pd.DataFrame(columns=reduce(lambda xs, ys: xs + ys, [[f"Acc_{i}", f"Time_{i}", f"Steps_{i}"] for i in range(10)]))
    mesures_df['epochs'] = [f"epoch_{e}" for e in range(args.epochs)]
    mesures_df.set_index('epochs', inplace=True)

    data_path = f'Datasets/{args.data_name}'
    db = Database.load_csv(data_path)

    model_dir = f'models/{args.data_name}/{args.kernel}_{args.depth}_{args.dim}_{args.num_samples}_{args.epochs}_{args.batch_size}_{args.seed}'

    # -# add Experiment description to model_dir
    if args.yuval_change != '':
        model_dir += f'experiment_{args.yuval_change}'

    os.makedirs(model_dir, exist_ok=True)

    sample_fct = ek_utlis.ek_sample_fct if args.kernel == 'EK' else mmd_utils.mmd_sample_fct

    Y, rows = db.get_labels()

    scores = []
    split = StratifiedShuffleSplit(train_size=0.9, random_state=0, n_splits=10)
    for i, (train_index, test_index) in enumerate(split.split(rows, Y)):
        samples = get_samples(db, args.depth, args.num_samples, sample_fct, args.yuval_change)
        row_idx = {r: i for i, r in enumerate(rows)}
        scheme_idx = {s: i for i, s in enumerate(samples.keys())}
        model = Forward(args.dim, len(samples), row_idx, scheme_idx)

        # loader = preproc_data(samples, model, args.batch_size) if not args.tryout else preproc_data_tryout(samples, model, args.batch_size)
        try:
            loader = preproc_data(samples, model, args.batch_size)
        except Exception as e:
            loader = preproc_data_tryout(samples, model, args.batch_size)
        train(model, loader, args.epochs)

        embedding = model.get_embedding()
        embedding = {r: embedding[i] for r, i in row_idx.items()}

        X_train = np.float32([embedding[rows[j]] for j in train_index])
        X_test = np.float32([embedding[rows[j]] for j in test_index])
        Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]

        clf = MLPClassifier(max_iter=1000) if args.classifier == 'NN' else SVC(kernel='rbf', C=1.0)
        clf = make_pipeline(StandardScaler(), clf)

        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)

        scores.append(float(score))

        if i == 0:
            io_utils.save_embedding(model_dir.replace("models", "Embeddings"), embedding)
            emb_time = strftime("%H:%M:%S", gmtime(time.time() - start))
        save_dict({'scores': scores, 'time': strftime("%H:%M:%S", gmtime(time.time() - start)), "emb_time": emb_time}, f'{model_dir}/results.json')
        print(f"Run {i}; Accuracy: {score:.2f}")

    print(f'Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')
    # total time taken
    print(f'Runtime of the program is {time.time() - start} seconds = {strftime("%H:%M:%S", gmtime(time.time() - start))}')
