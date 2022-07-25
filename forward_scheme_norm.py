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
import pickle
import mmd_utils
import ek_utlis
import random
import time
from time import strftime
from time import gmtime
import entropy_utils as entropy
from torch_scatter import scatter  # pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
import io_utils
import json
from datetime import datetime

SAVE_EMBEDDINGS_EACH_EPOCH = False  # True

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

    # def loss(self, y_pred, y_true):
    #     # -# changed for Dynamic scheme reduction
    #     return F.mse_loss(y_pred, y_true, reduction='none')

    def loss(self, y_pred, y_true):  # original
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

def check_corr_matrixs(model_A):
    B = model_A.reshape(-1, 10000)
    C = (B @ B.T).detach().numpy()
    C[np.arange(63),np.arange(63)] = 0
    return C

def train(model, data3, epochs, batch_size, args_train={"percent_remove": 0, "num_epochs": 1, "tryout": False, "start_time": 0, "stop_n_restart": -1}, exp_name=""):
    # args_train["tryout"] = False  ############# -#
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # init loss per scheme tensor to zeros
    loss_per_scheme, temp_out = torch.zeros(model.num_relations).to(device), torch.zeros(model.num_relations).to(device)

    schemes_removed = torch.tensor([])
    for e in range(epochs):
        if args_train["stop_n_restart"] == e:  # should be -1 if not in stop_n_restart
            break  # go to the tryout if and exit there
        if SAVE_EMBEDDINGS_EACH_EPOCH:
            embedding = model.get_embedding()
            embedding = {r: embedding[i] for r, i in row_idx.items()}
            io_utils.save_embedding(model_dir.replace("models", "Embeddings") + f"_epoch_{e}", embedding)
        loader = DataLoader(data3, batch_size=batch_size, shuffle=True, num_workers=8)

        bar = tqdm(desc=f'Epoch {e + 1} Mean Loss: _')
        bar.reset(total=len(loader))

        epoch_losses = []
        for (pairs, vals, scheme) in loader:
            pairs, vals, scheme = pairs.to(device), vals.to(device), scheme.to(device)

            opt.zero_grad()
            y_pred = model(pairs, scheme)  # forward pass
            loss = model.loss(y_pred, vals)
            # -# changed for Dynamic scheme reduction
            # temp_out.zero_()
            # scatter(loss, scheme, reduce="mean", out=temp_out)
            # loss_per_scheme += temp_out
            # loss_per_scheme += torch.cat((s, torch.zeros(len(loss_per_scheme)-len(s))), 0)

            # loss = torch.mean(loss)  # loss needs to be a scalar
            # loss.backward()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.cpu().detach().numpy())
            bar.set_description(desc=f'Epoch {e + 1} Mean Loss: {epoch_losses[-1]:.4f}')
            bar.update()

        bar.close()

        # if len(schemes_removed) < model.num_relations * args_train["percent_remove"]:
        #     # TODO: run to get order for exp
        #     # print(f"Ordered schemes from lowest loss to highest: {torch.sort(loss_per_scheme, descending=False).indices.tolist()}")
        #     # print(f"Dict: {dict(zip(torch.sort(loss_per_scheme, descending=False).indices.tolist(),torch.sort(loss_per_scheme, descending=False).values.tolist()))}")
        #     NUM_REMOVED = min(int(model.num_relations * args_train["percent_remove"] / args_train["num_epochs"]), int(model.num_relations * args_train["percent_remove"]-len(schemes_removed)))
        #     # descending=False means remove lowest and descending=True means remove highest
        #     schemes_to_remove = torch.tensor(
        #         [idx for idx in torch.sort(loss_per_scheme.cpu(), descending=False).indices if
        #          idx not in schemes_removed])[:NUM_REMOVED]
        #
        #     mask = ~(torch.sum(data3.tensors[2].view(-1, 1) == schemes_to_remove.view(1, -1), dim=1).bool())
        #
        #     data3 = data3[mask]
        #     data3 = TensorDataset(data3[0], data3[1], data3[2])
        #
        #     schemes_removed = torch.cat((schemes_removed, schemes_to_remove), 0)

        loss_per_scheme = torch.linalg.matrix_norm(model.A, dim=(1, 2))


    if args_train["tryout"]:
        # save Ordered schemes from lowest loss to highest + Dict
        with open(os.path.join("Sorted_schemes", f"{exp_name}.txt"), 'w') as f:
            json.dump({f"Ordered_schemes": torch.sort(loss_per_scheme, descending=False).indices.tolist(), f"Dict": dict(
                zip(torch.sort(loss_per_scheme, descending=False).indices.tolist(),
                    torch.sort(loss_per_scheme, descending=False).values.tolist()))}, f, indent=4)
        print(f'Time:{strftime("%H:%M:%S", gmtime(time.time() - args_train["start_time"]))}')
        exit()

    if SAVE_EMBEDDINGS_EACH_EPOCH:
        embedding = model.get_embedding()
        embedding = {r: embedding[i] for r, i in row_idx.items()}
        io_utils.save_embedding(model_dir.replace("models", "Embeddings") + "_epoch_10", embedding)
        exit()

    return model


def get_samples(db, depth, num_samples, sample_fct, yuval_change=''):
    # -# depth is used here
    tuples = [r for _, r, _ in db.iter_rows(db.predict_rel)]
    scheme_tuple_map = db.scheme_tuple_map(db.predict_rel, tuples, depth)
    # print(f"Try---{scheme_tuple_map.keys()}")
    # import json
    # scheme_tuple_map['target'] = {k: list(v) for k, v in scheme_tuple_map['target'].items()}
    # with open("scheme_tuple_map.json", "w") as f:
    #     json.dump(dict(scheme_tuple_map), f)

    # -# cut half of the schemes
    # scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if len(key.split(">")) == 2+1} #if len(key.split(">")) == 4}
    # random.seed(0)
    # rand_list = random.sample(range(30), 12)
    # scheme_tuple_map = {key: scheme_tuple_map[key] for key in [list(scheme_tuple_map.keys())[idx] for idx in [27, 12, 24, 13, 1, 8, 16, 15, 28, 9, 22, 11]]}
    # try try
    # include_keys = list(scheme_tuple_map.keys())[::2] #take every second key only
    # scheme_tuple_map = {key: value for key, value in scheme_tuple_map.items() if key in include_keys}
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
        subset_full_schemes = [scheme for scheme in subset_full_schemes if len(scheme.split("-")) == 3 + 1]
    elif 'r6_manual_reduce' in yuval_change:
        exp_num, rand_list = int(yuval_change.split('r6_manual_reduce')[-1]), []
        random.seed(0)
        for i in range(6):
            rand_list = random.sample(range(63), 30)
        subset_full_schemes = [subset_full_schemes[idx] for idx in rand_list]
        if exp_num == 1:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in remove_from_r6_to_reduce]
        elif exp_num == 2:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[0]]]
        elif exp_num == 3:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[1]]]
        elif exp_num == 4:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[2]]]
        elif exp_num == 5:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[3]]]
        elif exp_num == 6:
            subset_full_schemes = [subset_full_schemes[idx] for idx in random.sample(range(30), 29)]
        elif exp_num == 7:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if
                                   sc not in [remove_from_r6_to_reduce[1:]]]  # 1+2+3
        elif exp_num == 8:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country@province>Country@province-Country@city>city>Latitude@city"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in remove_from_r6_to_reduce]
        elif exp_num == 9:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[1:3]]]  # 1+2
        elif exp_num == 10:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if sc not in [remove_from_r6_to_reduce[2:]]]  # 2+3
        elif exp_num == 11:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Capital@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Name@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if
                                   sc not in [remove_from_r6_to_reduce[1]] + [remove_from_r6_to_reduce[3]]]  # 1+3
        elif exp_num == 12:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country1@borders>borders>Length@borders",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country"]
            subset_full_schemes = [sc for sc in subset_full_schemes if
                                   sc not in remove_from_r6_to_reduce]  # boreder_len + 1+2+3
        elif exp_num == 13:
            remove_from_r6_to_reduce = [
                "Country@target-Code@country>Code@country-Country1@borders>borders>Length@borders",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Population@country",
                "Country@target-Code@country>Code@country-Country2@borders>Country1@borders-Code@country>country>Province@country",
                "Country@target-Code@country>Code@country-Country@religion>religion>Percentage@religion"]
            subset_full_schemes = [sc for sc in subset_full_schemes if
                                   sc not in remove_from_r6_to_reduce]  # boreder_len + 1+2+3
        elif exp_num in range(100, 110):
            for i in range(100, exp_num):
                random.sample(range(30), 29)
            subset_full_schemes = [subset_full_schemes[idx] for idx in random.sample(range(30), 29)]
    elif 'conditional_entropy_removed_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('conditional_entropy_removed_')[-1])
        ordered_schemes = entropy.sorted_dict_by_max_value_in_list(
            entropy.get_schemes_to_entropies_dict(db, subset_full_schemes, entropy.XIY_conditional_entropy))
        subset_full_schemes = [scheme for scheme, v in ordered_schemes.items()][:-schemes_to_remove]
    elif 'conditional_entropy_from_low_to_high_removed_' in yuval_change:
        schemes_to_remove = int(yuval_change.split('conditional_entropy_from_low_to_high_removed_')[-1])
        ordered_schemes = entropy.sorted_dict_by_max_value_in_list(
            entropy.get_schemes_to_entropies_dict(db, subset_full_schemes, entropy.XIY_conditional_entropy))
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
        ordered_schemes = entropy.sorted_dict_by_max_value_in_list(
            entropy.get_schemes_to_entropies_dict(db, subset_full_schemes, entropy.XIY_conditional_entropy))
        subset_full_schemes = [scheme for scheme, v in ordered_schemes.items()][scheme_to_remain:scheme_to_remain + 1]
    elif 'sorted_by_loss_after_1_epoch_' in yuval_change:  # reverse order
        schemes_to_remove = int(yuval_change.split('sorted_by_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [39, 41, 44, 40, 42, 43, 38, 59, 60, 57, 35, 34, 33, 36, 37, 58, 29, 16, 10, 6, 8, 12, 7,
                                14, 13, 15, 9, 5, 11,
                                30, 24, 49, 0, 4, 47, 1, 2, 50, 3, 17, 28, 25, 27, 21, 45, 46, 31, 55, 56, 52, 62, 53,
                                18, 51, 54, 19, 32, 20,
                                61, 23, 48, 26, 22]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_by_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_by_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [39, 41, 44, 40, 42, 43, 38, 59, 60, 57, 35, 34, 33, 36, 37, 58, 29, 16, 10, 6, 8, 12, 7,
                                14, 13, 15, 9, 5, 11,
                                30, 24, 49, 0, 4, 47, 1, 2, 50, 3, 17, 28, 25, 27, 21, 45, 46, 31, 55, 56, 52, 62, 53,
                                18, 51, 54, 19, 32, 20,
                                61, 23, 48, 26, 22][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_correct_highest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_highest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 23, 61, 20, 32, 19, 54, 51, 18, 53, 62, 52, 56, 55, 31, 46, 45, 21, 27, 25,
                                28, 17, 3, 50, 2, 1, 47, 4, 0, 49, 24, 30, 11, 5, 9, 15, 13, 14, 7, 12, 8, 6, 10, 16,
                                29, 58, 37, 36, 33, 34, 35, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_correct_lowest_loss_after_1_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_lowest_loss_after_1_epoch_')[-1])
        ordered_schemes_idxs = [22, 26, 48, 23, 61, 20, 32, 19, 54, 51, 18, 53, 62, 52, 56, 55, 31, 46, 45, 21, 27, 25,
                                28, 17, 3, 50, 2, 1, 47, 4, 0, 49, 24, 30, 11, 5, 9, 15, 13, 14, 7, 12, 8, 6, 10, 16,
                                29, 58, 37, 36, 33, 34, 35, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_correct_highest_loss_after_10_epoch_' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_highest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [26, 22, 23, 61, 48, 32, 20, 51, 54, 62, 19, 53, 56, 18, 52, 55, 25, 45, 21, 31, 46, 28,
                                27, 17, 3, 30, 24, 50, 2, 47, 49, 4, 1, 0, 9, 11, 15, 5, 14, 13, 8, 29, 12, 7, 6, 10,
                                16, 36, 37, 33, 35, 34, 58, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_correct_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_correct_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [26, 22, 23, 61, 48, 32, 20, 51, 54, 62, 19, 53, 56, 18, 52, 55, 25, 45, 21, 31, 46, 28,
                                27, 17, 3, 30, 24, 50, 2, 47, 49, 4, 1, 0, 9, 11, 15, 5, 14, 13, 8, 29, 12, 7, 6, 10,
                                16, 36, 37, 33, 35, 34, 58, 57, 60, 59, 38, 43, 42, 40, 44, 41, 39][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_genes_correct_highest_loss_after_10_epoch' in yuval_change:  # highest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_correct_highest_loss_after_10_epoch')[-1])
        ordered_schemes_idxs = [6, 0, 28, 16, 22, 7, 10, 8, 5, 20, 3, 30, 18, 29, 25, 17, 13, 27, 15, 2, 12, 21, 24, 9,
                                31, 1, 19, 11, 23, 4, 26, 14]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    elif 'sorted_genes_correct_lowest_loss_after_10_epoch_' in yuval_change:  # lowest loss first
        schemes_to_remove = int(yuval_change.split('sorted_genes_correct_lowest_loss_after_10_epoch_')[-1])
        ordered_schemes_idxs = [6, 0, 28, 16, 22, 7, 10, 8, 5, 20, 3, 30, 18, 29, 25, 17, 13, 27, 15, 2, 12, 21, 24, 9,
                                31, 1, 19, 11, 23, 4, 26, 14][::-1]
        subset_full_schemes = [subset_full_schemes[idx] for idx in ordered_schemes_idxs][schemes_to_remove:]
    # elif 'EC_specific' in yuval_change:
    #    scheme_tuple_map = {key: scheme_tuple_map[key] for key in
    #                        [list(scheme_tuple_map.keys())[idx] for idx in [int(yuval_change[8:])]]}
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
    print(f'subset_full_schemes: \n{subset_full_schemes}')
    # print(f"Try---- {subset_full_schemes[22]}")
    # -#

    samples = {}
    for scheme, tuple_map in tqdm(scheme_tuple_map.items()):
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                full_scheme = f"{scheme}>{col_id}"
                if full_scheme in subset_full_schemes:
                    col_kernel = kernels_dict[col_id] if col_id in kernels_dict.keys() else kernels[
                        db.get_col_type(col_id)]
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
    # loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)
    return data

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
    # loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)
    return data

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
    parser.add_argument("--train", type=str, default=None, help="loss specification")
    parser.add_argument("--tryout", type=bool, default=False, help="Is tryout experiment")
    parser.add_argument("--pre_time", type=str, default='', help="pre processing time in str format")
    parser.add_argument("--stop_n_restart", type=int, default=-1, help="stop after ? epochs (before restart)")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    emb_time = 0
    start = time.time()
    if args.pre_time != '':
        pt = datetime.strptime(args.pre_time, '%H:%M:%S')
        start -= (pt.second + pt.minute * 60 + pt.hour * 3600)  # add in seconds

    data_path = f'Datasets/{args.data_name}'
    db = Database.load_csv(data_path)
    # print(f"Try-- {db.relations.keys()}")

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

        # loader = preproc_data(samples, model, args.batch_size)
        # data3 = preproc_data(samples, model, args.batch_size) if not args.tryout else preproc_data_tryout(samples, model, args.batch_size)
        try:
            data3 = preproc_data(samples, model, args.batch_size)
        except Exception as e:
            data3 = preproc_data_tryout(samples, model, args.batch_size)
        if args.stop_n_restart != -1:  # if we are in the stop_n_restart the above code can run only once
            start = time.time()
            if args.pre_time != '':
                pt = datetime.strptime(args.pre_time, '%H:%M:%S')
                start -= (pt.second + pt.minute * 60 + pt.hour * 3600)  # add in seconds

        train(model, data3, args.epochs, args.batch_size,
              {"percent_remove": 0, "num_epochs": 1, "tryout": args.tryout, "start_time": start, "stop_n_restart": args.stop_n_restart} if args.train is None else {
                  "percent_remove": int(args.train.split("%")[0]) / 100, "num_epochs": int(args.train.split("%")[-1]),
                  "tryout": args.tryout, "start_time": start, "stop_n_restart": args.stop_n_restart}, exp_name=args.yuval_change)
        embedding = model.get_embedding()
        embedding = {r: embedding[i] for r, i in row_idx.items()}

        # # save the embeddings to examine them afterwards
        # embedding = model.get_embedding()
        # embedding = {r: embedding[i] for r, i in row_idx.items()}
        # words = list(embedding.keys())
        # vectors = []
        # for word in embedding.keys():
        #     vectors.append(embedding[word])
        # m = models.KeyedVectors(np.array(vectors).shape[1])
        # m.add(words, vectors)
        # m.wv.save_word2vec_format(output_embeddings_file, binary=False)
        # # get from file: model1 = models.KeyedVectors.load_word2vec_format(input_file, binary=False)

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
    print(
        f'Runtime of the program is {time.time() - start} seconds = {strftime("%H:%M:%S", gmtime(time.time() - start))}')
