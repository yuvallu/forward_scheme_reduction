import math

from displey_res_ends_in_specific_column import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

INCLUDE_PREPROCESSING = True
DO_KVAR_MODIFICATION = False
FINAL_EXP_NUM = 3

def winapi_path(dos_path, encoding=None):
    if (not isinstance(dos_path, str) and encoding is not None):
        dos_path = dos_path.decode(encoding)
    path = os.path.abspath(dos_path)
    if path.startswith(u"\\\\"):
        return u"\\\\?\\UNC\\" + path[2:]
    return u"\\\\?\\" + path

data_names = ["mutagenesis", "world_B", "hepatitis", "genes", "mondial_original_target",
                  "mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
                  "mondial_target_Inflation_g6"]
exps = ["sorted_by_shuffle42", "remove_longest_schemes", "distribution_var", "stop_n_restart1", "conditional_entropy_from_low_to_high_removed", "yan81_mul0.33", "rev_min_mutual_information"]  #, "dynamic"]
data_name_to_num_schemes = {"mutagenesis": 58, "world": 20, "hepatitis": 21, "genes": 32, "genes_essential": 32,
                                "mondial_target_infant_mortality_g40": 63,
                                "mondial_target_continent": 63, "mondial_target_GDP_g8e3": 63,
                                "mondial_target_Inflation_g6": 63, "mondial_original_target": 63, "world_B": 60}
data_name_to_depth = {"mutagenesis": 4, "world": 3, "hepatitis": 3, "genes": 3, "genes_essential": 3,
                          "mondial_target_infant_mortality_g40": 3, "mondial_target_continent": 3,
                          "mondial_target_GDP_g8e3": 3, "world_B": 3,
                          "mondial_target_Inflation_g6": 3, "mondial_original_target": 3}
data_name_shortcut = {"mutagenesis": "mutagenesis", "world": "world", "world_B": "world", "hepatitis": "hepatitis", "genes": "genes", "mondial_target_infant_mortality_g40": "mondial infant-mortality","mondial_target_continent": "mondial continent", "mondial_target_GDP_g8e3": "mondial GDP","mondial_target_Inflation_g6": "mondial inflation", "mondial_original_target": "mondial religion","final_exps2\mutagenesis": "mutagenesis", "final_exps2\world": "world", "final_exps2\world_B": "world", "final_exps2\hepatitis": "hepatitis","final_exps2\genes": "genes","final_exps2\mondial_target_infant_mortality_g40": "mondial infant-mortality","final_exps2\mondial_target_continent": "mondial continent", "final_exps2\mondial_target_GDP_g8e3": "mondial GDP","final_exps2\mondial_target_Inflation_g6": "mondial inflation", "final_exps2\mondial_original_target": "mondial religion"
                      ,"final_exps3\mutagenesis": "mutagenesis", "final_exps3\world": "world", "final_exps3\world_B": "world", "final_exps3\hepatitis": "hepatitis","final_exps3\genes": "genes","final_exps3\mondial_target_infant_mortality_g40": "mondial infant-mortality","final_exps3\mondial_target_continent": "mondial continent", "final_exps3\mondial_target_GDP_g8e3": "mondial GDP","final_exps3\mondial_target_Inflation_g6": "mondial inflation", "final_exps3\mondial_original_target": "mondial religion"}
def exp_to_title(data_name, exp_name, short_title=True, seperator=" "):
    prefix = data_name_shortcut[data_name]
    if short_title:
        prefix = ""
    sufix = exp_name
    if "conditional_entropy" in exp_name:
        sufix = "ent"
    elif "distribution_var" in exp_name:
        sufix = "k_var"
    elif "SnR" in exp_name or "stop_n_restart1" in exp_name:
        sufix = "1ep"
    elif "longest" in exp_name:
        sufix = "len"
    elif "shuffle" in exp_name:
        sufix = "random"
    elif "yan81" in exp_name:
        sufix = "sampling"
    elif "rev_min_mutual_information" in exp_name:
        sufix = "MI"
    return prefix + seperator + sufix

def num_schemes_removed_to_percentage(data_name, num):
    candidate = [d for d in data_names if d in data_name]
    if len(candidate) == 0:
        return "?"
    elif len(candidate) > 1:
        candidate = [d for d in candidate if d == data_name.split("\\")[-1]]
        if len(candidate) != 1:
            return "?"
    return f"{(1 - num / data_name_to_num_schemes[candidate[0]]) * 100:.0f}%"

def k_var_fix(mesures_df, n_avg):
    sampling_time = mesures_df[f'Time_0'].iloc[0] * 0.40
    for i in range(n_avg):
        # we deduce sampling_time because we can optimize the code such that we only compute it once
        # (and not twice like we measured in the experiments)
        mesures_df[f'Time_{i}'] = mesures_df[f'Time_{i}'] - sampling_time
    return mesures_df

def entropy_fix(mesures_df, n_avg):
    times_0 = mesures_df[[f'Time_{i}' for i in range(n_avg)]].iloc[0]
    for i in range(n_avg):
        mesures_df[f'Time_{i}'] = mesures_df[f'Time_{i}'] + times_0[0] - times_0[i]
    return mesures_df

def remove_preprocessing_time(mesures_df, n_avg):
    times_0 = mesures_df[[f'Time_{i}' for i in range(n_avg)]].iloc[0]
    for i in range(n_avg):
        mesures_df[f'Time_{i}'] = mesures_df[f'Time_{i}'] - times_0[i]
    return mesures_df

def get_mesures_from_dirs(dirs, exp_name, data_name=""):
    """
    :return: scores_list (accuracies), times taken from experiment dirs
    """
    colors = cm.jet(np.linspace(0, 1, len(dirs)))
    try:
        dirs.sort(key=lambda exp: int(exp.split("_")[-1]))
    except ValueError as ve:
        dirs.sort(key=lambda exp: exp.split(exp_name+"_")[-1])
    if len([d for d in dirs if int(d.split(exp_name+'_')[-1]) == 0]) == 0:  # add blank exp to show the original F run
        dirs += dirs[0].split('experiment_')[0] + 'experiment_sorted_by_shuffle42_0'

    for i,(d, c) in list(enumerate(zip(dirs, colors)))[::-1]:
        label = f"{num_schemes_removed_to_percentage(data_name, int(d.split(exp_name + '_')[-1]))}"
        if int(d.split(exp_name+'_')[-1]) == 0:  # if regular forward
            d, label, c = d.split('experiment_')[0] + 'experiment_sorted_by_shuffle42_0', "100%", "gray"
        # if i!= 0 and i != len(dirs)-2:
        #     continue
        # mesures_df = pd.read_csv(f'{d}/mesures.csv')  # had problems with long file names
        mesures_df = pd.read_csv(winapi_path(f'{d}/mesures.csv'))
        n_avg = int(len(mesures_df.columns)/3)
        if not INCLUDE_PREPROCESSING:
            pass  # remove the preprocessing time
            mesures_df = remove_preprocessing_time(mesures_df, n_avg)
        elif label != "100%":
            if "conditional_entropy" in exp_name and data_name != "mutagenesis":
                mesures_df = entropy_fix(mesures_df, n_avg)
            elif DO_KVAR_MODIFICATION and "distribution_var" in exp_name:
                mesures_df = k_var_fix(mesures_df, n_avg)
        for col in mesures_df.columns:
            mesures_df[col] = mesures_df[col][:int(0.95*len(mesures_df[col]))]

        time_per_epoch = list(mesures_df[[f'Time_{i}' for i in range(n_avg)]].mean(axis=1))
        ###
        # start = time_per_epoch[0]
        # time_per_epoch = [t - start + 25 for t in time_per_epoch]
        ###
        acc_per_epoch = list(mesures_df[[f'Acc_{i}' for i in range(n_avg)]].mean(axis=1))
        if label == "100%":
            # baseline
            baseline_txt, baseline = "a", 0.95 * 100 * [v for v in acc_per_epoch if not np.isnan(v)][-1]  # "majority" if label == 'Accuracy' else "regular run"
            plt.axhline(y=baseline, color='r', linestyle='--', label=f"baseline ({baseline_txt})")
        plt.plot(time_per_epoch, [acc*100 for acc in acc_per_epoch], color=c, linestyle='dashed', linewidth=1 if label != "100%" else 4,  #[acc*100 for acc in acc_per_epoch]
                 marker='o', markerfacecolor=c, markersize=2 if label != "100%" else 6, label=label)

    if "shuffle" in exp_name or True:
        plt.legend(bbox_to_anchor=(1.0, 1))  # left:=bbox_to_anchor=(-0.08, 1)
    #plt.title(exp_to_title(data_name, exp_name), fontname="Times New Roman", size=28,fontweight="bold")  # f'{data_name} - {exp_name}')
    # plt.xlabel("Time (sec)")  # no need because i plot them side by side on the paper
    # plt.ylabel("Acc")
    plt.tight_layout()
    plt.xticks(fontsize=26)
    plt.locator_params(axis='x', nbins=7)  # don't have to put this but it looks nicer
    plt.locator_params(axis='y', nbins=7)
    plt.yticks(fontsize=26, rotation='vertical')

    f_name = exp_to_title(data_name, exp_name, short_title=False, seperator="_")
    plt.savefig(os.path.abspath(f"../../images_for_paper/first_try/{f_name}"))
    plt.title(f_name)
    plt.show()

def display_acc_per_time(data_name, exp_name):
    exp_dirs = get_all_exp_with_exp_name(exp_name, data_name)
    get_mesures_from_dirs(exp_dirs, exp_name, data_name)

def acc_per_time_visualization():
    # for data_name in ["mondial_original_target"]:
    for data_name in data_names:  # data_names:  # ["mondial_original_target", "hepatitis", "genes", "mutagenesis", "mondial_target_GDP_g8e3", "world_B", "mondial_target_infant_mortality_g40"]:
        # for exp in ["999_50000_0experiment_distribution_var"]:
        for exp in ["distribution_var", "sorted_by_shuffle42"]:  # ["yan81_mul0.33"]:  # exps:  # ["distribution_var", "conditional_entropy_from_low_to_high_removed", "stop_n_restart1"] ["min_mutual_information", "rev_min_mutual_information", "max_mutual_information", "rev_max_mutual_information"]
        # for exp in ["999_50000_0experiment_dynamic"]:
            # display_acc_per_time(f"final_exps{FINAL_EXP_NUM}\{data_name}", f"experiment_{exp}")
            display_acc_per_time(f"final_exps{FINAL_EXP_NUM}\{data_name}", exp)

################################################################################################################################################################################################
def get_ts_from_dirs(dirs, exp_name, data_name=""):
    """
    :return: dict from each exp name (dir) to its t (time to reach a*  or -1 if never reached)
    """
    ts, baseline = {}, 0  # placeholder
    try:
        dirs.sort(key=lambda exp: int(exp.split("_")[-1]))
    except ValueError as ve:
        dirs.sort(key=lambda exp: exp.split(exp_name+"_")[-1])
    if len([d for d in dirs if int(d.split(exp_name+'_')[-1]) == 0]) == 0:  # add blank exp to show the original F run
        dirs = [dirs[0].split('experiment_')[0] + 'experiment_sorted_by_shuffle42_0'] + dirs

    for i, d in enumerate(dirs):
        if "dynamic" not in d:
            num_schemes_removed = int(d.split(exp_name + '_')[-1].split('_')[-1])
            label = f"{num_schemes_removed_to_percentage(data_name, num_schemes_removed)}"
            if int(d.split(exp_name + '_')[-1].split('_')[-1]) == 0:  # if regular forward
                d, label = d.split('experiment_')[0] + 'experiment_sorted_by_shuffle42_0', "100%"
        else:
            label = d.split(exp_name + '_')[-1]
            if label == "0%in_1_ep":  # if regular forward
                d, label = d.split('experiment_')[0] + 'experiment_sorted_by_shuffle42_0', "100%"
        mesures_df = pd.read_csv(winapi_path(f'{d}/mesures.csv'))
        n_avg = int(len(mesures_df.columns)/3)
        if not INCLUDE_PREPROCESSING:
            # remove the preprocessing time
            mesures_df = remove_preprocessing_time(mesures_df, n_avg)
        elif label != "100%":
            if "conditional_entropy" in exp_name and data_name != "mutagenesis":
                mesures_df = entropy_fix(mesures_df, n_avg)
            elif DO_KVAR_MODIFICATION and "distribution_var" in exp_name:
                mesures_df = k_var_fix(mesures_df, n_avg)
        for col in mesures_df.columns:
            mesures_df[col] = mesures_df[col][:int(0.95*len(mesures_df[col]))]

        time_per_epoch = list(mesures_df[[f'Time_{i}' for i in range(n_avg)]].mean(axis=1))
        acc_per_epoch = list(mesures_df[[f'Acc_{i}' for i in range(n_avg)]].mean(axis=1))
        if label == "100%":
            baseline = 0.95 * [v for v in acc_per_epoch if not np.isnan(v)][-1]
            continue

        t= math.inf
        for idx, acc in enumerate(acc_per_epoch):
            if acc >= baseline:
                t = time_per_epoch[idx]
                break
        ts[d] = t
    return ts

def get_t_star(data_name, exp_name):
    """
    :return: the name of the best exp's dir from all different #schemes removed,
             t* (of this exp)
    """
    exp_dirs = get_all_exp_with_exp_name(exp_name, data_name)
    t_s = get_ts_from_dirs(exp_dirs, exp_name, data_name)  # dict from dir to t.
    t_star_dir = min(t_s, key=t_s.get)
    return t_star_dir, t_s[t_star_dir]

def compare_sr_strategies():
    task_exps_dict = {}
    for data_name in data_names:  #["mondial_target_infant_mortality_g40"]:  # data_names:  # ["mondial_original_target", "hepatitis", "genes", "mutagenesis", "mondial_target_GDP_g8e3", "world_B"]:
        exps_dict = {}
        for exp in ["distribution_var", "sorted_by_shuffle42", "stop_n_restart1", "yan81_mul0.33", "remove_longest_schemes"]:  #["distribution_var", "sorted_by_shuffle42"]:  # exps:  # ["distribution_var", "conditional_entropy_from_low_to_high_removed", "stop_n_restart1","dynamic"]
            d, t = get_t_star(f"final_exps{FINAL_EXP_NUM}\{data_name}", exp)
            exps_dict[exp_to_title(data_name, exp)] = {"t_star":t, "dir":d}
        task_exps_dict[data_name] = {k: v for k, v in sorted(exps_dict.items(), key=lambda item: item[1]["t_star"])}

    # solutions:
    #  run another random
    #  ignor the preprocessing time
    p=5
    for data_name in data_names:
        t_s = task_exps_dict[data_name]
        winer_exp = min(t_s, key=lambda x: t_s[x]["t_star"])
        print(f"For data_name {data_name:<35}, the winner is exp {winer_exp:<25} with t={t_s[winer_exp]['t_star']:5f} and dir={t_s[winer_exp]['dir']}")

    winners_table_df = pd.DataFrame.from_dict({data_name_shortcut[k]: {kk:vv["t_star"] for kk,vv in v.items()} for k,v in task_exps_dict.items()})
    winners_table_df.to_csv("temp_winners_table.csv")

    ##
    for data_name in data_names:
        t_s = task_exps_dict[data_name]
        t_s["baseline"] = {"t_star": -1, "dir": f'final_exps{FINAL_EXP_NUM}/{data_name}/EK_{data_name_to_depth[data_name]}_100_500_999_50000_0experiment_sorted_by_shuffle42_0'}
        colors = cm.jet(np.linspace(0, 1, len(t_s)))
        for i, (exp, c) in enumerate(zip(sorted(list(t_s.keys())), colors)):
            d = t_s[exp]["dir"]
            mesures_df = pd.read_csv(winapi_path(f'{d}/mesures.csv'))
            n_avg = int(len(mesures_df.columns) / 3)
            if not INCLUDE_PREPROCESSING:
                pass  # remove the preprocessing time
                mesures_df = remove_preprocessing_time(mesures_df, n_avg)
            elif i != 0:
                if "conditional_entropy" in exp and data_name != "mutagenesis":
                    mesures_df = entropy_fix(mesures_df, n_avg)
                elif DO_KVAR_MODIFICATION and "distribution_var" in exp:
                    mesures_df = k_var_fix(mesures_df, n_avg)
            for col in mesures_df.columns:
                mesures_df[col] = mesures_df[col][:int(0.95 * len(mesures_df[col]))]
            time_per_epoch = list(mesures_df[[f'Time_{i}' for i in range(n_avg)]].mean(axis=1))
            acc_per_epoch = list(mesures_df[[f'Acc_{i}' for i in range(n_avg)]].mean(axis=1))
            plt.plot(time_per_epoch, [acc * 100 for acc in acc_per_epoch], color=c if exp != "baseline" else "gray",
                     linestyle='dashed', linewidth=1 if exp != "baseline" else 4,
                     marker='o', markerfacecolor=c if exp != "baseline" else "gray", markersize=2 if exp != "baseline" else 6, label=exp_to_title(data_name, exp, short_title=True, seperator=""))
        plt.legend()
        plt.tight_layout()
        plt.xticks(fontsize=26)
        plt.locator_params(axis='y', nbins=7)
        plt.yticks(fontsize=26, rotation='vertical')
        f_name = f"winners_{data_name}"
        plt.savefig(os.path.abspath(f"../../images_for_paper/first_try/winners/{f_name}"))
        plt.title(data_name)
        plt.show()



if __name__ == '__main__':
    # acc_per_time_visualization()

    # remember to change INCLUDE_PREPROCESSING before you run
    compare_sr_strategies()