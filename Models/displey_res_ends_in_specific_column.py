# mondial
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pprint
import pandas as pd
import time_per_acc_vis as vis

'''
1 schemes with 0 length - 
5 schemes with 1 length - 
25 schemes with 2 length - 
33 schemes with 3 length - 
0 schemes with 4 length - 
---
'''
exp_baselines = {"mondial_target_infant_mortality_g40": 0.605042, "mondial_target_continent": 0.227273,
                 "mondial_target_population_growth": 0.567227, "world": 0.242678, "world_B": 0.242678,
                 "mondial_target_GDP_g8e3": 0.5, "mondial_target_Inflation_g6": 0.508403, "mutagenesis": 0.664894,
                 "mondial_original_target": 0.637255, "mondial": 0.637255, "genes": 0.426744,
                 "genes_essential": 0.647783, "hepatitis": 0.588}


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i], 3), ha='center', bbox=dict(facecolor='red', alpha=.7))


def plot_acc_to_textname(exp_names, scores_list, colors=('green', 'blue'), offset=0.5, label='Accuracy',
                         add_labels=True,
                         x_ticks=None, title="", baseline=0.637255, exp_num=0, exp_name=""):
    if exp_num == 0:
        fig = plt.figure()
        baseline_txt = "majority" if label == 'Accuracy' else "regular run"
        plt.axhline(y=baseline, color='r', linestyle=':', label=f"baseline ({baseline_txt})")
    # bar chart (previously)
    # plt.bar(exp_names, [s - offset for s in scores_list], bottom=offset, color=colors)
    # connected (dashed) plot
    plt.plot(exp_names, [s - offset for s in scores_list], color=colors[0], linestyle='dashed', linewidth=3, marker='o',
             markerfacecolor=colors[1], markersize=8, label=exp_name)

    if add_labels:
        addlabels(exp_names, scores_list)
    plt.ylabel(label)
    if x_ticks != None:
        plt.xticks(x_ticks)
    # plt.xlabel('Test name')
    if label == 'Accuracy':
        plt.ylim(ymin=0, ymax=1)
    plt.title(f'{title}')


def get_results_from_dirs(dirs, ret_dict=False):
    """
    :return: scores_list (accuracies), times taken from experiment dirs
    """
    scores_list = []
    for d in dirs:
        # if 'Inflation' in d and 'SnR' in d:
        #     i=0
        data = json.load(open(f'{d}/results.json'))
        scores = data["scores"]
        print(f'{d:<75} Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')
        scores_list.append(np.mean(scores))

    times, embTs = [], []
    for d in dirs:
        try:  # times
            data = json.load(open(f'{d}/results.json'))
            pt = datetime.strptime(data["time"], '%H:%M:%S')
            times += [(pt.second + pt.minute * 60 + pt.hour * 3600) / (60 * 60)]
        except FileNotFoundError:
            pass
        except:
            print(f"{d} result doesnt have time")
        try:  # emb_time
            data = json.load(open(f'{d}/results.json'))
            pt = datetime.strptime(data["emb_time"], '%H:%M:%S')
            embTs += [(pt.second + pt.minute * 60 + pt.hour * 3600) / (60 * 60)]
        except FileNotFoundError:
            pass
        except:
            print(f"{d} result doesnt have emb_time")

    if ret_dict:
        if len(embTs) != len(scores_list):
            return {d: (scores_list[idx], times[idx]) for idx, d in enumerate(dirs)}
        else:
            return {d: (scores_list[idx], times[idx], embTs[idx]) for idx, d in enumerate(dirs)}
    return scores_list, times


def display_res_ends_in_specific_column():
    data_name = "mondial"
    dirs = os.listdir(os.path.join(".", data_name))
    for dir in dirs.copy():
        if "EC" not in dir:
            # dirs.remove(dir)
            pass
        if ".py" in dir:
            dirs.remove(dir)
        # elif "random" in dir:
        #    dirs.remove(dir)
    dirs = [os.path.join(data_name, d) for d in dirs]
    # dirs =   # experiment_EC_r6andom{i} # [dirs[0]] + dirs[2:12] + dirs[20:]
    print(dirs)
    scores_list, times = get_results_from_dirs(dirs)
    print(len(dirs))
    # dirs = ["all\n(43)", "rand1\n(12)", "rand2\n(12)", "rand3\n(12)", "rand4\n(12)", "rand5\n(12)", "all2\n(12)", "all3\n(30)"]
    dirs = [d.split("EC_")[-1].replace("r6andom", "r").replace("scheme_len_eq_", "len") for d in dirs]
    colors = ['purple'] + ['blue'] * 5 + ['blue'] + ['green'] + ['blue'] * 3 + ['cyan'] * 3

    plot_acc_to_textname(exp_names=dirs, scores_list=scores_list, colors=colors, offset=0.5, label='Accuracy')
    plt.show()
    fig = plt.figure()
    # times = ['05:50:47', '00:59:18', '00:36:03', '01:25:01', '00:54:56', '01:16:49', '02:59:59', '02:20:12']
    # times = [5 + 5 / 6, 59/60, 36 / 60, 1 + 25 / 60, 55/60, 1+16/60, 2+59/60, 2 + 20 / 60]
    plt.bar(dirs, times, color=colors)
    # addlabels(dirs, ['05:50:47','00:36:03','00:36:03','00:36:03','00:36:03','00:36:03','02:20:12'])
    plt.ylabel('Hours')
    # plt.xlabel('Test name')
    plt.show()


def get_all_exp_with_exp_name(exp_name, data_name="mondial", must_contain=''):
    dirs = os.listdir(os.path.join(".", data_name))
    if 'sorted_mean_lowest_loss_after_SnR_mondial_target_Inflation_g6_stop_n_restart' in exp_name:
        exp_name = exp_name.replace('sorted_mean_lowest_loss_after_SnR', 'low_loss_SnR')
    for d in dirs.copy():
        if exp_name not in d or must_contain not in d:
            dirs.remove(d)
    print(dirs)
    return [os.path.join(data_name, d) for d in dirs]


def get_graph_color(exp_num):
    colors = [('green', 'blue'), ('black', 'red'), ('black', 'yellow'), ('black', 'orange'), ('black', 'pink'),
              ('black', 'purple'), ('black', 'cyan'), ('black', 'green')]
    if exp_num >= len(colors):
        return ('black', 'red')
    return colors[exp_num]


def save_exp_info_for_cmp(exp_names="experiment_conditional_entropy_removed", data_name="mondial",
                          dst_path="exp_info_for_cmp.csv"):
    exp_names = [exp_names] if type(exp_names) != list else exp_names
    res_df = pd.read_csv(dst_path) if os.path.exists(data_name) else pd.DataFrame(
        {'exp_name': [], 'accuracy': [], 'time': [], 'emb_time': []})
    for exp_num, exp_name in enumerate(exp_names):
        exp_dirs = get_all_exp_with_exp_name(exp_name, data_name)
        exp_results_dict = get_results_from_dirs(exp_dirs, ret_dict=True)
        # pprint.pprint(exp_results_dict)
        res_df = res_df.append(pd.DataFrame(
            {'exp_name': exp_results_dict.keys(), 'accuracy': [v[0] for v in exp_results_dict.values()],
             'time': [v[1] for v in exp_results_dict.values()], 'emb_time': [v[2] for v in exp_results_dict.values()]}))
    res_df.to_csv(dst_path, index=False)


def display_timeline_progression_scheme_reduction(exp_names="experiment_conditional_entropy_removed",
                                                  data_name="mondial", must_contain=''):
    exp_names = [exp_names] if type(exp_names) != list else exp_names

    for exp_num, exp_name in enumerate(exp_names):
        exp_dirs = get_all_exp_with_exp_name(exp_name, data_name, must_contain=must_contain)
        try:
            exp_results_dict = {int(k.split(exp_name + "_")[-1]): v for k, v in
                                sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                       key=lambda item: int(item[0].split(exp_name + "_")[-1]))}
        except ValueError as ve:
            try:
                exp_results_dict = {int(k.split(exp_name)[-1]): v for k, v in
                                sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                       key=lambda item: int(item[0].split(exp_name)[-1]))}
            except ValueError as ve:
                exp_results_dict = {int(k.split("_")[-1]): v for k, v in
                                sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                       key=lambda item: int(item[0].split("_")[-1]))}
        pprint.pprint(exp_results_dict)
        title = f'{data_name}---{exp_name}'
        plot_acc_to_textname(exp_names=exp_results_dict.keys(), scores_list=[v[0] for k, v in exp_results_dict.items()],
                             colors=get_graph_color(exp_num), offset=0.0,
                             label='Accuracy', add_labels=False,
                             x_ticks=list(exp_results_dict.keys()), title=title, baseline=exp_baselines[data_name],
                             exp_num=exp_num, exp_name=exp_name)
    plt.legend()
    pdf.savefig(plt.gcf())
    plt.show()

    fig = plt.figure()
    for exp_num, exp_name in enumerate(exp_names):
        exp_dirs = get_all_exp_with_exp_name(exp_name, data_name)
        try:
            exp_results_dict = {int(k.split(exp_name + "_")[-1]): v for k, v in
                                sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                       key=lambda item: int(item[0].split(exp_name + "_")[-1]))}
        except ValueError as ve:
            try:
                exp_results_dict = {int(k.split(exp_name)[-1]): v for k, v in
                                sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                       key=lambda item: int(item[0].split(exp_name)[-1]))}
            except ValueError as ve:
                exp_results_dict = {int(k.split("_")[-1]): v for k, v in
                                    sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                           key=lambda item: int(item[0].split("_")[-1]))}
        title = f'{data_name}---time_comparison'
        if len(exp_results_dict) == 0:
            continue
        if len(list(exp_results_dict.values())[0]) == 2:  # if no embT  # not 'EmbT' in exp_name:
            scores_list = [v[1] for k, v in exp_results_dict.items()]
            scores_list = [v / 10 for v in scores_list]
        else:
            scores_list = [v[2] for k, v in exp_results_dict.items()]
        full_epoch_time = scores_list[0] / 10
        if '10_epoch' in exp_name:
            scores_list = [v + 10 * full_epoch_time for v in scores_list]
        elif '1_epoch' in exp_name:
            scores_list = [v + 1 * full_epoch_time for v in scores_list]

        plot_acc_to_textname(exp_names=exp_results_dict.keys(), scores_list=scores_list,
                             colors=get_graph_color(exp_num), offset=0.0,
                             label='Hours', add_labels=False,
                             x_ticks=list(exp_results_dict.keys()), title=title, baseline=full_epoch_time * 10,
                             exp_num=exp_num, exp_name=exp_name)
    plt.legend()
    pdf.savefig(plt.gcf())
    plt.show()


def display_timeline_progression_scheme_reduction_before_time_disp(exp_names="experiment_conditional_entropy_removed",
                                                                   data_name="mondial"):
    ########################################################
    exp_names = [exp_names] if type(exp_names) != list else exp_names

    for exp_num, exp_name in enumerate(exp_names):
        exp_dirs = get_all_exp_with_exp_name(exp_name, data_name)
        try:
            exp_results_dict = {int(k.split(exp_name + "_")[-1]): v for k, v in
                                sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                       key=lambda item: int(item[0].split(exp_name + "_")[-1]))}
        except ValueError as ve:
            exp_results_dict = {int(k.split(exp_name)[-1]): v for k, v in
                                sorted(get_results_from_dirs(exp_dirs, ret_dict=True).items(),
                                       key=lambda item: int(item[0].split(exp_name)[-1]))}
        pprint.pprint(exp_results_dict)
        title = f'{data_name}---{exp_name}'
        plot_acc_to_textname(exp_names=exp_results_dict.keys(), scores_list=[v[0] for k, v in exp_results_dict.items()],
                             colors=get_graph_color(exp_num), offset=0.0,
                             label='Accuracy', add_labels=False,
                             x_ticks=list(exp_results_dict.keys()), title=title, baseline=exp_baselines[data_name],
                             exp_num=exp_num, exp_name=exp_name)
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.bar(exp_results_dict.keys(), [v[1] for k, v in exp_results_dict.items()], color="purple")
    plt.xticks(list(exp_results_dict.keys()))
    plt.ylabel('Hours')
    # plt.xlabel('Test name')
    plt.title(title)
    plt.show()

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

if __name__ == '__main__':
    pass
    # vis.acc_per_time_visualization()
    # exit()
    # display all in acc/experiment graph:
    # display_res_ends_in_specific_column()

    # Conditional Entropy – remove schemes by “weakest link” (the highest entropy in one of the edges)
    # display_timeline_progression_scheme_reduction(exp_names="experiment_conditional_entropy_removed")

    # Reverse Conditional Entropy – remove schemes by “weakest link” (the LOWEST highest entropy in one of the edges)
    #  for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  #, "mondial_original_target"]
    #      display_timeline_progression_scheme_reduction(exp_names="conditional_entropy_from_low_to_high_removed", data_name=data_name)

    # Conditional Entropy – from start table to end table - remove schemes with ***HIGH*** entropy
    # display_timeline_progression_scheme_reduction(exp_names="conditional_entropy_start_to_end_high_to_low_removed")

    # Conditional Entropy – from start table to end table - remove schemes with ***LOW*** entropy
    # display_timeline_progression_scheme_reduction(exp_names="conditional_entropy_start_to_end_low_to_high_removed")

    # remove schemes sorted by their loss after the first epoch (LOWEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  #, "mondial_original_target"]
    #    display_timeline_progression_scheme_reduction(exp_names="sorted_by_loss_after_1_epoch", data_name=data_name)

    # remove schemes sorted by their loss after the first epoch (HIGHEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3","mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #    display_timeline_progression_scheme_reduction(exp_names="sorted_by_highest_loss_after_1_epoch", data_name=data_name)

    # compare "LOWEST weakest link" to remove schemes sorted by their length
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["conditional_entropy_from_low_to_high_removed", "remove_longest_schemes"], data_name=data_name)

    # compare "LOWEST weakest link" to remove random schemes shuffle0
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["conditional_entropy_from_low_to_high_removed", "sorted_by_shuffle0"],data_name=data_name)

    # remove schemes sorted by their loss after the LAST (10) epoch (HIGHEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3","mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names="sorted_by_highest_loss_after_10_epoch", data_name=data_name)

    # remove schemes sorted by their loss after the LAST (10) epoch (LOWEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names="sorted_by_lowest_loss_after_10_epoch", data_name=data_name)

    # compare "LOWEST weakest link" to LAST (10) epoch (LOWEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["conditional_entropy_from_low_to_high_removed", "sorted_by_lowest_loss_after_10_epoch"], data_name=data_name)

    # compare LAST (10) epoch (HIGHEST LOSS) to LAST (10) epoch (LOWEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_by_highest_loss_after_10_epoch", "sorted_by_lowest_loss_after_10_epoch"], data_name=data_name)

    # compare remove random schemes shuffle0 to shuffle1
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["conditional_entropy_from_low_to_high_removed", "sorted_by_shuffle1"],data_name=data_name)

    # compare remove schemes sorted by their loss after the LAST (10) epoch (HIGHEST LOSS) to (LOWEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3","mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_correct_lowest_loss_after_10_epoch", "sorted_correct_highest_loss_after_10_epoch"], data_name=data_name)

    # # compare remove schemes sorted by their loss after the LAST (10) epoch (HIGHEST LOSS) to (LOWEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3","mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_by_loss_after_1_epoch", "sorted_by_shuffle1", "remove_longest_schemes", "conditional_entropy_from_low_to_high_removed"],data_name=data_name)

    # # run experiments on genes dataset
    # for data_name in ["genes"]:  # , "mondial_original_target"] # ,"correct_lowest_loss_after_10_epoch"
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_genes_mean_lowest_loss_after_1_epoch_", "sorted_by_shuffle1", "remove_longest_schemes", "conditional_entropy_from_low_to_high_removed"],data_name=data_name)

    # # Compare (10) epoch (mean LOWEST LOSS) to (sum LOWEST LOSS)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3","mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_correct_lowest_loss_after_10_epoch", "sorted_mean_lowest_loss_after_10_epoch"],data_name=data_name)

    # run experiments on hepatitis dataset
    # for data_name in ["hepatitis"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_by_shuffle0", "remove_longest_schemes","conditional_entropy_from_low_to_high_removed","sorted_hepatitis_correct_lowest_loss_after_10_epoch", "sorted_hepatitis_correct_highest_loss_after_10_epoch"], data_name=data_name)

    # # run experiments on hepatitis dataset
    # for data_name in ["hepatitis"]:  # , "mondial_original_target"]
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_hepatitis_mean_lowest_loss_after_1_epoch","sorted_by_shuffle0", "remove_longest_schemes","conditional_entropy_from_low_to_high_removed"
    #                                                              #"sorted_hepatitis_correct_lowest_loss_after_10_epoch",
    #                                                              ],data_name=data_name)

    # # run experiments on mutagenesis dataset
    # for data_name in ["mutagenesis"]:
    #     for exp in [["sorted_mutagenesis_mean_lowest_loss_after_1_epoch_",'sorted_by_shuffle2', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed']]: #, 'sorted_mutagenesis_mean_highest_loss_after_10_epoch', 'sorted_mutagenesis_mean_lowest_loss_after_10_epoch'
    #         display_timeline_progression_scheme_reduction(exp_names=exp, data_name=data_name)

    # # run experiments on world dataset
    # for data_name in ["world"]:
    #     for exp in [['sorted_world_mean_lowest_loss_after_1_epoch', 'sorted_by_shuffle0', 'remove_longest_schemes',
    #                 'conditional_entropy_from_low_to_high_removed',
    #                 'sorted_world_mean_highest_loss_after_10_epoch']]:
    #         display_timeline_progression_scheme_reduction(exp_names=exp, data_name=data_name)

    # # compare results mondial with time compare
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3","mondial_target_Inflation_g6", "mondial"]:
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_by_loss_after_1_epoch", "sorted_by_shuffle1", "remove_longest_schemes","conditional_entropy_from_low_to_high_removed"], data_name=data_name)

    # # compare results mondial with time compare
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3","mondial_target_Inflation_g6", "mondial"]:
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_mean_lowest_loss_after_10_epoch", "sorted_by_loss_after_1_epoch", "sorted_by_shuffle1"], data_name=data_name)

    # dynamic scheme reduction experiments
    # for data_name in ["mondial"]:
    #     display_timeline_progression_scheme_reduction(exp_names=["dynamic"],data_name=data_name)
    # TODO: display excel to see how much time we save
    # for data_name in ["mondial"]:
    #     save_exp_info_for_cmp(exp_names=["experiment_dynamic", "EK_3_100_500_10_50000_0experiment_remove_longest_schemes_0"],
    #                           data_name=data_name)

    # excel dynamic scheme reduction
    # for data_name in ["mondial_small_sample_data_EXP_mul" + mul_by for mul_by in ["0.1", "0.2", "0.4", "0.5", "0.6", "0.8"]]:
    #     save_exp_info_for_cmp(exp_names=["mul"], data_name=data_name)

    # TODO: display this in slides
    # for data_name in ["mondial"]:
    #     # display_timeline_progression_scheme_reduction(exp_names=["mul_highest_0.4", "mul_highest_0.5"], data_name=data_name)
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_by_shuffle1","sorted_mean_lowest_loss_after_1_epoch" , "tryout_mul_0.4", "tryout_mul_0.5", "conditional_entropy_from_low_to_high_removed"], data_name=data_name)

    # TOD: tryout 100
    # for data_name in ["mondial"]:
    #     display_timeline_progression_scheme_reduction(exp_names=["sorted_mean_lowest_loss_after_1_epoch", "sorted_by_shuffle1", "remove_longest_schemes", "conditional_entropy_from_low_to_high_removed", "num_samples100_mul_0.4"], data_name=data_name)

    # for data_name in ["hepatitis","mutagenesis"]:
    #     display_timeline_progression_scheme_reduction(exp_names=[f'sorted_by_shuffle0', f'remove_longest_schemes',f'conditional_entropy_from_low_to_high_removed',
    #             f'sorted_{data_name}_mean_highest_loss_after_10_epoch', f'sorted_{data_name}_mean_lowest_loss_after_10_epoch',
    #             f'sorted_{data_name}_mean_highest_loss_after_1_epoch', f'sorted_{data_name}_mean_lowest_loss_after_1_epoch'], data_name=data_name, must_contain='EK_5')
    #
    # for data_name in ["mondial"]:
    #     save_exp_info_for_cmp(exp_names=["dynamic_d4"], data_name=data_name)

    # for data_name in ["mondial_fuzz"]:
    #     save_exp_info_for_cmp(exp_names=["_"], data_name=data_name)
    # exit()

    # # france experiments # , "world_B"
    data_names = ["genes", "genes_essential", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial_original_target", "hepatitis", "mutagenesis"]
    regular_exps = ['sorted_by_shuffle1', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed']
    # creat the pdf:
    # for data_name in data_names:
    #     display_timeline_progression_scheme_reduction(exp_names=regular_exps + [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart1"] + [f"low_loss_tryout_{data_name}_try_out_mul0.4"], data_name=data_name)
    # pdf.close()
    # exit()
    # for data_name in data_names:
    #     save_exp_info_for_cmp(exp_names=["all"], data_name=data_name)
    # exit()
    # # "mutagenesis","hepatitis",
    # exp_names = ['all_schemes']
    # for data_name in data_names:
    #     # TODO: add here the tryout exp's
    #     display_timeline_progression_scheme_reduction(exp_names=regular_exps + [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart1"] + [f"low_loss_tryout_{data_name}_try_out_mul0.4"], data_name=data_name)
    # for data_name in ["mondial_original_target"]:
    #     display_timeline_progression_scheme_reduction(exp_names=regular_exps + ["sorted_norm", "sorted_rev_norm"], data_name=data_name)  #
    # for data_name in ["world_B"]:
    #     display_timeline_progression_scheme_reduction(exp_names=regular_exps + ["distribution_var", "rev_distribution_var"], data_name=data_name)
    pdf.close()