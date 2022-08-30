import subprocess
import re
from configparser import ConfigParser
import time

def set_backtrack_to(bol):
    config = ConfigParser()
    config.read("config.txt")
    config["BACKTRACK"]["backtrack"] = bol
    with open('config.txt', 'w') as configfile:
        config.write(configfile)

def write_log(s, backslash='\n'):
    with open('log.txt', 'a') as log:
        log.write(s + backslash)

if __name__ == '__main__':
    # out: world(regular), genes_essential,mutagenesis,hepatitis
    data_names = ["mutagenesis", "world_B", "hepatitis", "genes", "mondial_original_target",
                  "mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
                  "mondial_target_Inflation_g6"]
    data_name_to_num_schemes = {"mutagenesis": 58, "world": 20, "hepatitis": 21, "genes": 32, "genes_essential": 32,
                                "mondial_target_infant_mortality_g40": 63,
                                "mondial_target_continent": 63, "mondial_target_GDP_g8e3": 63,
                                "mondial_target_Inflation_g6": 63,
                                "mondial_original_target": 63, "world_B": 60}
    data_name_to_depth = {"mutagenesis": 4, "world": 3, "hepatitis": 3, "genes": 3, "genes_essential": 3,
                          "mondial_target_infant_mortality_g40": 3, "mondial_target_continent": 3,
                          "mondial_target_GDP_g8e3": 3, "world_B": 3,
                          "mondial_target_Inflation_g6": 3, "mondial_original_target": 3}
    data_name_to_total_time = {"mondial_target_infant_mortality_g40": 250, "mondial_target_continent": 250,
                               "mondial_target_GDP_g8e3": 250, "mondial_target_Inflation_g6": 250,
                               "mondial_original_target": 250, "genes": 370, "hepatitis": 250, "mutagenesis": 300,
                               "world_B": 370}
    data_name_to_is_backtrack = {"mutagenesis": "True", "hepatitis": "True", "genes": "False",
                                "mondial_target_infant_mortality_g40": "False",
                                "mondial_target_continent": "False", "mondial_target_GDP_g8e3": "False",
                                "mondial_target_Inflation_g6": "False",
                                "mondial_original_target": "False", "world_B": "True"}

    all_tasks = ["distribution_var", 'sorted_by_shuffle42', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed',
                 "sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}",  #TODO: work on Yan81 version
                 "sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}",
                 "dynamic"]
    done_taks = []
    first_tasks_to_run = ['conditional_entropy_from_low_to_high_removed']
    second_tasks_to_run = ["distribution_var", 'remove_longest_schemes', 'sorted_by_shuffle42']
    # tasks = first_tasks_to_run

    # tasks = ["rev_min_mutual_information", "min_mutual_information", "max_mutual_information", "rev_max_mutual_information"]
    # #write_log(f"Starting task {tasks[0]}")
    # for data_name in ["mondial_original_target"]:
    #     write_log(f"    Starting data_name {data_name}")
    #     for task in tasks:
    #         write_log(f"Starting task {task}")
    #         set_backtrack_to(data_name_to_is_backtrack[data_name])
    #         num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], \
    #                                          data_name_to_total_time[data_name]
    #         for i in range(0, num_schemes, 8):  # TODO: choose jumping
    #             subprocess.run(
    #                 f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    # exit()
    #####################################################################################
    shuffle_dict = {"mutagenesis": [0,52], "hepatitis": [0,4], "genes": [0,24],  "mondial_target_infant_mortality_g40": [0,48],
                                "mondial_target_continent": [0,48], "mondial_target_GDP_g8e3": [0,48],
                                "mondial_target_Inflation_g6": [0,36], "mondial_original_target": [0,40], "world_B": [0,36]}
    k_var_dict = {"mutagenesis": [48,44,20], "hepatitis": [12,8], "genes": [20,16,24], "mondial_target_infant_mortality_g40": [52,48],
                                "mondial_target_continent": [52,48], "mondial_target_GDP_g8e3": [44,48,36],
                                "mondial_target_Inflation_g6": [44,48,52], "mondial_original_target": [44,48], "world_B": [44,40]}
    len_dict = {"mutagenesis": [48], "hepatitis": [4], "genes": [20], "mondial_target_infant_mortality_g40": [16],
                  "mondial_target_continent": [20], "mondial_target_GDP_g8e3": [4],
                  "mondial_target_Inflation_g6": [12], "mondial_original_target": [20], "world_B": [48]}
    ep1_dict = {"mutagenesis": [36], "hepatitis": [4], "genes": [20], "mondial_target_infant_mortality_g40": [48],
                  "mondial_target_continent": [52], "mondial_target_GDP_g8e3": [48],
                  "mondial_target_Inflation_g6": [48], "mondial_original_target": [56], "world_B": [44]}
    smple_yan_dict = {"mutagenesis": [28, 24], "hepatitis": [12,16], "genes": [28,24], "mondial_target_infant_mortality_g40": [52,48],
                                "mondial_target_continent": [32,28], "mondial_target_GDP_g8e3": [44,40],
                                "mondial_target_Inflation_g6": [4,44,48], "mondial_original_target": [36,32], "world_B": [44,40]}

    # # Run online scheme reduction
    # write_log(f"Starting task - dynamic - online scheme reduction")
    # for data_name in data_names:
    #     set_backtrack_to(data_name_to_is_backtrack[data_name])
    #     write_log(f"    Starting data_name {data_name}")
    #     depth, total_time = data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #     for exp in ["dynamic"]:
    #         for percentage, num_epochs in [(66, 2), (85, 4), (95, 7)]:  # (0,1)
    #             write_log(f"        {(percentage, num_epochs)}", backslash=" ")
    #             subprocess.run(f"python forward_split_Loss_per_scheme_acc_each_epoch.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth} --epoch 999 --threshold {total_time}".split())
    #     write_log(f"")

    # tasks = ["rev_min_mutual_information"]
    # for task in tasks:
    #     write_log(f"Starting task {task}")
    #     for data_name in data_names[::-1]:
    #         write_log(f"    Starting data_name {data_name}")
    #         set_backtrack_to(data_name_to_is_backtrack[data_name])
    #         num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #         for i in range(4, num_schemes, 4):
    #             write_log(f"        {i}", backslash=" ")
    #             subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    #         write_log(f"")
    tasks = ["distribution_v"]
    for task in tasks:
        write_log(f"Starting task {task}")
        for num_smpls in [10, 25, 3]:
            for data_name in k_var_dict.keys():
                write_log(f"    Starting data_name {data_name}")
                set_backtrack_to(data_name_to_is_backtrack[data_name])
                num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
                for i in k_var_dict[data_name]:
                    subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}{num_smpls}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    exit()
    # # run yan81 mul sampling tryout
    # write_log(f"Starting task low_loss_yan81")
    # for data_name in smple_yan_dict.keys():  # data_names:
    #     set_backtrack_to(data_name_to_is_backtrack[data_name])
    #     write_log(f"    Starting data_name {data_name}")
    #     for mul_by in [0.33]:
    #         num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #         try_out_time_str = subprocess.check_output(f"python try_out_utils.py --method yan81 --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
    #         try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #         print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
    #         # for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
    #         for exp in [f"low_loss_yan81_{data_name}_yan81_mul{mul_by}"]:
    #             for i in smple_yan_dict[data_name]:
    #                 subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str} --tryout True --epoch 999 --threshold {total_time}".split())

    # for task in tasks:
    #     write_log(f"Starting task {task}")
    #     for data_name in len_dict.keys():
    #         write_log(f"    Starting data_name {data_name}")
    #         set_backtrack_to(data_name_to_is_backtrack[data_name])
    #         num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #         for i in len_dict[data_name]:
    #             subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    #
    # write_log(f"Starting task lowest_loss_after_1_epoch")
    # for data_name in ep1_dict.keys():
    #     set_backtrack_to(data_name_to_is_backtrack[data_name])
    #     write_log(f"    Starting data_name {data_name}")
    #     for epoch in [1]:
    #         num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #         try_out_time_str = subprocess.check_output(f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
    #         try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #         print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
    #         for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
    #             for i in ep1_dict[data_name]:
    #                 subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str} --tryout True --epoch 999 --threshold {total_time}".split())

    # tasks = ["sorted_by_shuffle42"]
    # for task in tasks:
    #     write_log(f"Starting task {task}")
    #     for data_name in ["mondial_original_target"]:
    #         write_log(f"    Starting data_name {data_name}")
    #         set_backtrack_to(data_name_to_is_backtrack[data_name])
    #         num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #         for i in [0]:
    #             ti = time.time()
    #             # print(f"start - {time.time()}")
    #             subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    #             # print(f"end   - {time.time()}")
    #             # print(f"total - {time.time() - ti}")

    exit()
    # tasks = ["distribution_var"]
    # for task in tasks:
    #     write_log(f"Starting task {task}")
    #     for data_name in shuffle_dict.keys():
    #         write_log(f"    Starting data_name {data_name}")
    #         set_backtrack_to(data_name_to_is_backtrack[data_name])
    #         num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #         for i in k_var_dict[data_name]:
    #             subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    exit()
    #####################################################################################
    """# run yan81 mul sampling tryout
    write_log(f"Starting task low_loss_yan81")
    for data_name in data_names:  # data_names:
        set_backtrack_to(data_name_to_is_backtrack[data_name])
        write_log(f"    Starting data_name {data_name}")
        for mul_by in [0.33]:
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            try_out_time_str = subprocess.check_output(f"python try_out_utils.py --method yan81 --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
            # for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
            for exp in [f"low_loss_yan81_{data_name}_yan81_mul{mul_by}"]:
                for i in range(0, num_schemes, 4):
                    subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str} --tryout True --epoch 999 --threshold {total_time}".split())
    """
    # # Run online scheme reduction
    # write_log(f"Starting task - dynamic - online scheme reduction")
    # for data_name in data_names:
    #     set_backtrack_to(data_name_to_is_backtrack[data_name])
    #     write_log(f"    Starting data_name {data_name}")
    #     depth, total_time = data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #     for exp in ["dynamic"]:
    #         for percentage, num_epochs in [(66, 2), (85, 4), (95, 7), (0, 1)]:
    #             subprocess.run(f"python forward_split_Loss_per_scheme_acc_each_epoch.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth} --epoch 999 --threshold {total_time}".split())
    exit()

    tasks = second_tasks_to_run
    for task in tasks:
        write_log(f"Starting task {task}")
        for data_name in data_names:
            write_log(f"    Starting data_name {data_name}")
            set_backtrack_to(data_name_to_is_backtrack[data_name])
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], \
                                             data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 4):  # TODO: choose jumping
                subprocess.run(
                    f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    exit()
    #TODO run the "Sampling" exp
    #TODO run a BAD exp for comparenson

    # run online scheme reduction on mondial_original_target and genes
    # for data_name in ["mondial_original_target", "genes"]:
    #     depth, total_time = data_name_to_depth[data_name], data_name_to_total_time[data_name]
    #     for exp in ["dynamic"]:
    #         for percentage, num_epochs in [(66, 2), (85, 4), (95, 7), (0, 1)]:
    #             subprocess.run(
    #                 f"python forward_split_Loss_per_scheme_acc_each_epoch.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth} --epoch 999 --threshold {total_time}".split())
    # exit()


    # # run simple mul tryout
    # for data_name in data_names:
    #     if data_name in ["mutagenesis", "hepatitis", "world_B", "world"]:
    #         continue
    #     for mul_by in [0.4]:  # [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
    #         try:
    #             num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #             try_out_time_str = subprocess.check_output(
    #                 f"python try_out_utils.py --method tryout --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
    #             try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #             print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
    #             # for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
    #             for exp in [f"low_loss_tryout_{data_name}_try_out_mul{mul_by}"]:
    #                 for i in range(0, num_schemes, 4):
    #                     subprocess.run(
    #                         f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())
    #         except Exception as e:
    #             with open('log.txt', 'a') as log:
    #                 log.write(f"Error at: python forward.py --data_name {data_name} --depth {depth} --yuval_change low_loss_tryout_{data_name}_try_out_mul{mul_by}_?\n{e}\n")


    # # run lowest_loss_after_1_epoch
    # for data_name in data_names:
    #     for epoch in [1]:
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         try_out_time_str = subprocess.check_output(
    #             f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
    #         try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #         print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
    #         for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
    #             for i in range(0, num_schemes, 4):
    #                 subprocess.run(
    #                     f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())

    # # run simple mul tryout
    # for data_name in ["hepatitis", "world_B"]:
    #     for mul_by in [0.4]:  # [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         try_out_time_str = subprocess.check_output(
    #             f"python try_out_utils.py --method tryout --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
    #         try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #         print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
    #         # for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
    #         for exp in [f"low_loss_tryout_{data_name}_try_out_mul{mul_by}"]:
    #             for i in range(0, num_schemes, 4):
    #                 subprocess.run(
    #                     f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())


def full_first_exps():
    # out: world(regular), genes_essential,mutagenesis,hepatitis
    data_names = ["mutagenesis", "world_B", "hepatitis", "genes", "mondial_original_target",
                  "mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
                  "mondial_target_Inflation_g6"]
    data_name_to_num_schemes = {"mutagenesis": 58, "world": 20, "hepatitis": 21, "genes": 32, "genes_essential": 32,
                                "mondial_target_infant_mortality_g40": 63,
                                "mondial_target_continent": 63, "mondial_target_GDP_g8e3": 63,
                                "mondial_target_Inflation_g6": 63,
                                "mondial_original_target": 63, "world_B": 60}
    data_name_to_depth = {"mutagenesis": 4, "world": 3, "hepatitis": 3, "genes": 3, "genes_essential": 3,
                          "mondial_target_infant_mortality_g40": 3, "mondial_target_continent": 3,
                          "mondial_target_GDP_g8e3": 3, "world_B": 3,
                          "mondial_target_Inflation_g6": 3, "mondial_original_target": 3}
    data_name_to_total_time = {"mondial_target_infant_mortality_g40": 250, "mondial_target_continent": 250,
                               "mondial_target_GDP_g8e3": 250, "mondial_target_Inflation_g6": 250,
                               "mondial_original_target": 250, "genes": 370, "hepatitis": 250, "mutagenesis": 300,
                               "world_B": 370}
    data_name_to_is_backtrack = {"mutagenesis": "True", "hepatitis": "True", "genes": "False",
                                "mondial_target_infant_mortality_g40": "False",
                                "mondial_target_continent": "False", "mondial_target_GDP_g8e3": "False",
                                "mondial_target_Inflation_g6": "False",
                                "mondial_original_target": "False", "world_B": "True"}

    all_tasks = ["distribution_var", 'sorted_by_shuffle42', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed',
                 "sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}",  #TODO: work on Yan81 version
                 "sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}",
                 "dynamic"]
    done_taks = []
    first_tasks_to_run = ['conditional_entropy_from_low_to_high_removed']
    second_tasks_to_run = ["distribution_var", 'remove_longest_schemes', 'sorted_by_shuffle42']
    tasks = first_tasks_to_run
    for task in tasks:
        write_log(f"Starting task {task}")
        for data_name in data_names:
            if data_name in ["mutagenesis", "world_B", "hepatitis", "genes"]:  # did it already
                continue
            write_log(f"    Starting data_name {data_name}")
            set_backtrack_to(data_name_to_is_backtrack[data_name])
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 4):  #TODO: choose jumping
                subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())

    # run lowest_loss_after_1_epoch
    write_log(f"Starting task lowest_loss_after_1_epoch")
    for data_name in data_names:
        set_backtrack_to(data_name_to_is_backtrack[data_name])
        write_log(f"    Starting data_name {data_name}")
        for epoch in [1]:
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            try_out_time_str = subprocess.check_output(
                f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
            for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
                for i in range(0, num_schemes, 4):
                    subprocess.run(
                        f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str} --tryout True --epoch 999 --threshold {total_time}".split())

    # Run online scheme reduction
    write_log(f"Starting task - dynamic - online scheme reduction")
    for data_name in ["mondial_original_target", "genes"]:
        set_backtrack_to(data_name_to_is_backtrack[data_name])
        write_log(f"    Starting data_name {data_name}")
        depth, total_time = data_name_to_depth[data_name], data_name_to_total_time[data_name]
        for exp in ["dynamic"]:
            for percentage, num_epochs in [(66, 2), (85, 4), (95, 7), (0, 1)]:
                subprocess.run(f"python forward_split_Loss_per_scheme_acc_each_epoch.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth} --epoch 999 --threshold {total_time}".split())

    tasks = second_tasks_to_run
    for task in tasks:
        write_log(f"Starting task {task}")
        for data_name in data_names:
            write_log(f"    Starting data_name {data_name}")
            set_backtrack_to(data_name_to_is_backtrack[data_name])
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], \
                                             data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 4):  # TODO: choose jumping
                subprocess.run(
                    f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    exit()

    tasks = ["rev_min_mutual_information"]
    write_log(f"Starting task {tasks[0]}")
    for data_name in data_names:
        for task in tasks:
            write_log(f"    Starting data_name {data_name}")
            set_backtrack_to(data_name_to_is_backtrack[data_name])
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], \
                                             data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 4)[::-1]:  # TODO: choose jumping
                subprocess.run(
                    f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())

def full_second_exps():
    # out: world(regular), genes_essential,mutagenesis,hepatitis
    data_names = ["mutagenesis", "world_B", "hepatitis", "genes", "mondial_original_target",
                  "mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
                  "mondial_target_Inflation_g6"]
    data_name_to_num_schemes = {"mutagenesis": 58, "world": 20, "hepatitis": 21, "genes": 32, "genes_essential": 32,
                                "mondial_target_infant_mortality_g40": 63,
                                "mondial_target_continent": 63, "mondial_target_GDP_g8e3": 63,
                                "mondial_target_Inflation_g6": 63,
                                "mondial_original_target": 63, "world_B": 60}
    data_name_to_depth = {"mutagenesis": 4, "world": 3, "hepatitis": 3, "genes": 3, "genes_essential": 3,
                          "mondial_target_infant_mortality_g40": 3, "mondial_target_continent": 3,
                          "mondial_target_GDP_g8e3": 3, "world_B": 3,
                          "mondial_target_Inflation_g6": 3, "mondial_original_target": 3}
    data_name_to_total_time = {"mondial_target_infant_mortality_g40": 250, "mondial_target_continent": 250,
                               "mondial_target_GDP_g8e3": 250, "mondial_target_Inflation_g6": 250,
                               "mondial_original_target": 250, "genes": 370, "hepatitis": 250, "mutagenesis": 300,
                               "world_B": 370}
    data_name_to_is_backtrack = {"mutagenesis": "True", "hepatitis": "True", "genes": "False",
                                "mondial_target_infant_mortality_g40": "False",
                                "mondial_target_continent": "False", "mondial_target_GDP_g8e3": "False",
                                "mondial_target_Inflation_g6": "False",
                                "mondial_original_target": "False", "world_B": "True"}

    all_tasks = ["distribution_var", 'sorted_by_shuffle42', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed',
                 "sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}",  #TODO: work on Yan81 version
                 "sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}", "rev_min_mutual_information",
                 "dynamic"]
    done_taks = []
    # first_tasks_to_run = ['conditional_entropy_from_low_to_high_removed']
    # second_tasks_to_run = ["rev_min_mutual_information", "distribution_var", 'remove_longest_schemes', 'sorted_by_shuffle42']
    forward_less_tasks = ["rev_min_mutual_information", "distribution_var", 'remove_longest_schemes', 'sorted_by_shuffle42']
    more_random_tasks = ['sorted_by_shuffle1', 'sorted_by_shuffle2']
    tasks = forward_less_tasks
    for task in tasks:
        write_log(f"Starting task {task}")
        for data_name in data_names:
            write_log(f"    Starting data_name {data_name}")
            set_backtrack_to(data_name_to_is_backtrack[data_name])
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            for i in range(4, num_schemes, 4):
                subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())

    # run lowest_loss_after_1_epoch
    write_log(f"Starting task lowest_loss_after_1_epoch")
    for data_name in data_names:
        set_backtrack_to(data_name_to_is_backtrack[data_name])
        write_log(f"    Starting data_name {data_name}")
        for epoch in [1]:
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            try_out_time_str = subprocess.check_output(f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
            for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
                for i in range(4, num_schemes, 4):
                    subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str} --tryout True --epoch 999 --threshold {total_time}".split())

    # run yan81 mul sampling tryout
    write_log(f"Starting task low_loss_yan81")
    for data_name in data_names:
        set_backtrack_to(data_name_to_is_backtrack[data_name])
        write_log(f"    Starting data_name {data_name}")
        for mul_by in [0.33]:
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            try_out_time_str = subprocess.check_output(f"python try_out_utils.py --method yan81 --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
            # for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
            for exp in [f"low_loss_yan81_{data_name}_yan81_mul{mul_by}"]:
                for i in range(4, num_schemes, 4):
                    subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str} --tryout True --epoch 999 --threshold {total_time}".split())

    # Run online scheme reduction
    write_log(f"Starting task - dynamic - online scheme reduction")
    for data_name in data_names:
        set_backtrack_to(data_name_to_is_backtrack[data_name])
        write_log(f"    Starting data_name {data_name}")
        depth, total_time = data_name_to_depth[data_name], data_name_to_total_time[data_name]
        for exp in ["dynamic"]:
            for percentage, num_epochs in [(66, 2), (85, 4), (95, 7)]:  # (0,1)
                subprocess.run(f"python forward_split_Loss_per_scheme_acc_each_epoch.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth} --epoch 999 --threshold {total_time}".split())
    exit()