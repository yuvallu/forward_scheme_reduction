import subprocess
import re
from configparser import ConfigParser

def set_backtrack_to(bol):
    config = ConfigParser()
    config.read("config.txt")
    config["BACKTRACK"]["backtrack"] = bol
    with open('config.txt', 'w') as configfile:
        config.write(configfile)

def write_log(s):
    with open('log.txt', 'a') as log:
        log.write(s + '\n')

if __name__ == '__main__':
    # out: world(regular), genes_essential,mutagenesis,hepatitis
    data_names = ["mutagenesis", "world_B", "hepatitis", "genes", "mondial_original_target"
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
    second_tasks_to_run = ['sorted_by_shuffle42', 'remove_longest_schemes', "distribution_var"]
    tasks = first_tasks_to_run
    for task in tasks:
        write_log(f"Starting task {task}")
        for data_name in data_names:
            write_log(f"Starting data_name {data_name}")
            set_backtrack_to(data_name_to_is_backtrack[data_name])
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 4):  #TODO: choose jumping
                subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())

    # run lowest_loss_after_1_epoch
    write_log(f"Starting task {task}")
    for data_name in data_names:
        write_log(f"Starting data_name {data_name}")
        for epoch in [1]:
            num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
            try_out_time_str = subprocess.check_output(f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
            for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
                for i in range(0, num_schemes, 4):
                    subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str} --tryout True --epoch 999 --threshold {total_time}".split())

    tasks = second_tasks_to_run
    for task in tasks:
        write_log(f"Starting task {task}")
        for data_name in data_names:
            write_log(f"Starting data_name {data_name}")
            set_backtrack_to(data_name_to_is_backtrack[data_name])
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], \
                                             data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 4):  # TODO: choose jumping
                subprocess.run(
                    f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {task}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    exit()
    # set_backtrack_to("False")
    # # TODO: run online scheme reduction on mondial_original_target and genes
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

