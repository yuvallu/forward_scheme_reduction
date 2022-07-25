import subprocess
import re
from configparser import ConfigParser

if __name__ == '__main__':
    # for i in [2,3,4]:
    #    subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change scheme_len_eq_1_{i}")
    # for i in range(2,6):
    #    subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change random{i}")
    # for i in [1, 2, 3, 4]:
    #     subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change scheme_len_eq_{i}")
    # for i in [1]:
    #     subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change specific{i}")
    #
    # subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change scheme_len_eq_3")
    #
    # # test r6 scheme on other column predictions
    # for exp in ["mondial_target_infant_mortality_g40","mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial_original_target"]:
    #     subprocess.run(f"python forward.py --data_name {exp}")
    #     subprocess.run(f"python forward.py --data_name {exp} --yuval_change random6")
    #
    # #test r9 (bad) scheme on other column predictions
    # for exp in ["mondial_target_infant_mortality_g40","mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial_original_target"]:
    #    subprocess.run(f"python forward.py --data_name {exp} --yuval_change random9")
    #
    # # check robustness of Reverse Conditional Entropy – remove schemes by “weakest link” (the LOWEST highest entropy in one of the edges)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
    #            "mondial_target_Inflation_g6"]:  #, "mondial_original_target"]:
    #     for i in range(3, 53, 4):
    #         subprocess.run(
    #             f"python forward.py --data_name {data_name} --yuval_change conditional_entropy_from_low_to_high_removed_{i}")

    # run tests on other datasets
    # # remove longest scheme
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change remove_longest_schemes_{i}")
    # # remove random
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change sorted_by_shuffle0_{i}")
    # # LOWEST "weakest link"
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change conditional_entropy_from_low_to_high_removed_{i}")
    # # sorted_correct_highest_loss_after_10_epoch
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change sorted_genes_correct_highest_loss_after_10_epoch{i}")
    # # sorted_correct_lowest_loss_after_10_epoch
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change sorted_genes_correct_lowest_loss_after_10_epoch_{i}")
    # # remove longest scheme
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change remove_longest_schemes_{i}")
    # # remove random
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_by_shuffle0_{i}")
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_by_shuffle1_{i}")
    # # LOWEST "weakest link"
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change conditional_entropy_from_low_to_high_removed_{i}")
    #     exit()
    # # sorted_correct_highest_loss_after_10_epoch
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_hepatitis_correct_highest_loss_after_10_epoch_{i}")
    # # sorted_correct_lowest_loss_after_10_epoch
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_hepatitis_correct_lowest_loss_after_10_epoch_{i}")

    # # mutagenesis
    # exps = ['sorted_by_shuffle0', 'remove_longest_schemes','conditional_entropy_from_low_to_high_removed',
    #         'sorted_mutagenesis_mean_highest_loss_after_10_epoch', 'sorted_mutagenesis_mean_lowest_loss_after_10_epoch'
    # ]
    # for exp in exps:
    #     for i in range(0, 15, 4):
    #         subprocess.run(f"python forward.py --data_name mutagenesis --depth 4 --yuval_change {exp}_{i}")
    #         exit()

    # # world
    # exps = ['sorted_by_shuffle0', 'remove_longest_schemes','conditional_entropy_from_low_to_high_removed', 'sorted_world_mean_highest_loss_after_10_epoch', 'sorted_world_mean_lowest_loss_after_10_epoch']
    # exps = ['conditional_entropy_from_low_to_high_removed']
    # for exp in exps:
    #     for i in range(0, 20, 4):
    #         subprocess.run(f"python forward.py --data_name world --yuval_change {exp}_{i}")

    # # run dynamic scheme reduction experiments
    # for data_name in ["mondial"]:
    #     for exp in ["dynamic"]:
    #         for percentage in [45, 60, 75]:
    #             for num_epochs in [1, 2, 3, 4]:
    #                 percentage = 60
    #                 num_epochs = 1
    #                 subprocess.run(
    #                     f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs}")
    #                 exit()

    # subprocess.run(f"python forward.py --data_name mondial --yuval_change conditional_entropy_from_low_to_high_removed_4")
    # subprocess.run(f"python forward.py --data_name mutagenesis --depth 5 --yuval_change sorted_by_shuffle0")
    # subprocess.run(f"python forward.py --data_name mondial --depth 4 --yuval_change sorted_by_shuffle0")

    ####################################################################################################################
    ################################################ run in france #####################################################
    ####################################################################################################################

    data_names = ["mutagenesis", "world", "hepatitis", "genes", "genes_essential",
                  "mondial_target_infant_mortality_g40",
                  "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6",
                  "mondial_original_target"]
    data_name_to_num_schemes = {"mutagenesis": 25, "world": 20, "hepatitis": 21, "genes": 32, "genes_essential": 32,
                                "mondial_target_infant_mortality_g40": 63,
                                "mondial_target_continent": 63, "mondial_target_GDP_g8e3": 63,
                                "mondial_target_Inflation_g6": 63,
                                "mondial_original_target": 63}
    data_name_to_depth = {"mutagenesis": 3, "world": 3, "hepatitis": 4, "genes": 3, "genes_essential": 3,
                          "mondial_target_infant_mortality_g40": 3, "mondial_target_continent": 3,
                          "mondial_target_GDP_g8e3": 3,
                          "mondial_target_Inflation_g6": 3, "mondial_original_target": 3}
    # # run regular forward once with all the schemes
    # for data_name in data_names:
    #     depth = data_name_to_depth[data_name]
    #     subprocess.run(
    #         f"python forward.py --data_name {data_name} --depth {depth} --yuval_change all_schemes --tryout True")
    #
    # regular_exps = ['sorted_by_shuffle0', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed']
    # for exp in regular_exps:
    #     for data_name in data_names:
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         for i in range(0, num_schemes, 4):
    #             subprocess.run(
    #                 f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True")
    #
    # # run lowest_loss_after_1_epoch
    # for data_name in data_names:
    #     for epoch in [1]:
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         try_out_time_str = subprocess.check_output(
    #             f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
    #         try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
    #         print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
    #         for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
    #             for i in range(0, num_schemes, 4):
    #                 subprocess.run(
    #                     f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())
    #
    # # run simple mul tryout
    # for data_name in data_names:
    #     data_name = "mondial_target_Inflation_g6"
    #     for mul_by in [0.4]:  # [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         try_out_time_str = subprocess.check_output(
    #             f"python try_out_utils.py --method tryout --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
    #         try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #         print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
    #         # for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
    #         for exp in [f"low_loss_tryout_{data_name}_try_out_mul{mul_by}"]:
    #             for i in range(0, num_schemes, 4):
    #                 i=16
    #                 subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())
    #                 exit()
    # # TODO run Jan81 tryout
    # # TODO run tryout with stop_n_restart
    #
    # # run dynamic scheme reduction
    # for data_name in data_names:
    #     depth = data_name_to_depth[data_name]
    #     for exp in ["dynamic"]:
    #         for percentage in [55, 65, 75, 85]:
    #             for num_epochs in [1, 2, 3, 4]:
    #                 subprocess.run(
    #                     f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth}".split())
    #         for percentage in [95]:
    #             for num_epochs in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #                 subprocess.run(
    #                     f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth}".split())
    ####################################################################################################################
    ################################################ bennys exps   #####################################################
    # run norm_loss_after_1_epoch
    for data_name in ["mondial_original_target"]:
        for epoch in [1]:  # ,10
            num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
            try_out_time_str = subprocess.check_output(
                f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch} --sorting_method norm".split())
            # try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
            for exp in [f"sorted_norm_after_SnR_{data_name}_stop_n_restart{epoch}"]:
                for i in range(0, num_schemes, 4):
                    subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())
    exit()
    ####################################################################################################################

    ####################################################################################################################
    ################################################ run in france #####################################################
    ####################################################################################################################
    # # run dynamic scheme reduction
    # for data_name in ["mondial_original_target"]:
    #     depth = data_name_to_depth[data_name]
    #     for exp in ["dynamic"]:
    #         for percentage in [55, 65, 75, 85]:
    #             for num_epochs in [1, 2, 3, 4]:
    #                 subprocess.run(
    #                     f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth}".split())
    #         for percentage in [95]:
    #             for num_epochs in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #                 subprocess.run(
    #                     f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth}".split())
    # run simple mul tryout
    for data_name in data_names:
        for mul_by in [0.4]:  # [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
            num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
            try_out_time_str = subprocess.check_output(f"python try_out_utils.py --method tryout --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
            # for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
            for exp in [f"low_loss_tryout_{data_name}_try_out_mul{mul_by}"]:
                for i in range(0, num_schemes, 4):
                    subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())

    config = ConfigParser()
    config.read("config.txt")
    config["BACKTRACK"]["backtrack"] = "True"
    with open('config.txt', 'w') as configfile:
        config.write(configfile)

    data_names = ["mutagenesis", "hepatitis"]
    # run regular forward once with all the schemes
    for data_name in data_names:
        depth = data_name_to_depth[data_name]
        subprocess.run(
            f"python forward.py --data_name {data_name} --depth {depth} --yuval_change all_schemes --tryout True")

    regular_exps = ['sorted_by_shuffle0', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed']
    for exp in regular_exps:
        for data_name in data_names:
            num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
            for i in range(0, num_schemes, 4):
                subprocess.run(
                    f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True")

    # run lowest_loss_after_1_epoch
    for data_name in data_names:
        for epoch in [1]:
            num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
            try_out_time_str = subprocess.check_output(
                f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
            try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
            print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
            for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
                for i in range(0, num_schemes, 4):
                    subprocess.run(
                        f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())


    # run same time but more time per scheme on less schemes (benny's idea 17.07)
    regular_exps = ['sorted_by_shuffle0', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed']
    for exp in regular_exps:
        for data_name in data_names:
            num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
            # STLS = Same Time Less Schemes
            if exp == 'sorted_by_shuffle0':
                subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change STLS{exp}_{0} --tryout True")
            subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --num_samples {500*2} --yuval_change STLS{exp}_{int(num_schemes*2/3)} --tryout True")

    # run lowest_loss_after_1_epoch
    for data_name in data_names:
        for epoch in [1]:
            num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
            try_out_time_str = subprocess.check_output(f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
            try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
            print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
            for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
                subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --num_samples {500*2} --yuval_change STLS{exp}_{int(num_schemes*2/3)} --pre_time {try_out_time_str}".split())
