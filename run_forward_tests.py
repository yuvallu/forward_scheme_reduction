import subprocess
import re
from configparser import ConfigParser

def set_backtrack_to(bol):
    config = ConfigParser()
    config.read("config.txt")
    config["BACKTRACK"]["backtrack"] = bol
    with open('config.txt', 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    pass
    # for i in [2,4]:
    #    subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change scheme_len_eq_1_{i}")
    # for i in range(2,6):
    #    subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change random{i}")
    # for i in [1,2,3,4]:
    #    subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change scheme_len_eq_{i}")
    # for i in [6,7,8,9]:
    #    subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change specific{i}")

    # subprocess.run(f"python forward.py --data_name genes --depth 4 --yuval_change scheme_len_eq_3")

    # forward - displey_res_ends_in_specific_column
    # for i in [1,2,3]:
    #    subprocess.run(f"python forward.py --data_name mondial --yuval_change EC_scheme_len_eq_{i}")
    # for i in range(1,6):
    # for i in range(40,50):
    #    subprocess.run(f"python forward.py --data_name mondial --yuval_change EC_r6andom{i}")
    # for i in range(20,40):
    #    subprocess.run(f"python forward.py --data_name mondial --yuval_change EC_r6_25andom{i}")

    # test r6 scheme on other column predictions
    # for exp in ["mondial_target_infant_mortality_g40","mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial_original_target"]:
    #    subprocess.run(f"python forward.py --data_name {exp}")
    #    subprocess.run(f"python forward.py --data_name {exp} --yuval_change random6")

    # subprocess.run(f"python forward.py --data_name mondial --yuval_change r6_manual_reduce13")

    # r6 manual reduce experiments
    # for i in range(104,110):
    #   subprocess.run(f"python forward.py --data_name mondial --yuval_change r6_manual_reduce{i}")

    # test r9 (bad) scheme on other column predictions
    # for exp in ["mondial_target_infant_mortality_g40","mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial_original_target"]:
    #   subprocess.run(f"python forward.py --data_name {exp} --yuval_change random9")

    # max edge entropy removal (weakest link in each scheme points out the weakest scheme)
    # for i in range(1,60,2):
    #   subprocess.run(f"python forward.py --data_name mondial --yuval_change conditional_entropy_removed_{i}")
    # reverse - minimum of max edge entropy removal 
    # for i in range(3,60,4):
    #   subprocess.run(f"python forward.py --data_name mondial --yuval_change conditional_entropy_from_low_to_high_removed_{i}")
    # for i in range(3,60,4):
    #   subprocess.run(f"python forward.py --data_name mondial --yuval_change conditional_entropy_start_to_end_high_to_low_removed_{i}")
    # for i in range(3,60,4):
    #   subprocess.run(f"python forward.py --data_name mondial --yuval_change conditional_entropy_start_to_end_low_to_high_removed_{i}")
    # subprocess.run(f"python forward.py --data_name mondial --yuval_change specific_entropy_50")
    # check robustness of Reverse Conditional Entropy – remove schemes by “weakest link” (the LOWEST highest entropy in one of the edges)
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
    #           "mondial_target_Inflation_g6"]:  #, "mondial_original_target"]:
    #    for i in range(3, 53, 4):
    #        subprocess.run(
    #            f"python forward.py --data_name {data_name} --yuval_change conditional_entropy_from_low_to_high_removed_{i}")

    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
    #           "mondial_target_Inflation_g6","mondial"]:  #, "mondial_original_target"]:
    #    for i in range(3, 63, 4):
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change sorted_by_highest_loss_after_1_epoch_{i}".split())

    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
    #           "mondial_target_Inflation_g6", "mondial"]:  #, "mondial_original_target"]:
    #    for i in range(0, 62, 4):
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change sorted_by_shuffle1_{i}".split())

    # get all the losses
    # subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name mondial --yuval_change all_schemes_get_losses".split())

    # # run sorted_correct_highest_loss_after_10_epoch_
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3",
    #           "mondial_target_Inflation_g6", "mondial"]:  #, "mondial_original_target"]:
    #    for i in range(0, 62, 4):
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change sorted_correct_highest_loss_after_10_epoch_{i}".split())
    #
    # # run sorted_correct_lowest_loss_after_10_epoch_
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]:
    #    for i in range(0, 62, 4):
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change sorted_correct_lowest_loss_after_10_epoch_{i}".split())

    # run tests on GENES datasets
    # # remove longest scheme
    # for i in range(8, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change remove_longest_schemes_{i}".split())
    # # LOWEST "weakest link"
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change conditional_entropy_from_low_to_high_removed_{i}".split())
    # # remove random
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change sorted_by_shuffle1_{i}".split())
    # # sorted_correct_highest_loss_after_10_epoch
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change sorted_genes_correct_highest_loss_after_10_epoch{i}".split())
    # # sorted_correct_lowest_loss_after_10_epoch
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change sorted_genes_correct_lowest_loss_after_10_epoch_{i}".split())

    # # run longest+entropy lowest weakest link to get the times
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]:
    #    for i in range(0, 62, 4):
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change remove_longest_schemes_{i}".split())
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change conditional_entropy_from_low_to_high_removed_{i}".split())

    # # run sorted_mean_lowest_loss_after_10_epoch_
    # for data_name in ["mondial_target_infant_mortality_g40", "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6", "mondial"]:  # , "mondial_original_target"]:
    #    for i in range(0, 62, 4):
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change sorted_mean_lowest_loss_after_10_epoch_{i}".split())

    # hepatitis dataset tests
    # # remove longest scheme
    # for i in range(0, 15, 4):
    #    subprocess.run(f"python forward.py --data_name hepatitis --yuval_change remove_longest_schemes_{i}".split())
    # # remove random
    # for i in range(0, 15, 4):
    #    subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_by_shuffle0_{i}".split())
    #    subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_by_shuffle1_{i}".split())
    # # LOWEST "weakest link"
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change conditional_entropy_from_low_to_high_removed_{i}".split())
    # # sorted_correct_highest_loss_after_10_epoch
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_hepatitis_correct_highest_loss_after_10_epoch_{i}".split())
    # # sorted_correct_lowest_loss_after_10_epoch
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_hepatitis_correct_lowest_loss_after_10_epoch_{i}".split())

    # # mutagenesis
    # exps = ['sorted_by_shuffle1'#, 'remove_longest_schemes',
    #     #'conditional_entropy_from_low_to_high_removed', 'sorted_mutagenesis_mean_highest_loss_after_10_epoch'
    #     # , 'sorted_mutagenesis_mean_lowest_loss_after_10_epoch'
    # ]
    # for exp in exps:
    #     for i in range(0, 15, 4):
    #         subprocess.run(f"python forward.py --data_name mutagenesis --yuval_change {exp}_{i}".split())
    #
    # # # world
    # exps = ['sorted_by_shuffle0', 'remove_longest_schemes','conditional_entropy_from_low_to_high_removed', 'sorted_world_mean_highest_loss_after_10_epoch', 'sorted_world_mean_lowest_loss_after_10_epoch']
    # exps = ['conditional_entropy_from_low_to_high_removed']
    # for exp in exps:
    #     for i in range(0, 20, 4):
    #         subprocess.run(f"python forward.py --data_name world --yuval_change {exp}_{i}".split())

    # run 1 epoch
    # for i in range(0, 32, 4):
    #     subprocess.run(f"python forward.py --data_name genes --yuval_change sorted_genes_mean_lowest_loss_after_1_epoch_{i}".split())
    # for i in range(0, 20, 4):
    #     subprocess.run(f"python forward.py --data_name world --yuval_change sorted_world_mean_lowest_loss_after_1_epoch_{i}".split())
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name mutagenesis --yuval_change sorted_mutagenesis_mean_lowest_loss_after_1_epoch_{i}".split())
    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name hepatitis --yuval_change sorted_hepatitis_mean_lowest_loss_after_1_epoch_{i}".split())

    # for i in range(0, 15, 4):
    #     subprocess.run(f"python forward.py --data_name mutagenesis --yuval_change sorted_by_shuffle2_{i}".split())
    #     subprocess.run(f"python forward.py --data_name mutagenesis --yuval_change sorted_by_shuffle3_{i}".split())

    # run dynamic scheme reduction experiments
    # subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name mondial --yuval_change dynamic_36%in_1_ep --train 36%1".split())
    # for data_name in ["mondial"]:
    #     for exp in ["dynamic"]:
    #         for percentage in [45, 60, 75]:
    #             for num_epochs in [1, 2, 3, 4]:
    #                 subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs}".split())
    # # subprocess.run(f"python try_out_utils.py".split())
    # subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name mondial --yuval_change dynamic_0%in_1_ep --train 0%1".split())
    # for data_name in ["mondial"]:
    #     for exp in ["dynamic"]:
    #         for percentage in [85]:
    #             for num_epochs in [1, 2, 3, 4, 5, 6, 7, 9]:
    #                 subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs}".split())

    # subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name mondial --yuval_change conditional_entropy_removed_43".split())
    # exit()

    # for mul_by in [0.4]:  # , 0.5]:  # [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
    #     try_out_time_str = subprocess.check_output(f"python try_out_utils.py --mul_by {mul_by}".split())
    #     try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
    #     print(f"try_out_time mul {mul_by} is {try_out_time_str}")
    #     for data_name in ["mondial"]:
    #         for exp in [f"sorted_mean_lowest_loss_after_tryout_num_samples100_mul_{mul_by}"]:
    #             for i in range(0, 62, 4):
    #                 subprocess.run(f"python forward.py --data_name {data_name} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())
    #
    # for data_name in ["mondial"]:
    #    for i in range(0, 62, 4):
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change sorted_by_shuffle1_{i}".split())
    #        subprocess.run(f"python forward.py --data_name {data_name} --yuval_change conditional_entropy_from_low_to_high_removed_{i}".split())
    # subprocess.run(f"python forward.py --data_name mondial --yuval_change try_num_samples100 --num_samples 100".split())
    # subprocess.run(f"python forward.py --data_name mondial --yuval_change try_num_samples20 --num_samples 20".split())
    # for data_name in ["mondial"]:
    #     subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change sorted_mean_lowest_loss_after_1_epoch_40".split())
    #     subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change sorted_mean_highest_loss_after_1_epoch_40".split())
    #     subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change sorted_mean_lowest_loss_after_10_epoch_40".split())
    #     subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change sorted_mean_highest_loss_after_10_epoch_40".split())

    # #run tests with depth 5
    # for data_name in ["mutagenesis", "hepatitis"]:
    #     exps = [f'sorted_by_shuffle0', f'remove_longest_schemes',f'conditional_entropy_from_low_to_high_removed',
    #             f'sorted_{data_name}_mean_highest_loss_after_10_epoch', f'sorted_{data_name}_mean_lowest_loss_after_10_epoch',
    #             f'sorted_{data_name}_mean_highest_loss_after_1_epoch', f'sorted_{data_name}_mean_lowest_loss_after_1_epoch']
    #     for exp in exps:
    #         for i in range(0, 25 if data_name == "mutagenesis" else 15, 4):
    #             subprocess.run(f"python forward.py --data_name {data_name} --depth 5 --yuval_change {exp}_{i}".split())

    #TODO remove
    # for data_name in ["hepatitis"]:
    #     exps = [f'sorted_{data_name}_mean_highest_loss_after_10_epoch', f'sorted_{data_name}_mean_lowest_loss_after_10_epoch', f'sorted_by_shuffle1']
    #     for exp in exps:
    #         for i in range(0, 25 if data_name == "mutagenesis" else 15, 4):
    #             subprocess.run(f"python forward.py --data_name {data_name} --depth 5 --yuval_change {exp}_{i}".split())

    # TODO: correct runs of mutagenesis because fliped (check also 10 epoch and switch 1 already checked)
    # subprocess.run(f"python forward.py --data_name mondial --depth 4 --yuval_change all_depth_4 --tryout True".split())  #432
    # subprocess.run(f"python forward.py --data_name mondial --depth 4 --yuval_change sorted_d4_mean_highest_loss_after_1_epoch_400 --tryout True".split())
    # subprocess.run(f"python forward.py --data_name mondial --depth 4 --yuval_change sorted_d4_mean_lowest_loss_after_1_epoch_350 --tryout True".split())
    # subprocess.run(f"python forward.py --data_name mondial --depth 4 --yuval_change sorted_d4_mean_lowest_loss_after_1_epoch_300 --tryout True".split())
    # subprocess.run(f"python forward.py --data_name mondial --depth 4 --yuval_change sorted_d4_mean_lowest_loss_after_1_epoch_250 --tryout True".split())
    # subprocess.run(f"python forward.py --data_name mondial --depth 5 --yuval_change all_depth_5 --tryout True".split())  #1182
    # subprocess.run(f"python forward.py --data_name mondial --depth 5 --yuval_change all_depth_5_1130 --tryout True".split())
    # for data_name in ["mutagenesis"]:
    #     exps = [f'sorted_{data_name}_mean_highest_loss_after_1_epoch', f'sorted_{data_name}_mean_lowest_loss_after_1_epoch', f'sorted_by_shuffle1']
    #     for exp in exps:
    #         for i in range(0, 25 if data_name == "mutagenesis" else 15, 4):
    #             subprocess.run(f"python forward.py --data_name {data_name} --depth 5 --yuval_change {exp}_{i}".split())

    # for data_name in ["mondial"]:
    #     for exp in ["dynamic_d4"]:
    #         for percentage in [65,75,85,95]:
    #             for num_epochs in [1, 2, 3, 4, 6]:
    #                 subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --depth 4 --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --tryout True".split())




    ####################################################################################################################
    ################################################ run in france #####################################################
    ####################################################################################################################
    data_names = ["mutagenesis", "world", "world_B", "hepatitis", "genes", "genes_essential",
                  "mondial_target_infant_mortality_g40",
                  "mondial_target_continent", "mondial_target_GDP_g8e3", "mondial_target_Inflation_g6",
                  "mondial_original_target"]
    data_name_to_num_schemes = {"mutagenesis": 58, "world": 20, "hepatitis": 21, "genes": 32, "genes_essential": 32,
                                "mondial_target_infant_mortality_g40": 63,
                                "mondial_target_continent": 63, "mondial_target_GDP_g8e3": 63,
                                "mondial_target_Inflation_g6": 63,
                                "mondial_original_target": 63, "world_B": 60}
    data_name_to_depth = {"mutagenesis": 4, "world": 3, "hepatitis": 3, "genes": 3, "genes_essential": 3,
                          "mondial_target_infant_mortality_g40": 3, "mondial_target_continent": 3,
                          "mondial_target_GDP_g8e3": 3, "world_B": 3,
                          "mondial_target_Inflation_g6": 3, "mondial_original_target": 3}
    # # run regular forward once with all the schemes
    # for data_name in data_names:
    #     depth = data_name_to_depth[data_name]
    #     subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change all_schemes --tryout True".split())
    #
    # regular_exps = ['sorted_by_shuffle0', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed']
    # for exp in regular_exps:
    #     for data_name in data_names:
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         for i in range(0, num_schemes, 4):
    #             subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True".split())
    #
    # # run lowest_loss_after_1_epoch
    # for data_name in data_names:
    #     for epoch in [1]:
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         try_out_time_str = subprocess.check_output(f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch}".split())
    #         try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
    #         print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
    #         for exp in [f"sorted_mean_lowest_loss_after_SnR_{data_name}_stop_n_restart{epoch}"]:
    #             for i in range(0, num_schemes, 4):
    #                 subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())

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
    #
    #
    # # run simple mul tryout
    # for data_name in data_names:
    #     for mul_by in [0.4]:  # [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
    #         try:
    #             num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #             try_out_time_str = subprocess.check_output(
    #                 f"python try_out_utils.py --method tryout --mul_by {mul_by} --data_name {data_name} --num_samples 100 --depth {depth}".split())
    #             try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
    #             print(f"try_out_time {data_name} - mul {mul_by} is {try_out_time_str}")
    #             for exp in [f"sorted_mean_lowest_loss_after_tryout_{data_name}_try_out_mul{mul_by}"]:
    #                 for i in range(0, num_schemes, 4):
    #                     subprocess.run(
    #                         f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())
    #                     print(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}")
    #         except Exception as e:
    #             print(f"data_name {data_name} crashed")
    # TODO run Jan81 tryout
    # TODO run tryout with stop_n_restart

    # TODO change mutagenesis to 5 and run again
    ####################################################################################################################
    ################################################ run in france #####################################################
    ####################################################################################################################
    # config = ConfigParser()
    # config.read("config.txt")
    # config["BACKTRACK"]["backtrack"] = "False"
    # with open('config.txt', 'w') as configfile:
    #     config.write(configfile)

    # for data_name in ["mondial_original_target"]:
    #     for epoch in [1]:  # ,10
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         try_out_time_str = subprocess.check_output(
    #             f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch} --sorting_method norm".split())
    #         # try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
    #         try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #         print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
    #         for exp in [f"sorted_norm_after_SnR_{data_name}_stop_n_restart{epoch}"]:
    #             for i in range(0, num_schemes, 4):
    #                 subprocess.run(
    #                     f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())

    #norm method:

    # for data_name in ["mondial_original_target"]:
    #     for epoch in [1]:  # ,10
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         try_out_time_str = subprocess.check_output(
    #             f"python try_out_utils.py --depth {depth} --data_name {data_name} --method stop_n_restart --epoch {epoch} --sorting_method norm".split())
    #         # try_out_time_str = str(try_out_time_str.split(b'Time:')[-1])[4:-6]
    #         try_out_time_str = re.findall('\d+:\d+:\d+', str(try_out_time_str.split(b'Time:')[-1]))[0]
    #         print(f"stop_n_restart epoch {epoch} is {try_out_time_str}")
    #         for exp in [f"sorted_rev_norm_after_SnR_{data_name}_stop_n_restart{epoch}"]:
    #             for i in range(0, num_schemes, 4):
    #                 subprocess.run(
    #                     f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --pre_time {try_out_time_str}".split())

    # #
    # subprocess.run(f"python forward_eval_each_epoch.py --data_name mondial_original_target --depth {data_name_to_depth['mondial_original_target']} --yuval_change distribution_var_60 --epoch 999 --tryout True".split())
    # for i in [0]:
    #     subprocess.run(
    #     f"python forward_eval_each_epoch.py --data_name genes --depth 3 --yuval_change distribution_var_{i} --tryout True".split())  # --epoch 999
    # exit()
    # run distribution_var on mondial original target and genes
    # for exp in ["distribution_var"]:  # "rev_distribution_var",
    #     for data_name in ["genes"]:
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         for i in range(0, num_schemes, 4)[::-1][1:]:
    #             subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True --epoch 999".split())
    # config = ConfigParser()
    # config.read("config.txt")
    # config["BACKTRACK"]["backtrack"] = "True"
    # with open('config.txt', 'w') as configfile:
    #     config.write(configfile)
    # subprocess.run(
    #     f"python forward_eval_each_epoch.py --data_name hepatitis --depth 3 --yuval_change distribution_var_0 --tryout True".split())
    # # TODO: find hepatitis max time to run
    data_name_to_total_time = {"mondial_original_target": 200, "mondial_target_GDP_g8e3": 200, "genes": 370, "hepatitis": 250, "mutagenesis": 300, "world_B":370}
    # set_backtrack_to("True")
    # subprocess.run(f"python forward_eval_each_epoch.py --data_name world_B --depth 3 --yuval_change distribution_var_0 --tryout True".split())
    # exit()
    # run distribution_var also on hepatitis and world_B and mutagenesis
    set_backtrack_to("True")
    for exp in ["distribution_var"]:  # "rev_distribution_var",
        for data_name in ["world_B"]:  # , "hepatitis", mutagenesis
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 8)[::-1]:
                subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True --epoch 999 --threshold {total_time}".split())

    set_backtrack_to("False")
    for exp in ["distribution_var"]:  # "rev_distribution_var",
        for data_name in ["mondial_target_GDP_g8e3"]:  # , "hepatitis", mutagenesis
            num_schemes, depth, total_time = data_name_to_num_schemes[data_name], data_name_to_depth[data_name], data_name_to_total_time[data_name]
            for i in range(0, num_schemes, 8)[::-1]:
                subprocess.run(f"python forward_eval_each_epoch.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True --epoch 999 --threshold {total_time}".split())
    exit()
    set_backtrack_to("False")
    # TODO: run online scheme reduction on mondial_original_target and genes
    for data_name in ["mondial_original_target", "genes"]:
        depth, total_time = data_name_to_depth[data_name], data_name_to_total_time[data_name]
        for exp in ["dynamic"]:
            for percentage, num_epochs in [(66, 2), (85, 4), (95, 7), (0, 1)]:
                subprocess.run(
                    f"python forward_split_Loss_per_scheme_acc_each_epoch.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth} --epoch 999 --threshold {total_time}".split())
    exit()
    # # run dynamic scheme reduction
    # for data_name in ["mondial_original_target"]:
    #     depth = data_name_to_depth[data_name]
    #     for exp in ["dynamic"]:
    #         for percentage in [55, 65, 75, 85]:
    #             for num_epochs in [1, 2, 3, 4]:
    #                 subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth}".split())
    #         for percentage in [95]:
    #             for num_epochs in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #                 subprocess.run(f"python forward_split_Loss_per_scheme.py --data_name {data_name} --yuval_change {exp}_{percentage}%in_{num_epochs}_ep --train {percentage}%{num_epochs} --depth {depth}".split())
    #"""
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

    #TODO: run later ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # change config
    # regular_exps = ['sorted_by_shuffle1']
    # for exp in regular_exps:
    #     for data_name in data_names:
    #         if data_name in ["mutagenesis", "hepatitis", "world_B"]:
    #             continue
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         for i in range(0, num_schemes, 4):
    #             subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True".split())
    #"""
    # config = ConfigParser()
    # config.read("config.txt")
    # config["BACKTRACK"]["backtrack"] = "True"
    # with open('config.txt', 'w') as configfile:
    #     config.write(configfile)
    #
    # data_names = ["mutagenesis", "hepatitis", "world_B"]
    # data_name_to_num_schemes["world_B"] = 60
    # data_name_to_depth["world_B"] = 3
    # run regular forward once with all the schemes
    # for data_name in data_names:
    #     depth = data_name_to_depth[data_name]
    #     subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change all_schemes --tryout True".split())
    #
    # regular_exps = ['sorted_by_shuffle0', 'remove_longest_schemes', 'conditional_entropy_from_low_to_high_removed']
    # for exp in regular_exps:
    #     for data_name in data_names:
    #         num_schemes, depth = data_name_to_num_schemes[data_name], data_name_to_depth[data_name]
    #         for i in range(0, num_schemes, 4):
    #             subprocess.run(f"python forward.py --data_name {data_name} --depth {depth} --yuval_change {exp}_{i} --tryout True".split())
    #
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

