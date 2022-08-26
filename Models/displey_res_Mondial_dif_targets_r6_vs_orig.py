# mondial_ ...
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

'''
1 schemes with 0 length - 
5 schemes with 1 length - 
25 schemes with 2 length - 
33 schemes with 3 length - 
0 schemes with 4 length - 
---
'''
WIDTH = 0.2


def addlabels(x, y, ha='center', delta=0.0, facecolor='red'):
    for i in range(len(x)):
        if i == 2 and facecolor=='green':
            delta -= 0.025
            plt.text(i, y[i] + delta, round(y[i], 3), ha=ha, bbox=dict(facecolor=facecolor, alpha=.7))
        elif i>=3:
            if facecolor=='green':
                delta = -0.025
            if facecolor=='red':
                delta = -0.03
                if i ==4:
                    delta = -0.065
            if facecolor=='deepskyblue':
                delta = 0.0
            plt.text(i, y[i] + delta, round(y[i], 3), ha=ha, bbox=dict(facecolor=facecolor, alpha=.7))
        elif abs(scores_list_all[i] - scores_list_r6[i]) < abs(delta) and ha == 'center':
            plt.text(i, y[i] + delta, round(y[i], 3), ha=ha, bbox=dict(facecolor=facecolor, alpha=.7))
        else:
            plt.text(i, y[i], round(y[i], 3), ha=ha, bbox=dict(facecolor=facecolor, alpha=.7))


mondial_exps = os.listdir(".")
for dir in mondial_exps.copy():
    if "mondial_" not in dir:
        mondial_exps.remove(dir)
print(mondial_exps)
scores_list_all, scores_list_r6, scores_list_r9 = [], [], []
for mon_exp in mondial_exps:
    with open(f'{mon_exp}/EK_3_100_500_10_50000_0/results.json') as f:
        data = json.load(f)
        scores = data["scores"]
        print(f'{"all " + mon_exp:<75} Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')
        scores_list_all.append(np.mean(scores))
    with open(f'{mon_exp}/EK_3_100_500_10_50000_0experiment_random6/results.json') as f:
        data = json.load(f)
        scores = data["scores"]
        print(f'{"r6  " + mon_exp:<75} Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')
        scores_list_r6.append(np.mean(scores))
    with open(f'{mon_exp}/EK_3_100_500_10_50000_0experiment_random9/results.json') as f:
        data = json.load(f)
        scores = data["scores"]
        print(f'{"r9  " + mon_exp:<75} Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')
        scores_list_r9.append(np.mean(scores))

times_all, times_r6, times_r9 = [], [], []
for mon_exp in mondial_exps:
    try:
        with open(f'{mon_exp}/EK_3_100_500_10_50000_0/results.json') as f:
            data = json.load(f)
            pt = datetime.strptime(data["time"], '%H:%M:%S')
            times_all += [(pt.second + pt.minute * 60 + pt.hour * 3600) / (60 * 60)]
        with open(f'{mon_exp}/EK_3_100_500_10_50000_0experiment_random6/results.json') as f:
            data = json.load(f)
            pt = datetime.strptime(data["time"], '%H:%M:%S')
            times_r6 += [(pt.second + pt.minute * 60 + pt.hour * 3600) / (60 * 60)]
        with open(f'{mon_exp}/EK_3_100_500_10_50000_0experiment_random9/results.json') as f:
            data = json.load(f)
            pt = datetime.strptime(data["time"], '%H:%M:%S')
            times_r9 += [(pt.second + pt.minute * 60 + pt.hour * 3600) / (60 * 60)]
    except FileNotFoundError:
        pass
    except:
        print(f"{mon_exp} result doesnt have time")

print(len(mondial_exps))
# dirs = ["all\n(43)", "rand1\n(12)", "rand2\n(12)", "rand3\n(12)", "rand4\n(12)", "rand5\n(12)", "all2\n(12)", "all3\n(30)"]
dirs = [d.replace("mondial_", "").replace("target_", "").replace("infant_mortality", "inf-mort") for d in mondial_exps]
colors = ['purple'] + ['blue'] * 5 + ['blue'] + ['green'] + ['blue'] * 3 + ['cyan'] * 3


def plot_acc_to_textname():
    fig = plt.figure()
    r = np.arange(len(scores_list_all))
    # plt.bar(r, [s - 0.4 for s in scores_list_all], bottom=0.4, color="blue", width=WIDTH, label='all')
    # plt.bar(r + WIDTH, [s - 0.4 for s in scores_list_r6], bottom=0.4, color="green", width=WIDTH, label='r6')
    # plt.bar(r + 2 * WIDTH, [s - 0.4 for s in scores_list_r9], bottom=0.4, color="red", width=WIDTH, label='r9')
    # plt.bar(r + 3 * WIDTH, [s - 0.4 for s in majority], bottom=0.4, color="purple", width=WIDTH, label='base line')
    plt.bar(r, scores_list_all, bottom=0.0, color="blue", width=WIDTH, label='all')
    plt.bar(r + WIDTH, scores_list_r6, bottom=0.0, color="green", width=WIDTH, label='r6')
    plt.bar(r + 2 * WIDTH, scores_list_r9, bottom=0.0, color="red", width=WIDTH, label='r9')
    # majority:
    majority = [0.637255, 0.227273
        , 0.5, 0.605042, 0.508403]
    plt.bar(r + 3 * WIDTH, majority, bottom=0.0, color="purple", width=WIDTH, label='base line')
    addlabels(dirs, scores_list_all, ha='center', delta=-0.1, facecolor='deepskyblue')
    addlabels(dirs, scores_list_r6, ha='left', delta=0.0, facecolor='green')
    addlabels(dirs, scores_list_r9, ha='left')
    plt.xticks(r, dirs)
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.xlabel('Test name')
    plt.show()


plot_acc_to_textname()
fig = plt.figure()
# times = ['05:50:47', '00:59:18', '00:36:03', '01:25:01', '00:54:56', '01:16:49', '02:59:59', '02:20:12']
# times = [5 + 5 / 6, 59/60, 36 / 60, 1 + 25 / 60, 55/60, 1+16/60, 2+59/60, 2 + 20 / 60]
r = np.arange(len(times_all))
plt.bar(r, times_all, color="blue", width=WIDTH, label='all')
plt.bar(r + WIDTH, times_r6, color="green", width=WIDTH, label='r6')
plt.bar(r + 2 * WIDTH, times_r9, color="red", width=WIDTH, label='r9')
# addlabels(dirs, ['05:50:47','00:36:03','00:36:03','00:36:03','00:36:03','00:36:03','02:20:12'])
plt.xticks(r, dirs)
plt.ylabel('Hours')
plt.legend()
# plt.xlabel('Test name')
plt.show()
