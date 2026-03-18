import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_local(path, seed):
    with open(path + str(seed) + ".json", "r") as file:
        data_dict = json.load(file)
    p1 = data_dict['models']['M1']['evaluation']['P1']
    p2 = data_dict['models']['M2']['evaluation']['P2']
    return p1, p2


def load_fed(path, seed):
    with open(path + str(seed) + ".json", "r") as file:
        data_dict = json.load(file)
    priv_param_num = priv_param(path, seed)
    matrix_p1, matrix_p2 = [], []
    for par_1 in range(priv_param_num):
        matrix_p1.append([])
        matrix_p2.append([])
        for par_2 in range(priv_param_num):
            matrix_p1[par_1].append(data_dict['experiments'][priv_param_num * par_1 + par_2]['rounds'][-1]['global_evaluation']["P1"])
            matrix_p2[par_1].append(data_dict['experiments'][priv_param_num * par_1 + par_2]['rounds'][-1]['global_evaluation']["P2"])
    return matrix_p1, matrix_p2


def normalize(baseline, federated):
    norm = [[], []]
    for p in [0, 1]:
        norm[p] = [[{"loss": baseline[p]['loss'] - cell["loss"], "accuracy": cell["accuracy"] - baseline[p]['accuracy']} for cell in row] for row in federated[p]]
    return norm[0], norm[1]


def priv_param(path, seed):
    with open(path + str(seed) + ".json", "r") as file:
        data_dict = json.load(file)
    return int(np.sqrt(len(data_dict['experiments'])))


def get_avg(path, seeds):
    with open(path + str(seeds[0]) + ".json", "r") as file:
        data_dict = json.load(file)
    priv_param_num = priv_param(path, seeds[0])
    avg1 = [[{} for i in range(priv_param_num)] for j in range(priv_param_num)]
    avg2 = [[{} for i in range(priv_param_num)] for j in range(priv_param_num)]
    for seed in seeds:
        local = load_local('1_local_baseline/', seed)
        fed   = load_fed(path, seed)
        tmp1, tmp2 = normalize(local, fed)
        for i in range(2):
            for j in range(2):
                avg1[i][j] = {key: avg1[i][j].get(key, 0) + tmp1[i][j].get(key, 0) / len(seeds) for key in set(avg1[i][j]) | set(tmp1[i][j])}
                avg2[i][j] = {key: avg2[i][j].get(key, 0) + tmp2[i][j].get(key, 0) / len(seeds) for key in set(avg2[i][j]) | set(tmp2[i][j])}
    return avg1, avg2


def split_matrix(matrix):
    matrix1 = [[cell['loss'] for cell in row] for row in matrix]
    matrix2 = [[cell['accuracy'] for cell in row] for row in matrix]
    return matrix1, matrix2


def plot(mx, ttl, params):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap([[0 if value < 0 else value for value in row] for row in mx], cmap="Greys_r", annot=True, cbar=True, square=True)
    ax.set_xticklabels(params)
    ax.set_yticklabels(params)
    plt.xlabel("Self")
    plt.ylabel("Other")
    plt.title(ttl)
    return plt


experiments = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # SET HERE seed parameters
supp = [14, 12, 10, 8, 6, 4, 2]  # CHANGE HERE privacy-SUP parameters
noise = [0.00, 0.33, 0.67, 1.00, 1.33, 1.67, 2.00]  # CHANGE HERE privacy-NOISE parameters
for j, setting in enumerate(['3_suppression/', '3_suppression/P1/', '3_suppression/P2/', '4_noise/', '4_noise/P1/', '4_noise/P2/']):
    if j < 3:
        method = 'Suppression'
        tick = supp
    else:
        method = 'Noise'
        tick = noise
    simulating_player = ''
    if j == 1 or j == 4:
        simulating_player = '1'
    if j == 2 or j == 5:
        simulating_player = '2'
    avg1, avg2 = get_avg(setting, experiments)
    p1_loss, p1_acc = split_matrix(avg1)
    p2_loss, p2_acc = split_matrix(avg2)
    for i, matrix in enumerate([p1_loss, p1_acc, p2_loss, p2_acc]):
        print('p1_loss')
        for row in matrix:
            print(" ".join(map(str, row)))
        if i == 0 or i == 1:
            player = '1'
        if i == 2 or i == 3:
            player = '2'
        if i == 0 or i == 2:
            metric = 'Loss'
        if i == 1 or i == 3:
            metric = 'Accuracy'
        if len(simulating_player) == 0:
            title = 'Client ' + player + '\'s ' + metric + ' with ' + method
            plt = plot(matrix, title, tick)
            plt.savefig('0_plots/' + method + '_' + metric + '_' + player + '.png', dpi=300)
        else:
            title = 'SD for Client ' + simulating_player + ': player ' + player + '\'s ' + metric + ' with ' + method
            plt = plot(matrix, title, tick)
            plt.savefig('0_plots/' + method + '_' + metric + '_' + simulating_player + '_' + player + '.png', dpi=300)
            plt.close()
