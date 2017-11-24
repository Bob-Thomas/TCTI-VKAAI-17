import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')


validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[
                           0, 1, 2, 3, 4, 5, 6, 7])
test_set = np.genfromtxt('dataset1.csv', delimiter=';',
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7])

training = np.genfromtxt('days.csv', delimiter=';',
                         usecols=[1, 2, 3, 4, 5, 6, 7])

labels = []
for label in test_set:
    if label[0] < 20000301:
        labels.append('winter')
    elif 20000301 <= label[0] < 20000601:
        labels.append('lente')
    elif 20000601 <= label[0] < 20000901:
        labels.append('zomer')
    elif 20000901 <= label[0] < 20001201:
        labels.append('herfst')
    else:  # from 01−12 to end of year
        labels.append('winter')


def get_season_2000(date):
    if date < 20000301:
        return 'winter'
    elif 20000301 <= date < 20000601:
        return 'lente'
    elif 20000601 <= date < 20000901:
        return 'zomer'
    elif 20000901 <= date < 20001201:
        return 'herfst'
    else:  # from 01−12 to end of year
        return 'winter'


def get_season_2001(date):
    if date < 20010301:
        return 'winter'
    elif 20010301 <= date < 20010601:
        return 'lente'
    elif 20010601 <= date < 20010901:
        return 'zomer'
    elif 20010901 <= date < 20011201:
        return 'herfst'
    else:  # from 01−12 to end of year
        return 'winter'

# [print (labels[x], test_set[x][1::]) for x in range(len(test_set))]
# print(training)


dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}

new_featues = [5, 7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

# plt.scatter(new_featues[0], new_featues[1])
# plt.show()


def k_nearest_neighbours(data, predict, k=1):
    if len(data) <= k:
        warnings.warn(
            "K is set to a value less than the total voting groups :'(")

    distances = []
    for group in range(len(data)):
        for features in data[group][1::]:
            distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([distance, data[group][0]])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


for k in range(1, 100):
    new_days_2000 = [
        get_season_2000(k_nearest_neighbours(test_set, trainer, k)) for trainer in training
    ]

    new_days_2001 = [
        get_season_2001(k_nearest_neighbours(test_set, trainer, k)) for trainer in training
    ]

    differences = [x for x in range(len(training)) if new_days_2000[x] != new_days_2001[x]]
    # print(new_days_2000)
    # print(new_days_2001)
    print('%d%% : %d' % ((100 - (len(training) / 100 * len(differences) * 100)), k))
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

# plt.scatter(new_featues[0], new_featues[1], color=result)
# plt.show()
