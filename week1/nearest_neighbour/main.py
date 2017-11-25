import numpy as np
from math import sqrt
import warnings
from collections import Counter
import random


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


def k_nearest_neighbours(data, predict, k=1):
    if len(data) <= k:
        warnings.warn(
            "K is set to a value less than the total voting groups :'(")

    distances = []
    predict = [predict[0], predict[1]]
    for group in range(len(data)):
        distances.append([
            sqrt(sum([(a - b) ** 2 for a, b in zip(data[group][1::], predict)])),
            data[group][0]]
        )
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return get_season_2000(vote_result)



if __name__ == "__main__":
    validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
    test_set = np.genfromtxt('dataset1.csv', delimiter=';',
                            usecols=[0, 1, 2, 3, 4, 5, 6, 7])

    training = np.genfromtxt('validation1.csv', delimiter=';',
                            usecols=[1, 2, 3, 4, 5, 6, 7])


    validation_labels = [
        get_season_2001(trainer) for trainer in validation
    ]

    for k in range(1, 100):
        new_days_2000 = [
            k_nearest_neighbours(test_set, trainer, k) for trainer in training
        ]
        print(
            '%d%% for k = %d' % (
                (len(new_days_2000) - len([x for x in range(len(new_days_2000)) if new_days_2000[x] == validation_labels[x]])),
                k
            )
        )
