import operator
import random
import warnings
from collections import Counter
from math import sqrt

import numpy as np


def get_season_2000(date):
    """get season based on date in 2000
    Arguments:
        date {int} -- number corresponding to a date yearmonthday
    """

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
    """get season based on date in 2001
    Arguments:
        date {int} -- number corresponding to a date yearmonthday
    """
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


def get_neighbors(data, predict, k=1):
    """return knn nearest neighbour of data and prediction

    Run the knn algorithm over data and fit the prediction closest to the data.

    Arguments:
        data {array} -- array filled with data to iterate over
        predict {tuple} -- tuple containing the similar data to a data element

    Keyword Arguments:
        k {int} -- the K value you want to use (default: {1})

    Returns:
        string -- season label (zomer,herfst,winter,lente)
    """

    if len(data) <= k:
        warnings.warn(
            "K is set to a value less than the total voting groups :'(")

    neighbors = []
    # predict = [predict[0], predict[1]]
    for group in range(len(data)):
        neighbors.append([
            sqrt(sum([(a - b) ** 2 for a, b in zip(predict, data[group][1::])])),
            data[group][0]]
        )
    neighbors.sort(key=operator.itemgetter(0))
    return neighbors[:k]


def get_most_common_label(neighbors):
    """Find the most frequent label in the set of neighbors
    Arguments:
        neighbors {array} -- array of neighbors for this example  array of integers corresponding to a date

    Returns:
        String -- A season label (winter, lente, zomer, herfst)
    """

    classVotes = {}
    for x in range(len(neighbors)):
        response = get_season_2000(neighbors[x][1])
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def knn(data, point, k):
    """knn run knn and get most common label for that oint

    Arguments:
        data {array} -- array of tuples representing data points
        point {array|tuple} -- The data point you want to place using the data
        k {int} -- what k you are using for knn

    Returns:
        String -- A season label (winter, lente, zomer, herfst)
    """

    return get_most_common_label(get_neighbors(data, point, k))


if __name__ == "__main__":
    validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0],
                               converters={
        5: lambda x: 0 if x == '-1' else float(x),
        7: lambda x: 0 if x == '-1' else float(x)
    })

    test_set = np.genfromtxt('dataset1.csv', delimiter=';',
                             usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                             converters={
                                 5: lambda x: 0 if x == '-1' else float(x),
                                 7: lambda x: 0 if x == '-1' else float(x)
                             })

    training = np.genfromtxt('validation1.csv', delimiter=';',
                             usecols=[1, 2, 3, 4, 5, 6, 7],
                             converters={
                                 5: lambda x: 0 if x == '-1' else float(x),
                                 7: lambda x: 0 if x == '-1' else float(x)
                             })

    days = np.genfromtxt('days.csv', delimiter=';',
                         usecols=[1, 2, 3, 4, 5, 6, 7],
                         converters={
                             5: lambda x: 0 if x == '-1' else float(x),
                             7: lambda x: 0 if x == '-1' else float(x)
                         })

    validation_labels = [
        get_season_2001(trainer) for trainer in validation
    ]

    # run nearest neighbour 100 times over the training set and compare it to validation
    log = open('nearest_neighbor.txt', mode='w')
    results = []
    print("Running 100 iterations for knn over data")
    for k in range(1, 100):
        predicted_seasons = [
            knn(test_set, trainer, k) for trainer in training
        ]
        accuracy = len([x for x in range(len(predicted_seasons))
                        if predicted_seasons[x] == validation_labels[x]])
        results.append((accuracy, k))

    winner = sorted(results, key=lambda tup: tup[0], reverse=True)[0]
    log.write("Best accuracy has been achieved by k = %d with %d%% accuracy\n" % (
        winner[1], winner[0]))
    for result in results:
        log.write('%d%% for k = %d\n' % (result[0], result[1]))
    log.close()
    print("Done iterating")

    # predict the season of the random days
    print("Now predicting seasons by random weather data")
    log = open('predicted_days.txt', mode='w')
    predicted_season_on_weather = [
        knn(test_set, day, winner[1]) for day in days
    ]
    log.write("k used is %d\n" % (winner[1]))
    [log.write('%d : %s -> %s\n' % (x, days[x], predicted_season_on_weather[x]))
     for x in range(len(predicted_season_on_weather))]
    log.close()

    print("DONE")
