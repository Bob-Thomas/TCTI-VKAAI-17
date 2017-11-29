from math import sqrt
import random
import numpy as np
from collections import Counter


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


def euclidian_distance(data, prediction):
    return sqrt(sum([(a - b) ** 2 for a, b in zip(data, prediction)]))


validation = np.genfromtxt(
    'validation1.csv',
    delimiter=';',
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],
    converters={
        5: lambda x: 0 if x == '-1' else float(x),
        7: lambda x: 0 if x == '-1' else float(x)
    }
)

training = np.genfromtxt(
    'dataset1.csv',
    delimiter=';',
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],
    converters={
        5: lambda x: 0 if x == '-1' else float(x),
        7: lambda x: 0 if x == '-1' else float(x)
    }
)


class Cluster:
    def __init__(self, centroid):
        self.centroid = centroid
        self.group = []

    def add(self, point):
        self.group.append(point)


k = 4
training = [
    {'label': get_season_2000(point[0]), 'data': point} for point in training
]

clusters = [Cluster(random.choice(training)) for i in range(k)]


def generate_centroids(clusters, k):
    new_centroids = []
    for point in training:
        distances = []
        for cluster in clusters:
            distances.append([
                euclidian_distance(point['data'][1::],
                                   cluster.centroid['data'][1::]),
                cluster
            ])
        nearest = Counter(
            [i[1] for i in sorted(distances, key=lambda t: t[0])[:k]]
        ).most_common(1)[0][0]
        nearest.add(point)
    for cluster in clusters:
        for g in cluster.group:
            new_centroids.append(
                [sum(x) / len(cluster.group)
                 for x in list(zip(g['data'][1::]))]
            )
    return new_centroids


for cluster in generate_centroids(clusters, k):
    print(cluster)
