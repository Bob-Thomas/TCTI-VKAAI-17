from math import sqrt
import random
import numpy as np
from collections import Counter
from functools import reduce
import copy as c
import time
from collections import defaultdict

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


class Point:
    def __init__(self, label, data):
        self.label = label
        self.data = data

    def __repr__(self):
        return "%s -> %s" % (self.label, self.data)


class Cluster:
    def __init__(self, centroid, group=[]):
        self.centroid = centroid
        self.group = []

    def add(self, point):
        self.group.append(point)

    def get_group_size(self):
        return len(self.group)

    def __repr__(self):
        return 'Centroid: \n\t%s\nPoints\n\t%s' % (self.centroid, ''.join(["%s\n" % x for x in self.group]))


def new_centroids(clusters, training, k):
    copy = c.deepcopy(clusters)
    for cluster in copy:
        cluster.group = []
    for point in training:
        distances = []
        for cluster in copy:
            distances.append(
                [
                    euclidian_distance(point.data, cluster.centroid.data),
                    cluster
                ]
            )
        nearest = min(distances, key=lambda x: x[0])
        nearest[1].add(point)
    newCentroids = []
    for cluster in copy:
        cluster.centroid = Point('', [sum(x)/len(cluster.group) for x in list(zip(*[x.data for x in cluster.group]))])
    return copy


def kmeans(clusters, traning, k):
    find_new = True
    current_clusters = new_centroids(clusters, training, k)
    while find_new:
        new_clusters = new_centroids(current_clusters, training, k)
        for current_cluster in current_clusters:
            for new_cluster in new_clusters:
                if np.array_equiv(np.array(new_cluster.centroid.data), np.array(current_cluster.centroid.data)):
                    find_new = False
                else:
                    find_new = True
        if find_new:
            current_clusters = new_clusters
    return current_clusters


k = 4
training = [
    Point(get_season_2000(point[0]), point[1::]) for point in training
]

clusters = [Cluster(random.choice(training)) for i in range(k)]
result = kmeans(clusters, training, k)

label_list = []
for cluster in result:
    appearances = defaultdict(int)
    for curr in cluster.group:
        appearances[curr.label] += 1
    label_list.append([cluster, appearances])

for index in range(len(label_list)):
    print("cluster -  %d - %s" % (index, (max(label_list[index][1], key=label_list[index][1].get))))
    print(label_list[index][0].get_group_size())

"""
    Given:
        • Training set X of examples {x~1,..., ~xn} wherea
        – x¯i
        is the feature vector of example i
        • A set K of centroids {c~1,...,~ck}
    Do:
        1. For each point ~xi:
        (a) Find the nearest centroid ~cj;
        (b) Assign point ~xi to cluster j;
        2. For each cluster j = 1,..., k:
        (a) Calculate new centroid ~cj as the mean of all points ~xi
        that are assigned
        to cluster j.
"""
