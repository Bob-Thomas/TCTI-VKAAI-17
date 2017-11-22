import numpy as np
import math
import operator
import random
dataset = list(np.genfromtxt('dataset1.csv', delimiter=';',
                             usecols=[1, 2, 3, 4, 5, 6, 7]))

dates = list(np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0]))

labels_2000 = []
for label in dates:
    if label < 20000301:
        labels_2000.append('winter')
    elif 20000301 <= label < 20000601:
        labels_2000.append('lente')
    elif 20000601 <= label < 20000901:
        labels_2000.append('zomer')
    elif 20000901 <= label < 20001201:
        labels_2000.append('herfst')
    else: # from 01−12 to end of year
        labels_2000.append('winter')

labels_2001 = []
for label in dates:
    if label < 20010301:
        labels_2001.append('winter')
    elif 20010301 <= label < 20010601:
        labels_2001.append('lente')
    elif 20010601 <= label < 20010901:
        labels_2001.append('zomer')
    elif 20010901 <= label < 20011201:
        labels_2001.append('herfst')
    else: # from 01−12 to end of year
        labels_2001.append('winter')

def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x) - 1):
        d += pow((float(x[i]) - float(xi[i])), 2)
        d = math.sqrt(d)
    return d


[print(x) for x in dates]
[print(x) for x in dataset]
