import numpy as np
from math import e
import random


def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 - (e**(-x)))


def derivative_sigmoid(x):
    """Expects input x to be already sigmoid-ed"""
    return x * (1 - x)


def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (1 - e ** (-2 * x)) / (1 + e ** (-2 * x))


def derived_tanh(x):
    """Expects input x to already be tanh-ed."""
    return 1 - tanh(x)


def forward(inputs, weights, function=sigmoid, step=-1):
    """Function needed to calculate activation on a particular layer.
    step=-1 calculates all layers, thus provides the output of the network
    step=0 returns the inputs
    any step in between, returns the output vector of that particular (hidden) layer"""
    previous = np.array(inputs)
    for index in range(len(weights)):
        if step == 0:
            break
        else:
            previous = function(np.dot(weights[index], previous))
            previous = np.insert(np.array(previous), len(previous), [1])
    print('=================>', previous.shape, step)
    print(previous)
    # previous = np.delete(previous, 0)
    return previous


def backprop(inputs, outputs, weights, function=sigmoid, derivative=derivative_sigmoid, eta=0.01):
    """
    Function to calculate deltas matrix based on gradient descent / backpropagation algorithm.
    Deltas matrix represents the changes that are needed to be performed to the weights (per layer) to
    improve the performance of the neural net.
    :param inputs: (numpy) array representing the input vector.
    :param outputs:  (numpy) array representing the output vector.
    :param weights:  list of numpy arrays (matrices) that represent the weights per layer.
    :param function: activation function to be used, e.g. sigmoid or tanh
    :param derivative: derivative of activation function to be used.
    :param learnrate: rate of learning.
    :return: list of numpy arrays representing the delta per weight per layer.
    """
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    deltas = []
    layers = len(weights)  # set current layer to output layer
    # activation on current layer
    a_now = forward(inputs, weights, function, layers)
    for i in range(0, layers):
        print('==============>', i)
        # calculate activation of previous layer
        a_prev = forward(inputs, weights, function, layers - i - 1)
        if i == 0:
            # calculate error on output
            error = np.array(derivative(a_now) * (outputs - a_now)).T
        else:
            print(weights[-1].shape, error.shape)
            # calculate error on current layer
            error = np.expand_dims(derivative(
                a_now), axis=1) * weights[-i].T.dot(error)[1:]
        # calculate adjustments to weights
        delta = eta * np.expand_dims(np.append(1, a_prev), axis=1) * error.T
        deltas.insert(0, delta.T)  # store adjustments
        a_now = a_prev  # move one layer backwards

    return deltas

# updating weights:
# given an array w of numpy arrays per layer and the deltas calculated by backprop, do
# for index in range(len(w)):
#     w[index] = w[index] + deltas[index]


identity = [2, 2, 1]
network = []
# return np.array([np.array([0.5, 0.5])])
for layer in range(0, len(identity)):
    network.append(np.array([
                            [random.random()
                             for __ in range(identity[layer - 1])]
                            for _ in range(identity[layer])
                            ]))

for i in range(len(network)):
    network[i] = np.append(network[i], random.random(), axis=1)

new = backprop(np.array([0, 1]), np.array([1]), network)
print(new)
for n in range(len(network)):
    network[n] = network[n] + new[n]
