#Worked together with nerinai(github.com/nerinai/HU_AI
import numpy as np
from numpy import matrix, array, random, power
from pprint import pprint
from math import e


def make_network(identity):
    """
    Creates an algebraic neural network based on the identity list.
    Example identity = [2, 2, 1]
    Inputlayer = 2 neurons, Hiddenlayer = 2 neurons, Outputlayer = 1 neuron
    [[Input, Input], [Hidden, Hidden], [Output]]

    :param identity: (int) list representing the layers of the network
    :return: A list of layers in a networ withc each layer containt the apropriate amount of neurons.
    """
    network = []
    for layer in range(1,len(identity)):
        network.append(array([
                                [random.random() for __ in range(identity[layer-1] + 1)]
                                for _ in range(identity[layer])
                                ]))
    return network


def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 - (power(e,-x)))

def derivative_sigmoid(x):
    """Expects input x to be already sigmoid-ed"""
    return x * (1 - x)

def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (1- power(e,(-2*x))) / (1 + power(e,(-2*x)))

def derived_tanh(x):
    """Expects input x to already be tanh-ed."""
    return 1 - tanh(x)


def forward(inputs, network, function=sigmoid, step=-1):
    previous = array(inputs)
    #print(inputs)
    cstep = 0
    for layer in network:
        if cstep == step:
            print('step prevous', previous)
            return previous
        #print(layer.shape, np.insert(previous, 0, 1).shape)
        previous = function(np.dot(layer, np.insert(previous, 0, 1)))
        cstep += 1
    print('previous', previous)
    return previous



def calculate_cost(network, _input, expected_output):
    expected_output = matrix(expected_output)
    output = calculate_output(_input, network)
    return np.inner(output-expected_output,output-expected_output)

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
    layers = len(weights) # set current layer to output layer
    a_now = forward(inputs, weights, function, layers) # activation on current layer
    for i in range(0, layers):
        a_prev = forward(inputs, weights, function, layers-i-1) # calculate activation of previous layer
        if i == 0:
            error = np.array(derivative(a_now) * (outputs - a_now))  # calculate error on output
        else:
            error = derivative(a_now) * (weights[-i].T).dot(error)[1:] # calculate error on current layer
        delta = eta * np.expand_dims(np.append(1, a_prev), axis=1) * error # calculate adjustments to weights
        deltas.insert(0, delta.T) # store adjustments
        a_now = a_prev # move one layer backwards

    return deltas

network = make_network([2, 2, 1])
pprint(network)
training_set = [
((0,0),0),
((0,1),1),
((1,0),1),
((1,1),0)
]

pprint(backprop([0,0],[1],network))
