import math
import helper_functions as helper
from bias import Bias
# ik sla het gewicht van de inputs op in de neuron die ze ontvangt omdat het op deze manier overzichtelijker is
# welk gewicht bij welke input hoort en ik niet het gewicht in mijn voorgaande neuronen hoef aan te passen op het moment
# dat ik de deltaregel toepas.
class Neuron:
    """
    Neuron class for neural networks,

    This class represents a neuron in a larger neural network.
    It has the ability to handle a varriable amount of inputs and outputs.
    """

    def __init__(self, input = 0, input_weights = None, neuron_type = 'input'):
        """
        Constructor for the neuron class.

        :param input: List of input objects. is either a single integer value as input or a list of neurons if the neuron
        in question is not an input neuron.
        :param input_weights: weights the input values are multiplied by
        example: input = 1, weight = 0.3 result = 1 * 0.3 = 0.3
        :param neuron_type: String representation of the type of neuron.
        Can be either:
            Input: A neuron that takes a single float value as input. this input is a raw value and has no weight.
            Hidden: A neuron that is intermediary between the input and output layers. Can have multiple inputs and outputs.
            Delta is calculated based on the delta of the previous nodes.
            Output: A neuron with multiple inputs and a single output. Delta is calculated by checking the output and the expected output.
        """
        if neuron_type == 'input':
            assert(type(input) == int or type(input) == float)
        else:
            assert(len(input) == len(input_weights))
            for x in input:
                x.register(self)

        self.outputs = list()
        self.input = input
        self.input_weights = input_weights
        self.neuron_type = neuron_type
        self.delta = None

    def __repr__(self):
        """

        :return: The type of neuron it is (str)
        """
        # return "%s %s" % (self.neuron_type, self.input)
        return self.neuron_type

    def register(self, neuron):
        """
        Allows for registration the neuron sends it's outputs to so we can acces it when needed

        :param neuron: The neuron to register
        :return: Nothing
        """
        self.outputs.append(neuron)

    def get_output(self):
        """
        Function that returns the weighted sum of its input
        :return: (float)
        """
        if self.neuron_type == 'input':
            return self.input
        else:
            inputs = list()
            for i in self.input:
                 if isinstance(i, Neuron) or isinstance(i, Bias):
                     inputs.append(i.get_output())
                 else:
                     inputs.append(i)
            return math.tanh(helper.weighted_sum(inputs, self.input_weights))

    def delta_update(self, learning_rate, oracle = None):
        """
        Function allows a neuron to recalculate its its input weights acording to the delta rule
        and the use of back propegation

        :param learning_rate: Stepsizes (float)
        :param oracle: Expected value (list(int))

        :return: A list of the new weights for the input values (list(float))
        """
        templist = list()
        if self.neuron_type != "input":
            activation_cost = helper.weighted_sum([x.get_output() for x in self.input], self.input_weights)
            for i in range(len(self.input_weights)):
                if self.neuron_type == 'output':
                    assert(oracle != None)
                    self.delta = helper.g_accent(activation_cost) * (oracle - self.get_output())
                    new_weight = self.input_weights[i] + learning_rate * self.input[i].get_output() * self.delta
                elif self.neuron_type == 'hidden':
                    self.delta = helper.g_accent(activation_cost) * self.back_propegation(list(map(lambda x: x.delta, self.outputs)))
                    new_weight = self.input_weights[i] + learning_rate * self.input[i].get_output() * self.delta
                #print(new_weight)
                templist.append(new_weight)


            return templist
            #self.input_weights = templist

    def back_propegation(self, previous_delta):
        """
        Allows for calculation of new weight using the delta of above lying nodes

        :param previous_delta: sum of the above lying nodes delta's (float)
        :return: new weight value (float)
        """
        weights = []
        #outputs = []
        for x in self.outputs:
            weights.append(x.input_weights[x.input.index(self)])
            #outputs.append(x.get_output())
        if self.neuron_type != 'input':
            #print(g_accent(sum(self.input_weights)) * sum(weights) * previous_delta)
            return  helper.weighted_sum(previous_delta, weights) #calculating
        return float('inf')
