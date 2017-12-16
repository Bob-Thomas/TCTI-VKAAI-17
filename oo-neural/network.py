import random

from neuron import Neuron
from bias import Bias

class Network:
    def __init__(self, layers=[2,2,1]):
        self.layers = []
        for layer_number in range(len(layers)):
            if layer_number == 0:
                self.create_layer(layer_number, 'input', layers[layer_number])
            elif layer_number == len(layers)-1:
                self.create_layer(layer_number, 'output', layers[layer_number])
            else:
                self.create_layer(layer_number, 'hidden', layers[layer_number])
        self.trained = 0

    def train_network(self, learning_rate, training_set):
        """
        learning_rate: rate of learning
        training_set: [ ([input], oracle) ]
        """
        reversed_layers = self.layers[::-1]
        for training_tuple in training_set:
            #settign the input of the input nodes
            for x in range(len(reversed_layers[-1]) -1):
                reversed_layers[-1][x].input = training_tuple[0][x]

            only_neurons = list(filter(lambda x: isinstance(x, Neuron), reversed_layers[0]))
            try:
                deltalist = [[only_neurons[x].delta_update(learning_rate, training_tuple[1][x]) for x in range(len(only_neurons))]]
            except:
                deltalist = [[x.delta_update(learning_rate, training_tuple[1]) for x in only_neurons]]


            for layer in reversed_layers[1:]:
                only_neurons = list(filter(lambda x: isinstance(x, Neuron), layer))
                #removing bias nodes because they dont need their delta calculatedo
                deltalist.append(
                    [x.delta_update(learning_rate, training_tuple[1]) for x in only_neurons]
                )

            for layer_index in range(len(reversed_layers)):
                only_neurons = list(filter(lambda x: isinstance(x, Neuron), reversed_layers[layer_index]))
                for neuron_index in range(len(only_neurons)):
                    reversed_layers[layer_index][neuron_index].input_weights = deltalist[layer_index][neuron_index]
        self.trained += 1

    def run(self, inputs):
        assert len(inputs) == len(self.layers[0]) - 1

        for x in range(len(self.layers[0])-1):
            self.layers[0][x].input = inputs[x]

        return tuple([z.get_output() for z in self.layers[-1]])


    def create_neuron(self, layer_number, neuron_type):
        if neuron_type == 'input':
            return Neuron()
        else:  #if hidden or output or special snow flake
            return Neuron(self.layers[layer_number-1], [random.uniform(-1, 1) for x in range(len(self.layers[layer_number -1]))], neuron_type)

    def create_layer(self, layer_number, neuron_type, amount_neurons):
        self.layers.append([self.create_neuron(layer_number, neuron_type) for x in range(amount_neurons)])
        if neuron_type != "output":
            self.layers[-1].append(Bias())

    def create_graph(self):
        from graphviz import Graph
        dot = Graph()

        for i in range(len(self.layers)):
            for y in self.layers[i]:
                dot.node(str(id(y)),str(y))
                if i > 0:
                    if not isinstance(y,Bias):
                        for x in range(len(self.layers[i-1])):
                            dot.edge(
                                str(id(self.layers[i-1][x])),
                                str(id(y)),
                                label=str(round(y.input_weights[x],2))
                            )
        return dot
