class Bias:
    """
    A simplified neuron class that has no input and always outputs 1
    """
    def __init__(self):
        self.outputs = []
        self.input = ['biasinput']
        self.input_weights = ['biasweights']

    def __repr__(self):
        return 'bias'

    def register(self, neuron):
        self.outputs.append(neuron)

    def get_output(self):
        return 1
