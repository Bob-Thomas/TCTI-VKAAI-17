import math
import functools
import operator
import enum


class Seasons(enum.Enum):
    lente = [1, 0, 0, 0]
    summer = [0, 1, 0, 0]
    herfst = [0, 0, 1, 0]
    winter = [0, 0, 0, 1]


class Iris(enum.Enum):
    setosa = [1, 0, 0]
    versicolor = [0, 1, 0]
    virginica = [0, 0, 1]


def g_accent(cijfer):
    return 1 - math.tanh(math.tanh(cijfer))


def weighted_sum(inputs, input_weights):
    return functools.reduce(operator.add, list(map(lambda x, y: x * y, inputs, input_weights)))


def validate(network, training_data):
    for x in training_data:
        print("Expected ~{} got {}".format(x[1], network.run(x[0])))


def validate_seasons(network, training_data):
    for x in training_data:
        result = network.run(x[0])
        try:
            print("Expected {} got {}".format(Seasons(x[1]) if x[1] else "not that much really",
                                              Seasons([int(round(x)) for x in result])))
        except ValueError:
            print("Couldn't determine season confidently got {}".format(result))

def validate_irises(network, training_data):
    correct = 0
    for x in training_data:
        result = network.run(x[0])
        best = min(
                ([sum(list(map(lambda x, y: (x-y)**2, x.value,result))), x] for x in Iris), key=lambda tup:tup[0]
            )
        try:
            if best[1] == Iris(x[1]):
                correct+=1
            # print("Expected {} got {}".format(Iris(x[1]) if x[1] else "not that much really",
            #                                   best[1]))
        except ValueError:
            pass# print("Couldn't determine iris confidently got {}".format(result))
    print(correct/len(training_data) * 100)
    return correct/len(training_data) * 100
