# made together with David(Nerinai) and Robert (sqrtroot) github.com/nerinai/HU_AI
import argparse
import csv
import pickle
import random

import helper_functions as helper
from network import Network

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--load", type=str, help="Load previous training")
parser.add_argument("-s", "--save", type=str, help="Save training data to file or\
                                                                        load file", default="")
parser.add_argument("-r", "--learning-rate", type=float, help="Learning rate", default=0.1)
parser.add_argument("-t", "--times", type=int, help="Amount of times to train", default=0)
parser.add_argument("-a", "--accuracy", type=int, help="The amount of accuracy you want it to get trained to",
                    default=90)
parser.add_argument("-d", "--date", help="Indicates if csv has date field", action="store_true")
parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")

parser.add_argument("csv", type=str, help="csv file to load")

args = parser.parse_args()

if args.load:
    try:
        with open(args.load, 'rb') as f:
            network = pickle.load(f)
            if args.verbose:
                print("Loaded old network that has been trained {} times".format(network.trained))
    except:
        print("couldn't load file {}".format(args.load))
else:
    network = Network([4, 6, 3, 3])
    if args.verbose:
        print("Created new network")

dataset = []
# with open(args.csv) as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=';')
#     for row in spamreader:
#         tmp = [list(map(int,row[1:]))]
#         if args.date and row[0]:
#             if int(row[0][4:]) < 301:
#                 tmp.append(helper.Seasons.winter.value) #Winter
#             elif 301 <= int(row[0][4:]) <  601:
#                 tmp.append(helper.Seasons.lente.value) #Winter
#             elif 601 <= int(row[0][4:]) < 901:
#                 tmp.append(helper.Seasons.summer.value) #Winter
#             elif 901 <= int(row[0][4:]) < 1201:
#                 tmp.append(helper.Seasons.herfst.value) #Winter
#             else:
#                 tmp.append(helper.Seasons.winter.value)
#         else:
#             tmp.append(None)
#         dataset.append(tmp)
with open(args.csv) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        tmp = [list(map(float, row[:-1]))]
        try:
            if row[-1] == 'Iris-virginica':
                tmp.append(helper.Iris.virginica.value)
            elif row[-1] == 'Iris-versicolor':
                tmp.append(helper.Iris.versicolor.value)
            elif row[-1] == 'Iris-setosa':
                tmp.append(helper.Iris.setosa.value)
        except:
            tmp.append(None)
        dataset.append(tmp)

training_groups = [
    list(filter(lambda x: x[1] == helper.Iris.setosa.value, dataset)),
    list(filter(lambda x: x[1] == helper.Iris.virginica.value, dataset)),
    list(filter(lambda x: x[1] == helper.Iris.versicolor.value, dataset))
]
training_set = []

for l in range(len(training_groups)):
    for x in range(17):
        training_set.append(random.choice(training_groups[l - 1]))


def trainsu(args, dataset):
    if args.verbose:
        print("Read dataset")
    random.shuffle(training_set)
    for i in range(args.times):
        if args.verbose:
            print("{} times left to train. Did a total of {} trainings".format(args.times - i, network.trained))
        network.train_network(args.learning_rate, training_set)
        if args.save:
            with open(args.save if args.save else args.load, 'wb') as f:
                pickle.dump(network, f)
                if args.verbose:
                    print("Saved new trained network")
    if args.verbose and args.times:
        print("Trained from dataset")


# network.create_graph().render('test-output/round-table.gv', view=True)
# helper.validate_seasons(network, dataset[:5])
while (helper.validate_irises(network, dataset) < args.accuracy):
    trainsu(args, training_set)
    # print(network.trained)

# todo - command line tool and cached neural
