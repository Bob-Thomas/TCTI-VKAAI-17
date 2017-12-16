import argparse
import functools
import operator
import random
import sys

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--population", type=int, help="Population", default=30)
parser.add_argument("-g", "--geno-types", type=int, help="Amount of genotypes", default=10)
parser.add_argument("-m", "--mutation-rate", type=float, help="Mutation rate", default=0.1)
parser.add_argument("-r", "--recombination-rate", type=float, help="Recombination rate", default=0.5)
parser.add_argument("-t", "--tournaments", type=int, help="Amount of tournaments", default=1000)
parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")
args = parser.parse_args()

population = args.population  # how many genes
geno_type = args.geno_types  # mutations of genes
mutation_rate = args.mutation_rate  # rate of mution
recombination_rate = args.recombination_rate  # rate of recombination
tournaments = args.tournaments  # how many fights

pile_0_target = 36  # excepted result of pile 0
pile_1_target = 360  # expected result of pile 1

genes = []  # build the gene structure
for g in range(population):
    genes.append([0] * geno_type)


def evaluate(gene):
    """
    Evaluate the error of the given gene.
    Calculate it's error with the sum pile and multiply pile
    and return the combined error

    :param int gene:
    :return float error of gene:
    """
    sum = 0
    multiply = 1
    for g in range(geno_type):
        if genes[gene][g] == 0:
            sum += (1 + g)
        else:
            multiply *= (1 + g)
    sum_error = (sum - pile_0_target) / pile_0_target
    multiply_error = (multiply - pile_1_target) / pile_1_target
    combined_error = abs(sum_error) + abs(multiply_error)
    return combined_error


def create_pop():
    """
        Generate the gene population with random 0 or 1 for genotype in the gene
    """
    for p in range(population):
        for g in range(geno_type):
            if random.random() < 0.5:
                genes[p][g] = 0
            else:
                genes[p][g] = 1


def display(tour, gene, won=False):
    """
    Show the given gene and what kinds of cards it chose.
    Show the calculation of those cards and show it it's the winning gene if so exit
    :param tour how many tournaments
    :param gene which gene is displayed
    :param won  if the gene is a winner
    """
    print("=" * 30)
    print("after %d tournaments, solution sum cards are" % tour)
    for x in range(geno_type):
        if genes[gene][x] == 0:
            print('[%d]' % (x + 1), end='')
    try:
        print(' = %d' % functools.reduce(operator.add, [x + 1 for x in range(geno_type) if genes[gene][x] == 0]))
    except:
        print('nope')

    print("Solution multiply cards")
    for x in range(geno_type):
        if genes[gene][x] == 1:
            print('[%d]' % (x + 1), end='')
    try:
        print(' = %d' % functools.reduce(operator.mul, [x + 1 for x in range(geno_type) if genes[gene][x] == 1]))
    except:
        print('nope')

    if won:
        print("WINNING GENE GOOD BOY")
        sys.exit()


def evolve(gene, winner, loser):
    """
    Evolve the geno by recombination or mutation
    :param gene:
    :param winner:
    :param loser:
    :return new geno:
    """
    if random.random() < recombination_rate:
        genes[loser][gene] = genes[winner][gene]  # recombine with the winner
    genes[loser][gene] = 1 - genes[loser][gene]  # flip a bit
    return genes[loser][gene]


def fight():
    """
    Let the genes fight for their lives.
    Create the population and select 2 random genes.
    eval them both and check which one is the winner.

    for every genotype in the gene check if it will mutate or recombine.
    Check if the gene is our winning gene and display it
    """
    create_pop()

    for tournament in range(tournaments):
        a = int(population * random.random())
        b = int(population * random.random())
        if evaluate(a) < evaluate(b):
            winner = a
            loser = b
        else:
            winner = b
            loser = a

        for i in range(geno_type):
            new_geno = evolve(i, winner, loser)
            if evaluate(new_geno) == 0:
                display(tournament, new_geno, True)
        if args.verbose:
            display(tournament, winner)
    print("NO WINNER :(")


fight()
