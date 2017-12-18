import argparse
import functools
import operator
import random

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--population", type=int, help="Population", default=30)
parser.add_argument("-g", "--geno-types", type=int, help="Amount of genotypes", default=10)
parser.add_argument("-r", "--retain-rate", type=float, help="Amount of genotypes", default=0.2)
parser.add_argument("-m", "--mutation-rate", type=float, help="Mutation rate", default=0.01)
parser.add_argument("-c", "--recombination-rate", type=float, help="Recombination rate", default=0.05)
parser.add_argument("-s", "--random-survival-rate", type=float, help="Recombination rate", default=0.9)
parser.add_argument("-t", "--tournaments", type=int, help="Amount of tournaments", default=1000)
parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")
args = parser.parse_args()

population_length = args.population  # how many population
geno_type = args.geno_types  # mutations of population
mutation_rate = args.mutation_rate  # rate of mution
recombination_rate = args.recombination_rate  # rate of recombination
tournaments = args.tournaments  # how many fights
retain = args.retain_rate
survival_rate = args.random_survival_rate
target = 0
pile_0_target = 36  # excepted result of pile 0
pile_1_target = 360  # expected result of pile 1


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
        if gene[g] == 0:
            sum += (1 + g)
        else:
            multiply *= (1 + g)
    sum_error = (sum - pile_0_target) / pile_0_target
    multiply_error = (multiply - pile_1_target) / pile_1_target
    combined_error = abs(sum_error) + abs(multiply_error)
    return combined_error


def evolve(population, target=0, retain=0.2, random_select=0.05, mutate=0.01):
    """
    Function for evolving a population , that is, creating
    offspring (next generation population) from combining
    (crossover) the fittest individuals of the current
    population
    :param population: the current population
    :param target: the value that we are aiming for
    :param retain: the portion of the population that we
    allow to spawn offspring
    :param random_select: the portion of individuals that
    are selected at random, not based on their score
    :param mutate: the amount of random change we apply to
    new offspring
    :return: next generation population
    """
    graded = [(evaluate(x), x) for x in population]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)
    desired_length = population_length - len(parents)
    children = []
    while len(children) < desired_length:
        male = random.randint(0, len(parents) - 1)
        female = random.randint(0, len(parents) - 1)
        if male != female:
            male = population[male]
            female = population[female]
            half = int(len(male) / 2)
            child = male[:half] + female[half:]
            children.append(child)

    for individual in children:
        if mutate > random.random():
            pos_to_mutate = random.randint(0, len(individual) - 1)
            individual[pos_to_mutate] = 1 - individual[pos_to_mutate]  # flip a bit

    parents.extend(children)

    return parents


def create_pop():
    """
        Generate the gene population with random 0 or 1 for genotype in the gene
    """
    population = []
    for g in range(population_length):
        population.append([0] * geno_type)

    for p in range(population_length):
        for g in range(geno_type):
            if random.random() < 0.5:
                population[p][g] = 0
            else:
                population[p][g] = 1
    return population


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
        if gene[x] == 0:
            print('[%d]' % (x + 1), end='')
    try:
        print(' = %d' % functools.reduce(operator.add, [x + 1 for x in range(geno_type) if gene[x] == 0]))
    except:
        print('nope')

    print("Solution multiply cards")
    for x in range(geno_type):
        if gene[x] == 1:
            print('[%d]' % (x + 1), end='')
    try:
        print(' = %d' % functools.reduce(operator.mul, [x + 1 for x in range(geno_type) if gene[x] == 1]))
    except:
        print('nope')

    if won:
        print("WINNING GENE GOOD BOY")


def natural_selection(population, survival_rate):
    fighters = list(population)
    winners = []
    while len(fighters) > 0:
        selection = []
        for _ in range(2):
            pos = random.randint(0, len(fighters) - 1)
            selection.append(fighters[pos])
            fighters.remove(fighters[pos])
        if evaluate(selection[0]) < evaluate(selection[1]):
            winners.append(selection[0])
            if survival_rate > random.random():
                winners.append(selection[1])
        else:
            winners.append(selection[1])
    return winners


def fight(population):
    """
    Let the population fight for their lives.
    Create the population and select 2 random population.
    eval them both and check which one is the winner.

    for every genotype in the gene check if it will mutate or recombine.
    Check if the gene is our winning gene and display it
    """
    create_pop()
    for tournament in range(tournaments):
        population = evolve(natural_selection(population, survival_rate), target, retain, recombination_rate,
                            mutation_rate)
        winner = min([[x, evaluate(x)] for x in population], key=lambda tup: tup[0])
        if winner[1] == 0:
            display(tournament, winner[0], True)
            return 1
        if args.verbose:
            print("Average target round %d target: %f" % (
            tournament, float(sum([evaluate(x) for x in population]) / len(population))))

    print("NO WINNER :( after: %d rounds" % tournaments)
    if args.verbose:
        display(tournaments, winner[0])
    return 0

tournament_counter = 0
while not fight(create_pop()):
    print(tournament_counter)
    tournament_counter += 1
