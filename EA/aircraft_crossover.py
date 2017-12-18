# lift = (A - B)^2 + (C + D)^2 - (A-30)^3 - (C -40)^3
# A B C D = 0-63
import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--population", type=int, help="Population", default=30)
parser.add_argument("-g", "--geno-types", type=int, help="Amount of genotypes", default=4)
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

def bitfield(n):
    return list(map(int, list('{0:06b}'.format(n))))


def get_bit_int(n):
    out = 0
    for bit in n:
        out = (out << 1) | bit
    return out


def evolve(population, retain=0.2, random_select=0.05, mutate=0.01):
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
        for x in range(len(individual)):
            for y in range(len(individual[x])):
                if mutate > random.random():
                    pos_to_mutate = random.randint(0, len(individual[x]) - 1)
                    individual[x][pos_to_mutate] = 1 - individual[x][pos_to_mutate]  # flip a bit

    parents.extend(children)

    return parents


def natural_selection(population, survival_rate):
    fighters = list(population)
    winners = []
    while len(fighters) > 0:
        selection = []
        for _ in range(2):
            pos = random.randint(0, len(fighters) - 1)
            selection.append(fighters[pos])
            fighters.remove(fighters[pos])
        if evaluate(selection[0]) > evaluate(selection[1]):
            winners.append(selection[0])
            if survival_rate > random.random():
                winners.append(selection[1])
        else:
            winners.append(selection[1])
    return winners



def calculate_lift(A, B, C, D):
    return (A - B) ** 2 + (C + D) ** 2 - (A - 30) ** 3 - (C - 40) ** 3


def create_pop():
    population = []
    for gene in range(population_length):
        population.append([bitfield(0)] * geno_type)

    for p in range(population_length):
        for g in range(geno_type):
            population[p][g] = bitfield(random.randrange(0, 64))

    return population

def evaluate(gene):
    return calculate_lift(get_bit_int(gene[0]), get_bit_int(gene[1]), get_bit_int(gene[2]), get_bit_int(gene[3]))


def fight(tournaments, population):
    """
    Let the genes fight for their lives.
    Create the population and select 2 random genes.
    eval them both and check which one is the winner.

    for every genotype in the gene check if it will mutate or recombine.
    Check if the gene is our winning gene and display it
    """
    best = [0,0]
    for tournament in range(tournaments):
        population = evolve(natural_selection(population, survival_rate), retain, recombination_rate,
                            mutation_rate)
        winner = min([[x, evaluate(x)] for x in population], key=lambda tup: tup[0])
        if not best:
            best = winner
        elif winner[1] > best[1]:
            best = winner
        if args.verbose:
            print("Average target round %d target: %f" % (
            tournament, float(sum([evaluate(x) for x in population]) / len(population))))

    print("best gene is %s" % list(map(lambda x: get_bit_int(x), best[0])))
    print("Maximal lift is %d" % best[1])


fight(tournaments, create_pop())
