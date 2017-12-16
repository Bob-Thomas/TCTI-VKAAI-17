# lift = (A - B)^2 + (C + D)^2 - (A-30)^3 - (C -40)^3
# A B C D = 0-63
import random


def bitfield(n):
    return list(map(int, list('{0:07b}'.format(n))))


def get_bit_int(n):
    out = 0
    for bit in n:
        out = (out << 1) | bit
    return out


def calculate_lift(A, B, C, D):
    return (A - B) ** 2 + (C + D) ** 2 - (A - 30) ** 3 - (C - 40) ** 3


pop = 30

genes = []
geno_type = 4
recombination_rate = 0.5

for gene in range(pop):
    genes.append([bitfield(0)] * geno_type)


def create_pop():
    for p in range(pop):
        for g in range(geno_type):
            genes[p][g] = bitfield(random.randrange(0, 64))


def evolve(winner, loser):
    """
    Evolve the geno by recombination or mutation
    :param gene:
    :param winner:
    :param loser:
    :return new geno:
    """
    for gene in range(geno_type):
        if random.random() < recombination_rate:
            genes[loser][gene] = genes[winner][gene]  # recombine with the winner
        if random.random() < 0.1:
            genes[loser][gene] = bitfield(random.randrange(0, 64))
    return genes[loser]


def evaluate(gene):
    return calculate_lift(get_bit_int(gene[0]), get_bit_int(gene[1]), get_bit_int(gene[2]), get_bit_int(gene[3]))


def fight(tournaments):
    """
    Let the genes fight for their lives.
    Create the population and select 2 random genes.
    eval them both and check which one is the winner.

    for every genotype in the gene check if it will mutate or recombine.
    Check if the gene is our winning gene and display it
    """
    create_pop()
    best = None
    for tournament in range(tournaments):
        a = int(pop * random.random())
        b = int(pop * random.random())
        if evaluate(genes[a]) > evaluate(genes[b]):
            winner = a
            loser = b
        else:
            winner = b
            loser = a
        if not best:
            best = genes[winner]
        for i in range(geno_type):
            new_gene = evolve(winner, loser)
            if evaluate(new_gene) > evaluate(best):
                best = new_gene

    print("best gene is %s" % best)
    print("Maximal lift is %d" % evaluate(best))


fight(1000)
