from ga.traffic_simulation import simulate_traffic


def make_congestion_fitness(decoder):
    """Placeholder fitness function"""

    def fitness_fn(genotype):
        A = decoder.decode(genotype)
        return -float((A ** 2).sum())

    return fitness_fn


def make_genttp_fitness(decoder, model):
    """Fitness = minus total travel time predicted by GenTTP"""

    def fitness_fn(genotype):
        A = decoder.decode(genotype)
        _, TT = simulate_traffic(model, A)
        return -TT

    return fitness_fn