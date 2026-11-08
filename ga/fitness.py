from ga.traffic_simulation import simulate_traffic, simulate_traffic_batch
import numpy as np

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

def make_genttp_batch_fitness(decoder, model, device="cpu"):
    """Fitness for evaluating genotypes in batches: minus total travel time predicted by GenTTP"""

    def fitness_fn(genotypes):
        A_batch = np.stack([decoder.decode(g) for g in genotypes])
        _, TT = simulate_traffic_batch(model, A_batch, device=device)
        return [-float(tt) for tt in TT]

    return fitness_fn

