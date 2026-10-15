def make_fitness(decoder):
    """Placeholder fitness"""

    def fitness_fn(genotype):
        A = decoder.decode(genotype)
        return -float((A ** 2).sum())

    return fitness_fn