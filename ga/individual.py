import random


class Individual:
    """Class representing a genotype and fitness of an individual in the population."""

    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    @classmethod
    def create(cls, allowed_routes):
        """Creates an individual with a random allowed route for each agent."""
        genotype = [random.choice(options) for options in allowed_routes]
        return cls(genotype)

    def mutate(self, mutation_rate, allowed_routes):
        """Mutates the genotype by changing route of agents with a given mutation rate."""
        for i in range(len(self.genotype)):
            if random.random() < mutation_rate:
                self.genotype[i] = random.choice(allowed_routes[i])
                self.fitness = None

    def crossover(self, other, method="uniform"):
        """Performs crossover with another individual and returns two offspring.

        method="uniform":      each agent's route is taken from a randomly chosen parent.
        method="single_point": genes before a random point come from one parent, the rest from the other.
        """
        if method == "uniform":
            child1_genotype = []
            child2_genotype = []
            for gene1, gene2 in zip(self.genotype, other.genotype):
                if random.random() < 0.5:
                    child1_genotype.append(gene1)
                    child2_genotype.append(gene2)
                else:
                    child1_genotype.append(gene2)
                    child2_genotype.append(gene1)

        elif method == "single_point":
            point = random.randint(1, len(self.genotype) - 1)
            child1_genotype = self.genotype[:point] + other.genotype[point:]
            child2_genotype = other.genotype[:point] + self.genotype[point:]

        else:
            raise ValueError(f"Unknown crossover method: {method}")

        return Individual(child1_genotype), Individual(child2_genotype)

    def evaluate(self, fitness_fn):
        """Evaluates the individual with the given fitness function."""
        self.fitness = fitness_fn(self.genotype)
        return self.fitness