import random

from .individual import Individual


class Population:
    def __init__(self, population_size, allowed_routes):
        self.population_size = population_size
        self.allowed_routes = allowed_routes
        self.individuals = [Individual.create(allowed_routes) for _ in range(population_size)]

    def evaluate(self, fitness_fn):
        """Evaluate all individuals in the population."""
        for ind in self.individuals:
            ind.evaluate(fitness_fn)

    def select(self, elite_size, selection_type="tournament"):
        """Selects individuals that pass to the next generation, the elite are directly copied."""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        elite = self.individuals[:elite_size]
        parents = []
        while len(parents) < (self.population_size - elite_size):
            if selection_type == "tournament":
                parent = self.tournament_selection()
            elif selection_type == "roulette":
                parent = self.roulette_wheel()
            else:
                raise ValueError(f"Unknown selection_type: {selection_type}")

            if len(parents) % 2 == 0:  # first parent of the new couple is added
                parents.append(parent)
            else:  # second parent of the new couple is added only if different from the first one
                if parents[-1] != parent:
                    parents.append(parent)
        return elite, parents

    def crossover(self, crossover_rate, elite_percent=0.2, method="uniform"):
        """Performs crossover or copies parents to create a new generation."""
        new_generation = []
        elite, parents = self.select(int(self.population_size * elite_percent))
        new_generation.extend(elite)
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                if random.random() < crossover_rate:
                    child1, child2 = parents[i].crossover(parents[i + 1], method=method)
                else:
                    child1 = Individual(parents[i].genotype[:])
                    child2 = Individual(parents[i + 1].genotype[:])
                new_generation.extend([child1, child2])

        while len(new_generation) < self.population_size:
            ind = random.choice(self.individuals)
            new_generation.append(Individual(ind.genotype[:]))
        self.individuals = new_generation

    # mutation might happen to certain genes independently from other individuals and genes in the same individual
    def mutate(self, mutation_rate, elite_size=0):
        for individual in self.individuals[elite_size:]:
            individual.mutate(mutation_rate, self.allowed_routes)

    # mutate elite just if there is no improvement for a certain number of generations
    def mutate_elite(self, mutation_rate, elite_size):
        for individual in self.individuals[:elite_size]:
            individual.mutate(mutation_rate, self.allowed_routes)

    def get_best_individual(self):
        return max(self.individuals, key=lambda ind: ind.fitness)

    def roulette_wheel(self):
        """Selects an individual using roulette wheel selection (requires fitness >= 0)."""
        total_fitness = sum(ind.fitness for ind in self.individuals)
        if total_fitness == 0:
            return random.choice(self.individuals)
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in self.individuals:
            current += ind.fitness
            if current >= pick:
                return ind
        return random.choice(self.individuals)

    def tournament_selection(self, tournament_size=3):
        """Selects the best individual among a random subset."""
        tournament = random.sample(self.individuals, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)