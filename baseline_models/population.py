from __future__ import annotations

import random
from typing import List, Tuple

from .genotype import Individual



class Population:
    """Represents a population of individuals for a genetic algorithm."""

    def __init__(self, population_size: int, num_agents: int, num_paths: int) -> None:
        """
        Initialize a population with randomly created individuals.

        Args:
            population_size: Number of individuals in the population.
            num_agents: Number of agents per individual.
            num_paths: Number of paths per individual.
        """
        self.population_size = population_size
        self.num_agents = num_agents
        self.num_paths = num_paths
        self.individuals: List[Individual] = [
            Individual.create(num_agents, num_paths)
            for _ in range(population_size)
        ]

    def evaluate(self, env, agent_ids) -> None:
        """
        Evaluate all individuals in the population.

        Args:
            env: Environment used for evaluation.
            agent_ids: Identifiers of agents to be evaluated.
        """
        for idx, individual in enumerate(self.individuals):
            individual.evaluate(env, agent_ids)
            print(f"  Individual {idx}: fitness={individual.fitness}")

    def select(
        self,
        elite_size: int,
        selection_type: str = "tournament",
    ) -> Tuple[List[Individual], List[Individual]]:
        """
        Select individuals for the next generation.

        Elite individuals are copied directly. The remaining parents
        are selected using the chosen selection strategy.

        Args:
            elite_size: Number of elite individuals to preserve.
            selection_type: Selection strategy ("tournament" or "roulette").

        Returns:
            A tuple of (elite_individuals, selected_parents).
        """
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        elite = self.individuals[:elite_size]
        parents: List[Individual] = []

        while len(parents) < self.population_size - elite_size:
            if selection_type == "tournament":
                parent = self.tournament_selection()
            elif selection_type == "roulette":
                parent = self.roulette_wheel()
            else:
                raise ValueError(f"Unknown selection_type: {selection_type}")

            # Ensure parents in a pair are not identical
            if len(parents) % 2 == 0 or parents[-1] != parent:
                parents.append(parent)

        return elite, parents

    def crossover(self, crossover_rate: float, elite_percent: float = 0.2) -> None:
        """
        Create a new generation using crossover and elitism.

        Args:
            crossover_rate: Probability of performing crossover.
            elite_percent: Fraction of the population preserved as elite.
        """
        elite_size = int(self.population_size * elite_percent)
        elite, parents = self.select(elite_size)

        new_generation: List[Individual] = list(elite)

        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                break

            parent1, parent2 = parents[i], parents[i + 1]

            if random.random() < crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = Individual(parent1.genotype[:])
                child2 = Individual(parent2.genotype[:])

            new_generation.extend((child1, child2))

        while len(new_generation) < self.population_size:
            new_generation.append(random.choice(self.individuals))

        self.individuals = new_generation

    def mutate(self, mutation_rate: float, elite_size: int = 0) -> None:
        """
        Mutate non-elite individuals in the population.

        Args:
            mutation_rate: Probability of mutation.
            elite_size: Number of elite individuals to exclude from mutation.
        """
        for individual in self.individuals[elite_size:]:
            individual.mutate(mutation_rate, self.num_paths)

    def mutate_elite(self, mutation_rate: float, elite_size: int) -> None:
        """
        Mutate elite individuals.

        Args:
            mutation_rate: Probability of mutation.
            elite_size: Number of elite individuals to mutate.
        """
        for individual in self.individuals[:elite_size]:
            individual.mutate(mutation_rate, self.num_paths)

    def get_best_individual(self) -> Individual:
        """
        Return the best individual in the population.

        Returns:
            Individual with the highest fitness.
        """
        return max(self.individuals, key=lambda ind: ind.fitness)

def roulette_wheel(self) -> Individual:
        """
        Select an individual using roulette wheel selection.

        Individuals with higher fitness have a higher probability
        of being selected.

        Returns:
            Selected individual.
        """
        total_fitness = sum(ind.fitness for ind in self.individuals)

        if total_fitness <= 0:
            return random.choice(self.individuals)

        pick = random.uniform(0.0, total_fitness)
        current = 0.0

        for individual in self.individuals:
            current += individual.fitness
            if current >= pick:
                return individual

        return random.choice(self.individuals)

def tournament_selection(self, tournament_size: int = 3) -> Individual:
    """
    Select an individual using tournament selection.

    Args:
        tournament_size: Number of individuals in the tournament.

    Returns:
        Best individual from the tournament.
    """
    tournament = random.sample(self.individuals, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

def restart(self, keep_elite: bool = True, elite_percent: float = 0.2) -> None:
    """
    Restart the population, optionally preserving elite individuals.

    Args:
        keep_elite: Whether to keep elite individuals.
        elite_percent: Fraction of population preserved as elite.
    """
    if keep_elite:
        elite_size = int(self.population_size * elite_percent)
        elite, _ = self.select(elite_size)
        self.individuals = elite + [
            Individual.create(self.num_agents, self.num_paths)
            for _ in range(self.population_size - len(elite))
        ]
    else:
        self.individuals = [
            Individual.create(self.num_agents, self.num_paths)
            for _ in range(self.population_size)
        ]