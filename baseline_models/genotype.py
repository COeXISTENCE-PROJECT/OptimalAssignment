from __future__ import annotations

import random
from typing import Dict, List, Optional


class Individual:
    """
    Represents an individual genotype in a genetic algorithm.

    Each individual assigns a path index to each agent and can be
    evaluated, mutated, and crossed over with other individuals.
    """

    def __init__(self, genotype: List[int]) -> None:
        """
        Initialize an individual.

        Args:
            genotype: List of path indices, one per agent.
        """
        self.genotype: List[int] = genotype
        self.fitness: Optional[float] = None

    @classmethod
    def create(cls, num_agents: int, num_paths: int) -> Individual:
        """
        Create a random individual.

        Args:
            num_agents: Number of agents.
            num_paths: Number of available paths.

        Returns:
            A newly created individual with a random genotype.
        """
        genotype = [
            random.randint(0, num_paths - 1)
            for _ in range(num_agents)
        ]
        return cls(genotype)

    def mutate(self, mutation_rate: float, num_paths: int) -> None:
        """
        Mutate the genotype in-place.

        Each gene has a probability of being replaced with a new
        random path index.

        Args:
            mutation_rate: Probability of mutation per gene.
            num_paths: Number of available paths.
        """
        for i, _ in enumerate(self.genotype):
            if random.random() < mutation_rate:
                self.genotype[i] = random.randint(0, num_paths - 1)

    def crossover(self, other: Individual) -> tuple[Individual, Individual]:
        """
        Perform single-point crossover with another individual.

        Args:
            other: The other parent individual.

        Returns:
            Two offspring individuals.
        """
        if len(self.genotype) < 2:
            return (
                Individual(self.genotype[:]),
                Individual(other.genotype[:]),
            )

        point = random.randint(1, len(self.genotype) - 1)

        child1_genotype = self.genotype[:point] + other.genotype[point:]
        child2_genotype = other.genotype[:point] + self.genotype[point:]

        return Individual(child1_genotype), Individual(child2_genotype)

    def evaluate(self, env, agent_ids: List) -> float:
        """
        Evaluate the individual in the given environment.

        The fitness is computed as the negative average travel time
        across all agents.

        Args:
            env: Environment to evaluate in.
            agent_ids: Ordered list of agent identifiers.

        Returns:
            Computed fitness value.
        """
        env.reset()

        total_travel_time = 0.0
        agent_count = 0

        path_assignments: Dict = {
            agent_id: self.genotype[i] if i < len(self.genotype) else 0
            for i, agent_id in enumerate(agent_ids)
        }

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                travel_time = -reward if reward is not None else 0.0
                total_travel_time += travel_time
                agent_count += 1
                action = None
            else:
                action = path_assignments.get(agent, 0)

            env.step(action)

        self.fitness = -total_travel_time / max(agent_count, 1)
        return self.fitness
