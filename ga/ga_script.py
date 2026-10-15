import json
import os
import random

from ga.decoder import Decoder
from ga.fitness import make_fitness
from ga.population import Population
from ga.scenario import load_scenario


POPULATION_SIZE = 20
GENERATIONS = 100
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
ELITE_PERCENT = 0.2
SEED = 42
OUTPUT_DIR = "outputs/ga/"


if __name__ == "__main__":
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenario = load_scenario()
    decoder = Decoder(scenario)
    fitness_fn = make_fitness(decoder)

    population = Population(POPULATION_SIZE, scenario.allowed_routes)
    elite_size = int(POPULATION_SIZE * ELITE_PERCENT)

    history = []
    for generation in range(GENERATIONS):
        population.evaluate(fitness_fn)
        best = population.get_best_individual()
        mean = sum(ind.fitness for ind in population.individuals) / POPULATION_SIZE
        history.append({"generation": generation, "best": best.fitness, "mean": mean})
        print(f"gen {generation:3d}  best {best.fitness:.2f}  mean {mean:.2f}")

        population.crossover(CROSSOVER_RATE, ELITE_PERCENT)
        population.mutate(MUTATION_RATE, elite_size)

    population.evaluate(fitness_fn)
    best = population.get_best_individual()
    with open(f"{OUTPUT_DIR}/best_genotype.json", "w") as f:
        json.dump({"fitness": best.fitness, "genotype": best.genotype}, f)
    with open(f"{OUTPUT_DIR}/history.json", "w") as f:
        json.dump(history, f)
    print("best fitness:", best.fitness)