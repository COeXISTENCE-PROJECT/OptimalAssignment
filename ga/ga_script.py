import json
import os
import random
import time
from ga.decoder import Decoder
from ga.fitness import make_genttp_fitness, make_genttp_batch_fitness
from ga.model_loader import load_model
from ga.population import Population
from ga.scenario import load_scenario
import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

POPULATION_SIZE = 10
GENERATIONS = 20
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
ELITE_PERCENT = 0.2
SEED = 42
OUTPUT_DIR = "outputs/ga"


if __name__ == "__main__":
    print(f"Running GA on device: {DEVICE}")
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenario = load_scenario()
    decoder = Decoder(scenario)
    model = load_model(device=DEVICE)
    fitness_fn = make_genttp_batch_fitness(decoder, model, device=DEVICE)

    population = Population(POPULATION_SIZE, scenario.allowed_routes)
    elite_size = int(POPULATION_SIZE * ELITE_PERCENT)

    history = []
    for generation in range(GENERATIONS):
        population.evaluate_batch(fitness_fn)
        best = population.get_best_individual()
        mean = sum(ind.fitness for ind in population.individuals) / POPULATION_SIZE
        history.append({"generation": generation, "best": best.fitness, "mean": mean})
        print(f"gen {generation:3d}  best {best.fitness:.2f}  mean {mean:.2f}", flush=True)

        population.crossover(CROSSOVER_RATE, ELITE_PERCENT)
        population.mutate(MUTATION_RATE, elite_size)

    population.evaluate_batch(fitness_fn)
    best = population.get_best_individual()
    with open(f"{OUTPUT_DIR}/best_genotype.json", "w") as f:
        json.dump({"fitness": best.fitness, "genotype": best.genotype}, f)
    with open(f"{OUTPUT_DIR}/history.json", "w") as f:
        json.dump(history, f)
    print("best fitness:", best.fitness)