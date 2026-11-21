import os
import random
import time
from ga.decoder import Decoder
from ga.fitness import make_genttp_batch_fitness
from ga.model_loader import load_model, MODEL
from ga.population import Population
from ga.scenario import load_scenario
import argparse
from ga.utils import choose_device, save_config, save_results, save_plot, next_run_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic algorithm with GenTTP fitness.")
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--crossover_rate", type=float, default=0.9)
    parser.add_argument("--crossover_method", type=str, default="uniform",
                        choices=["uniform", "single_point"])
    parser.add_argument("--selection_type", type=str, default="tournament",
                        choices=["tournament", "roulette"])
    parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument("--elite_percent", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    args.device = choose_device(args.device)
    if args.output_dir is None:
        args.output_dir = next_run_dir()
    
    print(f"Running GA on device: {args.device}", flush=True)
    print(f"Saving results to: {args.output_dir}", flush=True)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args)

    scenario = load_scenario()
    decoder = Decoder(scenario)
    model = load_model(device=args.device)
    fitness_fn = make_genttp_batch_fitness(decoder, model, device=args.device)

    population = Population(args.population_size, scenario.allowed_routes)
    elite_size = int(args.population_size * args.elite_percent)

    history = []
    total_start = time.time()
    for generation in range(args.generations):
        gen_start = time.time()
        population.evaluate_batch(fitness_fn)
        gen_time = time.time() - gen_start

        best = population.get_best_individual()
        mean = sum(ind.fitness for ind in population.individuals) / args.population_size
        history.append({"generation": generation, "best": best.fitness, "mean": mean,
                        "time": gen_time, "best_genotype": best.genotype})
        print(f"gen {generation:3d}  best {best.fitness:.2f}  mean {mean:.2f}  time {gen_time:.1f}s", flush=True)

        save_results(args.output_dir, history, best)
        save_plot(args.output_dir, history)

        population.crossover(args.crossover_rate, args.elite_percent, method=args.crossover_method)
        population.mutate(args.mutation_rate, elite_size)

    population.evaluate_batch(fitness_fn)
    best = population.get_best_individual()
    save_results(args.output_dir, history, best)
    save_plot(args.output_dir, history)
    print(f"total time: {time.time() - total_start:.1f}s", flush=True)
    print("best fitness:", best.fitness)