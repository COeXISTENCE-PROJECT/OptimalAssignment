import torch
import json
from ga.model_loader import MODEL
import matplotlib.pyplot as plt
import os
def choose_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def save_config(args):
    config = vars(args) | {"model": MODEL}
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

def save_results(output_dir, history, best):
    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(history, f)
    with open(f"{output_dir}/best_genotype.json", "w") as f:
        json.dump({"fitness": best.fitness, "genotype": best.genotype}, f)

def save_plot(output_dir, history):
    generations = [h["generation"] for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(generations, [-h["best"] for h in history], label="best individual")
    plt.plot(generations, [-h["mean"] for h in history], label="mean population")
    plt.xlabel("generation")
    plt.ylabel("TT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/history.png", dpi=150)
    plt.close()

def next_run_dir(base="outputs/ga"):
    """Return the next free numbered folder, e.g. outputs/ga/004."""
    os.makedirs(base, exist_ok=True)
    numbers = [int(name) for name in os.listdir(base) if name.isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"{base}/{next_number:03d}"

