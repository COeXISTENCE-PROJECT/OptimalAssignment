import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from models.ppo import PPO
from utils import *
import argparse
import ast
import json
import logging
import random
import shutil
import numpy as np
import pandas as pd
import torch

from routerl import TrafficEnvironment
from tqdm import tqdm

os.environ["PROJ_LIB"] = "/opt/homebrew/share/proj"


def initialize_env(params, paths, args):
    """Konfiguruje i zwraca środowisko TrafficEnvironment."""
    # Pobieranie danych OD
    with open(os.path.join(paths["network"], f"od_{args.net}.txt")) as f:
        od_data = ast.literal_eval(f.read())

    agents_df = pd.read_csv(os.path.join(paths["network"], "agents.csv"))
    agents_df.to_csv(os.path.join(paths["records"], "agents.csv"), index=False)

    # Obliczanie faz
    phases = [
        1,
        params["human_learning_episodes"],
        int(params["training_eps"]) + params["human_learning_episodes"],
    ]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]

    env = TrafficEnvironment(
        seed=args.env_seed,
        create_agents=False,
        create_paths=True,
        agent_parameters={
            "new_machines_after_mutation": int(
                len(agents_df) * params["ratio_machines"]
            ),
            "human_parameters": {"model": params["human_model"]},
            "machine_parameters": {
                "behavior": params["av_behavior"],
                "observation_type": "previous_agents_plus_start_time",
            },
        },
        simulator_parameters={
            "network_name": args.net,
            "custom_network_folder": paths["network"],
            "sumo_type": "sumo",
            "simulation_timesteps": agents_df["start_time"].max(),
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "records_folder": paths["records"],
            "plots_folder": paths["plots"],
            "smooth_by": params.get("smooth_by", 5),
            "plot_choices": params.get("plot_choices", []),
        },
        path_generation_parameters={
            "origins": od_data["origins"],
            "destinations": od_data["destinations"],
            "number_of_paths": params["number_of_paths"],
            "beta": params["path_gen_beta"],
            "num_samples": params["num_samples"],
        },
    )
    return env, agents_df


def setup_experiment(args):
    """
    Przygotowuje środowisko uruchomieniowe: foldery, nasiona (seeds) i parametry.
    Zwraca krotkę (params, device, paths).
    """
    # 1. Konfiguracja ścieżek
    paths = {
        "network": f"networks/{args.net}",
        "records": f"results/{args.id}",
        "snapshots": os.path.join(f"results/{args.id}", "snapshots"),
        "plots": f"results/{args.id}/plots",
    }

    for path in [paths["records"], paths["snapshots"], paths["plots"]]:
        os.makedirs(path, exist_ok=True)

    # 2. Logowanie i system
    print(f"### STARTING EXPERIMENT: {args.id} ###")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # 3. Determinizm (Seeds)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 4. Ładowanie parametrów
    params = {}
    configs = [
        f"config/algo_config/rl_alg/{args.alg_conf}.json",
        f"config/env_config/{args.env_conf}.json",
        f"config/task_config/{args.task_conf}.json",
    ]

    for conf in configs:
        with open(conf, "r") as f:
            params.update(json.load(f))

    if "desc" in params:
        del params["desc"]

    return params, device, paths


def init_ppo_agents(env, params, device):
    """Inicjalizuje modele PPO dla wszystkich agentów maszynowych."""
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    action_size = env.machine_agents[0].action_space_size

    for idx, agent in enumerate(env.machine_agents):
        agent.model = PPO(
            state_size=obs_size,
            action_space_size=action_size,
            agent_id=idx,
            device=device,
            batch_size=params["batch_size"],
            lr=params["lr"],
            num_epochs=params["num_epochs"],
            hidden_sizes=params["widths"],
            clip_eps=params["clip_eps"],
            total_training_eps=params["training_eps"],
        )
    return {str(agent.id): agent for agent in env.machine_agents}


def run_av_learning(env, agent_lookup, params, pbar):
    """Przeprowadza proces uczenia agentów AV."""
    pbar.set_description("AV learning")
    no_snaps = 4000

    for episode in range(params["training_eps"]):
        env.reset()
        env.machine_agents[0].model.update_params(episode)
        snapshots_taken = 0

        for agent_id in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()

            if snapshots_taken < no_snaps:
                env.unwrapped.simulator.save_snapshot_two(episode, snapshots_taken)
                snapshots_taken += 1

            if term or trunc:
                if agent_id in agent_lookup:
                    agent_lookup[agent_id].model.push(reward)
                    if episode % params["update_every"] == 0:
                        agent_lookup[agent_id].model.learn()
                action = None
            else:
                action = agent_lookup[agent_id].model.act(obs)
            env.step(action)
        pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--env-conf", type=str, default="config1")
    parser.add_argument("--task-conf", type=str, required=True)
    parser.add_argument("--alg-conf", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    args = parser.parse_args()

    params, device, paths = setup_experiment(args)

    env, agents_df = initialize_env(params, paths, args)

    env.start()
    env.reset()

    total_eps = (
        params["human_learning_episodes"] + params["training_eps"] + params["test_eps"]
    )
    pbar = tqdm(total=total_eps, desc="Human learning")

    for _ in range(params["human_learning_episodes"]):
        env.step()
        pbar.update()

    env.mutation(
        disable_human_learning=not params["should_humans_adapt"],
        mutation_start_percentile=-1,
    )
    agent_lookup = init_ppo_agents(env, params, device)

    run_av_learning(env, agent_lookup, params, pbar)

    pbar.set_description("Testing phase")
    for _ in range(params["test_eps"]):
        env.reset()
        for agent_id in env.agent_iter():
            obs, _, term, trunc, _ = env.last()
            action = None if (term or trunc) else agent_lookup[agent_id].model.act(obs)
            env.step(action)
        pbar.update()

    env.stop_simulation()

    if os.path.exists(os.path.join(paths["records"], "SUMO_output")):
        shutil.copytree(
            os.path.join(paths["records"], "SUMO_output"),
            paths["snapshots"],
            dirs_exist_ok=True,
        )
    pbar.close()
