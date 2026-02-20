import os
import sys

# Ensure that the script always runs with the repository root
# added to PYTHONPATH, regardless of where it is executed from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random
import optuna

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from routerl import TrafficEnvironment
from tqdm import tqdm

from baseline_models import BaseLearningModel
from utils import clear_SUMO_files, print_agent_counts

import xml.etree.ElementTree as ET

import torch.nn.functional as F

import shutil

class ContextAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Warstwa wyliczająca wynik "ważności" dla każdego elementu wektora wejściowego
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid() # Wagi od 0 do 1
        )

    def forward(self, x):
        # x to połączony wektor [local_stats, global_stats]
        weights = self.attention_weights(x)
        # Mnożymy cechy przez ich wyuczone wagi (element-wise)
        return x * weights, weights


def update_agent_context(agent_embeddings, exp_id, episode_num, agents_df, device):
    xml_detailed_path = f"../results/{exp_id}/SUMO_output/detailed_sumo_stats_{episode_num}.xml"
    xml_global_path = f"../results/{exp_id}/SUMO_output/sumo_stats_{episode_num}.xml"
    csv_path = f"../results/{exp_id}/episodes/ep{episode_num}.csv"
    
    try:
        # --- PUNKT 1: Ekstrakcja Globalnych Statystyk ---
        global_tree = ET.parse(xml_global_path)
        g_root = global_tree.getroot()
        g_stats = g_root.find('vehicleTripStatistics')
        
        # Wektor globalny: [avg_speed, avg_waitingTime, avg_timeLoss]
        global_vector = torch.FloatTensor([
            float(g_stats.get('speed')),
            float(g_stats.get('waitingTime')),
            float(g_stats.get('timeLoss'))
        ]).to(device)

        # --- PUNKT 3: Travel Time Ratio (Normalizacja) ---
        df_recent = pd.read_csv(csv_path)
        detailed_tree = ET.parse(xml_detailed_path)
        d_root = detailed_tree.getroot()
        
        trip_data = []
        for trip in d_root.findall('tripinfo'):
            duration = float(trip.get('duration'))
            time_loss = float(trip.get('timeLoss'))
            # Czas idealny = czas faktyczny - straty (korki/światła)
            ideal_time = max(duration - time_loss, 1.0) 
            
            trip_data.append({
                'id': int(trip.get('id')),
                'time_ratio': duration / ideal_time, # 1.0 = trasa idealna
                'waitingTime': float(trip.get('waitingTime')),
                'timeLoss': time_loss
            })
        
        df_xml = pd.DataFrame(trip_data)
        combined = df_recent.merge(df_xml, on='id', how='left')
        
        # Agregacja po trasach
        route_stats = combined.groupby(['origin', 'destination']).agg({
            'time_ratio': 'mean',
            'waitingTime': 'mean',
            'timeLoss': 'mean'
        }).reset_index()

        # Mapowanie na wszystkich agentów
        final_context = agents_df.merge(route_stats, on=['origin', 'destination'], how='left').fillna(0)
        
        # Aktualizacja featurizera
        cols = ['time_ratio', 'waitingTime', 'timeLoss']
        agent_embeddings.route_context = torch.FloatTensor(final_context[cols].values).to(device)
        agent_embeddings.global_context = global_vector # Nowe pole
        
        print(f"--- Context updated: Global Speed: {global_vector[0]:.2f}, Mean Ratio: {final_context['time_ratio'].mean():.2f} ---")
        
    except Exception as e:
        print(f"Error updating context: {e}")


class ContextualAgentEmbedder(nn.Module):
    def __init__(self, agents_df, embed_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.route_context = torch.zeros((len(agents_df), 3)).to(device)
        self.global_context = torch.zeros(3).to(device)
        
        num_locs = max(agents_df['origin'].max(), agents_df['destination'].max()) + 1
        self.loc_embedding = nn.Embedding(num_locs, 10)
        self.raw_locs = torch.LongTensor(agents_df[['origin', 'destination']].values).to(device)

        # Moduł uwagi dla 6 statystyk (3 lokalne + 3 globalne)
        self.attention = ContextAttention(input_dim=6)

        # Wejście do FC: 10(O) + 10(D) + 6(Attended Stats) = 26
        self.fc = nn.Sequential(
            nn.Linear(26, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, agent_idx):
        if not isinstance(agent_idx, torch.Tensor):
            agent_idx = torch.tensor(agent_idx, device=self.device)
        if agent_idx.dim() == 0:
            agent_idx = agent_idx.unsqueeze(0)
            
        o_emb = self.loc_embedding(self.raw_locs[agent_idx, 0])
        d_emb = self.loc_embedding(self.raw_locs[agent_idx, 1])
        
        # Przygotowanie statystyk
        local_context = self.route_context[agent_idx]
        global_context_batch = self.global_context.repeat(agent_idx.shape[0], 1)
        stats = torch.cat([local_context, global_context_batch], dim=-1)
        
        # ZASTOSOWANIE ATTENTION
        # stats_weighted to przefiltrowane informacje
        stats_weighted, attn_scores = self.attention(stats)
        
        combined = torch.cat([o_emb, d_emb, stats_weighted], dim=-1)
        
        out = self.fc(combined)
        return out.squeeze(0) if agent_idx.shape[0] == 1 else out
        

class HyperNetwork(nn.Module):    
    """
    HyperNetwork that generates parameters (weights and biases)
    of a policy network conditioned on an agent embedding.
    """
    def __init__(self, agent_embed_dim, state_dim, action_dim, hidden_sizes):
        """
        Parameters
        ----------
        agent_embed_dim : int
            Dimensionality of agent embedding.
        state_dim : int
            Input dimension of the policy network.
        action_dim : int
            Output dimension (number of actions).
        hidden_sizes : list[int]
            Hidden layer sizes of the policy network.
        """
        super().__init__()

        # Full policy network architecture
        layer_sizes = [state_dim] + hidden_sizes + [action_dim]
        self.layer_sizes = layer_sizes

        # Compute parameter counts per layer (weights + biases)
        self.param_sizes = [
            layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
            for i in range(len(layer_sizes) - 1)
        ]
        self.total_params = sum(self.param_sizes)

        # Hypernetwork MLP that outputs all policy parameters
        self.net = nn.Sequential(
            nn.Linear(agent_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.total_params)
        )

    def forward(self, agent_embed):
        """
        Generate flattened policy parameters from an agent embedding.
        """
        return self.net(agent_embed)

def functional_mlp(x, weights, biases):
    """
    Stateless forward pass of an MLP using explicitly provided
    weights and biases.

    This is used because the policy network parameters
    are generated dynamically by a HyperNetwork.
    """
    for W, b in zip(weights[:-1], biases[:-1]):
        x = torch.relu(x @ W + b)
    return x @ weights[-1] + biases[-1]

class PPO(BaseLearningModel):
    """
    Proximal Policy Optimization (PPO) agent whose policy parameters
    are generated by a shared HyperNetwork.
    """
    def __init__(self,
            state_size, 
            action_space_size, 
            agent_id, hypernet, 
            agent_embeddings,
            device="cpu", 
            batch_size=16, 
            lr=0.003, 
            num_epochs=4, 
            hidden_sizes=[32, 64, 32], 
            clip_eps=0.2, 
            normalize_advantage=True, 
            entropy_coef=0.3, 
            total_training_eps=10000
        ):
        """
        Initialize PPO agent with shared HyperNetwork.
        """
        super().__init__()
        self.device = device
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.agent_id = agent_id
        
        self.hypernet = hypernet
        self.agent_embeddings = agent_embeddings
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_eps = clip_eps
        self.normalize_advantage = normalize_advantage
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef
        self.hidden_sizes = hidden_sizes
        self.total_training_eps = total_training_eps

        # Optimizer jointly updates hypernetwork and agent embeddings
        self.optimizer = optim.Adam(
            list(self.hypernet.parameters()) + list(self.agent_embeddings.parameters()),
            lr=lr
        )
        
        # Linear learning-rate decay
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_training_eps)
        
        self.memory = []
        self.loss = []
        self.deterministic = False

    def update_params(self, episode):
        """
        Update learning rate and entropy coefficient schedule.
        """
        self.scheduler.step()
        self.entropy_coef = self.initial_entropy_coef * max(
            0.01,
            (1 -episode / self.total_training_eps)
        )

    def _get_weights(self):
        """
        Generate policy network weights and biases for this agent.
        """
        agent_embed = self.agent_embeddings(self.agent_id)
        params = self.hypernet(agent_embed)
        
        weights, biases = [], []
        idx = 0
        layer_sizes = [self.state_size] + self.hidden_sizes + [self.action_space_size]
        
        for i in range(len(layer_sizes) - 1):
            w_size = layer_sizes[i] * layer_sizes[i+1]
            b_size = layer_sizes[i+1]
            
            weights.append(params[idx:idx + w_size].view(layer_sizes[i], layer_sizes[i+1]))
            idx += w_size
            
            biases.append(params[idx:idx + b_size])
            idx += b_size

        return weights, biases


    def act(self, state):
        """
        Sample or select an action given the current state.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        weights, biases = self._get_weights()

        logits = functional_mlp(state, weights, biases)
        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = torch.argmax(probs) if self.deterministic else dist.sample()

        # Store transition data for PPO update
        self.last_state = state.detach().cpu().numpy()
        self.last_action = action.item()
        self.last_log_prob = dist.log_prob(action).item()

        return action.item()
    
    def push(self, reward):
        """
        Store transition reward in replay buffer.
        """
        self.memory.append((
            self.last_state,
            self.last_action,
            self.last_log_prob,
            reward
        ))

    def learn(self):
        """
        Perform PPO update using stored transitions.
        """
        if len(self.memory) < self.batch_size:
            return

        step_losses = []

        for _ in range(self.num_epochs):
            batch = random.sample(self.memory, self.batch_size)
            states, actions, old_log_probs, rewards = zip(*batch)

            # Remove extra singleton dimension from states
            states = torch.FloatTensor(np.array(states)).to(self.device).squeeze(1)
            actions = torch.LongTensor(actions).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)

            weights, biases = self._get_weights()
            logits = functional_mlp(states, weights, biases)
            dist = torch.distributions.Categorical(logits=logits)
            
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Advantage normalization
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if self.normalize_advantage else rewards

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            # PPO objective with entropy regularization
            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist.entropy().mean()

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for shared parameters
            torch.nn.utils.clip_grad_norm_(list(self.hypernet.parameters()) + list(self.agent_embeddings.parameters()), 1)
            self.optimizer.step()
            step_losses.append(loss.item())

        self.loss.append(np.mean(step_losses))
        self.memory.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    args = parser.parse_args()

    ALGORITHM = "hyp_ippo"
    exp_id = args.id
    env_config = args.env_conf
    task_config = args.task_conf
    alg_config = args.alg_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Torch seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is:", device)
    params = {}
    params.update(json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json")))
    params.update(json.load(open(f"../config/env_config/{env_config}.json")))
    params.update(json.load(open(f"../config/task_config/{task_config}.json")))
    del params["desc"]

    for k, v in params.items():
        globals()[k] = v

    custom_network_folder = f"../networks/{network}"
    records_folder = f"../results/{exp_id}"
    snapshots_dir = os.path.join(records_folder, "snapshots")
    plots_folder = f"{records_folder}/plots"
    os.makedirs(records_folder, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]

    with open(os.path.join(custom_network_folder, f"od_{network}.txt")) as f:
        od_data = ast.literal_eval(f.read())

    origins = od_data["origins"]
    destinations = od_data["destinations"]

    agents_csv = os.path.join(custom_network_folder, "agents.csv")
    agents_df = pd.read_csv(agents_csv)
    agents_df.to_csv(os.path.join(records_folder, "agents.csv"), index=False)

    num_agents = len(agents_df)
    max_start_time = agents_df["start_time"].max()
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps

    dump_config = params.copy()
    dump_config.update({
        "network": network,
        "env_seed": env_seed,
        "torch_seed": torch_seed,
        "algorithm": ALGORITHM,
        "num_agents": num_agents,
        "num_machines": num_machines,
        "script": os.path.abspath(__file__)
    })

    with open(os.path.join(records_folder, "exp_config.json"), "w") as f:
        json.dump(dump_config, f, indent=4)

    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {"model": human_model},
            "machine_parameters": {
                "behavior": av_behavior,
                "observation_type": "previous_agents_plus_start_time"
            }
        },
        environment_parameters={"save_every": save_every},
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "smooth_by": smooth_by,
            "plot_choices": plot_choices,
            "records_folder": records_folder,
            "plots_folder": plots_folder
        },
        path_generation_parameters={
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": path_gen_beta,
            "num_samples": num_samples,
            "visualize_paths": False,
        }
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    pbar = tqdm(total=total_episodes, desc="Human learning")
    for _ in range(human_learning_episodes):
        env.step()
        pbar.update()

    env.mutation(
        disable_human_learning=not should_humans_adapt,
        mutation_start_percentile=-1
    )
    print_agent_counts(env)
    
    models_folder = os.path.join(records_folder, "saved_models")
    os.makedirs(models_folder, exist_ok=True)

    post_mutation_path = os.path.join(models_folder, "model_post_mutation.pth")
    #env.machine_agents[0].model.save(post_mutation_path)
    print(f"Model zapisany po mutacji: {post_mutation_path}")
    
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    action_size = env.machine_agents[0].action_space_size

    machine_ids = [int(a.id) for a in env.machine_agents]
    machine_features_df = agents_df[agents_df['id'].isin(machine_ids)].reset_index(drop=True)
    
    agent_embed_dim = 64

    agent_embeddings = ContextualAgentEmbedder(
            machine_features_df, 
            agent_embed_dim, 
            device=device
        ).to(device)
    
    hypernet = HyperNetwork(
        agent_embed_dim,
        obs_size,
        action_size,
        hidden_sizes=widths
    ).to(device)

    for idx, agent in enumerate(env.machine_agents):
        agent.model = PPO(
            state_size=obs_size,
            action_space_size=action_size,
            agent_id=idx,
            hypernet=hypernet,
            agent_embeddings=agent_embeddings,
            device=device,
            batch_size=batch_size,
            lr=lr,
            num_epochs=num_epochs,
            hidden_sizes=widths,
            clip_eps=clip_eps,
            normalize_advantage=normalize_advantage,
            entropy_coef=entropy_coef,
            total_training_eps=training_eps
        )

    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}

    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)

    for episode in range(training_eps):
        if episode > 0 and episode % update_every == 0:
            update_agent_context(
                agent_embeddings=agent_embeddings,
                exp_id=exp_id,
                episode_num=episode,
                agents_df=machine_features_df,
                device=device
            )
        
        env.reset()
        env.machine_agents[0].model.update_params(episode) 
        
        snapshot_interval = max(1, max_start_time // 10)
        snapshots_taken = 0
        
        for agent_id in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            current_sim_step = env.unwrapped.simulator.timestep
            
            if snapshots_taken < 10 and current_sim_step >= (snapshots_taken * snapshot_interval):
                env.unwrapped.simulator.save_snapshot_two(episode, snapshots_taken)
                snapshots_taken += 1
            
            if term or trunc:
                agent_lookup[agent_id].model.push(reward)
                if episode % update_every == 0:
                    agent_lookup[agent_id].model.learn()
                action = None
            else:
                action = agent_lookup[agent_id].model.act(obs)

            env.step(action)

        if episode % plot_every == 0:
            env.plot_results()

        pbar.update()

    for agent in env.machine_agents:
        agent.model.deterministic = True

    pbar.set_description("Testing")
    for _ in range(test_eps):
        env.reset()
        for agent_id in env.agent_iter():
            obs, _, term, trunc, _ = env.last()
            action = None if (term or trunc) else agent_lookup[agent_id].model.act(obs)
            env.step(action)
        pbar.update()
    pbar.close()
    env.plot_results()

    losses = pd.DataFrame([
        {"id": agent.id, "losses": agent.model.loss}
        for agent in env.machine_agents
    ])
    losses.to_csv(os.path.join(records_folder, "losses.csv"), index=False)

    env.stop_simulation()
    
    shutil.copytree(
        os.path.join(records_folder, "SUMO_output"),
        snapshots_dir,
        dirs_exist_ok=True
    )
    
    clear_SUMO_files(
        os.path.join(records_folder, "SUMO_output"),
        os.path.join(records_folder, "episodes"),
        remove_additional_files=True
    )