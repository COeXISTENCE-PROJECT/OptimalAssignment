import xml.etree.ElementTree as ET
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

import traci
import networkx as nx
from torch_geometric.data import Data

import json


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


def calculate_component2_loss(reconstructed_seq, target_seq, device='cpu'):
    # Proporcjonalnie rosnące wagi dla kroków czasowych
    t_weights = torch.linspace(1.0, 5.0, steps=reconstructed_seq.shape[0]).to(device)
    
    # MSE dla każdego kroku czasowego uśrednione po węzłach
    mse_per_step = ((reconstructed_seq - target_seq)**2).mean(dim=1)
    
    # Średnia ważona
    return (mse_per_step * t_weights).mean()


# def calculate_component2_loss(predicted_A_seq, target_A_seq):
#     """
#     Loss with weight increasing with time t.
#     Loss = sum_{t} (weight_t * ||A_t_pred - A_t_target||^2)
#     """
#     T = len(predicted_A_seq)
#     total_loss = 0
#     for t in range(T):
#         weight_t = (t + 1) / T 
#         step_loss = F.mse_loss(predicted_A_seq[t], target_A_seq[t])
#         total_loss += weight_t * step_loss
#     return total_loss
def optimize_latent_assignment(world_model, autoencoder, edge_index, latent_dim, num_nodes, device='cpu', num_iterations=200):
    latent_vector = torch.randn(1, latent_dim, requires_grad=True, device=device)
    optimizer = optim.Adam([latent_vector], lr=0.05)
    
    world_model.eval()
    autoencoder.eval()
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Rekonstrukcja CAŁEJ sekwencji z latent vector
        reconstructed_seq_flat = autoencoder.decoder(latent_vector) 
        reconstructed_seq = reconstructed_seq_flat.view(50, num_nodes) 
        
        # Wybieramy OSTATNI stan (G^T), aby ocenić koszt końcowy
        # (Zgodnie z dokumentacją: optymalizujemy stan końcowy)
        A_final = reconstructed_seq[-1].view(-1, 1)
        
        # f0(fd(z))
        z_world = world_model.encoder(A_final, edge_index)
        predicted_r = world_model.travel_time_head(z_world.mean(dim=0))
        
        loss = predicted_r
        loss.backward()
        optimizer.step()
        
    return latent_vector.detach()

def retrieve_optimal_solution(optimized_z, autoencoder, inverse_model, initial_state_A0, no_snaps=50, device='cpu'):
    # 1. Odzyskujemy pełną sekwencję G^T z wektora latentnego
    optimal_G_T_flat = autoencoder.decoder(optimized_z)
    # Zmieniamy kształt na [no_snaps, num_nodes]
    optimal_G_T = optimal_G_T_flat.view(no_snaps, -1)
    
    # 2. Problem: autoencoder(initial_state_A0) wybuchnie, bo spodziewa się 50 kroków.
    # Musimy stworzyć "pustą" sekwencję, gdzie tylko pierwszy krok to A0,
    # albo po prostu podać A0 powtórzone 50 razy, aby przejść przez wymiar warstwy Linear.
    dummy_seq = initial_state_A0.repeat(no_snaps, 1) # [50, 853]
    _, z_start = autoencoder(dummy_seq)
    
    # 3. Component 1.5: Porównujemy latent stanu początkowego z optymalnym
    optimal_assignment_demand = inverse_model.retrieve_assignment(z_start, optimized_z)
    
    return optimal_G_T, optimal_assignment_demand

def update_agent_context(agent_embeddings, exp_id, episode_num, agents_df, device):
    xml_detailed_path = f"results/{exp_id}/SUMO_output/detailed_sumo_stats_{episode_num}.xml"
    xml_global_path = f"results/{exp_id}/SUMO_output/sumo_stats_{episode_num}.xml"
    
    csv_path = f"results/{exp_id}/episodes/ep{episode_num}.csv"
    
    if not os.path.exists(xml_detailed_path) or os.path.getsize(xml_detailed_path) == 0:
        print(f"Warning: XML {episode_num} is empty or missing. Skipping context update.")
        return

    detailed_tree = ET.parse(xml_detailed_path)
    d_root = detailed_tree.getroot()
    
    trip_data = []
    for trip in d_root.findall('tripinfo'):
        trip_data.append({
            'id': int(trip.get('id')),
            'time_ratio': float(trip.get('duration')) / max(float(trip.get('duration')) - float(trip.get('timeLoss')), 1.0),
            'waitingTime': float(trip.get('waitingTime')),
            'timeLoss': float(trip.get('timeLoss'))
        })
    
    if not trip_data:
        print(f"No trip data found in XML for ep {episode_num}")
        return

    df_xml = pd.DataFrame(trip_data)
    
    if not os.path.exists(csv_path):
        print(f"CSV {csv_path} missing.")
        return
        
    df_recent = pd.read_csv(csv_path)
    df_recent['id'] = df_recent['id'].astype(int)
    combined = df_recent.merge(df_xml, on='id', how='inner')
    route_stats = combined.groupby(['origin', 'destination']).agg({
        'time_ratio': 'mean',
        'waitingTime': 'mean',
        'timeLoss': 'mean'
    }).reset_index()
    final_context = agents_df.merge(route_stats, on=['origin', 'destination'], how='left').fillna(0)
    cols = ['time_ratio', 'waitingTime', 'timeLoss']
    agent_embeddings.route_context = torch.FloatTensor(final_context[cols].values).to(device)
    agent_embeddings.global_context = global_vector
    
    print(f"--- Context updated: Global Speed: {global_vector[0]:.2f}, Mean Ratio: {final_context['time_ratio'].mean():.2f} ---")


def extract_dynamic_graph_sequence(episode_num, snapshots_dir, edge_to_idx, max_snaps=10):
    """
    Realizuje Component 1: Buduje sekwencję A_t oraz D_t.
    Return: 
        A_seq: Tensor (T, num_edges, feature_dim) - cechy krawędzi w czasie
        D_seq: Tensor (T, num_nodes) - popyt wchodzący w czasie
    """
    num_edges = len(edge_to_idx)
    # k_a = 2 (count, avg_speed)
    A_seq = torch.zeros((max_snaps, num_edges, 2)) 
    D_seq = torch.zeros((max_snaps, num_edges))

    file_pattern = f"es_ep{episode_num}_"
    snaps = sorted([f for f in os.listdir(snapshots_dir) if f.startswith(file_pattern)],
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for t, snap_file in enumerate(snaps[:max_snaps]):
        df = pd.read_csv(os.path.join(snapshots_dir, snap_file))
        
        # A_t: Liczba pojazdów na krawędzi (Wymiar 2 z LaTeX)
        counts = df['edge_id'].value_counts()
        for edge_id, count in counts.items():
            if edge_id in edge_to_idx:
                idx = edge_to_idx[edge_id]
                A_seq[t, idx, 0] = count  # Feature 0: Count
        
        # D_t: Popyt pojawiający się w systemie
        # Zakładamy, że D_t to pojazdy, które są w pierwszym snapshocie na swojej pierwszej krawędzi
        # (Można to doprecyzować sprawdzając czas odjazdu z agents_df)
        if 'is_new_departure' in df.columns: # Wymaga dodania flagi w save_snapshot_two
            new_departures = df[df['is_new_departure'] == True]['edge_id'].value_counts()
            for edge_id, count in new_departures.items():
                if edge_id in edge_to_idx:
                    D_seq[t, edge_to_idx[edge_id]] = count

    return A_seq, D_seq



def plot_traffic_heatmap(episode_num, snapshots_dir, edge_coords, top_n=10):
    """
    Tworzy heatmapę i zapisuje CSV z dodatkowymi kolumnami coord_x, coord_y.
    """
    file_pattern = f"es_ep{episode_num}_"
    snaps = sorted([f for f in os.listdir(snapshots_dir) if f.startswith(file_pattern) and f.endswith('.csv')])
    heatmap_data = {}
    for i, snap_file in enumerate(snaps):
        df = pd.read_csv(os.path.join(snapshots_dir, snap_file))
        counts = df['edge_id'].value_counts().to_dict()
        
        for edge_id, count in counts.items():
            if edge_id not in heatmap_data:
                heatmap_data[edge_id] = [0] * len(snaps)
            heatmap_data[edge_id][i] = count

    full_df = pd.DataFrame.from_dict(heatmap_data, orient='index')
    full_df.columns = [f"Step {i}" for i in range(len(snaps))]

    # Mapujemy ID krawędzi (index) na x i y ze słownika edge_coords
    
    full_df['coord_x'] = full_df.index.map(lambda x: edge_coords.get(str(x), (0.0, 0.0))[0])
    full_df['coord_y'] = full_df.index.map(lambda x: edge_coords.get(str(x), (0.0, 0.0))[1])
    
    cols = ['coord_x', 'coord_y'] + [c for c in full_df.columns if c not in ['coord_x', 'coord_y']]
    full_df = full_df[cols]
    
    csv_file = f"{snapshots_dir}/traffic_heatmap_data_ep{episode_num}.csv"
    full_df.to_csv(csv_file)

    step_cols = [c for c in full_df.columns if c.startswith('Step')]
    top_edges = full_df[step_cols].sum(axis=1).nlargest(top_n).index
    df_filtered = full_df.loc[top_edges, step_cols]

    plt.figure(figsize=(14, 10))
    sns.heatmap(df_filtered, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={'label': 'Vehicles'})
    
    plt.title(f"Traffic flow heatmap \n(Top {top_n} edges)")

    plt.xlabel("Timesteps")
    plt.ylabel("Edge ID")
    
    save_path = f"{snapshots_dir}/traffic_heatmap_ep{episode_num}.png"
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

# def get_edge_coords(env):
#     """Pobiera współrzędne środka krawędzi w sposób bezpieczny."""
#     edge_coords = {}
#     all_edges = traci.edge.getIDList()
    
#     for eid in all_edges:
#         try:
#             lane_id = f"{eid}_0"
#             shape = traci.lane.getShape(lane_id)
            
#             if shape:
#                 # Obliczamy centroid (środek) łamanej pasu
#                 avg_x = sum(p[0] for p in shape) / len(shape)
#                 avg_y = sum(p[1] for p in shape) / len(shape)
#                 edge_coords[eid] = (avg_x, avg_y)
#         except (traci.exceptions.TraCIException, Exception):
#             edge_coords[eid] = (0.0, 0.0)
            
#     return edge_coords

def get_edge_coords(env):
    """Pobiera współrzędne X, Y i normalizuje je tak, by (0,0) był w lewym dolnym rogu."""
    edge_coords_raw = {}
    
    all_edges = traci.edge.getIDList()
    
    min_x = float('inf')
    min_y = float('inf')

    for eid in all_edges:
        try:
            lane_id = f"{eid}_0"
            shape = traci.lane.getShape(lane_id)
            if shape:
                avg_x = sum(p[0] for p in shape) / len(shape)
                avg_y = sum(p[1] for p in shape) / len(shape)
                edge_coords_raw[eid] = (avg_x, avg_y)
                
                if avg_x < min_x: min_x = avg_x
                if avg_y < min_y: min_y = avg_y
        except:
            continue

    normalized_coords = {}
    for eid, (raw_x, raw_y) in edge_coords_raw.items():
        norm_x = raw_x - min_x
        norm_y = raw_y - min_y
        normalized_coords[eid] = (norm_x, norm_y)
            
    return normalized_coords

def get_edge_adjacency_matrix(env, network_name):
    cache_path = f"networks/{network_name}/adj_matrix.npy"
    dict_path = f"networks/{network_name}/edge_to_idx.json"
    
    if os.path.exists(cache_path) and os.path.exists(dict_path):
        print(f"Loading adjacency matrix from cache for {network_name}")
        adj_matrix = np.load(cache_path)
        with open(dict_path, 'r') as f:
            edge_to_idx = json.load(f)
        return adj_matrix, edge_to_idx

    print("Generating adjacency matrix from TraCI")
    conn = env.unwrapped.simulator.sumo_connection 
    edges = [e for e in conn.edge.getIDList() if not e.startswith(':')]
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    n = len(edges)
    adj_matrix = np.zeros((n, n), dtype=int)

    for edge_id in edges:
        try:
            num_lanes = conn.edge.getLaneNumber(edge_id)
            for lane_idx in range(num_lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                links = conn.lane.getLinks(lane_id)
                for link in links:
                    target_lane = link[0]
                    if target_lane:
                        target_edge = target_lane.split('_')[0]
                        if target_edge in edge_to_idx:
                            adj_matrix[edge_to_idx[edge_id], edge_to_idx[target_edge]] = 1
        except Exception:
            continue
            
    np.save(cache_path, adj_matrix)
    with open(dict_path, 'w') as f:
        json.dump(edge_to_idx, f)
        
    return adj_matrix, edge_to_idx

# def visualize_traffic_graph(data, edge_to_idx, exp_id, threshold=1):
    G = nx.DiGraph()
    idx_to_edge = {v: k for k, v in edge_to_idx.items()}
    
    edge_index = data.edge_index.t().tolist()
    weights = data.edge_attr.view(-1).tolist() if data.edge_attr is not None else [1] * len(edge_index)

    for i, (u_idx, v_idx) in enumerate(edge_index):
        if weights[i] > threshold:
            G.add_edge(idx_to_edge[u_idx], idx_to_edge[v_idx], weight=weights[i])

    if G.number_of_edges() == 0:
        print(f"Warning: No edges with flow > {threshold} found. Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.2, iterations=30, seed=42)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=15, node_color="lightgray", alpha=0.7, ax=ax)
    
    edges_plot = nx.draw_networkx_edges(
        G, pos, 
        edge_color=edge_weights, 
        edge_cmap=plt.cm.YlOrRd,
        width=1.5, 
        arrowsize=10,
        ax=ax
    )
    
    if edge_weights:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Traffic Flow (Vehicles)', rotation=270, labelpad=15)

    plt.title(f"Traffic Flow Network Analysis: {exp_id}\n(Active Edges Only, Flow > {threshold})", fontsize=14)
    ax.set_axis_off()
    
    save_path = f"results/{exp_id}/traffic_analysis_{exp_id}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Graph visualization saved to: {save_path}")
    plt.show()

    

def visualize_traffic_graph(data, edge_to_idx, exp_id, threshold=0.1):

    G = nx.DiGraph()
    idx_to_edge_name = {v: k for k, v in edge_to_idx.items()}
    
    # Pobieramy flow z data.x (bo tam trafiają wyniki z autoenkodera/SUMO)
    # Zakładamy, że data.x ma kształt [num_nodes, 1]
    node_flows = data.x.view(-1).tolist()
    
    # Dodajemy węzły (drogi) i ich flow
    for i, flow in enumerate(node_flows):
        if flow > threshold:
            G.add_node(i, name=idx_to_edge_name[i], flow=flow)
    
    # Dodajemy krawędzie (połączenia między drogami) tylko jeśli łączą aktywne drogi
    edge_index = data.edge_index.t().tolist()
    for u, v in edge_index:
        if u in G and v in G:
            G.add_edge(u, v)

    if G.number_of_nodes() == 0:
        print(f"Warning: No nodes with flow > {threshold} found.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    # spring_layout jest ok, ale 'kamada_kawai' często lepiej oddaje struktury drogowe
    pos = nx.kamada_kawai_layout(G)
    
    # Kolorujemy WĘZŁY (drogi) według natężenia ruchu
    node_colors = [G.nodes[n]['flow'] for n in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=50, 
        node_color=node_colors, 
        cmap=plt.cm.YlOrRd, 
        alpha=0.9, 
        ax=ax
    )
    
    edges_plot = nx.draw_networkx_edges(
        G, pos, 
        edge_color="gray", 
        alpha=0.3, 
        arrowsize=8,
        ax=ax
    )
    
    # Pasek kolorów dla natężenia
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Traffic Intensity (per road)', rotation=270, labelpad=15)

    plt.title(f"Traffic State: {exp_id}\n(Nodes = Roads, Edges = Transitions)", fontsize=14)
    ax.set_axis_off()
    
    # Dynamiczne tworzenie folderu jeśli nie istnieje
    os.makedirs(f"results/{exp_id}", exist_ok=True)
    save_path = f"results/{exp_id}/traffic_analysis_{exp_id}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def aggregate_episode_flow(episode_num, records_folder, edge_to_idx):
    """Tworzy macierz przejść na podstawie snapshotów 'es_ep'."""
    n = len(edge_to_idx)
    flow_matrix = np.zeros((n, n))
    file_pattern = f"es_ep{episode_num}_"
    snaps = sorted([f for f in os.listdir(records_folder) if f.startswith(file_pattern) and f.endswith('.csv')])
    
    if not snaps:
        print(f"Warning: No snapshot files found for episode {episode_num} with pattern '{file_pattern}'")
        return flow_matrix

    print(f"Processing {len(snaps)} snapshots for episode {episode_num}")
    prev_positions = {} 

    for snap_file in snaps:
        file_path = os.path.join(records_folder, snap_file)

        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {snap_file}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
                
            current_positions = dict(zip(df['agent_id'], df['edge_id']))
            current_positions = dict(zip(df['agent_id'], df['edge_id']))
            
            for a_id, current_edge in current_positions.items():
                if a_id in prev_positions:
                    prev_edge = prev_positions[a_id]
                    if prev_edge != current_edge:
                        u, v = edge_to_idx.get(prev_edge), edge_to_idx.get(current_edge)
                        if u is not None and v is not None:
                            flow_matrix[u, v] += 1
                
                prev_positions[a_id] = current_edge
        except pd.errors.EmptyDataError:
            print(f"Warning: {snap_file} is empty or corrupted. Skipping.")
            continue
        
    return flow_matrix

def create_pyg_graph(adj_matrix, flow_matrix):

    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    node_features = torch.tensor(flow_matrix.sum(axis=1), dtype=torch.float).view(-1, 1)
    
    edge_weights = torch.tensor(flow_matrix[adj_matrix > 0], dtype=torch.float)
    
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)
    
    return data

def build_pyg_data(adj_matrix, flow_matrix):
    
    edge_index = torch.tensor(np.argwhere(adj_matrix == 1).T, dtype=torch.long)
    
    node_features = torch.tensor(flow_matrix.sum(axis=1), dtype=torch.float).unsqueeze(1)
    
    weights = []
    for u, v in edge_index.t().tolist():
        weights.append(flow_matrix[u, v])
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def get_episodes(ep_path: str) -> list[int]:
    """Get the episodes data

    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    eps = list()
    if os.path.exists(ep_path):
        for file in os.listdir(ep_path):
            episode = int(file.split("ep")[1].split(".csv")[0])
            eps.append(episode)
    else:
        raise FileNotFoundError(f"Episodes folder does not exist!")

    return sorted(eps)


def clear_SUMO_files(sumo_path, ep_path, remove_additional_files=False):
    '''
        Clear SUMO files that are empty or not in the episodes folder.
        Works only for the consecutive files with the same name.
        The files are named as <file_name>_<episode>.xml

        This is a destructive function, it will remove files from the directory!
    '''
    file_id = 1
    episode = 1

    file_name = "detailed_sumo_stats"
    
    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    file_id = 1
    episode = 1

    file_name = "sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <vehicle loaded=0>
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("loaded") == "0":
                # remove the file
                os.remove(file_path)
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                file_id += 1
        else:
            break
        episode += 1
    if remove_additional_files:
        episodes = get_episodes(ep_path)
        # remove SUMO files that are not in the episodes
        for file in os.listdir(sumo_path):
            if file.endswith(".xml"):
                episode = int(file.split("_")[-1].split(".")[0])
                if episode not in episodes:
                    os.remove(os.path.join(sumo_path, file))
                    
                    
def print_agent_counts(env):
    print(f"""
    ----------------------------------------------------
                    Agents in traffic
    ----------------------------------------------------
    Total agents           | {len(env.all_agents)}
    Human agents           | {len(env.human_agents)}
    AV agents              | {len(env.machine_agents)}
    ----------------------------------------------------
    """)
