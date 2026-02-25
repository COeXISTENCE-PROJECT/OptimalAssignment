import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_edge_coords(env):
    """Pobiera współrzędne X, Y i normalizuje je tak, by (0,0) był w lewym dolnym rogu."""
    edge_coords_raw = {}

    all_edges = traci.edge.getIDList()

    min_x = float("inf")
    min_y = float("inf")

    for eid in all_edges:
        try:
            lane_id = f"{eid}_0"
            shape = traci.lane.getShape(lane_id)
            if shape:
                avg_x = sum(p[0] for p in shape) / len(shape)
                avg_y = sum(p[1] for p in shape) / len(shape)
                edge_coords_raw[eid] = (avg_x, avg_y)

                if avg_x < min_x:
                    min_x = avg_x
                if avg_y < min_y:
                    min_y = avg_y
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
        with open(dict_path, "r") as f:
            edge_to_idx = json.load(f)
        return adj_matrix, edge_to_idx

    print("Generating adjacency matrix from TraCI")
    conn = env.unwrapped.simulator.sumo_connection
    edges = [e for e in conn.edge.getIDList() if not e.startswith(":")]
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
                        target_edge = target_lane.split("_")[0]
                        if target_edge in edge_to_idx:
                            adj_matrix[
                                edge_to_idx[edge_id], edge_to_idx[target_edge]
                            ] = 1
        except Exception:
            continue

    np.save(cache_path, adj_matrix)
    with open(dict_path, "w") as f:
        json.dump(edge_to_idx, f)

    return adj_matrix, edge_to_idx

    # def visualize_traffic_graph(data, edge_to_idx, exp_id, threshold=1):
    G = nx.DiGraph()
    idx_to_edge = {v: k for k, v in edge_to_idx.items()}

    edge_index = data.edge_index.t().tolist()
    weights = (
        data.edge_attr.view(-1).tolist()
        if data.edge_attr is not None
        else [1] * len(edge_index)
    )

    for i, (u_idx, v_idx) in enumerate(edge_index):
        if weights[i] > threshold:
            G.add_edge(idx_to_edge[u_idx], idx_to_edge[v_idx], weight=weights[i])

    if G.number_of_edges() == 0:
        print(f"Warning: No edges with flow > {threshold} found. Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.2, iterations=30, seed=42)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=15, node_color="lightgray", alpha=0.7, ax=ax
    )

    edges_plot = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_weights,
        edge_cmap=plt.cm.YlOrRd,
        width=1.5,
        arrowsize=10,
        ax=ax,
    )

    if edge_weights:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.YlOrRd,
            norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Traffic Flow (Vehicles)", rotation=270, labelpad=15)

    plt.title(
        f"Traffic Flow Network Analysis: {exp_id}\n(Active Edges Only, Flow > {threshold})",
        fontsize=14,
    )
    ax.set_axis_off()

    save_path = f"results/{exp_id}/traffic_analysis_{exp_id}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
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
    node_colors = [G.nodes[n]["flow"] for n in G.nodes()]

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=50,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        alpha=0.9,
        ax=ax,
    )

    edges_plot = nx.draw_networkx_edges(
        G, pos, edge_color="gray", alpha=0.3, arrowsize=8, ax=ax
    )

    # Pasek kolorów dla natężenia
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd,
        norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Traffic Intensity (per road)", rotation=270, labelpad=15)

    plt.title(
        f"Traffic State: {exp_id}\n(Nodes = Roads, Edges = Transitions)", fontsize=14
    )
    ax.set_axis_off()

    # Dynamiczne tworzenie folderu jeśli nie istnieje
    os.makedirs(f"results/{exp_id}", exist_ok=True)
    save_path = f"results/{exp_id}/traffic_analysis_{exp_id}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def aggregate_episode_flow(episode_num, records_folder, edge_to_idx):
    """Tworzy macierz przejść na podstawie snapshotów 'es_ep'."""
    n = len(edge_to_idx)
    flow_matrix = np.zeros((n, n))
    file_pattern = f"es_ep{episode_num}_"
    snaps = sorted(
        [
            f
            for f in os.listdir(records_folder)
            if f.startswith(file_pattern) and f.endswith(".csv")
        ]
    )

    if not snaps:
        print(
            f"Warning: No snapshot files found for episode {episode_num} with pattern '{file_pattern}'"
        )
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

            current_positions = dict(zip(df["agent_id"], df["edge_id"]))
            current_positions = dict(zip(df["agent_id"], df["edge_id"]))

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

    node_features = torch.tensor(flow_matrix.sum(axis=1), dtype=torch.float).unsqueeze(
        1
    )

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
    """
    Clear SUMO files that are empty or not in the episodes folder.
    Works only for the consecutive files with the same name.
    The files are named as <file_name>_<episode>.xml

    This is a destructive function, it will remove files from the directory!
    """
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
    print(
        f"""
    ----------------------------------------------------
                    Agents in traffic
    ----------------------------------------------------
    Total agents           | {len(env.all_agents)}
    Human agents           | {len(env.human_agents)}
    AV agents              | {len(env.machine_agents)}
    ----------------------------------------------------
    """
    )
