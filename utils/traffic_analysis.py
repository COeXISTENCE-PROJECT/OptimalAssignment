import os
import xml.etree.ElementTree as ET
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def update_agent_context(agent_embeddings, exp_id, episode_num, agents_df, device):
    xml_detailed_path = (
        f"results/{exp_id}/SUMO_output/detailed_sumo_stats_{episode_num}.xml"
    )
    xml_global_path = f"results/{exp_id}/SUMO_output/sumo_stats_{episode_num}.xml"

    csv_path = f"results/{exp_id}/episodes/ep{episode_num}.csv"

    if not os.path.exists(xml_detailed_path) or os.path.getsize(xml_detailed_path) == 0:
        print(
            f"Warning: XML {episode_num} is empty or missing. Skipping context update."
        )
        return

    detailed_tree = ET.parse(xml_detailed_path)
    d_root = detailed_tree.getroot()

    trip_data = []
    for trip in d_root.findall("tripinfo"):
        trip_data.append(
            {
                "id": int(trip.get("id")),
                "time_ratio": float(trip.get("duration"))
                / max(float(trip.get("duration")) - float(trip.get("timeLoss")), 1.0),
                "waitingTime": float(trip.get("waitingTime")),
                "timeLoss": float(trip.get("timeLoss")),
            }
        )

    if not trip_data:
        print(f"No trip data found in XML for ep {episode_num}")
        return

    df_xml = pd.DataFrame(trip_data)

    if not os.path.exists(csv_path):
        print(f"CSV {csv_path} missing.")
        return

    df_recent = pd.read_csv(csv_path)
    df_recent["id"] = df_recent["id"].astype(int)
    combined = df_recent.merge(df_xml, on="id", how="inner")
    route_stats = (
        combined.groupby(["origin", "destination"])
        .agg({"time_ratio": "mean", "waitingTime": "mean", "timeLoss": "mean"})
        .reset_index()
    )
    final_context = agents_df.merge(
        route_stats, on=["origin", "destination"], how="left"
    ).fillna(0)
    cols = ["time_ratio", "waitingTime", "timeLoss"]
    agent_embeddings.route_context = torch.FloatTensor(final_context[cols].values).to(
        device
    )
    agent_embeddings.global_context = global_vector

    print(
        f"--- Context updated: Global Speed: {global_vector[0]:.2f}, Mean Ratio: {final_context['time_ratio'].mean():.2f} ---"
    )


def extract_dynamic_graph_sequence(
    episode_num, snapshots_dir, edge_to_idx, max_snaps=10
):
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
    snaps = sorted(
        [f for f in os.listdir(snapshots_dir) if f.startswith(file_pattern)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    for t, snap_file in enumerate(snaps[:max_snaps]):
        df = pd.read_csv(os.path.join(snapshots_dir, snap_file))

        # A_t: Liczba pojazdów na krawędzi (Wymiar 2 z LaTeX)
        counts = df["edge_id"].value_counts()
        for edge_id, count in counts.items():
            if edge_id in edge_to_idx:
                idx = edge_to_idx[edge_id]
                A_seq[t, idx, 0] = count  # Feature 0: Count

        # D_t: Popyt pojawiający się w systemie
        # Zakładamy, że D_t to pojazdy, które są w pierwszym snapshocie na swojej pierwszej krawędzi
        # (Można to doprecyzować sprawdzając czas odjazdu z agents_df)
        if "is_new_departure" in df.columns:  # Wymaga dodania flagi w save_snapshot_two
            new_departures = df[df["is_new_departure"] == True][
                "edge_id"
            ].value_counts()
            for edge_id, count in new_departures.items():
                if edge_id in edge_to_idx:
                    D_seq[t, edge_to_idx[edge_id]] = count

    return A_seq, D_seq


def plot_traffic_heatmap(episode_num, snapshots_dir, edge_coords, top_n=10):
    """
    Tworzy heatmapę i zapisuje CSV z dodatkowymi kolumnami coord_x, coord_y.
    """
    file_pattern = f"es_ep{episode_num}_"
    snaps = sorted(
        [
            f
            for f in os.listdir(snapshots_dir)
            if f.startswith(file_pattern) and f.endswith(".csv")
        ]
    )
    heatmap_data = {}
    for i, snap_file in enumerate(snaps):
        df = pd.read_csv(os.path.join(snapshots_dir, snap_file))
        counts = df["edge_id"].value_counts().to_dict()

        for edge_id, count in counts.items():
            if edge_id not in heatmap_data:
                heatmap_data[edge_id] = [0] * len(snaps)
            heatmap_data[edge_id][i] = count

    full_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    full_df.columns = [f"Step {i}" for i in range(len(snaps))]

    # Mapujemy ID krawędzi (index) na x i y ze słownika edge_coords

    full_df["coord_x"] = full_df.index.map(
        lambda x: edge_coords.get(str(x), (0.0, 0.0))[0]
    )
    full_df["coord_y"] = full_df.index.map(
        lambda x: edge_coords.get(str(x), (0.0, 0.0))[1]
    )

    cols = ["coord_x", "coord_y"] + [
        c for c in full_df.columns if c not in ["coord_x", "coord_y"]
    ]
    full_df = full_df[cols]

    csv_file = f"{snapshots_dir}/traffic_heatmap_data_ep{episode_num}.csv"
    full_df.to_csv(csv_file)

    step_cols = [c for c in full_df.columns if c.startswith("Step")]
    top_edges = full_df[step_cols].sum(axis=1).nlargest(top_n).index
    df_filtered = full_df.loc[top_edges, step_cols]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        df_filtered,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Vehicles"},
    )

    plt.title(f"Traffic flow heatmap \n(Top {top_n} edges)")

    plt.xlabel("Timesteps")
    plt.ylabel("Edge ID")

    save_path = f"{snapshots_dir}/traffic_heatmap_ep{episode_num}.png"
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
