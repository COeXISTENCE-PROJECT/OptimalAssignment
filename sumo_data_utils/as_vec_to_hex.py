import pandas as pd
import numpy as np
import ast
import h3
import sumolib
import folium
import os


H3_RES = 10

# --- 1. Load Data and Mappings ---
# Load your hex mapping created in the previous step
hex_map_df = pd.read_csv("hex_mapping.csv")
hex_to_idx = dict(zip(hex_map_df["hex_id"], hex_map_df["matrix_index"]))
num_hexes = len(hex_to_idx)


net = sumolib.net.readNet(
    "../../../networks/ingolstadt_custom/ingolstadt_custom.net.xml"
)
flow_data = np.load("flow_batch_0_9.npz")
flow_matrix = flow_data["exp_0"]

ordered_edges = (
    pd.read_csv("edge_mapping.csv").sort_values("matrix_index")["edge_id"].tolist()
)


edge_to_hex = {}
for edge in net.getEdges():
    shape = edge.getShape()
    lon, lat = net.convertXY2LonLat(shape[0][0], shape[0][1])

    # NEWEST H3 (v4): latlng_to_cell
    # MIDDLE H3: latlng_to_h3
    # OLD H3 (v3): geo_to_h3
    try:
        hex_id = h3.latlng_to_cell(lat, lon, H3_RES)
    except AttributeError:
        try:
            hex_id = h3.latlng_to_h3(lat, lon, H3_RES)
        except AttributeError:
            hex_id = h3.geo_to_h3(lat, lon, H3_RES)

    edge_to_hex[edge.getID()] = hex_id


# Load the edge-to-hex mapping (re-run the loop from your eth.py or load a saved json)
# Assuming you have the 'edge_to_hex' dict from the previous script session
# If not, you'll need to regenerate it using the net.getEdges() loop.

# Load departures
df = pd.read_csv("all_departures.csv")
df["path"] = df["path"].apply(ast.literal_eval)

# --- 2. Process Paths into Hex-Transitions ---


def get_hex_transitions(edge_path, edge_to_hex):
    # Convert edges to hex IDs
    hex_path = [edge_to_hex[edge] for edge in edge_path if edge in edge_to_hex]

    # Remove consecutive duplicates (staying in same hex)
    # e.g., [H1, H1, H1, H2, H2, H3] -> [H1, H2, H3]
    reduced_path = []
    if hex_path:
        reduced_path.append(hex_path[0])
        for i in range(1, len(hex_path)):
            if hex_path[i] != hex_path[i - 1]:
                reduced_path.append(hex_path[i])
    return reduced_path


for batch_idx in range(100):
    start_exp = 10 * batch_idx
    end_exp = start_exp + 9

    # Filter batch
    df_batch = df[(df["exp_id"] >= start_exp) & (df["exp_id"] <= end_exp)].copy()

    if df_batch.empty:
        continue

    # Apply hex transformation
    df_batch["hex_path"] = df_batch["path"].apply(
        lambda p: get_hex_transitions(p, edge_to_hex)
    )

    hex_results = {}
    grouped = df_batch.groupby(["exp_id", "time"])

    for (exp_id, time), group in grouped:
        # One matrix per unique (Experiment + Timestep)
        matrix = np.zeros(num_hexes, dtype=np.int8)  # int8 saves space

        for h_path in group["hex_path"]:
            # Use 'j' here to avoid conflict with 'batch_idx' or 'i'
            for j in range(len(h_path) - 1):
                u_hex = h_path[j]

                if u_hex in hex_to_idx:
                    u_idx = hex_to_idx[u_hex]
                    matrix[u_idx] += 1

        # Only save if there's actually a transition (optional, but saves space)
        if np.any(matrix):
            hex_results[f"exp{exp_id}_t{time}"] = matrix

    # Save Batch
    if hex_results:
        fname = f"hex_as_vec_{start_exp}_{end_exp}.npz"
        np.savez_compressed(fname, **hex_results)
        print(f"Saved {len(hex_results)} matrices to {fname}")
