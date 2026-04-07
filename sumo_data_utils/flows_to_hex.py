import h3
import pandas as pd
import sumolib
import numpy as np
import folium
import os

# --- SETTINGS ---
H3_RES = 10

# --- 1. FUNCTIONS ---


def aggregate_flow_to_hex(flow_matrix, edge_to_hex, ordered_edges):
    unique_hexes = sorted(list(set(edge_to_hex.values())))
    hex_to_idx = {h: i for i, h in enumerate(unique_hexes)}
    hex_flow = np.zeros((len(unique_hexes), flow_matrix.shape[1]))

    for i, edge_id in enumerate(ordered_edges):
        if edge_id in edge_to_hex:
            hex_id = edge_to_hex[edge_id]
            hex_idx = hex_to_idx[hex_id]
            hex_flow[hex_idx, :] += flow_matrix[i, :]

    return hex_flow, hex_to_idx


def visualize_hex_flow(hex_flow_matrix, hex_to_idx):
    total_flow = np.sum(hex_flow_matrix, axis=1)
    m = folium.Map(location=[48.7667, 11.4250], zoom_start=13, tiles="cartodbpositron")

    for hex_id, idx in hex_to_idx.items():
        flow_value = total_flow[idx]
        if flow_value == 0:
            continue

        # NEWEST H3 (v4): cell_to_boundary
        # OLD H3 (v3): h3_to_geo_boundary
        try:
            polygons = h3.cell_to_boundary(hex_id)
        except AttributeError:
            polygons = h3.h3_to_geo_boundary(hex_id)

        folium.Polygon(
            locations=polygons,
            fill=True,
            fill_color="YlOrRd",
            fill_opacity=0.6,
            color="gray",
            weight=1,
            tooltip=f"Hex: {hex_id}<br>Total Flow: {flow_value}",
        ).add_to(m)

    m.save("hex_flow_map.html")
    print("Success! Map saved to hex_flow_map.html.")


# --- 2. EXECUTION ---

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

hex_flow_matrix, hex_to_idx = aggregate_flow_to_hex(
    flow_matrix, edge_to_hex, ordered_edges
)

# Save
np.save("hex_flow_exp_0.npy", hex_flow_matrix)
pd.DataFrame(list(hex_to_idx.items()), columns=["hex_id", "matrix_index"]).to_csv(
    "hex_mapping.csv", index=False
)

# Visualize
visualize_hex_flow(hex_flow_matrix, hex_to_idx)
