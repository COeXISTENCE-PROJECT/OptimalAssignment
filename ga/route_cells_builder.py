import json
from pathlib import Path

import h3
import pandas as pd
import sumolib

H3_RES = 10  # resolution of H3 hexagons
DIR = "network/ingolstadt_custom"


def build_edge_to_hex(net_file):
    """Map each network edge to the H3 cell of its first shape point."""
    net = sumolib.net.readNet(str(net_file))
    edge_to_hex = {}
    for edge in net.getEdges():
        x, y = edge.getShape()[0]
        lon, lat = net.convertXY2LonLat(x, y)
        edge_to_hex[edge.getID()] = h3.latlng_to_cell(lat, lon, H3_RES)
    return edge_to_hex


def route_to_cells(path, edge_to_hex, hex_to_idx):
    """Convert a route to cell indices."""
    hex_path = [edge_to_hex[e] for e in path if e in edge_to_hex]
    reduced = [h for i, h in enumerate(hex_path) if i == 0 or h != hex_path[i - 1]]
    return [int(hex_to_idx[h]) for h in reduced[:-1] if h in hex_to_idx]


if __name__ == "__main__":
    hex_df = pd.read_csv(f"{DIR}/hex_mapping.csv")
    hex_to_idx = dict(zip(hex_df["hex_id"], hex_df["matrix_index"]))
    edge_to_hex = build_edge_to_hex(f"{DIR}/ingolstadt_custom.net.xml")

    routes_df = pd.read_csv(f"{DIR}/ingolstadt_custom_clusters_representants.csv")
    routes_df["cells"] = routes_df["path"].str.split(",").apply(
        lambda p: route_to_cells(p, edge_to_hex, hex_to_idx)
    )

    route_cells = {
        f"{r.origins}|{r.destinations}|{r.cluster}": r.cells
        for r in routes_df.itertuples()
    }
    with open(f"{DIR}/route_cells.json", "w") as f:
        json.dump(route_cells, f)
    print("number of saved routes:", len(route_cells))