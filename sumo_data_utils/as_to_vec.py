import pandas as pd
import numpy as np
import ast
import os

# --- 1. Load Data ---
df = pd.read_csv("all_departures.csv")
df["path"] = df["path"].apply(ast.literal_eval)

# --- 2. Handle Persistent Edge Mapping ---
mapping_file = "edge_mapping.csv"

if os.path.exists(mapping_file):
    print(f"Loading existing mapping from {mapping_file}...")
    mapping_df = pd.read_csv(mapping_file)
    # Convert DataFrame back to a dictionary {edge_id: index}
    edge_to_idx = dict(zip(mapping_df["edge_id"], mapping_df["matrix_index"]))
else:
    print("No mapping found. Generating new mapping from current data...")
    all_edges = sorted(list(set(edge for path in df["path"] for edge in path)))
    edge_to_idx = {edge: i for i, edge in enumerate(all_edges)}

    # Save it immediately so it persists for next time
    mapping_df = pd.DataFrame(
        list(edge_to_idx.items()), columns=["edge_id", "matrix_index"]
    )
    mapping_df.to_csv(mapping_file, index=False)
    print(f"Mapping saved to {mapping_file}")

num_edges = len(edge_to_idx)
print(f"Total unique edges in mapping: {num_edges}")

# --- 3. Filter for Specific Experiments ---
# (Adjust your range as needed)
for series in range(100):  # Assuming 10 series of experiments
    df2 = df[
        (df["exp_id"] < (series * 10 + 10)) & (df["exp_id"] >= (series * 10))
    ].copy()

    # --- 4. Generate Adjacency Matrices ---
    results = {}
    grouped = df2.groupby(["exp_id", "time"])

    for (exp_id, time), group in grouped:
        # Use the persistent num_edges for consistent matrix shape
        matrix = np.zeros(num_edges, dtype=int)

        for path in group["path"]:
            for i in range(len(path) - 1):
                try:
                    u = edge_to_idx[path[i]]
                    matrix[u] = 1
                except KeyError as e:
                    # This happens if a new edge appears in the data
                    # that wasn't in the original mapping file.
                    print(
                        f"Warning: Edge {e} not found in loaded mapping. Skipping transition."
                    )

        results[(exp_id, time)] = matrix

    # --- 5. Save Results ---
    save_dict = {f"exp{exp}_t{t}": mat for (exp, t), mat in results.items()}
    np.savez_compressed(f"as_vec{series}.npz", **save_dict)
    print(f"Saved {len(save_dict)} matrices to as_vec{series}.npz")
