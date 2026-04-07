import pandas as pd
import numpy as np
import os

# 1. Load persistent mapping
mapping_file = "edge_mapping.csv"
if not os.path.exists(mapping_file):
    raise FileNotFoundError("edge_mapping.csv not found. Generate it first!")

mapping_df = pd.read_csv(mapping_file)
ordered_edges = mapping_df.sort_values("matrix_index")["edge_id"].tolist()
num_edges = len(ordered_edges)

# 2. Load the snapshot data
snapshot_path = "all_snapshots.csv"
df = pd.read_csv(snapshot_path)

# Get sorted list of all unique experiment IDs
all_exp_ids = sorted(df["exp_id"].unique())
batch_size = 10

# 3. Iterate over experiments in batches of 10
for i in range(0, len(all_exp_ids), batch_size):
    batch_ids = all_exp_ids[i : i + batch_size]
    batch_dict = {}

    print(f"Processing batch: Experiments {batch_ids[0]} to {batch_ids[-1]}...")

    for exp_id in batch_ids:
        exp_data = df[df["exp_id"] == exp_id]

        # Count vehicles per edge per timestep
        # .size() counts occurrences, .unstack() moves 'time' to columns
        flow_df = exp_data.groupby(["edge_id", "time"]).size().unstack(fill_value=0)

        # Reindex to ensure ALL edges from mapping are rows, in the correct order
        flow_df = flow_df.reindex(index=ordered_edges, fill_value=0)

        # Ensure timesteps (columns) are sorted numerically
        flow_df = flow_df.reindex(sorted(flow_df.columns), axis=1)

        # Store as numpy array to save space
        batch_dict[f"exp_{exp_id}"] = flow_df.values

    # 4. Save this batch to a dedicated file
    batch_filename = f"flow_batch_{batch_ids[0]}_{batch_ids[-1]}.npz"
    np.savez_compressed(batch_filename, **batch_dict)
    print(f"Saved {batch_filename}")

print("\nAll experiments processed and saved in batches.")
# import numpy as np

# batch = np.load("flow_batch_40_49.npz")
# for i in range(batch["exp_44"].shape[1]):
#     print(f"time: {i}, flow: {batch['exp_44'][:, i].sum()}")
