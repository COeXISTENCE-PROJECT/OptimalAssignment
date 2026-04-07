import os
import numpy as np
import pandas as pd

# 1. Setup Directories
base_dir = "data_processed"
flow_dir = os.path.join(base_dir, "flows")
assign_dir = os.path.join(base_dir, "assignments")

os.makedirs(flow_dir, exist_ok=True)
os.makedirs(assign_dir, exist_ok=True)

# 2. Load the batch data (Assuming you have flow_batch_0_9.npz and hex_adjacency_matrices.npz)
# In a real scenario, you'd loop through all your batch files here
flow_batches = [
    f"flow_batch_{10*i}_{10*i+9}.npz" for i in range(100)
]  # Add your other batch filenames here
adj_batches = [
    f"hex_adjacency_matrices_{10*i}_{10*i+9}.npz" for i in range(100)
]  # Add your other batch filenames here

print("Starting export to individual files...")

# Process Flows
for batch_file in flow_batches:
    data = np.load(batch_file)
    for key in data.files:
        # Save as data_processed/flows/exp_0.npy
        np.save(os.path.join(flow_dir, f"{key}.npy"), data[key])

# Process Assignments (Adjacency)
# Note: Adjacency is often saved per timestep, but OptimalAssignment
# usually prefers one 'Temporal Adjacency' or a list per experiment.
# Here we save the stacked matrices for the whole experiment.
for batch_file in adj_batches:
    data = np.load(batch_file)

    # Adjacency keys are usually 'exp0_t6'. We need to group them by exp.
    exp_groups = {}
    for key in data.files:
        exp_id = key.split("_t")[0]  # 'exp0'
        if exp_id not in exp_groups:
            exp_groups[exp_id] = []
        exp_groups[exp_id].append(data[key])

    for exp_id, matrices in exp_groups.items():
        # Stack into a single 3D array: [Time, Hex, Hex]
        stacked_adj = np.stack(matrices)
        np.save(os.path.join(assign_dir, f"{exp_id}.npy"), stacked_adj)

print(f"Export complete. Check the '{base_dir}' folder.")
