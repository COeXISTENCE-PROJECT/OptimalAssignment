import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Settings
BASE_DIR = Path("data")
OLD_FLOWS = BASE_DIR / "flows"
OLD_ASSIGNS = BASE_DIR / "assignments"

NEW_FLOWS = BASE_DIR / "flows_30s"
NEW_ASSIGNS = BASE_DIR / "assignments_30s"

# Create new directories
NEW_FLOWS.mkdir(parents=True, exist_ok=True)
NEW_ASSIGNS.mkdir(parents=True, exist_ok=True)

# Aggregation factor
WINDOW = 30

print(f"Starting aggregation (Window: {WINDOW}s)...")

# We use the flows folder as the master list
flow_files = sorted(list(OLD_FLOWS.glob("*.npy")))
TARGET_NODES = 195

for f_path in tqdm(flow_files):
    f_name = f_path.name
    a_path = OLD_ASSIGNS / f_name

    if not a_path.exists():
        continue

    flow_raw = np.load(f_path)
    assign_raw = np.load(a_path)

    # 1. Standardize the Node Dimension (Padding)
    # We need assign to be [Time, TARGET_NODES, TARGET_NODES]
    curr_t, curr_n, _ = assign_raw.shape
    if curr_n != TARGET_NODES:
        # Create a blank canvas of the right size
        padded_a = np.zeros((curr_t, TARGET_NODES, TARGET_NODES), dtype=np.float32)
        # Copy the small matrix into the top-left corner
        take_n = min(curr_n, TARGET_NODES)
        padded_a[:, :take_n, :take_n] = assign_raw[:, :take_n, :take_n]
        assign_raw = padded_a

    # 2. Trim to a clean multiple of WINDOW (30)
    num_windows = curr_t // WINDOW
    if num_windows == 0:
        continue

    # Trim Time
    assign_trimmed = assign_raw[: num_windows * WINDOW, :, :]

    # 3. Aggregate
    # Reshape: (Windows, 30s_Steps, Nodes, Nodes) -> Mean over 30s_Steps
    assign_30s = assign_trimmed.reshape(
        num_windows, WINDOW, TARGET_NODES, TARGET_NODES
    ).mean(axis=1)

    # Do the same for Flow
    flow_trimmed = flow_raw[:TARGET_NODES, : num_windows * WINDOW]
    # If flow was smaller than 195, pad it too
    if flow_trimmed.shape[0] < TARGET_NODES:
        f_padded = np.zeros((TARGET_NODES, num_windows * WINDOW))
        f_padded[: flow_trimmed.shape[0], :] = flow_trimmed
        flow_trimmed = f_padded

    flow_30s = flow_trimmed.reshape(TARGET_NODES, num_windows, WINDOW).sum(axis=2)

    # Save
    np.save(NEW_FLOWS / f_name, flow_30s.astype(np.float32))
    np.save(NEW_ASSIGNS / f_name, assign_30s.astype(np.float32))
