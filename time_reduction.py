import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path("data")
OLD_FLOWS = BASE_DIR / "flows"
OLD_ASSIGNS = BASE_DIR / "assignments"

NEW_FLOWS = BASE_DIR / "flows_30s"
NEW_ASSIGNS = BASE_DIR / "assignments_30s"

NEW_FLOWS.mkdir(parents=True, exist_ok=True)
NEW_ASSIGNS.mkdir(parents=True, exist_ok=True)

WINDOW = 30

print(f"Starting aggregation (Window: {WINDOW}s)...")

flow_files = sorted(list(OLD_FLOWS.glob("*.npy")))
TARGET_NODES = 195

for f_path in tqdm(flow_files):
    f_name = f_path.name
    a_path = OLD_ASSIGNS / f_name

    if not a_path.exists():
        continue

    flow_raw = np.load(f_path)
    assign_raw = np.load(a_path)

    curr_t, curr_n, _ = assign_raw.shape
    if curr_n != TARGET_NODES:
        padded_a = np.zeros((curr_t, TARGET_NODES, TARGET_NODES), dtype=np.float32)
        take_n = min(curr_n, TARGET_NODES)
        padded_a[:, :take_n, :take_n] = assign_raw[:, :take_n, :take_n]
        assign_raw = padded_a

    num_windows = curr_t // WINDOW
    if num_windows == 0:
        continue

    assign_trimmed = assign_raw[: num_windows * WINDOW, :, :]
    assign_30s = assign_trimmed.reshape(
        num_windows, WINDOW, TARGET_NODES, TARGET_NODES
    ).mean(axis=1)

    flow_trimmed = flow_raw[:TARGET_NODES, : num_windows * WINDOW]
    if flow_trimmed.shape[0] < TARGET_NODES:
        f_padded = np.zeros((TARGET_NODES, num_windows * WINDOW))
        f_padded[: flow_trimmed.shape[0], :] = flow_trimmed
        flow_trimmed = f_padded

    flow_30s = flow_trimmed.reshape(TARGET_NODES, num_windows, WINDOW).sum(axis=2)

    np.save(NEW_FLOWS / f_name, flow_30s.astype(np.float32))
    np.save(NEW_ASSIGNS / f_name, assign_30s.astype(np.float32))
