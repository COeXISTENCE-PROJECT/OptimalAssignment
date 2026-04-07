from dataset_utils.DataLoader import SumoFolderDataset
from __init__ import *
import torch
import util
from model import gwnet
from engine import Trainer
import numpy as np
import util
import os
from pathlib import Path

# import SumoFolderDataset
from torch.utils.data import DataLoader
from pathlib import Path

# 1. Config
device = torch.device("mps")  # Use "cuda" if on Linux/Windows with GPU
num_nodes = 195  # Match your hex count
seq_len = 12  # History length
batch_size = 32

# 2. Data
seq_len_q = 5  # Lower these
seq_len_a = 5
seq_len_y = 1


class IdentityScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


scaler = IdentityScaler()

print("--- Loading Data ---")
dataset = SumoFolderDataset("data", seq_length_q=5, seq_length_a=5, seq_length_y=1)
print(f"Total samples found: {len(dataset)}")

# Split 80/20
train_size = int(0.8 * len(dataset))
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)


# --- 1. Generate Static Adjacency ---
def create_sumo_adj(assignment_dir, num_nodes):
    print(f"Creating adjacency matrix from {assignment_dir}...")
    adj = np.zeros((num_nodes, num_nodes))
    # Look at a few files to see which hexagons are connected
    files = list(Path(assignment_dir).glob("*.npy"))[:20]
    for f in files:
        # Summing transitions over time and experiments
        adj += np.sum(np.load(f), axis=0)

    # Simple binary adjacency: 1 if any vehicle ever crossed, else 0
    adj = (adj > 0).astype(np.float32)
    adj += np.eye(num_nodes)  # Ensures no row-sum is zero
    print("Adjacency matrix created.")
    return adj


# Create the raw matrix
raw_adj = create_sumo_adj("data/assignments_30s", num_nodes=195)

# GraphWaveNet expects normalized "supports"
# 'doubletransition' is the default in the paper/repo
adj_normalized = util.asym_adj(raw_adj)
supports = [torch.tensor(adj_normalized).to(device)]

# In run_sumo.py

# 1. Define your parameters clearly
in_dim = 1
seq_length = 5  # Must match what you used in SumoFolderDataset
num_nodes = 195
nhid = 32
dropout = 0.3
lrate = 0.001
wdecay = 0.0001
batch_size = 32
device = torch.device("mps")

# 2. Initialize the Model
model = gwnet(
    device=device,
    num_nodes=num_nodes,
    in_dim=in_dim,
    out_dim=1,  # Predicting 1 step ahead
    supports=supports,
    gcn_bool=True,
    addaptadj=True,
)

# 3. Initialize the Trainer (Matches the 13 missing arguments)
# Note: Check your engine.py for the exact order, but usually it follows this:
engine = Trainer(
    scaler,
    in_dim,
    seq_length,
    num_nodes,
    nhid,
    dropout,
    lrate,
    wdecay,
    device,
    supports,
    True,  # gcn_bool
    True,  # addaptadj
    None,  # aptinit (can be None)
    2,  # kernel_size (default is usually 2)
    4,  # blocks (default is usually 4)
    2,  # layers (default is usually 2)
)

print("--- Initializing Model and Trainer ---")
# (model and engine setup)
print("--- Starting Training ---")

# 4. Training Loop
for epoch in range(1, 11):
    train_loss = []
    print(f"Epoch {epoch} started...")
    for iter, batch in enumerate(train_loader):
        if iter % 10 == 0:
            print(f"  Batch {iter} processed...")
        # Shape adjustment: [B, T, N, 1] -> [B, 1, N, T]
        x = batch["x"]["q"].permute(0, 3, 2, 1).to(device)
        y = batch["y"].permute(0, 3, 2, 1).to(device)

        # Train one step
        loss = engine.train(x, y[:, :, :, 0:1])  # Predicting next step
        train_loss.append(loss)
        if iter % 10 == 0:
            print(f"  Batch {iter} | Current Loss: {loss}")

    print(f"Epoch {epoch} | Mean Loss: {np.mean(train_loss)}")
