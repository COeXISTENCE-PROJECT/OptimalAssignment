from dataset_utils.DataLoader import SumoFolderDataset
from dataset_utils.DataLoader import make_qA_loader
#from __init__ import *
import torch
import util
from model import gwnet
from engine import TrainerADTTP
import numpy as np
import util
import os
from pathlib import Path

from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("mps")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_nodes = 195
seq_len = 12
batch_size = 32

# 2. Data
seq_len_q = 5
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

train_size = int(0.8 * len(dataset))
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)

train_loader = make_qA_loader(train_set, batch_size=batch_size, shuffle=True)
val_loader = make_qA_loader(val_set, batch_size=batch_size)


def create_sumo_adj(assignment_dir, num_nodes):
    print(f"Creating adjacency matrix from {assignment_dir}...")
    adj = np.zeros((num_nodes, num_nodes))
    files = list(Path(assignment_dir).glob("*.npy"))[:20]
    for f in files:
        adj += np.sum(np.load(f), axis=0)

    adj = (adj > 0).astype(np.float32)
    adj += np.eye(num_nodes)
    print("Adjacency matrix created.")
    return adj


raw_adj = create_sumo_adj("data/assignments_30s", num_nodes=195)

adj_normalized = util.asym_adj(raw_adj)
supports = [torch.tensor(adj_normalized).to(device)]


in_dim = 1
seq_length = 5
num_nodes = 195
nhid = 32
dropout = 0.3
lrate = 0.001
wdecay = 0.0001
batch_size = 32
device = torch.device("mps")

model = gwnet(
    device=device,
    num_nodes=num_nodes,
    in_dim=in_dim,
    out_dim=1,  # Predicting 1 step ahead
    supports=supports,
    gcn_bool=True,
    addaptadj=True,
)

engine = TrainerADTTP(
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

for epoch in range(1, 11):
    train_loss = []
    print(f"Epoch {epoch} started...")
    for iter, batch in enumerate(train_loader):
        if iter % 10 == 0:
            print(f"  Batch {iter} processed...")
        # Shape adjustment: [B, T, N, 1] -> [B, 1, N, T]
        x = batch["x"]["q"].permute(0, 3, 2, 1).to(device)
        y = batch["y"].permute(0, 3, 2, 1).to(device)

        loss = engine.train(x, y[:, :, :, 0:1])
        train_loss.append(loss)
        if iter % 10 == 0:
            print(f"  Batch {iter} | Current Loss: {loss}")

    print(f"Epoch {epoch} | Mean Loss: {np.mean(train_loss)}")
