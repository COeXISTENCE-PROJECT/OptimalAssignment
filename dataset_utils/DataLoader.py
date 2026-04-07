from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import random


class SumoFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        seq_length_q: int = 5,
        seq_length_a: int = 5,
        seq_length_y: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.flow_dir = self.root_dir / "flows_30s"
        self.assign_dir = self.root_dir / "assignments_30s"
        self.dtype = dtype

        self.seq_length_q = seq_length_q
        self.seq_length_a = seq_length_a
        self.seq_length_y = seq_length_y

        self.exp_files = sorted([f.name for f in self.flow_dir.glob("*.npy")])
        self.samples = []
        for f_name in self.exp_files:
            flow_data = np.load(self.flow_dir / f_name, mmap_mode="r")
            n_timesteps = flow_data.shape[1]
            max_start_t = n_timesteps - max(seq_length_q, seq_length_a) - seq_length_y

            for t in range(max_start_t):
                self.samples.append((f_name, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        f_name, t_start = self.samples[idx]

        TARGET_NODES = 195
        flow = np.load(self.flow_dir / f_name, mmap_mode="r")
        assign = np.load(self.assign_dir / f_name, mmap_mode="r")

        current_nodes = flow.shape[0]
        nodes_to_copy = min(current_nodes, TARGET_NODES)

        q_padded = np.zeros((self.seq_length_q, TARGET_NODES, 1))
        q_slice = flow[:nodes_to_copy, t_start : t_start + self.seq_length_q].T
        q_padded[:, :nodes_to_copy, 0] = q_slice

        a_padded = np.zeros((self.seq_length_a, TARGET_NODES, TARGET_NODES))
        a_end = t_start + self.seq_length_a

        if assign.shape[0] >= a_end:
            a_slice = assign[t_start:a_end, :nodes_to_copy, :nodes_to_copy]
            a_padded[:, :nodes_to_copy, :nodes_to_copy] = a_slice

        y_padded = np.zeros((self.seq_length_y, TARGET_NODES, 1))
        y_t = t_start + self.seq_length_q
        if y_t < flow.shape[1]:
            y_slice = flow[:nodes_to_copy, y_t : y_t + self.seq_length_y].T
            y_padded[:, :nodes_to_copy, 0] = y_slice

        return {
            "x": {
                "q": torch.from_numpy(q_padded.copy()).to(self.dtype),
                "a": torch.from_numpy(a_padded.copy()).to(self.dtype),
            },
            "y": torch.from_numpy(y_padded.copy()).to(self.dtype),
        }


class QAPairedDataset(Dataset):
    """
    Pojedynczy sample:
        x = {
            "q": (Lq, N, Cq),
            "a": (La, N, N)   lub (La, N, N, M)
        }
        y = (1, N, Cy)  # dla targetu q_{t+1}, jeśli seq_length_y=1
    """

    def __init__(
        self,
        npz_path: str | Path,
        dtype: torch.dtype = torch.float32,
        return_time_meta: bool = False,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.dtype = dtype
        self.return_time_meta = return_time_meta

        data = np.load(self.npz_path, allow_pickle=False)

        self.x_q = data["x_q"].astype(np.float32, copy=False)
        self.x_a = data["x_a"].astype(np.float32, copy=False)
        self.y = data["y"].astype(np.float32, copy=False)

        self.q_window_start_t = data["q_window_start_t"].astype(np.int64, copy=False)
        self.q_window_end_t = data["q_window_end_t"].astype(np.int64, copy=False)
        self.a_window_start_t = data["a_window_start_t"].astype(np.int64, copy=False)
        self.a_window_end_t = data["a_window_end_t"].astype(np.int64, copy=False)

        self.meta = {
            "seq_length_q": int(data["seq_length_q"][0]),
            "seq_length_a": int(data["seq_length_a"][0]),
            "seq_length_y": int(data["seq_length_y"][0]),
            "y_start": int(data["y_start"][0]),
            "assignment_mode_code": int(data["assignment_mode_code"][0]),
            "reduce_agents_code": int(data["reduce_agents_code"][0]),
        }

        data.close()

        n = self.x_q.shape[0]
        if self.x_a.shape[0] != n or self.y.shape[0] != n:
            raise ValueError(
                f"Inconsistent number of samples in {self.npz_path}: "
                f"x_q={self.x_q.shape[0]}, x_a={self.x_a.shape[0]}, y={self.y.shape[0]}"
            )

    def __len__(self) -> int:
        return self.x_q.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        q_hist = torch.from_numpy(self.x_q[idx]).to(self.dtype)  # (Lq, N, C)
        a_hist = torch.from_numpy(self.x_a[idx]).to(
            self.dtype
        )  # (La, N, N) lub (La, N, N, M)
        y = torch.from_numpy(self.y[idx]).to(self.dtype)  # (Ly, N, C)

        sample = {
            "x": {
                "q": q_hist,
                "a": a_hist,
            },
            "y": y,
        }

        if self.return_time_meta:
            sample["time_meta"] = {
                "q_window_start_t": torch.tensor(
                    self.q_window_start_t[idx], dtype=torch.long
                ),
                "q_window_end_t": torch.tensor(
                    self.q_window_end_t[idx], dtype=torch.long
                ),
                "a_window_start_t": torch.tensor(
                    self.a_window_start_t[idx], dtype=torch.long
                ),
                "a_window_end_t": torch.tensor(
                    self.a_window_end_t[idx], dtype=torch.long
                ),
            }

        return sample


def make_qA_loader(
    root_dir: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    dataset = SumoFolderDataset(root_dir=root_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Speeds up transfers to GPU/MPS
    )


# def make_qA_loader(
#     npz_path: str | Path,
#     batch_size: int,
#     shuffle: bool,
#     num_workers: int = 0,
#     pin_memory: bool = False,
#     drop_last: bool = False,
#     return_time_meta: bool = False,
# ) -> DataLoader:
#     dataset = QAPairedDataset(
#         npz_path=npz_path,
#         return_time_meta=return_time_meta,
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         drop_last=drop_last,
#         persistent_workers=(num_workers > 0),
#     )
#     return loader


def make_qA_loaders(
    root_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    return_time_meta: bool = False,
) -> dict[str, DataLoader]:
    root_dir = Path(root_dir)

    return {
        "train": make_qA_loader(
            root_dir / "train.npz",
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            return_time_meta=return_time_meta,
        ),
        "val": make_qA_loader(
            root_dir / "val.npz",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            return_time_meta=return_time_meta,
        ),
        "test": make_qA_loader(
            root_dir / "test.npz",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            return_time_meta=return_time_meta,
        ),
    }
