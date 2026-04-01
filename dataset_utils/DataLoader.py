from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
        q_hist = torch.from_numpy(self.x_q[idx]).to(self.dtype)   # (Lq, N, C)
        a_hist = torch.from_numpy(self.x_a[idx]).to(self.dtype)   # (La, N, N) lub (La, N, N, M)
        y = torch.from_numpy(self.y[idx]).to(self.dtype)          # (Ly, N, C)

        sample = {
            "x": {
                "q": q_hist,
                "a": a_hist,
            },
            "y": y,
        }

        if self.return_time_meta:
            sample["time_meta"] = {
                "q_window_start_t": torch.tensor(self.q_window_start_t[idx], dtype=torch.long),
                "q_window_end_t": torch.tensor(self.q_window_end_t[idx], dtype=torch.long),
                "a_window_start_t": torch.tensor(self.a_window_start_t[idx], dtype=torch.long),
                "a_window_end_t": torch.tensor(self.a_window_end_t[idx], dtype=torch.long),
            }

        return sample


def make_qA_loader(
    npz_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    return_time_meta: bool = False,
) -> DataLoader:
    dataset = QAPairedDataset(
        npz_path=npz_path,
        return_time_meta=return_time_meta,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    return loader


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