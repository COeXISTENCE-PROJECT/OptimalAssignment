from __future__ import annotations
from torch.utils.data import DataLoader

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict


class SumoFolderDataset(Dataset):
    def __init__(
        self,
        flow_dir: str | Path,
        assign_dir: str | Path,
        file_names: list[str] | None = None,
        seq_length_q: int = 15,
        seq_length_a: int = 30,
        seq_length_y: int = 1,
        target_nodes: int = 195,
        dtype: torch.dtype = torch.float32,
    ) -> None:

        self.flow_dir = Path(flow_dir)
        self.assign_dir = Path(assign_dir)
        self.dtype = dtype
        self.target_nodes = target_nodes

        self.cache_size = 32
        self._flow_cache = OrderedDict()
        self._assign_cache = OrderedDict()

        self.seq_length_q = seq_length_q
        self.seq_length_a = seq_length_a
        self.seq_length_y = seq_length_y

        if not self.flow_dir.exists():
            raise FileNotFoundError(f"flow_dir does not exist: {self.flow_dir}")
        if not self.assign_dir.exists():
            raise FileNotFoundError(f"assign_dir does not exist: {self.assign_dir}")

        flow_files = {f.name: f for f in self.flow_dir.glob("*.npy")}
        assign_files = {f.name: f for f in self.assign_dir.glob("*.npy")}

        common_files = sorted(set(flow_files.keys()) & set(assign_files.keys()))

        if not common_files:
            raise RuntimeError(
                f"No common .npy files found between {self.flow_dir} and {self.assign_dir}"
            )

        if file_names is None:
            self.exp_files = common_files
        else:
            selected = [Path(f).name for f in file_names]
            missing = sorted(set(selected) - set(common_files))

            if missing:
                raise RuntimeError(
                    f"Some selected files are missing in flow/assign dirs, e.g. {missing[:5]}"
                )

            self.exp_files = selected

        self.samples = []

        for f_name in self.exp_files:
            flow_data = np.load(flow_files[f_name], mmap_mode="r")

            # flow as (N, T)
            if flow_data.ndim != 2:
                raise ValueError(
                    f"flow file {f_name} must have shape (N, T), got {flow_data.shape}"
                )

            n_timesteps = flow_data.shape[1]

            history_len = max(seq_length_q, seq_length_a)
            first_t_end = history_len - 1
            last_t_end = n_timesteps - seq_length_y - 1

            for t_end in range(first_t_end, last_t_end + 1):
                self.samples.append((f_name, t_end))
        self.dtype = dtype

        self.seq_length_q = seq_length_q
        self.seq_length_a = seq_length_a
        self.seq_length_y = seq_length_y

    def _cached_load(self, cache, path):
        key = path.name

        arr = cache.get(key)
        if arr is not None:
            cache.move_to_end(key)
            return arr

        arr = np.load(path, mmap_mode="r")
        cache[key] = arr

        if len(cache) > self.cache_size:
            cache.popitem(last=False)

        return arr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        f_name, t_end = self.samples[idx]

        q_start = t_end - self.seq_length_q + 1
        a_start = t_end - self.seq_length_a + 1
        y_start = t_end + 1
        y_end = y_start + self.seq_length_y

        flow = self._cached_load(self._flow_cache, self.flow_dir / f_name)
        assign = self._cached_load(self._assign_cache, self.assign_dir / f_name)

        current_nodes = flow.shape[0]
        nodes_to_copy = min(current_nodes, self.target_nodes)

        # q -> shape (Tq, N)
        q_padded = np.zeros((self.seq_length_q, self.target_nodes), dtype=np.float32)
        q_slice = flow[:nodes_to_copy, q_start:t_end + 1].T
        q_padded[:, :nodes_to_copy] = q_slice

        # a -> shape (Ta, N)
        a_padded = np.zeros((self.seq_length_a, self.target_nodes), dtype=np.float32)
        a_end = t_end + 1


        if assign.ndim == 2:
            # assignment as (N,T)
            if assign.shape[0] == current_nodes:
                if assign.shape[1] >= a_end:
                    a_slice = assign[:nodes_to_copy, a_start:a_end].T
                    a_padded[:, :nodes_to_copy] = a_slice
                else:
                    raise ValueError(
                        f"assign has too few timesteps: shape={assign.shape}, a_end={a_end}"
                    )

            elif assign.shape[1] == current_nodes:
                if assign.shape[0] >= a_end:
                    a_slice = assign[a_start:a_end, :nodes_to_copy]
                    a_padded[:, :nodes_to_copy] = a_slice
                else:
                    raise ValueError(
                        f"assign has too few timesteps: shape={assign.shape}, a_end={a_end}"
                    )
            else:
                raise ValueError(f"Unsupported assign shape: {assign.shape}")

        elif assign.ndim == 3 and assign.shape[-1] == 1:
            if assign.shape[0] == current_nodes:
                if assign.shape[1] >= a_end:
                    a_slice = assign[:nodes_to_copy, a_start:a_end, 0].T
                    a_padded[:, :nodes_to_copy] = a_slice
                else:
                    raise ValueError(
                        f"assign has too few timesteps: shape={assign.shape}, a_end={a_end}"
                    )

            elif assign.shape[1] == current_nodes:
                if assign.shape[0] >= a_end:
                    a_slice = assign[a_start:a_end, :nodes_to_copy, 0]
                    a_padded[:, :nodes_to_copy] = a_slice
                else:
                    raise ValueError(
                        f"assign has too few timesteps: shape={assign.shape}, a_end={a_end}"
                    )
            else:
                raise ValueError(f"Unsupported assign shape: {assign.shape}")

        else:
            raise ValueError(
                f"assign must have shape (N, T), (T, N), (N, T, 1) or (T, N, 1), got {assign.shape}"
            )

        y_padded = np.zeros((self.seq_length_y, self.target_nodes), dtype=np.float32)
        if y_end <= flow.shape[1]:
            y_slice = flow[:nodes_to_copy, y_start:y_end].T
            y_padded[:, :nodes_to_copy] = y_slice
        else:
            raise ValueError(
                f"flow has too few timesteps for target: shape={flow.shape}, y_end={y_end}"
            )

        if self.seq_length_y == 1:
            y_out = y_padded[0]  # shape (N,)
        else:
            y_out = y_padded  # shape (Hy, N)

        return {
            "x": {
                "q": torch.from_numpy(q_padded).to(self.dtype),
                "a": torch.from_numpy(a_padded).to(self.dtype),
            },
            "y": torch.from_numpy(y_out).to(self.dtype),
        }



def make_qA_loader(
    flow_dir: str | Path,
    assign_dir: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    seq_length_q: int = 15,
    seq_length_a: int = 30,
    seq_length_y: int = 1,
    target_nodes: int = 195,
) -> DataLoader:
    dataset = SumoFolderDataset(
        flow_dir=flow_dir,
        assign_dir=assign_dir,
        seq_length_q=seq_length_q,
        seq_length_a=seq_length_a,
        seq_length_y=seq_length_y,
        target_nodes=target_nodes,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )