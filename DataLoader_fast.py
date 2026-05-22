from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate


@dataclass(frozen=True)
class _FileInfo:
    name: str
    flow_path: Path
    assign_path: Path
    n_nodes: int
    n_timesteps: int
    first_t_end: int
    n_windows: int
    assign_layout: str  # NT, TN, NT1, TN1


def _fast_collate(batch):
    """Return pre-batched output from Dataset.__getitems__ unchanged.

    PyTorch's DataLoader may call Dataset.__getitems__(list[int]) with the full
    batch index list. In that case the dataset already returns tensors shaped
    [B, ...], so the normal collate step must be skipped. If PyTorch falls back
    to repeated __getitem__ calls, default_collate still works.
    """
    if isinstance(batch, dict):
        return batch
    return default_collate(batch)


class FileWindowBatchSampler(Sampler[list[int]]):
    """Batches windows grouped by source file.

    This keeps your leakage protection: train/val/test are still split by file
    before windows are generated. It also avoids the slowest access pattern:
    random windows jumping between many .npy memory maps.
    """

    def __init__(
        self,
        dataset: "SumoFolderDatasetFast",
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        chunk_size: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size) if chunk_size is not None else self.batch_size * 16
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        file_order = np.arange(len(self.dataset.files))
        if self.shuffle:
            rng.shuffle(file_order)

        for file_idx in file_order:
            file_idx = int(file_idx)
            global_start = self.dataset.file_global_start(file_idx)
            n_windows = self.dataset.files[file_idx].n_windows

            chunk_starts = np.arange(0, n_windows, self.chunk_size)
            if self.shuffle:
                rng.shuffle(chunk_starts)

            for chunk_start in chunk_starts:
                chunk_start = int(chunk_start)
                chunk_end = min(chunk_start + self.chunk_size, n_windows)

                # Keep time order inside chunk for locality; chunk/file order is shuffled.
                indices = np.arange(
                    global_start + chunk_start,
                    global_start + chunk_end,
                    dtype=np.int64,
                )

                for start in range(0, len(indices), self.batch_size):
                    batch = indices[start:start + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        yield batch.tolist()

    def __len__(self) -> int:
        total = 0
        for info in self.dataset.files:
            if self.drop_last:
                total += info.n_windows // self.batch_size
            else:
                total += (info.n_windows + self.batch_size - 1) // self.batch_size
        return total


class SumoFolderDatasetFast(Dataset):
    def __init__(
        self,
        flow_dir: str | Path,
        assign_dir: str | Path,
        file_names: Sequence[str] | None = None,
        seq_length_q: int = 15,
        seq_length_a: int = 30,
        seq_length_y: int = 1,
        target_nodes: int = 195,
        dtype: torch.dtype = torch.float32,
        cache_size: int = 64,
        load_to_ram: bool = False,
    ) -> None:
        self.flow_dir = Path(flow_dir)
        self.assign_dir = Path(assign_dir)
        self.dtype = dtype
        self.target_nodes = int(target_nodes)
        self.seq_length_q = int(seq_length_q)
        self.seq_length_a = int(seq_length_a)
        self.seq_length_y = int(seq_length_y)
        self.cache_size = int(cache_size)
        self.load_to_ram = bool(load_to_ram)

        self._flow_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._assign_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        if self.seq_length_q <= 0 or self.seq_length_a <= 0 or self.seq_length_y <= 0:
            raise ValueError("All sequence lengths must be positive")
        if self.target_nodes <= 0:
            raise ValueError("target_nodes must be positive")
        if not self.flow_dir.exists():
            raise FileNotFoundError(f"flow_dir does not exist: {self.flow_dir}")
        if not self.assign_dir.exists():
            raise FileNotFoundError(f"assign_dir does not exist: {self.assign_dir}")

        flow_files = {f.name: f for f in self.flow_dir.glob("*.npy")}
        assign_files = {f.name: f for f in self.assign_dir.glob("*.npy")}
        common_files = sorted(set(flow_files) & set(assign_files))
        if not common_files:
            raise RuntimeError(f"No common .npy files found between {self.flow_dir} and {self.assign_dir}")

        if file_names is None:
            selected = common_files
        else:
            selected = [Path(f).name for f in file_names]
            missing = sorted(set(selected) - set(common_files))
            if missing:
                raise RuntimeError(f"Some selected files are missing in flow/assign dirs, e.g. {missing[:5]}")

        self.files: list[_FileInfo] = []
        counts: list[int] = []
        history_len = max(self.seq_length_q, self.seq_length_a)
        first_t_end = history_len - 1

        for name in selected:
            flow_shape, flow_ndim = self._array_shape(flow_files[name])
            if flow_ndim != 2:
                raise ValueError(f"flow file {name} must have shape (N, T), got {flow_shape}")

            n_nodes, n_timesteps = int(flow_shape[0]), int(flow_shape[1])
            assign_shape, assign_ndim = self._array_shape(assign_files[name])
            assign_layout, assign_timesteps = self._infer_assign_layout(assign_shape, assign_ndim, n_nodes, name)

            last_t_end = n_timesteps - self.seq_length_y - 1
            if last_t_end < first_t_end:
                continue

            if assign_timesteps < last_t_end + 1:
                raise ValueError(
                    f"assign file {name} has too few timesteps: "
                    f"assign_T={assign_timesteps}, required={last_t_end + 1}"
                )

            n_windows = last_t_end - first_t_end + 1
            self.files.append(
                _FileInfo(
                    name=name,
                    flow_path=flow_files[name],
                    assign_path=assign_files[name],
                    n_nodes=n_nodes,
                    n_timesteps=n_timesteps,
                    first_t_end=first_t_end,
                    n_windows=n_windows,
                    assign_layout=assign_layout,
                )
            )
            counts.append(n_windows)

        if not self.files:
            raise RuntimeError("No files contain enough timesteps to build at least one window")

        self.cum_windows = np.cumsum(np.asarray(counts, dtype=np.int64))
        self._first_t_ends = np.asarray([f.first_t_end for f in self.files], dtype=np.int64)

    @staticmethod
    def _array_shape(path: Path) -> tuple[tuple[int, ...], int]:
        arr = np.load(path, mmap_mode="r")
        return tuple(arr.shape), arr.ndim

    @staticmethod
    def _infer_assign_layout(
        shape: tuple[int, ...],
        ndim: int,
        n_flow_nodes: int,
        file_name: str,
    ) -> tuple[str, int]:
        if ndim == 2:
            if shape[0] == n_flow_nodes:
                return "NT", int(shape[1])
            if shape[1] == n_flow_nodes:
                return "TN", int(shape[0])
        elif ndim == 3 and shape[-1] == 1:
            if shape[0] == n_flow_nodes:
                return "NT1", int(shape[1])
            if shape[1] == n_flow_nodes:
                return "TN1", int(shape[0])

        raise ValueError(
            f"Unsupported assign shape for {file_name}: {shape}. Expected "
            "(N,T), (T,N), (N,T,1) or (T,N,1)."
        )

    def _cached_load(self, cache: OrderedDict[str, np.ndarray], path: Path) -> np.ndarray:
        key = str(path)
        arr = cache.get(key)
        if arr is not None:
            cache.move_to_end(key)
            return arr

        arr = np.load(path, mmap_mode=None if self.load_to_ram else "r")
        cache[key] = arr
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return arr

    def __len__(self) -> int:
        return int(self.cum_windows[-1])

    def file_global_start(self, file_idx: int) -> int:
        return int(self.cum_windows[file_idx - 1]) if file_idx > 0 else 0

    def _locate_indices(self, indices: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        idx = np.where(idx < 0, idx + len(self), idx)
        if np.any(idx < 0) or np.any(idx >= len(self)):
            raise IndexError("Dataset index out of range")

        file_idx = np.searchsorted(self.cum_windows, idx, side="right").astype(np.int64)
        prev = np.zeros_like(idx)
        mask = file_idx > 0
        prev[mask] = self.cum_windows[file_idx[mask] - 1]
        offsets = idx - prev
        t_ends = self._first_t_ends[file_idx] + offsets
        return file_idx, t_ends

    @staticmethod
    def _windows_from_NT(arr_nt: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
        # arr_nt: [N, T] -> [B, length, N]
        windows = np.lib.stride_tricks.sliding_window_view(arr_nt, window_shape=length, axis=1)
        return windows[:, starts, :].transpose(1, 2, 0)

    @staticmethod
    def _windows_from_TN(arr_tn: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
        # arr_tn: [T, N] -> [B, length, N]
        windows = np.lib.stride_tricks.sliding_window_view(arr_tn, window_shape=length, axis=0)
        return windows[starts, :, :].transpose(0, 2, 1)

    def _assign_windows(self, assign: np.ndarray, info: _FileInfo, starts: np.ndarray, nodes: int) -> np.ndarray:
        if info.assign_layout == "NT":
            return self._windows_from_NT(assign[:nodes, :], starts, self.seq_length_a)
        if info.assign_layout == "TN":
            return self._windows_from_TN(assign[:, :nodes], starts, self.seq_length_a)
        if info.assign_layout == "NT1":
            return self._windows_from_NT(assign[:nodes, :, 0], starts, self.seq_length_a)
        if info.assign_layout == "TN1":
            return self._windows_from_TN(assign[:, :nodes, 0], starts, self.seq_length_a)
        raise RuntimeError(f"Unexpected assign layout: {info.assign_layout}")

    def __getitem__(self, idx: int):
        if isinstance(idx, (list, tuple, np.ndarray)):
            return self._get_batch(idx)
        batch = self._get_batch([int(idx)])
        return {"x": {"q": batch["x"]["q"][0], "a": batch["x"]["a"][0]}, "y": batch["y"][0]}

    def __getitems__(self, indices: Sequence[int]):
        # DataLoader can call this with full batch indices; much faster than per-sample __getitem__.
        return self._get_batch(indices)

    def _get_batch(self, indices: Sequence[int] | np.ndarray) -> dict:
        idx = np.asarray(indices, dtype=np.int64)
        batch_size = len(idx)
        file_idx, t_ends = self._locate_indices(idx)

        q_batch = np.zeros((batch_size, self.seq_length_q, self.target_nodes), dtype=np.float32)
        a_batch = np.zeros((batch_size, self.seq_length_a, self.target_nodes), dtype=np.float32)
        if self.seq_length_y == 1:
            y_batch = np.zeros((batch_size, self.target_nodes), dtype=np.float32)
        else:
            y_batch = np.zeros((batch_size, self.seq_length_y, self.target_nodes), dtype=np.float32)

        for fidx in np.unique(file_idx):
            fidx = int(fidx)
            pos = np.flatnonzero(file_idx == fidx)
            info = self.files[fidx]
            flow = self._cached_load(self._flow_cache, info.flow_path)
            assign = self._cached_load(self._assign_cache, info.assign_path)

            nodes = self.target_nodes
            t = t_ends[pos]

            q_starts = t - self.seq_length_q + 1
            q_values = self._windows_from_NT(flow, q_starts, self.seq_length_q)
            q_batch[pos] = np.asarray(q_values, dtype=np.float32)

            a_starts = t - self.seq_length_a + 1
            a_values = self._assign_windows(assign, info, a_starts)
            a_batch[pos] = np.asarray(a_values, dtype=np.float32)

            y_starts = t + 1
            if self.seq_length_y == 1:
                y_values = flow[:, y_starts].T
                y_batch[pos] = np.asarray(y_values, dtype=np.float32)
            else:
                y_values = self._windows_from_NT(flow[:nodes, :], y_starts, self.seq_length_y)
                y_batch[pos, :, :nodes] = np.asarray(y_values, dtype=np.float32)

        q = torch.from_numpy(q_batch)
        a = torch.from_numpy(a_batch)
        y = torch.from_numpy(y_batch)
        if self.dtype != torch.float32:
            q = q.to(self.dtype)
            a = a.to(self.dtype)
            y = y.to(self.dtype)

        return {"x": {"q": q, "a": a}, "y": y}


# Backward-compatible alias if other files import SumoFolderDataset.
SumoFolderDataset = SumoFolderDatasetFast


def make_subset_loader(
    q_dir: str | Path,
    a_dir: str | Path,
    selected_files: Sequence[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    seq_length_q: int = 15,
    seq_length_a: int = 30,
    seq_length_y: int = 1,
    target_nodes: int = 195,
    dtype: torch.dtype = torch.float32,
    cache_size: int = 64,
    load_to_ram: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 4,
    group_by_file: bool = True,
    drop_last: bool = False,
    seed: int = 0,
    chunk_size: int | None = None,
) -> DataLoader:
    dataset = SumoFolderDatasetFast(
        flow_dir=q_dir,
        assign_dir=a_dir,
        file_names=selected_files,
        seq_length_q=seq_length_q,
        seq_length_a=seq_length_a,
        seq_length_y=seq_length_y,
        target_nodes=target_nodes,
        dtype=dtype,
        cache_size=cache_size,
        load_to_ram=load_to_ram,
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    kwargs = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(persistent_workers) if num_workers > 0 else False,
        "collate_fn": _fast_collate,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    if group_by_file:
        kwargs["batch_sampler"] = FileWindowBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            chunk_size=chunk_size,
        )
    else:
        kwargs.update({"batch_size": batch_size, "shuffle": shuffle, "drop_last": drop_last})

    return DataLoader(**kwargs)


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
    **kwargs,
) -> DataLoader:
    flow_files = sorted(Path(flow_dir).glob("*.npy"))
    return make_subset_loader(
        q_dir=flow_dir,
        a_dir=assign_dir,
        selected_files=[p.name for p in flow_files],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seq_length_q=seq_length_q,
        seq_length_a=seq_length_a,
        seq_length_y=seq_length_y,
        target_nodes=target_nodes,
        **kwargs,
    )
