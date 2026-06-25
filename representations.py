from __future__ import annotations
import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import torch

"""
==========================
Assignment representations
==========================                                  
A1: {-1, 0, 1}^{N x N x m x T}
A2: {0, 1}^{N x N x m x T}
A3: {-1, 0, 1}^{N x m x T}
A5: {0, 1}^{N x m x T}
A6: [m]_0^{N x T}
==========================
"""

def f_1_to_2(a1: np.ndarray) -> np.ndarray:
    """Remove signs element-wise: B_ijht = A_ijht^2."""
    return np.square(a1).astype(np.int8)


def f_1_to_3(a1: np.ndarray) -> np.ndarray:
    """Aggregate signed representation over the second node index j."""
    return np.sum(a1, axis=1).astype(np.int16)


def f_2_to_5(a2: np.ndarray) -> np.ndarray:
    """Reduce binary edge-agent-time tensor to binary node-agent-time tensor."""
    return np.max(a2, axis=1).astype(np.int8)


def f_5_to_6(a5: np.ndarray) -> np.ndarray:
    """Count how many agents use each node at each timestep."""
    if a5.ndim == 1:
        a5 = a5[:, np.newaxis]
    return np.sum(a5, axis=1).astype(np.int16)


def f_2_to_6(a2: np.ndarray) -> np.ndarray:
    """A2 -> A5 -> A6."""
    return f_5_to_6(f_2_to_5(a2))


def load_assignment_as_a6(path: str | Path, num_nodes: int | None = None) -> np.ndarray:
    """Load assignment counts as A6 with shape (N, T).

    Repository data commonly stores assignments as (N, T), (T, N),
    (N, T, 1), or (T, N, 1). This helper normalizes them to (N, T).
    """
    arr = np.load(path)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            f"Assignment must be 2D or 3D with singleton channel, got {arr.shape}"
        )

    if num_nodes is not None:
        if arr.shape[0] == num_nodes:
            out = arr
        elif arr.shape[1] == num_nodes:
            out = arr.T
        else:
            raise ValueError(
                f"Cannot infer node axis for shape {arr.shape} and num_nodes={num_nodes}"
            )
    else:
        out = arr

    return np.asarray(out, dtype=np.float32)
