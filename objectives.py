"""
  objectives.py
      ObjectiveWeights
      Score
      adttp_from_a6
      structured_assignment_loss_a6_torch
      relaxed_objective_a6_torch
      score_a6
"""

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
from gradient_search import *

def adttp_from_a6(a6: np.ndarray, delta_t: float = 1.0) -> float:
    """Default ADTTP on Representation 6: sum_t t * sum_i A6[i,t]."""
    if a6.ndim == 1:
        a6 = a6[:, np.newaxis]

    _, t_count = a6.shape
    weights = np.arange(t_count, dtype=np.float32) * float(delta_t)
    return float(np.sum(a6 * weights[None, :]))


@dataclass
class ObjectiveWeights:
    od: float = 0.0
    origin: float = 10_000.0
    admissibility: float = 10_000.0
    destination: float = 10_000.0
    connectivity: float = 10_000.0

@dataclass
class Score:
    objective: float
    adttp: float
    g1: float
    g2: float

def _node_mask(nodes, n_nodes: int, dtype, device) -> torch.Tensor:
    """Return a length-N mask with 1 on selected graph nodes."""
    mask = torch.zeros(n_nodes, dtype=dtype, device=device)

    if nodes is None:
        return mask

    nodes = list(nodes)
    if len(nodes) == 0:
        return mask

    idx = torch.as_tensor(nodes, dtype=torch.long, device=device)
    mask[idx] = 1.0
    return mask


def _prepare_adjacency(
    adjacency,
    n_nodes: int,
    dtype,
    device,
    add_self_loops: bool = True,
) -> torch.Tensor | None:
    """
    Prepare graph adjacency for differentiable connectivity penalties.

    adjacency[i, j] = 1 means mass can move from node i at time t
    to node j at time t+1.
    """
    if adjacency is None:
        return None

    A = torch.as_tensor(adjacency, dtype=dtype, device=device)

    if A.shape != (n_nodes, n_nodes):
        raise ValueError(
            f"adjacency must have shape {(n_nodes, n_nodes)}, got {tuple(A.shape)}"
        )

    A = (A > 0).to(dtype)

    if add_self_loops:
        A = torch.clamp(
            A + torch.eye(n_nodes, dtype=dtype, device=device),
            min=0.0,
            max=1.0,
        )

    return A


def score_a6(
    candidate_a6: np.ndarray,
    weights: ObjectiveWeights,
    delta_t: float = 1.0,
    target_od: np.ndarray | None = None,
    origins: Iterable[int] | None = None,
    destinations: Iterable[int] | None = None,
    total_agents_by_t: np.ndarray | None = None,
    m: int | None = None,
    model: torch.nn.Module | None = None,
    device=None,
    seed_q_tn: np.ndarray | None = None,
    seq_length_q: int = 15,
    seq_length_a: int = 30,
    adjacency: np.ndarray | torch.Tensor | None = None,
) -> Score:
    candidate_a6 = np.asarray(candidate_a6, dtype=np.float32)

    if model is None:
        adttp = adttp_from_a6(candidate_a6, delta_t=delta_t)
    else:
        if seed_q_tn is None or device is None:
            raise ValueError("seed_q_tn and device are required when scoring with GenTTP.")
        adttp = rollout_tt_with_genttp(
            model=model,
            device=device,
            seed_q_tn=seed_q_tn,
            candidate_a6=candidate_a6,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
            delta_t=delta_t,
        )

    if torch is None:
        raise RuntimeError("PyTorch is required for structured scoring.")

    score_device = device if device is not None else "cpu"

    with torch.no_grad():
        a6_t = torch.as_tensor(candidate_a6, dtype=torch.float32, device=score_device)
        penalties = structured_assignment_loss_a6_torch(
            a6_t,
            origins=origins,
            destinations=destinations,
            total_agents_by_t=total_agents_by_t,
            m=m,
            adjacency=adjacency,
            add_self_loops=True,
        )

        origin = float(penalties["origin"].detach().cpu())
        destination = float(penalties["destination"].detach().cpu())
        admissibility = float(penalties["admissibility"].detach().cpu())
        connectivity = float(penalties["connectivity"].detach().cpu())

    objective = float(
        adttp
        + weights.origin * origin
        + weights.destination * destination
        + weights.admissibility * admissibility
        + weights.connectivity * connectivity
    )

    return Score(
        objective=objective,
        adttp=float(adttp),
        g1=origin + destination + connectivity,
        g2=admissibility,
    )


def relaxed_objective_a6_torch(
    relaxed_a6: torch.Tensor,
    weights: ObjectiveWeights,
    delta_t: float = 1.0,
    target_od: np.ndarray | None = None,   # can be kept, but no longer needed
    origins: Iterable[int] | None = None,
    destinations: Iterable[int] | None = None,
    total_agents_by_t: np.ndarray | None = None,
    m: int | None = None,
    model: torch.nn.Module | None = None,
    seed_q_tn: np.ndarray | None = None,
    seq_length_q: int = 15,
    seq_length_a: int = 30,
    use_genttp_gradient: bool = False,
    adjacency: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:

    dtype = relaxed_a6.dtype
    device = relaxed_a6.device

    if use_genttp_gradient and model is not None:
        if seed_q_tn is None:
            raise ValueError("seed_q_tn is required for differentiable GenTTP gradient.")
        travel_time = differentiable_genttp_total_time(
            model=model,
            seed_q_tn=seed_q_tn,
            candidate_a6=relaxed_a6,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
            delta_t=delta_t,
        )
    else:
        t_count = relaxed_a6.shape[1]
        time_weights = torch.arange(t_count, dtype=dtype, device=device) * float(delta_t)
        travel_time = torch.sum(relaxed_a6 * time_weights[None, :])

    penalties = structured_assignment_loss_a6_torch(
        relaxed_a6,
        origins=origins,
        destinations=destinations,
        total_agents_by_t=total_agents_by_t,
        m=m,
        adjacency=adjacency,
        add_self_loops=True,
    )

    return (
        travel_time
        + weights.origin * penalties["origin"]
        + weights.destination * penalties["destination"]
        + weights.admissibility * penalties["admissibility"]
        + weights.connectivity * penalties["connectivity"]
    )

def structured_assignment_loss_a6_torch(
    relaxed_a6: torch.Tensor,
    *,
    origins,
    destinations,
    total_agents_by_t=None,
    m=None,
    adjacency=None,
    add_self_loops=True,
    eps=1e-6,
):
    """
    relaxed_a6: shape (N,T), continuous nonnegative assignment counts.

    Loss factors:
      L_origin:       all initial mass must start in origin nodes
      L_destination:  all final mass must end in destination nodes
      L_admissibility: integer-ish, bounded, correct total mass
      L_connectivity: aggregate mass can only move through graph edges
    """
    dtype = relaxed_a6.dtype
    device = relaxed_a6.device
    n_nodes, t_count = relaxed_a6.shape

    if total_agents_by_t is not None:
        total = torch.as_tensor(total_agents_by_t, dtype=dtype, device=device)
        if total.numel() != t_count:
            raise ValueError(f"total_agents_by_t must have length {t_count}")
    else:
        # If no total is provided, use the current relaxed totals only for scaling.
        total = relaxed_a6.sum(dim=0).detach()

    total_mass = torch.clamp(total.sum(), min=eps)
    target_origin_mass = torch.clamp(total[0], min=0.0)
    target_destination_mass = torch.clamp(total[-1], min=0.0)

    origin_mask = _node_mask(origins, n_nodes, dtype, device)
    destination_mask = _node_mask(destinations, n_nodes, dtype, device)

    # 1. Origin correctness:
    # penalize mass outside origins at t=0 and missing/excess mass inside origins.
    x0 = relaxed_a6[:, 0]
    origin_inside = torch.sum(origin_mask * x0)
    origin_outside = torch.sum((1.0 - origin_mask) * x0)
    L_origin = (
        origin_outside
        + torch.abs(origin_inside - target_origin_mass)
    ) / torch.clamp(target_origin_mass, min=1.0)

    # 2. Destination correctness:
    xT = relaxed_a6[:, -1]
    dest_inside = torch.sum(destination_mask * xT)
    dest_outside = torch.sum((1.0 - destination_mask) * xT)
    L_destination = (
        dest_outside
        + torch.abs(dest_inside - target_destination_mass)
    ) / torch.clamp(target_destination_mass, min=1.0)

    # 3. Admissibility:
    # A6 entries should be integer counts in [0,m], and timestep totals should match.
    L_integer = torch.sin(math.pi * relaxed_a6).pow(2).sum() / torch.clamp(
        torch.tensor(float(n_nodes * t_count), dtype=dtype, device=device),
        min=1.0,
    )

    L_bounds = torch.relu(-relaxed_a6).sum()
    if m is not None:
        L_bounds = L_bounds + torch.relu(relaxed_a6 - float(m)).sum()
    L_bounds = L_bounds / total_mass

    if total_agents_by_t is not None:
        L_total = torch.abs(relaxed_a6.sum(dim=0) - total).sum() / total_mass
    else:
        L_total = torch.zeros((), dtype=dtype, device=device)

    L_admissibility = L_integer + L_bounds + L_total

    # 4. Connectivity:
    # Forward condition: mass at node j at t+1 must be reachable from predecessors at t.
    # Backward condition: mass at node i at t must have at least one reachable successor at t+1.
    A = _prepare_adjacency(
        adjacency,
        n_nodes,
        dtype,
        device,
        add_self_loops=add_self_loops,
    )

    if A is None:
        L_connectivity = torch.zeros((), dtype=dtype, device=device)
    else:
        prev_mass = relaxed_a6[:, :-1]       # (N,T-1)
        next_mass = relaxed_a6[:, 1:]        # (N,T-1)

        reachable_next_capacity = A.T @ prev_mass
        unreachable_next = torch.relu(next_mass - reachable_next_capacity)

        reachable_prev_capacity = A @ next_mass
        stranded_prev = torch.relu(prev_mass - reachable_prev_capacity)

        L_connectivity = (
            unreachable_next.sum() + stranded_prev.sum()
        ) / torch.clamp(total[:-1].sum(), min=1.0)

    return {
        "origin": L_origin,
        "destination": L_destination,
        "admissibility": L_admissibility,
        "connectivity": L_connectivity,
    }
