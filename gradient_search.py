#!/usr/bin/env python3
"""Gradient search over Step 2 assignment Representation 6.

This version repairs the gradient-search path and logging in the original
`gradient_search.py` from the `step_2_branch` branch.

Key fixes:
- `score_a6` now scores A6 candidates consistently.
- Broken names such as `Objectiveweights`, `a1_original`, `constraint_matrix`,
  `objective`, and `best_score` are removed or initialized correctly.
- The optimization loop uses PyTorch autograd on a differentiable relaxed A1
  surrogate, then evaluates the hard projected A6 candidate.
- `target_od` is preserved instead of being overwritten with None.
- `optimization_history.csv` is written for all runs, not only checkpoint runs.
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

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ============================================================
# Assignment representations
# ============================================================
# A1: {-1, 0, 1}^{N x N x m x T}
# A2: {0, 1}^{N x N x m x T}
# A3: {-1, 0, 1}^{N x m x T}
# A5: {0, 1}^{N x m x T}
# A6: [m]_0^{N x T}


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


def pad_or_crop_a6_time(
    a6: np.ndarray,
    target_timesteps: int,
    pad_value: float = 0.0,
    *,
    name: str = "A6",
    allow_crop: bool = False,
) -> np.ndarray:
    """Return A6 with exactly target_timesteps columns.

    Autoencoders need a fixed input shape, while SUMO simulations can stop at
    different horizons. For shorter runs we append zero-count timesteps. Cropping
    is disabled by default because it discards simulated assignment data.
    """
    arr = np.asarray(a6, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D A6=(N,T), got {arr.shape}")

    target_timesteps = int(target_timesteps)
    if target_timesteps <= 0:
        raise ValueError("target_timesteps must be positive")

    n_nodes, current_timesteps = arr.shape
    if current_timesteps == target_timesteps:
        return arr.astype(np.float32, copy=False)

    if current_timesteps < target_timesteps:
        pad_width = target_timesteps - current_timesteps
        padding = np.full((n_nodes, pad_width), float(pad_value), dtype=np.float32)
        return np.concatenate([arr, padding], axis=1).astype(np.float32)

    if not allow_crop:
        raise ValueError(
            f"{name} has T={current_timesteps}, which is longer than target_timesteps={target_timesteps}. "
            "Use a larger --ae_target_timesteps or pass --ae_allow_crop only if truncation is intended."
        )

    print(
        f"WARNING: cropping {name} from T={current_timesteps} to T={target_timesteps}. "
        "This discards late timesteps."
    )
    return arr[:, :target_timesteps].astype(np.float32)


def save_a6(path: str | Path, a6: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(a6, dtype=np.float32))


# ============================================================
# Objective components
# ============================================================


def od_from_a6(
    a6: np.ndarray,
    origins: Iterable[int] | None,
    destinations: Iterable[int] | None,
) -> np.ndarray | None:
    """Simple OD proxy from A6 counts.

    If origins/destinations are supplied, the OD vector is represented by the
    total assignment count on origin nodes at t=0 and destination nodes at T-1.
    This is intentionally lightweight because the assignment files store
    node-time counts, not full paths.
    """
    if origins is None or destinations is None:
        return None

    origins = list(origins)
    destinations = list(destinations)

    if len(origins) == 0 or len(destinations) == 0:
        return None

    origin_mass = float(np.sum(a6[origins, 0]))
    destination_mass = float(np.sum(a6[destinations, -1]))
    return np.array([origin_mass, destination_mass], dtype=np.float32)


def g1_od_penalty(
    a6: np.ndarray,
    target_od: np.ndarray | None = None,
    origins: Iterable[int] | None = None,
    destinations: Iterable[int] | None = None,
) -> float:
    """g1: OD consistency penalty."""
    if target_od is None:
        return 0.0

    od = od_from_a6(a6, origins=origins, destinations=destinations)
    if od is None:
        return 0.0

    return float(np.linalg.norm(od - target_od, ord=1))


def g2_admissibility_penalty(
    a6: np.ndarray,
    total_agents_by_t: np.ndarray | None = None,
    m: int | None = None,
) -> float:
    """g2: penalize non-admissible count assignments.

    In A6, each entry should be an integer in [0, m]. Optionally, the total
    number of agents at every timestep should match total_agents_by_t.
    """
    penalty = 0.0

    rounded = np.rint(a6)
    penalty += float(np.sum(np.abs(a6 - rounded)))

    if m is not None:
        penalty += float(np.sum(np.maximum(0.0, -a6)))
        penalty += float(np.sum(np.maximum(0.0, a6 - m)))

    if total_agents_by_t is not None:
        penalty += float(np.sum(np.abs(np.sum(a6, axis=0) - total_agents_by_t)))

    return penalty

def init_a6_logits_from_base(
    base_a6: np.ndarray,
    m: int | None = None,
    total_agents_by_t: np.ndarray | None = None,
    eps: float = 0.05,
) -> np.ndarray:
    """
    Initialize unconstrained logits so that the first relaxed A6 is close to base_a6.
    If total_agents_by_t is used, logits are interpreted column-wise through softmax.
    Otherwise, logits are interpreted through sigmoid/softplus.
    """
    base = np.asarray(base_a6, dtype=np.float32)

    if total_agents_by_t is not None:
        # Softmax over nodes per timestep. log(base + eps) reconstructs the distribution.
        return np.log(base + eps).astype(np.float32)

    if m is not None:
        # A6 = m * sigmoid(logits)
        p = np.clip(base / float(m), eps, 1.0 - eps)
        return np.log(p / (1.0 - p)).astype(np.float32)

    # A6 = softplus(logits), inverse softplus approximately.
    return np.log(np.exp(base) - 1.0 + eps).astype(np.float32)


def logits_to_relaxed_a6(
    logits: torch.Tensor,
    m: int | None = None,
    total_agents_by_t: np.ndarray | None = None,
) -> torch.Tensor:
    """
    Convert unconstrained logits to continuous relaxed A6.

    If preserve_total_by_t is active, each timestep column sums exactly to
    the original total number of agents.
    """
    if total_agents_by_t is not None:
        totals = torch.as_tensor(
            total_agents_by_t,
            dtype=logits.dtype,
            device=logits.device,
        )
        probs = torch.softmax(logits, dim=0)
        return probs * totals[None, :]

    if m is not None:
        return float(m) * torch.sigmoid(logits)

    return torch.nn.functional.softplus(logits)


def differentiable_genttp_total_time(
    model,
    seed_q_tn: np.ndarray,
    candidate_a6: torch.Tensor,
    seq_length_q: int,
    seq_length_a: int,
    delta_t: float,
) -> torch.Tensor:
    """
    Differentiable GenTTP rollout wrt candidate_a6.

    candidate_a6: torch tensor with shape (N,T)
    seed_q_tn: numpy array with shape (T,N)
    """
    assign_tn = candidate_a6.T
    real_q_tn = torch.as_tensor(
        seed_q_tn,
        dtype=assign_tn.dtype,
        device=assign_tn.device,
    )

    if tuple(assign_tn.shape) != tuple(real_q_tn.shape):
        raise ValueError(
            f"q and assignment shapes must match as (T,N): "
            f"{tuple(real_q_tn.shape)} vs {tuple(assign_tn.shape)}"
        )

    seed_steps = max(seq_length_q, seq_length_a)
    t_count, n_nodes = real_q_tn.shape

    if t_count <= seed_steps:
        raise ValueError(f"Too few timesteps: T={t_count}, seed_steps={seed_steps}")

    generated = [real_q_tn[t] for t in range(seed_steps)]

    for t in range(seed_steps, t_count):
        q_window = torch.stack(generated[t - seq_length_q:t], dim=0).unsqueeze(0)
        a_window = assign_tn[t - seq_length_a:t].unsqueeze(0)

        pred = model(q_window, a_window)

        if pred.dim() == 2:
            pred_step = pred[0]
        elif pred.dim() == 3 and pred.shape[1] == 1:
            pred_step = pred[0, 0]
        else:
            raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")

        generated.append(pred_step)

    generated_tn = torch.stack(generated, dim=0)
    return float(delta_t) * torch.sum(generated_tn[seed_steps:, :n_nodes])

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

def structured_assignment_loss_a1_torch(
    p: torch.Tensor,
    *,
    edge_mask: torch.Tensor,
    origins,
    destinations,
    threshold=0.5,
    temperature=12.0,
):
    """
    p: relaxed A1, shape (N,N,m,T)
       p[i,j,h,t] means agent h uses transition i -> j at time t.

    edge_mask[i,j] = 1 if transition i -> j is legal.
    """
    dtype = p.dtype
    device = p.device
    n_nodes, _, m, t_count = p.shape

    # Soft active edge indicator.
    x = torch.sigmoid(temperature * (torch.abs(p) - threshold))

    edge_mask = edge_mask.to(dtype=dtype, device=device)
    origin_mask = _node_mask(origins, n_nodes, dtype, device)
    dest_mask = _node_mask(destinations, n_nodes, dtype, device)

    # No illegal edges.
    L_edge = (x * (1.0 - edge_mask[:, :, None, None])).sum()

    # Each agent should choose exactly one transition per timestep.
    per_agent_time = x.sum(dim=(0, 1))  # (m,T)
    L_one_transition = torch.abs(per_agent_time - 1.0).sum()

    # Start/end node occupancy per agent.
    start_occ = x.sum(dim=1)  # (N,m,T), start node i
    end_occ = x.sum(dim=0)    # (N,m,T), end node j

    # Origin at t=0.
    origin_per_agent = (origin_mask[:, None] * start_occ[:, :, 0]).sum(dim=0)
    L_origin = torch.abs(origin_per_agent - 1.0).sum()

    # Destination at final represented consistently with A6 as node occupancy at T-1.
    dest_per_agent = (dest_mask[:, None] * start_occ[:, :, -1]).sum(dim=0)
    L_destination = torch.abs(dest_per_agent - 1.0).sum()

    # Exact path continuity:
    # end node at t must equal start node at t+1 for every agent.
    L_connectivity = torch.abs(end_occ[:, :, :-1] - start_occ[:, :, 1:]).sum()

    return {
        "origin": L_origin,
        "destination": L_destination,
        "admissibility": L_edge + L_one_transition,
        "connectivity": L_connectivity,
    }

def project_a6_counts(
    relaxed_a6: np.ndarray,
    m: int | None = None,
    total_agents_by_t: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project continuous A6 to integer count A6.

    Ińf total_agents_by_t is given, each timestep column is repaired so that
    sum_i A6[i,t] equals the requested total.
    """
    raw = np.asarray(relaxed_a6, dtype=np.float32)
    projected = np.rint(raw).astype(np.float32)
    projected = np.maximum(projected, 0.0)

    if m is not None:
        projected = np.minimum(projected, float(m))

    if total_agents_by_t is None:
        return projected.astype(np.float32)

    target = np.rint(total_agents_by_t).astype(int)
    n_nodes, t_count = projected.shape

    for t in range(t_count):
        target_t = int(max(0, target[t]))
        if m is not None:
            target_t = min(target_t, n_nodes * int(m))

        current_t = int(np.sum(projected[:, t]))

        while current_t < target_t:
            capacity = np.inf if m is None else float(m) - projected[:, t]
            eligible = np.where(capacity > 0)[0]
            if len(eligible) == 0:
                break

            # Add to nodes whose relaxed value most wants to be higher.
            residual = raw[eligible, t] - projected[eligible, t]
            i = eligible[int(np.argmax(residual))]
            projected[i, t] += 1.0
            current_t += 1

        while current_t > target_t:
            eligible = np.where(projected[:, t] > 0)[0]
            if len(eligible) == 0:
                break

            # Remove from nodes whose relaxed value least supports current count.
            residual = projected[eligible, t] - raw[eligible, t]
            i = eligible[int(np.argmax(residual))]
            projected[i, t] -= 1.0
            current_t -= 1

    return projected.astype(np.float32)


def perform_gradient_search_a6(
    base_a6: np.ndarray,
    weights: ObjectiveWeights,
    learning_rate: float = 0.01,
    iterations: int = 100,
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
    use_genttp_gradient: bool = False,
    log_every: int = 1,
    adjacency: np.ndarray | torch.Tensor | None = None,
    projection_samples: int = 64,
    projection_noise: float = 0.75,
) -> tuple[np.ndarray, Score, list[dict], str]:
    """
    Proper gradient search directly in relaxed A6 space.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for gradient search.")

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    log_every = max(1, int(log_every))

    if device is None:
        optim_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        optim_device = device
    else:
        optim_device = torch.device(device)

    if model is not None:
        model = model.to(optim_device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    initial_logits = init_a6_logits_from_base(
        base_a6,
        m=m,
        total_agents_by_t=total_agents_by_t,
    )
    rng = np.random.default_rng(42)
    initial_logits = initial_logits + rng.normal(
        loc=0.0,
        scale=0.2,
        size=initial_logits.shape,
    ).astype(np.float32)
    
    logits = torch.as_tensor(
        initial_logits,
        dtype=torch.float32,
        device=optim_device,
    ).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([logits], lr=learning_rate)

    best_score = Score(
        objective=math.inf,
        adttp=math.inf,
        g1=math.inf,
        g2=math.inf,
    )
    best_a6: np.ndarray | None = None
    best_name = "gradient_step_0000.npy"
    history: list[dict] = []

    origins = list(origins or [])
    destinations = list(destinations or [])

    for iteration in range(iterations):
        optimizer.zero_grad()

        relaxed_a6 = logits_to_relaxed_a6(
            logits,
            m=m,
            total_agents_by_t=total_agents_by_t,
        )

        surrogate_loss = relaxed_objective_a6_torch(
            relaxed_a6,
            weights=weights,
            delta_t=delta_t,
            target_od=target_od,
            origins=origins,
            destinations=destinations,
            total_agents_by_t=total_agents_by_t,
            m=m,
            model=model,
            seed_q_tn=seed_q_tn,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
            use_genttp_gradient=use_genttp_gradient,
        )

        surrogate_loss.backward()
        optimizer.step()

        should_log = iteration % log_every == 0 or iteration == iterations - 1
        if not should_log:
            continue

        with torch.no_grad():
            relaxed_eval = logits_to_relaxed_a6(
                logits,
                m=m,
                total_agents_by_t=total_agents_by_t,
            )
            relaxed_np = relaxed_eval.detach().cpu().numpy()

        candidate_list = sample_projected_a6_candidates(
            relaxed_np,
            n_samples=64,
            noise_scale=0.75,
            m=m,
            total_agents_by_t=total_agents_by_t,
            rng=rng,
        )

        score = None
        candidate_a6 = None

        for cand in candidate_list:
            cand_score = score_a6(
            cand,
            weights=weights,
            delta_t=delta_t,
            target_od=target_od,
            origins=origins,
            destinations=destinations,
            total_agents_by_t=total_agents_by_t,
            m=m,
            model=model,
            device=optim_device,
            seed_q_tn=seed_q_tn,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
            adjacency=adjacency,
        )

        if score is None or cand_score.objective < score.objective:
            score = cand_score
            candidate_a6 = cand.copy()

        improved = score.objective < best_score.objective
        if improved:
            best_score = score
            best_a6 = candidate_a6.copy()
            best_name = f"gradient_step_{iteration:04d}.npy"

        history.append(
            {
                "iteration": iteration,
                "surrogate_loss": float(surrogate_loss.detach().cpu().item()),
                "objective": score.objective,
                "adttp": score.adttp,
                "g1_od": score.g1,
                "g2_admissibility": score.g2,
                "best_objective": best_score.objective,
                "improved": bool(improved),
            }
        )

        print(
            f"Iteration {iteration:04d}: "
            f"surrogate={float(surrogate_loss.detach().cpu().item()):.4f}, "
            f"objective={score.objective:.4f}, "
            f"adttp={score.adttp:.4f}, "
            f"g1={score.g1:.4f}, "
            f"g2={score.g2:.4f}, "
            f"best={best_score.objective:.4f}",
            flush=True,
        )

    if best_a6 is None:
        with torch.no_grad():
            relaxed_eval = logits_to_relaxed_a6(
                logits,
                m=m,
                total_agents_by_t=total_agents_by_t,
            )
        best_a6 = project_a6_counts(
            relaxed_eval.detach().cpu().numpy(),
            m=m,
            total_agents_by_t=total_agents_by_t,
        )
        best_score = score_a6(
            best_a6,
            weights=weights,
            delta_t=delta_t,
            target_od=target_od,
            origins=origins,
            destinations=destinations,
            total_agents_by_t=total_agents_by_t,
            m=m,
            model=model,
            device=optim_device,
            seed_q_tn=seed_q_tn,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
        )

    return best_a6, best_score, history, best_name



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

def _node_mask(nodes, n_nodes, dtype, device):
    mask = torch.zeros(n_nodes, dtype=dtype, device=device)
    if nodes is not None:
        nodes = list(nodes)
        if len(nodes) > 0:
            idx = torch.as_tensor(nodes, dtype=torch.long, device=device)
            mask[idx] = 1.0
    return mask

def _prepare_adjacency(adjacency, n_nodes, dtype, device, add_self_loops=True):
    """
    adjacency[i,j] = 1 means an agent can move from node i to node j in one time step.
    """
    if adjacency is None:
        return None

    A = torch.as_tensor(adjacency, dtype=dtype, device=device)
    if A.shape != (n_nodes, n_nodes):
        raise ValueError(f"adjacency must have shape {(n_nodes, n_nodes)}, got {tuple(A.shape)}")

    A = (A > 0).to(dtype)

    if add_self_loops:
        A = torch.clamp(A + torch.eye(n_nodes, dtype=dtype, device=device), 0.0, 1.0)

    return A

def sample_projected_a6_candidates(
    relaxed_a6: np.ndarray,
    *,
    n_samples: int,
    noise_scale: float,
    m: int | None,
    total_agents_by_t: np.ndarray | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Create hard integer A6 candidates by perturbing relaxed A6 before projection.
    This forces jumps across rounding/projection boundaries.
    """
    candidates = []

    # Always include the direct projection.
    candidates.append(
        project_a6_counts(
            relaxed_a6,
            m=m,
            total_agents_by_t=total_agents_by_t,
        )
    )

    for _ in range(max(0, n_samples - 1)):
        noisy = relaxed_a6 + rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=relaxed_a6.shape,
        ).astype(np.float32)

        candidate = project_a6_counts(
            noisy,
            m=m,
            total_agents_by_t=total_agents_by_t,
        )
        candidates.append(candidate)

    return candidates

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


# ============================================================
# Gradient search in relaxed A1 space
# ============================================================


def project_relaxed_a1_to_a1(a1_relaxed: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Project relaxed A1 values to {-1, 0, 1}.

    `np.sign` alone is unsafe because almost every small nonzero random value
    becomes +/-1, producing dense assignments. The threshold keeps weak latent
    values at zero.
    """
    a1_relaxed = np.asarray(a1_relaxed, dtype=np.float32)
    active = np.abs(a1_relaxed) >= threshold
    return (np.sign(a1_relaxed) * active).astype(np.int8)


def a1_to_a6(a1: np.ndarray) -> np.ndarray:
    """A1 -> A2 -> A6."""
    return f_2_to_6(f_1_to_2(a1)).astype(np.float32)


def initialize_search_a1_from_a6(
    base_a6: np.ndarray,
    m: int,
    threshold: float = 0.5,
    noise: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a feasible-ish latent A1 from an A6 count matrix.

    A6 does not contain path/edge signs. This initializes counts on diagonal
    i->i entries. That maps back to the same node-time counts through
    A1 -> A2 -> A6.
    """
    if rng is None:
        rng = np.random.default_rng()

    base_a6 = np.rint(np.asarray(base_a6)).astype(int)
    n_nodes, t_count = base_a6.shape

    p = rng.normal(
        0.0,
        noise,
        size=(n_nodes, n_nodes, m, t_count),
    ).astype(np.float32)

    for i in range(n_nodes):
        for t in range(t_count):
            count = int(np.clip(base_a6[i, t], 0, m))
            if count > 0:
                p[i, i, :count, t] = threshold + 1.0

    return p


def relaxed_a1_to_a6_torch(
    p: torch.Tensor,
    threshold: float = 0.5,
    temperature: float = 12.0,
) -> torch.Tensor:
    """Differentiable surrogate for A1 -> A2 -> A6.

    Hard projection is used only for evaluation/logging. The optimizer sees this
    smooth approximation.
    """
    a2_soft = torch.sigmoid(temperature * (torch.abs(p) - threshold))
    a5_soft = torch.amax(a2_soft, dim=1)
    a6_soft = torch.sum(a5_soft, dim=1)
    return a6_soft


def relaxed_objective_a1_torch(
    p: torch.Tensor,
    weights: ObjectiveWeights,
    delta_t: float = 1.0,
    target_od: np.ndarray | None = None,
    origins: Iterable[int] | None = None,
    destinations: Iterable[int] | None = None,
    total_agents_by_t: np.ndarray | None = None,
    m: int | None = None,
    threshold: float = 0.5,
    temperature: float = 12.0,
) -> torch.Tensor:
    """Differentiable surrogate objective used by the optimizer."""
    a6 = relaxed_a1_to_a6_torch(
        p,
        threshold=threshold,
        temperature=temperature,
    )

    dtype = p.dtype
    device = p.device
    t_count = a6.shape[1]

    time_weights = torch.arange(t_count, dtype=dtype, device=device) * float(delta_t)
    adttp = torch.sum(a6 * time_weights[None, :])

    origins = list(origins or [])
    destinations = list(destinations or [])

    g1 = torch.zeros((), dtype=dtype, device=device)
    if target_od is not None and len(origins) > 0 and len(destinations) > 0:
        target = torch.as_tensor(target_od, dtype=dtype, device=device)
        od = torch.stack([
            torch.sum(a6[origins, 0]),
            torch.sum(a6[destinations, -1]),
        ])
        g1 = torch.sum(torch.abs(od - target))

    g2 = torch.zeros((), dtype=dtype, device=device)
    if total_agents_by_t is not None:
        total = torch.as_tensor(total_agents_by_t, dtype=dtype, device=device)
        g2 = g2 + torch.sum(torch.abs(torch.sum(a6, dim=0) - total))

    if m is not None:
        g2 = g2 + torch.relu(-a6).sum()
        g2 = g2 + torch.relu(a6 - float(m)).sum()

    return adttp + weights.od * g1 + weights.admissibility * g2


def perform_gradient_search_a1(
    initial_a1: np.ndarray,
    weights: ObjectiveWeights,
    learning_rate: float = 0.01,
    iterations: int = 100,
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
    projection_threshold: float = 0.5,
    temperature: float = 12.0,
    log_every: int = 1,
) -> tuple[np.ndarray, Score, list[dict], str]:
    """Run gradient search and evaluate/log projected A6 candidates.

    The gradient is computed on a differentiable surrogate. Logged scores are
    computed on the hard projected candidate actually saved as A6.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for gradient search.")

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    log_every = max(1, int(log_every))

    if device is None:
        optim_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        optim_device = device
    else:
        optim_device = torch.device(device)

    p = torch.as_tensor(
        initial_a1,
        dtype=torch.float32,
        device=optim_device,
    ).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([p], lr=learning_rate)

    best_score = Score(
        objective=math.inf,
        adttp=math.inf,
        g1=math.inf,
        g2=math.inf,
    )
    best_a6: np.ndarray | None = None
    best_name = "gradient_step_0000.npy"
    history: list[dict] = []

    origins = list(origins or [])
    destinations = list(destinations or [])

    for iteration in range(iterations):
        optimizer.zero_grad()

        surrogate_loss = relaxed_objective_a1_torch(
            p,
            weights=weights,
            delta_t=delta_t,
            target_od=target_od,
            origins=origins,
            destinations=destinations,
            total_agents_by_t=total_agents_by_t,
            m=m,
            threshold=projection_threshold,
            temperature=temperature,
            adjacency=adjacency,
        )

        surrogate_loss.backward()
        optimizer.step()

        should_log = iteration % log_every == 0 or iteration == iterations - 1
        if not should_log:
            continue

        p_np = p.detach().cpu().numpy()
        a1_discrete = project_relaxed_a1_to_a1(
            p_np,
            threshold=projection_threshold,
        )
        candidate_a6 = a1_to_a6(a1_discrete)

        score = score_a6(
            candidate_a6,
            weights=weights,
            delta_t=delta_t,
            target_od=target_od,
            origins=origins,
            destinations=destinations,
            total_agents_by_t=total_agents_by_t,
            m=m,
            model=model,
            device=device,
            seed_q_tn=seed_q_tn,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
        )

        improved = score.objective < best_score.objective
        if improved:
            best_score = score
            best_a6 = candidate_a6.copy()
            best_name = f"gradient_step_{iteration:04d}.npy"

        history.append({
            "iteration": iteration,
            "surrogate_loss": float(surrogate_loss.detach().cpu().item()),
            "objective": score.objective,
            "adttp": score.adttp,
            "g1_od": score.g1,
            "g2_admissibility": score.g2,
            "best_objective": best_score.objective,
            "improved": bool(improved),
        })

        print(
            f"Iteration {iteration:04d}: "
            f"surrogate={float(surrogate_loss.detach().cpu().item()):.4f}, "
            f"objective={score.objective:.4f}, "
            f"adttp={score.adttp:.4f}, "
            f"g1={score.g1:.4f}, "
            f"g2={score.g2:.4f}, "
            f"best={best_score.objective:.4f}"
        )

    if best_a6 is None:
        p_np = p.detach().cpu().numpy()
        best_a6 = a1_to_a6(
            project_relaxed_a1_to_a1(p_np, threshold=projection_threshold)
        )
        best_score = score_a6(
            best_a6,
            weights=weights,
            delta_t=delta_t,
            target_od=target_od,
            origins=origins,
            destinations=destinations,
            total_agents_by_t=total_agents_by_t,
            m=m,
            model=model,
            device=device,
            seed_q_tn=seed_q_tn,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
        )

    return best_a6, best_score, history, best_name


# ============================================================
# Optional grid/candidate helpers kept for compatibility
# ============================================================


def parse_int_list(text: str | None) -> list[int]:
    if text is None or text.strip() == "":
        return []

    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            chunks = [c.strip() for c in part.split(":")]
            if len(chunks) not in (2, 3):
                raise ValueError(f"Bad range: {part}")
            start = int(chunks[0]) if chunks[0] else 0
            stop = int(chunks[1])
            step = int(chunks[2]) if len(chunks) == 3 and chunks[2] else 1
            out.extend(range(start, stop, step))
        else:
            out.append(int(part))

    return list(dict.fromkeys(out))


def maybe_build_genttp(args):
    if args.checkpoint is None:
        return None, None

    if torch is None:
        raise RuntimeError("PyTorch is required for --checkpoint mode")

    from inference import build_trainer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainer = build_trainer(args, device)
    trainer.model.eval()
    return trainer.model, device


def no_grad_if_available(fn):
    if torch is None:
        return fn
    return torch.no_grad()(fn)


@no_grad_if_available
def rollout_tt_with_genttp(
    model,
    device,
    seed_q_tn: np.ndarray,
    candidate_a6: np.ndarray,
    seq_length_q: int,
    seq_length_a: int,
    delta_t: float,
) -> float:
    """Use GenTTP to estimate total travel time for an A6 candidate."""
    if model is None:
        return adttp_from_a6(candidate_a6, delta_t=delta_t)

    assign_tn = np.asarray(candidate_a6.T, dtype=np.float32)
    real_q_tn = np.asarray(seed_q_tn, dtype=np.float32)

    if assign_tn.shape != real_q_tn.shape:
        raise ValueError(
            f"q and assignment shapes must match as (T,N): "
            f"{real_q_tn.shape} vs {assign_tn.shape}"
        )

    seed_steps = max(seq_length_q, seq_length_a)
    t_count, n_nodes = real_q_tn.shape

    if t_count <= seed_steps:
        raise ValueError(f"Too few timesteps: T={t_count}, seed_steps={seed_steps}")

    generated = np.zeros_like(real_q_tn, dtype=np.float32)
    generated[:seed_steps] = real_q_tn[:seed_steps]

    for t in range(seed_steps, t_count):
        q_window = torch.from_numpy(generated[t - seq_length_q:t]).unsqueeze(0).to(device)
        a_window = torch.from_numpy(assign_tn[t - seq_length_a:t]).unsqueeze(0).to(device)
        pred = model(q_window, a_window)

        if pred.dim() == 2:
            pred_step = pred[0]
        elif pred.dim() == 3 and pred.shape[1] == 1:
            pred_step = pred[0, 0]
        else:
            raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")

        generated[t] = pred_step.detach().cpu().numpy()

    return float(delta_t * np.sum(generated[seed_steps:, :n_nodes]))


# ============================================================
# Main CLI
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gradient search over Step 2 assignment Representation 6."
    )

    parser.add_argument(
        "--base_assignment",
        type=str,
        required=True,
        help="Base .npy assignment, normalized to A6=(N,T).",
    )
    parser.add_argument("--output_dir", type=str, default="./gradient_search_out")
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Maximum agents per timestep, used for [m]_0 bounds and latent A1 size.",
    )
    parser.add_argument("--delta_t", type=float, default=1.0)
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=100,
        help="Backward-compatible iteration budget when --iterations is omitted.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--projection_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--admissibility_weight", type=float, default=1_000.0)
    parser.add_argument("--od_weight", type=float, default=0.0)
    parser.add_argument("--origin_weight", type=float, default=10_000.0)
    parser.add_argument("--destination_weight", type=float, default=10_000.0)
    parser.add_argument("--connectivity_weight", type=float, default=10_000.0)
    parser.add_argument("--origins", type=str, default="")
    parser.add_argument("--destinations", type=str, default="")
    parser.add_argument("--projection_samples", type=int, default=64)
    parser.add_argument("--projection_noise", type=float, default=0.75)
    parser.add_argument(
        "--target_od",
        type=str,
        default=None,
        help="Two comma-separated values: origin_mass,destination_mass.",
    )
    parser.add_argument(
        "--preserve_total_by_t",
        action="store_true",
        help="Penalize candidates whose sum_i A6[i,t] differs from base.",
    )

    # Optional GenTTP args, compatible with inference.py/build_trainer.
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--q_seed",
        type=str,
        default=None,
        help="Required with --checkpoint; flow seed file as (N,T) or (T,N).",
    )
    parser.add_argument("--adjdata", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seq_length_q", type=int, default=15)
    parser.add_argument("--seq_length_a", type=int, default=30)
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gcn_bool", action="store_true")
    parser.add_argument("--addaptadj", action="store_true")
    parser.add_argument("--randomadj", action="store_true")
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--sequence_model",
        type=str,
        default="gru",
        choices=["lstm", "gru", "attention"],
    )
    parser.add_argument(
        "--fuse_method",
        type=str,
        default="attention",
        choices=["concatenate", "attention", "wavenet_only", "assignment_only", "hadamard"],
    )
    parser.add_argument("--a_embedding_size", type=int, default=32)
    parser.add_argument("--a_hidden_size", type=int, default=64)
    parser.add_argument("--q_rep_dim", type=int, default=32)
    parser.add_argument("--fused_dim", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--attention_num_heads", type=int, default=4)
    parser.add_argument("--attention_ff_dim", type=int, default=128)
    parser.add_argument("--fuse_attention_num_heads", type=int, default=4)
    parser.add_argument("--fuse_attention_ff_dim", type=int, default=None)
    parser.add_argument("--fuse_gated_update", action="store_true")
    parser.add_argument(
        "--use_genttp_gradient",
        action="store_true",
        help=(
            "Backpropagate through GenTTP rollout when --checkpoint is used. "
            "Without this, GenTTP is used only to score projected candidates."
        ),
    )

    return parser

def load_adjacency_matrix(path: str | Path, num_nodes: int) -> np.ndarray:
    path = Path(path)

    if path.suffix == ".npy":
        A = np.load(path).astype(np.float32)
    else:
        # For CSV adjacency with row labels, matching utilities.load_csv_adj style.
        A = pd.read_csv(path, index_col=0).to_numpy(dtype=np.float32)

    if A.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Adjacency must have shape {(num_nodes, num_nodes)}, got {A.shape}"
        )

    return A


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup base configuration.
    base_a6 = load_assignment_as_a6(args.base_assignment, num_nodes=args.num_nodes)
    if args.num_nodes is None:
        args.num_nodes = int(base_a6.shape[0])
    adjacency = None
    if args.adjdata is not None:
        adjacency = load_adjacency_matrix(args.adjdata, args.num_nodes)

    weights = ObjectiveWeights(
        od=args.od_weight,
        origin=args.origin_weight,
        destination=args.destination_weight,
        admissibility=args.admissibility_weight,
        connectivity=args.connectivity_weight,
    )

    original_base_timesteps = int(base_a6.shape[1])
    total_by_t = None  # computed after any AE time-axis padding

    target_od = None
    if args.target_od is not None:
        values = [float(x.strip()) for x in args.target_od.split(",") if x.strip()]
        if len(values) != 2:
            raise ValueError("--target_od must contain exactly two comma-separated values")
        target_od = np.array(values, dtype=np.float32)

    # 2. Optional GenTTP model.
    model, device = maybe_build_genttp(args)
    print("[status] GenTTP checkpoint loaded; starting candidate search", flush=True)
    if device is None:
        device = torch.device(args.device if torch is not None and torch.cuda.is_available() else "cpu")

    q_seed_tn = None
    q_seed_a6 = None
    if model is not None:
        if args.q_seed is None:
            raise ValueError("--q_seed is required when --checkpoint is used")
        q_seed_a6 = load_assignment_as_a6(args.q_seed, num_nodes=args.num_nodes)
        q_seed_tn = q_seed_a6.T.astype(np.float32)

    rng = np.random.default_rng(args.seed)
    origins = parse_int_list(args.origins)
    destinations = parse_int_list(args.destinations)

    iterations = args.iterations
    if iterations is None:
        iterations = int(args.max_candidates)

    m_latent = args.m
    if m_latent is None:
        m_latent = max(1, int(np.nanmax(base_a6)))

    autoencoder = None
    autoencoder_path = None

    if q_seed_a6 is not None and tuple(q_seed_a6.shape) != tuple(base_a6.shape):
        raise ValueError(
            f"q_seed and base_assignment must have the same A6 shape. "
            f"Got q_seed={tuple(q_seed_a6.shape)}, base={tuple(base_a6.shape)}"
        )
    total_by_t = np.sum(base_a6, axis=0) if args.preserve_total_by_t else None

    if args.checkpoint is not None and not args.use_genttp_gradient:
        print(
            "WARNING: --checkpoint is used only for scoring projected candidates. "
            "The gradient direction uses the simple differentiable A6 surrogate. "
            "Pass --use_genttp_gradient to backpropagate through GenTTP.",
            flush=True,
        )

    if args.preserve_total_by_t and args.checkpoint is None and target_od is None:
        print(
            "WARNING: --preserve_total_by_t makes simple ADTTP constant with respect "
            "to spatial redistribution. Without GenTTP gradient or OD/end-point target, "
            "the gradient signal may be weak.",
            flush=True,
        )
    best_a6, best_score, history, best_name = perform_gradient_search_a6(
        base_a6=base_a6,
        weights=weights,
        learning_rate=args.learning_rate,
        iterations=iterations,
        delta_t=args.delta_t,
        target_od=target_od,
        origins=origins,
        destinations=destinations,
        total_agents_by_t=total_by_t,
        m=args.m,
        model=model,
        device=device,
        seed_q_tn=q_seed_tn,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
        use_genttp_gradient=args.use_genttp_gradient,
        log_every=args.log_every,
        adjacency=adjacency,
        projection_samples=args.projection_samples,
        projection_noise=args.projection_noise,
    )
    # 5. Save outputs.
    best_path = out_dir / "best_assignment_a6.npy"
    save_a6(best_path, best_a6)

    history_path = out_dir / "optimization_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        "best_name": best_name,
        "best_assignment": str(best_path),
        "best_objective": best_score.objective,
        "best_adttp": best_score.adttp,
        "best_g1_od": best_score.g1,
        "best_g2_admissibility": best_score.g2,
        "iterations": iterations,
        "learning_rate": args.learning_rate,
        "projection_threshold": args.projection_threshold,
        "temperature": args.temperature,
        "m_latent": m_latent,
        "m_constraint": args.m,
        "original_base_timesteps": original_base_timesteps,
        "effective_timesteps": int(base_a6.shape[1]),
        "ae_target_timesteps": int(autoencoder.input_shape[1]) if autoencoder is not None else None,
        "ae_checkpoint": str(autoencoder_path) if autoencoder_path is not None else None,
    }

    summary_path = out_dir / "optimization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Gradient search complete.")
    print(f"History saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
