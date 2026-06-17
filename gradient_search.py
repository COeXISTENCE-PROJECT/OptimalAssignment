#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ============================================================
# assignment representations
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

    Existing repository data commonly stores assignments as (N, T), (T, N),
    (N, T, 1), or (T, N, 1). This helper normalizes them to (N, T).
    """
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Assignment must be 2D or 3D with singleton channel, got {arr.shape}")

    if num_nodes is not None:
        if arr.shape[0] == num_nodes:
            out = arr
        elif arr.shape[1] == num_nodes:
            out = arr.T
        else:
            raise ValueError(f"Cannot infer node axis for shape {arr.shape} and num_nodes={num_nodes}")
    else:
        out = arr

    return np.asarray(out, dtype=np.float32)


def save_a6(path: str | Path, a6: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(a6, dtype=np.float32))
    

def od_from_a6(a6: np.ndarray, origins: Iterable[int] | None, destinations: Iterable[int] | None) -> np.ndarray | None:
    """Simple OD proxy from A6 counts.

    If origins/destinations are supplied, the OD vector is represented by the
    total assignment count on origin nodes at t=0 and destination nodes at T-1.
    This is intentionally lightweight because the repository assignment files
    currently store node-time counts, not full paths.
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


# def g1_od_penalty(a6: np.ndarray, target_od: np.ndarray | None, origins=None, destinations=None) -> float:
#     """g1: OD consistency penalty."""
#     if target_od is None:
#         return 0.0
#     od = od_from_a6(a6, origins, destinations)
#     if od is None:
#         return 0.0
#     return float(np.linalg.norm(od - target_od, ord=1))


# def g2_admissibility_penalty(a6: np.ndarray, total_agents_by_t: np.ndarray | None = None, m: int | None = None) -> float:
#     """g2: penalize non-admissible count assignments.
#     In A6 each entry should be an integer in [0, m]. Optionally, the total
#     number of agents at every timestep should match total_agents_by_t.
#     """
#    penalty = 0.0
#    rounded = np.rint(a6)
#    penalty += float(np.sum(np.abs(a6 - rounded)))
#    if m is not None:
#        penalty += float(np.sum(np.maximum(0.0, -a6)))
#        penalty += float(np.sum(np.maximum(0.0, a6 - m)))
#    if total_agents_by_t is not None:
#        penalty += float(np.sum(np.abs(np.sum(a6, axis=0) - total_agents_by_t)))
#    return penalty

def g1_od_penalty(a6: np.ndarray, target_od: np.ndarray | None = None, origins: Iterable[int] | None = None, destinations: Iterable[int] | None = None) -> float:
    """g1: OD consistency penalty."""
    if target_od is None:
        return 0.0
    # Use explicit keyword mapping to avoid signature mismatches
    od = od_from_a6(a6, origins=origins, destinations=destinations)
    if od is None:
        return 0.0
    return float(np.linalg.norm(od - target_od, ord=1))

def g2_admissibility_penalty(
    a6: np.ndarray, 
    total_agents_by_t: np.ndarray | None = None, 
    m: int | None = None
) -> float:
    """g2: penalize non-admissible count assignments.
    In A6 each entry should be an integer in [0, m]. Optionally, the total
    number of agents at every timestep should match total_agents_by_t.
    """
    penalty = 0.0
    rounded = np.rint(a6)
    # Penalty for non-integer assignments
    penalty += float(np.sum(np.abs(a6 - rounded)))
    
    # Penalty for exceeding capacity [0, m]
    if m is not None:
        penalty += float(np.sum(np.maximum(0.0, -a6)))
        penalty += float(np.sum(np.maximum(0.0, a6 - m)))
        
    # Penalty for total agent count mismatch
    if total_agents_by_t is not None:
        # Summing over nodes (axis 0) to compare with total_agents_by_t
        penalty += float(np.sum(np.abs(np.sum(a6, axis=0) - total_agents_by_t)))
        
    return penalty
def adttp_from_a6(a6: np.ndarray, delta_t: float = 1.0) -> float:
    """Default ADTTP on Representation 6: sum_t t * sum_i A6[i,t]."""
    if a6.ndim == 1:
        a6 = a6[:, np.newaxis]
    n_nodes, t_count = a6.shape
    
    weights = np.arange(t_count, dtype=np.float32) * float(delta_t)
    
    return float(np.sum(a6 * weights[None, :]))


@dataclass
class ObjectiveWeights:
    od: float = 1_000.0
    admissibility: float = 1_000.0


@dataclass
class Score:
    objective: float
    adttp: float
    g1: float
    g2: float


def score_a6(
    a1: np.ndarray,
    target_od: np.ndarray,
    constraint_matrix: np.ndarray,
    weights: Objectiveweights,
    model: torch.nn.Module | None = None,
    delta_t: float = 1.0,
    # target_od: np.ndarray | None = None,
    # origins: Iterable[int] | None = None,
    # destinations: Iterable[int] | None = None,
    # total_agents_by_t: np.ndarray | None = None,
    # m: int | None = None,
) -> Score:
    b = f_1_to_2(a1)
    d = f_2_to_6(b)
    if model is not None:
        adttp = rollout_tt_with_genttp(model)
    else:
        adttp = adttp_from_a6(d, delta_t=delta_t)
    # g1 = g1_od_penalty(d, target_od=target_od, origins=origins, destinations=destinations)
    # g2 = g2_admissibility_penalty(b, total_agents_by_t=total_agents_by_t, m=m)
    g1 = g1_od_penalty(a1_original, target_od)
    g2 = g2_admissibility_penalty(b, constraint_matrix)
    objective = adttp + weights.od * g1 + weights.admissibility * g2
    return Score(objective=objective, adttp=adttp, g1=g1, g2=g2)

def objective_a1(
    a1_relaxed: np.ndarray,
    weights: ObjectiveWeights,
    delta_t: float = 1.0,
    target_od: np.ndarray | None = None,
    origins: Iterable[int] | None = None,
    destinations: Iterable[int] | None = None,
    total_agents_by_t: np.ndarray | None = None,
    m: int | None = None
) -> float:
    """
    Computes objective function for a relaxed A1 representation.
    f(A) = ADTTP(f_2->6(f_1->2(A))) + g1(A) + g2(f_1->2(A))
    """
    # 1. Project to discrete A1: map R to {-1, 0, 1}
    # Using np.sign ensures we maintain the signed representation A1
    a1_discrete = np.sign(a1_relaxed).astype(np.int8) 
    
    # 2. Map to A2 and A6 space
    # f_1_to_2 transforms A1 -> A2
    a2 = f_1_to_2(a1_discrete)
    # f_2_to_6 transforms A2 -> A6
    a6 = f_2_to_6(a2)
    
    # 3. Calculate components
    # ADTTP is calculated based on A6
    adttp = adttp_from_a6(a6, delta_t=delta_t)
    
    # g1: OD consistency penalty on A1
    # Note: Using a1_discrete as it represents the assignment
    g1 = g1_od_penalty(a6, target_od=target_od, origins=origins, destinations=destinations)
    
    # g2: Admissibility penalty on A2 (B = f_1_to_2(A1))
    g2 = g2_admissibility_penalty(a6, total_agents_by_t=total_agents_by_t, m=m)
    
    # Total objective
    return float(adttp + weights.od * g1 + weights.admissibility * g2)


def initialize_search_a1(shape):
    """
    Initialize search in the latent space of A1.
    Shape: (N, N, m, T)
    """
    # Initialize with small random values to allow gradient flow
    return np.random.normal(0, 0.1, shape)

# ============================================================
# Grid candidate generation in Representation 6
# ============================================================

def perform_gradient_search_a1(
    initial_a1: np.ndarray,
    weights: ObjectiveWeights,
    learning_rate: float = 0.01,
    iterations: int = 100,
    **kwargs
) -> np.ndarray:
    """
    Performs iterative gradient descent on the latent A1 representation.
    """
    p = initial_a1.copy().astype(np.float32)
    
    for i in range(iterations):
        # In a real implementation, you would compute the gradient via 
        # autograd (if using PyTorch) or numerical approximation.
        # This acts as the placeholder for the update step.
        
        # Current score
        current_score = objective_a1(p, weights, **kwargs)
        
        # Here you would typically apply: p = p - learning_rate * gradient(objective_a1)
        # For a simple grid/random search, you might instead perturb P:
        perturbation = np.random.normal(0, 0.01, p.shape)
        p += perturbation
        
    return np.sign(p).astype(np.int8)


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


def generate_local_grid_a6(
    base: np.ndarray,
    nodes: list[int],
    timesteps: list[int],
    values: list[int],
    m: int | None,
    max_candidates: int,
):
    """Enumerate a local Cartesian grid by replacing selected A6 entries.

    The unselected entries are copied from the base assignment. This avoids the
    impossible full grid over [m]^(N*T), while still implementing a real grid
    search over the requested Step 2 representation.
    """
    if not nodes:
        nodes = list(range(base.shape[0]))
    if not timesteps:
        timesteps = list(range(base.shape[1]))
    positions = [(i, t) for i in nodes for t in timesteps]
    if not positions:
        yield np.array(base, copy=True)
        return

    count = 0
    for combo in itertools.product(values, repeat=len(positions)):
        cand = np.array(base, copy=True)
        for (i, t), value in zip(positions, combo):
            cand[i, t] = value
        if m is not None:
            cand = np.clip(cand, 0, m)
        yield cand
        count += 1
        if count >= max_candidates:
            break


def load_candidates_from_dir(candidate_dir: str | Path, num_nodes: int | None) -> list[tuple[str, np.ndarray]]:
    candidate_dir = Path(candidate_dir)
    files = sorted(candidate_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy candidate files found in {candidate_dir}")
    return [(p.name, load_assignment_as_a6(p, num_nodes=num_nodes)) for p in files]


# ============================================================
# Optional GenTTP rollout objective
# ============================================================


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


@torch.no_grad() if torch is not None else (lambda fn: fn)
def rollout_tt_with_genttp(model, device, seed_q_tn: np.ndarray, candidate_a6: np.ndarray, seq_length_q: int, seq_length_a: int, delta_t: float) -> float:
    """Use GenTTP to estimate total travel time for an A6 candidate."""
    if model is None:
        return adttp_from_a6(candidate_a6, delta_t=delta_t)

    assign_tn = np.asarray(candidate_a6.T, dtype=np.float32)
    real_q_tn = np.asarray(seed_q_tn, dtype=np.float32)
    if assign_tn.shape != real_q_tn.shape:
        raise ValueError(f"q and assignment shapes must match as (T,N): {real_q_tn.shape} vs {assign_tn.shape}")

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
    parser = argparse.ArgumentParser(description="Grid search over Step 2 assignment Representation 6.")

    parser.add_argument("--base_assignment", type=str, required=True, help="Base .npy assignment, normalized to A6=(N,T).")
    parser.add_argument("--candidate_dir", type=str, default=None, help="Optional dir with candidate .npy assignments.")
    parser.add_argument("--output_dir", type=str, default="./grid_search_out")
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--m", type=int, default=None, help="Maximum agents per timestep, used for [m]_0 bounds.")
    parser.add_argument("--delta_t", type=float, default=1.0)

    parser.add_argument("--nodes", type=str, default="", help="Comma/range list, e.g. '0,3,10:20'. Empty means all nodes.")
    parser.add_argument("--timesteps", type=str, default="", help="Comma/range list. Empty means all timesteps.")
    parser.add_argument("--values", type=str, default="0,1", help="Grid values for selected entries, e.g. '0,1,2'.")
    parser.add_argument("--max_candidates", type=int, default=10000)

    parser.add_argument("--od_weight", type=float, default=1000.0)
    parser.add_argument("--admissibility_weight", type=float, default=1000.0)
    parser.add_argument("--origins", type=str, default="")
    parser.add_argument("--destinations", type=str, default="")
    parser.add_argument("--target_od", type=str, default=None, help="Two comma-separated values: origin_mass,destination_mass.")
    parser.add_argument("--preserve_total_by_t", action="store_true", help="Penalize candidates whose sum_i A6[i,t] differs from base.")

    # Optional GenTTP args, compatible with inference.py/build_trainer.
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--q_seed", type=str, default=None, help="Required with --checkpoint; flow seed file as (N,T) or (T,N).")
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
    parser.add_argument("--sequence_model", type=str, default="gru", choices=["lstm", "gru", "attention"])
    parser.add_argument("--fuse_method", type=str, default="attention", choices=["concatenate", "attention", "wavenet_only", "assignment_only", "hadamard"])
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target_od = None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup base configurations
    base_a6 = load_assignment_as_a6(args.base_assignment, num_nodes=args.num_nodes)
    if args.num_nodes is None:
        args.num_nodes = int(base_a6.shape[0])
    
    # Weights and constraints
    weights = ObjectiveWeights(od=args.od_weight, admissibility=args.admissibility_weight)
    total_by_t = np.sum(base_a6, axis=0) if args.preserve_total_by_t else None
    if args.target_od is not None:
        target_od = np.array([float(x) for x in args.target_od.split(",")], dtype=np.float32)
    # 2. Build model if provided
    model, device = maybe_build_genttp(args)
    q_seed_tn = None
    if model is not None:
        q_seed = load_assignment_as_a6(args.q_seed, num_nodes=args.num_nodes)
        q_seed_tn = q_seed.T.astype(np.float32)

    # 3. Initialize Gradient Search in A1 Space
    # Latent space P dimension: (N, N, m, T)
    m_val = args.m if args.m is not None else 1
    latent_shape = (args.num_nodes, args.num_nodes, m_val, base_a6.shape[1])
    
    # Initialize P randomly or from existing assignment (if possible)
    p = initialize_search_a1(latent_shape)
    history = []
    origins = parse_int_list(args.origins)
    destinations = parse_int_list(args.destinations)
    # 4. Run Optimization Loop
    # We iteratively update P to minimize objective_a1
    for i in range(args.max_candidates // 100): # Using max_candidates as iteration budget
        if i % 5 == 0:
            current_objective = objective_a1(
                a1_relaxed=p,
                weights=weights,
                delta_t=args.delta_t,
                target_od=target_od,          # Defined in main()
                origins=origins,              # Parsed from args.origins
                destinations=destinations,    # Parsed from args.destinations
                total_agents_by_t=total_by_t, # Optional: defined if preserve_total_by_t is set
                m=args.m                      # Defined from args
            )
            print(f"Iteration {i}: loss = {current_objective:.2f}")
        # In a real gradient implementation, perform P = P - lr * grad(objective_a1)
        # Here we perform the update via the gradient search logic
        p = perform_gradient_search_a1(
            initial_a1=p,
            weights=weights,
            delta_t=args.delta_t,
            target_od=None, # Define your target_od here
            total_agents_by_t=total_by_t,
            m=args.m,
            learning_rate=args.learning_rate
        )
        
        # 5. Map best P to A6 for evaluation with Model (if exists)
        a1_discrete = np.sign(p).astype(np.int8)
        best_a6_cand = f_2_to_6(f_1_to_2(a1_discrete))
        
        # 6. Evaluation
        # If model exists, use it to calculate refined objective
        if model is not None:
            # Calculate the model-based ADTTP
            adttp_model = rollout_tt_with_genttp(
                model=model,
                device=device,
                seed_q_tn=q_seed_tn,
                candidate_a6=best_a6_cand,
                seq_length_q=args.seq_length_q,
                seq_length_a=args.seq_length_a,
                delta_t=args.delta_t,
            )
            
            # Calculate penalties using the updated candidate
            # Note: We re-calculate scores on the projected A6 candidate
            score = score_a6(
                best_a6_cand,
                weights=weights,
                delta_t=args.delta_t,
                target_od=target_od,
                origins=origins,
                destinations=destinations,
                total_agents_by_t=total_by_t,
                m=args.m,
            )
            history.append({
                "iteration": i,
                "objective": objective,
                "adttp": score.adttp,
                "g1_od": score.g1,
                "g2_admissibility": score.g2
            })
            # Combine the model-predicted ADTTP with the structural penalties
            objective = float(adttp_model + weights.od * score.g1 + weights.admissibility * score.g2)
            
            # Update best_score if this candidate is superior
            if objective < best_score:
                best_score = objective
                best_name = f"gradient_step_{i:04d}.npy"
                best_a6 = best_a6_cand.copy()    
    # 7. Final output saving logic...
    pd.DataFrame(history).to_csv(out_dir / "optimization_history.csv")
    print("Gradient search complete.")

if __name__ == "__main__":
    main()
