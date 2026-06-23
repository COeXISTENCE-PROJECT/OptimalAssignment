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


def adttp_from_a6(a6: np.ndarray, delta_t: float = 1.0) -> float:
    """Default ADTTP on Representation 6: sum_t t * sum_i A6[i,t]."""
    if a6.ndim == 1:
        a6 = a6[:, np.newaxis]

    _, t_count = a6.shape
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
) -> Score:
    """Score one projected A6 candidate consistently."""
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

    g1 = g1_od_penalty(
        candidate_a6,
        target_od=target_od,
        origins=origins,
        destinations=destinations,
    )
    g2 = g2_admissibility_penalty(
        candidate_a6,
        total_agents_by_t=total_agents_by_t,
        m=m,
    )

    objective = float(adttp + weights.od * g1 + weights.admissibility * g2)
    return Score(objective=objective, adttp=float(adttp), g1=float(g1), g2=float(g2))


def objective_a1(
    a1_relaxed: np.ndarray,
    weights: ObjectiveWeights,
    delta_t: float = 1.0,
    target_od: np.ndarray | None = None,
    origins: Iterable[int] | None = None,
    destinations: Iterable[int] | None = None,
    total_agents_by_t: np.ndarray | None = None,
    m: int | None = None,
    projection_threshold: float = 0.5,
) -> float:
    """Non-differentiable projected objective retained for debugging."""
    a1_discrete = project_relaxed_a1_to_a1(
        a1_relaxed,
        threshold=projection_threshold,
    )
    a6 = a1_to_a6(a1_discrete)
    return score_a6(
        a6,
        weights=weights,
        delta_t=delta_t,
        target_od=target_od,
        origins=origins,
        destinations=destinations,
        total_agents_by_t=total_agents_by_t,
        m=m,
    ).objective


# ============================================================
# Gradient search in relaxed A1 space
# ============================================================


def initialize_search_a1(shape, rng: np.random.Generator | None = None) -> np.ndarray:
    """Initialize latent relaxed A1 values."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0.0, 0.1, shape).astype(np.float32)


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




# ============================================================
# Assignment autoencoder and latent-space optimisation
# ============================================================


class AssignmentAutoencoder(torch.nn.Module):
    """MLP autoencoder fd(fe(A6)) for assignment count snapshots.

    Input/output tensors use A6 shape (N, T). The decoder output is continuous
    A6; hard rounding/projection is applied only for scoring/saving candidates.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        latent_dim: int = 32,
        hidden_dims: Sequence[int] = (512, 256),
        output_activation: str = "sigmoid",
        output_scale: float = 1.0,
    ):
        super().__init__()
        if torch is None:
            raise RuntimeError("PyTorch is required for AssignmentAutoencoder")

        self.input_shape = tuple(int(x) for x in input_shape)
        self.input_dim = int(np.prod(self.input_shape))
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.output_activation = output_activation
        self.output_scale = float(output_scale)

        encoder_layers: list[torch.nn.Module] = []
        last = self.input_dim
        for h in self.hidden_dims:
            encoder_layers.append(torch.nn.Linear(last, h))
            encoder_layers.append(torch.nn.ReLU())
            last = h
        encoder_layers.append(torch.nn.Linear(last, self.latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers: list[torch.nn.Module] = []
        last = self.latent_dim
        for h in reversed(self.hidden_dims):
            decoder_layers.append(torch.nn.Linear(last, h))
            decoder_layers.append(torch.nn.ReLU())
            last = h
        decoder_layers.append(torch.nn.Linear(last, self.input_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def encode(self, a6: torch.Tensor) -> torch.Tensor:
        if a6.dim() == 2:
            a6 = a6.unsqueeze(0)
        return self.encoder(a6.reshape(a6.shape[0], -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        raw = self.decoder(z)
        if self.output_activation == "sigmoid":
            out = torch.sigmoid(raw) * self.output_scale
        elif self.output_activation == "softplus":
            out = torch.nn.functional.softplus(raw)
        elif self.output_activation == "relu":
            out = torch.relu(raw)
        elif self.output_activation == "identity":
            out = raw
        else:
            raise ValueError(f"Unknown output activation: {self.output_activation}")
        return out.reshape(z.shape[0], *self.input_shape)

    def forward(self, a6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(a6)
        return self.decode(z), z


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    if text is None or str(text).strip() == "":
        return ()
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def list_autoencoder_training_files(
    ae_manifest: str | None,
    ae_train_dir: str | None,
    candidate_dir: str | None = None,
) -> list[Path]:
    files: list[Path] = []

    if ae_manifest is not None:
        manifest_path = Path(ae_manifest)
        df = pd.read_csv(manifest_path)
        if "assignment_path" not in df.columns:
            raise ValueError("--ae_manifest must contain an assignment_path column")
        files.extend(Path(x) for x in df["assignment_path"].astype(str).tolist())

    train_dir = ae_train_dir or candidate_dir
    if train_dir is not None:
        files.extend(sorted(Path(train_dir).glob("*.npy")))

    # Preserve order but remove duplicates.
    seen: set[str] = set()
    unique: list[Path] = []
    for f in files:
        key = str(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)

    if not unique:
        raise ValueError(
            "Autoencoder training requires --ae_manifest, --ae_train_dir, "
            "or --candidate_dir with .npy assignments."
        )
    return unique


def load_a6_stack(
    paths: Sequence[Path],
    num_nodes: int | None,
    target_timesteps: int | None = None,
    min_timesteps: int | None = None,
    pad_value: float = 0.0,
    allow_crop: bool = False,
) -> np.ndarray:
    """Load many assignments as a fixed-shape stack (B,N,T).

    SUMO episodes often have different lengths. The autoencoder still requires
    one fixed tensor shape, so this loader pads the time axis to a shared target
    length. By default the target is max(T over training files, min_timesteps).
    """
    loaded: list[tuple[Path, np.ndarray]] = []
    n_expected: int | None = None
    max_t = int(min_timesteps or 0)

    total_paths = len(paths)
    for idx, path in enumerate(paths, start=1):
        arr = load_assignment_as_a6(path, num_nodes=num_nodes)
        if n_expected is None:
            n_expected = int(arr.shape[0])
        elif int(arr.shape[0]) != n_expected:
            raise ValueError(
                f"All AE training assignments must have same number of nodes. "
                f"Expected N={n_expected}, got N={arr.shape[0]} for {path}"
            )
        max_t = max(max_t, int(arr.shape[1]))
        loaded.append((path, arr.astype(np.float32)))
        if idx == 1 or idx % 100 == 0 or idx == total_paths:
            print(
                f"Loaded AE assignment {idx}/{total_paths}: {path.name} "
                f"shape={tuple(arr.shape)}, current max T={max_t}",
                flush=True,
            )

    if not loaded:
        raise ValueError("No autoencoder training assignments were loaded")

    common_t = int(target_timesteps) if target_timesteps is not None else max_t
    arrays: list[np.ndarray] = []
    original_lengths: list[int] = []
    for path, arr in loaded:
        original_lengths.append(int(arr.shape[1]))
        arrays.append(
            pad_or_crop_a6_time(
                arr,
                target_timesteps=common_t,
                pad_value=pad_value,
                name=str(path),
                allow_crop=allow_crop,
            )
        )

    if len(set(original_lengths)) > 1:
        print(
            "AE training assignments have variable simulation lengths: "
            f"min T={min(original_lengths)}, max T={max(original_lengths)}. "
            f"Padding all to T={common_t}.",
            flush=True,
        )
    else:
        print(f"AE training assignments all have T={common_t}.", flush=True)

    return np.stack(arrays, axis=0).astype(np.float32)


class AssignmentA6Dataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def make_time_reconstruction_weights(
    t_count: int,
    power: float,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Later timesteps receive larger reconstruction weight, as in the PDF note."""
    base = torch.linspace(1.0 / max(t_count, 1), 1.0, t_count, device=device, dtype=dtype)
    weights = torch.pow(base, float(power))
    return weights / torch.mean(weights)


def weighted_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    time_weights: torch.Tensor,
) -> torch.Tensor:
    # pred/target: (B, N, T), time_weights: (T,)
    sq = (pred - target) ** 2
    return torch.mean(sq * time_weights[None, None, :])


def save_autoencoder_checkpoint(
    path: str | Path,
    autoencoder: AssignmentAutoencoder,
    optimizer_state: dict | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": autoencoder.state_dict(),
        "config": {
            "input_shape": list(autoencoder.input_shape),
            "latent_dim": autoencoder.latent_dim,
            "hidden_dims": list(autoencoder.hidden_dims),
            "output_activation": autoencoder.output_activation,
            "output_scale": autoencoder.output_scale,
        },
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    if extra is not None:
        checkpoint["extra"] = extra
    torch.save(checkpoint, path)


def load_autoencoder_checkpoint(path: str | Path, device) -> AssignmentAutoencoder:
    if torch is None:
        raise RuntimeError("PyTorch is required for autoencoder loading")
    checkpoint = torch.load(path, map_location=device)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Autoencoder checkpoint must contain 'config' and 'model_state_dict'. "
            "Train it with --train_autoencoder using this script."
        )
    cfg = checkpoint["config"]
    autoencoder = AssignmentAutoencoder(
        input_shape=tuple(cfg["input_shape"]),
        latent_dim=int(cfg["latent_dim"]),
        hidden_dims=tuple(cfg.get("hidden_dims", [])),
        output_activation=cfg.get("output_activation", "sigmoid"),
        output_scale=float(cfg.get("output_scale", 1.0)),
    ).to(device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()
    return autoencoder


def train_assignment_autoencoder(
    training_a6: np.ndarray,
    output_path: str | Path,
    latent_dim: int,
    hidden_dims: Sequence[int],
    output_activation: str,
    output_scale: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    time_weight_power: float,
    device,
    seed: int = 42,
    history_csv_path: str | Path | None = None,
) -> tuple[AssignmentAutoencoder, list[dict]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for autoencoder training")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = np.asarray(training_a6, dtype=np.float32)
    input_shape = tuple(x.shape[1:])
    model = AssignmentAutoencoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        output_activation=output_activation,
        output_scale=output_scale,
    ).to(device)

    dataset = AssignmentA6Dataset(x)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    time_weights = make_time_reconstruction_weights(
        input_shape[1],
        power=time_weight_power,
        device=device,
    )

    if history_csv_path is not None:
        history_csv_path = Path(history_csv_path)
        history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["epoch", "reconstruction_loss"]).to_csv(history_csv_path, index=False)

    num_params = sum(p.numel() for p in model.parameters())
    print(
        "Starting AE training: "
        f"samples={int(x.shape[0])}, input_shape={input_shape}, "
        f"latent_dim={latent_dim}, hidden_dims={list(hidden_dims)}, "
        f"batch_size={batch_size}, epochs={epochs}, parameters={num_params:,}",
        flush=True,
    )

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, z = model(batch)
            loss = weighted_reconstruction_loss(recon, batch, time_weights)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu().item()) * int(batch.shape[0])
            total_count += int(batch.shape[0])

        mean_loss = total_loss / max(total_count, 1)
        row = {"epoch": epoch, "reconstruction_loss": mean_loss}
        history.append(row)
        if history_csv_path is not None:
            pd.DataFrame([row]).to_csv(
                history_csv_path,
                mode="a",
                header=False,
                index=False,
            )
        print(f"AE epoch {epoch:04d}: reconstruction_loss={mean_loss:.6f}", flush=True)

    save_autoencoder_checkpoint(
        output_path,
        model,
        optimizer_state=opt.state_dict(),
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "time_weight_power": time_weight_power,
            "num_training_samples": int(x.shape[0]),
        },
    )
    model.eval()
    return model, history


def differentiable_genttp_total_time(
    model,
    device,
    seed_q_tn: np.ndarray,
    candidate_a6: torch.Tensor,
    seq_length_q: int,
    seq_length_a: int,
    delta_t: float,
) -> torch.Tensor:
    """Differentiable GenTTP rollout wrt candidate_a6.

    candidate_a6 shape is (N,T). seed_q_tn shape is (T,N). Model parameters are
    expected to be frozen by the caller; gradients flow to candidate_a6/z.
    """
    if seed_q_tn is None:
        raise ValueError("seed_q_tn is required for differentiable GenTTP objective")

    assign_tn = candidate_a6.T.to(device)
    real_q_tn = torch.as_tensor(seed_q_tn, dtype=assign_tn.dtype, device=device)

    if tuple(assign_tn.shape) != tuple(real_q_tn.shape):
        raise ValueError(
            f"q and assignment shapes must match as (T,N): "
            f"{tuple(real_q_tn.shape)} vs {tuple(assign_tn.shape)}"
        )

    seed_steps = max(seq_length_q, seq_length_a)
    t_count, n_nodes = real_q_tn.shape
    if t_count <= seed_steps:
        raise ValueError(f"Too few timesteps: T={t_count}, seed_steps={seed_steps}")

    generated: list[torch.Tensor] = [real_q_tn[t] for t in range(seed_steps)]

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


def differentiable_objective_a6_torch(
    a6: torch.Tensor,
    weights: ObjectiveWeights,
    delta_t: float = 1.0,
    target_od: np.ndarray | None = None,
    origins: Iterable[int] | None = None,
    destinations: Iterable[int] | None = None,
    total_agents_by_t: np.ndarray | None = None,
    m: int | None = None,
    latent_z: torch.Tensor | None = None,
    latent_reference: torch.Tensor | None = None,
    latent_l2_weight: float = 0.0,
    genttp_model: torch.nn.Module | None = None,
    genttp_device=None,
    seed_q_tn: np.ndarray | None = None,
    seq_length_q: int = 15,
    seq_length_a: int = 30,
    use_genttp_gradient: bool = False,
) -> torch.Tensor:
    """Differentiable objective on decoded continuous A6."""
    if a6.dim() == 3:
        a6 = a6.squeeze(0)
    dtype = a6.dtype
    device = a6.device
    t_count = a6.shape[1]

    if use_genttp_gradient and genttp_model is not None:
        adttp = differentiable_genttp_total_time(
            model=genttp_model,
            device=genttp_device or device,
            seed_q_tn=seed_q_tn,
            candidate_a6=a6,
            seq_length_q=seq_length_q,
            seq_length_a=seq_length_a,
            delta_t=delta_t,
        )
    else:
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

    # Encourage decoded continuous counts to become integer-like without making
    # the whole objective non-differentiable. round(.) is detached intentionally.
    g2 = g2 + torch.sum(torch.abs(a6 - torch.round(a6.detach())))

    if total_agents_by_t is not None:
        total = torch.as_tensor(total_agents_by_t, dtype=dtype, device=device)
        g2 = g2 + torch.sum(torch.abs(torch.sum(a6, dim=0) - total))

    if m is not None:
        g2 = g2 + torch.relu(-a6).sum()
        g2 = g2 + torch.relu(a6 - float(m)).sum()

    latent_penalty = torch.zeros((), dtype=dtype, device=device)
    if latent_z is not None and latent_reference is not None and latent_l2_weight > 0.0:
        latent_penalty = float(latent_l2_weight) * torch.sum((latent_z - latent_reference) ** 2)

    return adttp + weights.od * g1 + weights.admissibility * g2 + latent_penalty


def project_a6_counts(
    a6: np.ndarray,
    m: int | None = None,
    total_agents_by_t: np.ndarray | None = None,
) -> np.ndarray:
    """Round decoded continuous A6 into integer count representation [m]_0^{N x T}."""
    raw = np.asarray(a6, dtype=np.float32)
    projected = np.rint(raw).astype(np.float32)
    projected = np.maximum(projected, 0.0)
    if m is not None:
        projected = np.minimum(projected, float(m))

    if total_agents_by_t is not None:
        target = np.rint(total_agents_by_t).astype(int)
        n_nodes, t_count = projected.shape
        for t in range(t_count):
            target_t = int(max(0, target[t]))
            current_t = int(np.sum(projected[:, t]))

            # Greedily repair column sum using the decoded fractional values as priorities.
            guard = 0
            while current_t < target_t and guard < n_nodes * max(target_t + 1, 1):
                capacity = np.full(n_nodes, np.inf, dtype=np.float32)
                if m is not None:
                    capacity = float(m) - projected[:, t]
                candidates = np.where(capacity > 0)[0]
                if len(candidates) == 0:
                    break
                residual = raw[candidates, t] - projected[candidates, t]
                i = int(candidates[np.argmax(residual)])
                projected[i, t] += 1.0
                current_t += 1
                guard += 1

            guard = 0
            while current_t > target_t and guard < n_nodes * max(current_t + 1, 1):
                candidates = np.where(projected[:, t] > 0)[0]
                if len(candidates) == 0:
                    break
                residual = projected[candidates, t] - raw[candidates, t]
                i = int(candidates[np.argmax(residual)])
                projected[i, t] -= 1.0
                current_t -= 1
                guard += 1

    return projected.astype(np.float32)


def perform_latent_search(
    autoencoder: AssignmentAutoencoder,
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
    log_every: int = 1,
    latent_init: str = "encode_base",
    latent_restarts: int = 1,
    latent_init_noise: float = 0.1,
    latent_l2_weight: float = 0.0,
    use_genttp_gradient: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Score, list[dict], str, np.ndarray, np.ndarray]:
    """Optimise z in R^k, decode A6 = fd(z), then score projected assignments."""
    if torch is None:
        raise RuntimeError("PyTorch is required for latent search")

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    log_every = max(1, int(log_every))
    latent_restarts = max(1, int(latent_restarts))

    if rng is None:
        rng = np.random.default_rng()

    if device is None:
        optim_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        optim_device = device
    else:
        optim_device = torch.device(device)

    autoencoder = autoencoder.to(optim_device)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    if model is not None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    base_t = torch.as_tensor(base_a6, dtype=torch.float32, device=optim_device)
    with torch.no_grad():
        encoded_base = autoencoder.encode(base_t).squeeze(0)

    origins = list(origins or [])
    destinations = list(destinations or [])

    best_score = Score(math.inf, math.inf, math.inf, math.inf)
    best_a6: np.ndarray | None = None
    best_soft_a6: np.ndarray | None = None
    best_z: np.ndarray | None = None
    best_name = "latent_restart_000_step_0000.npy"
    history: list[dict] = []

    for restart in range(latent_restarts):
        if latent_init == "encode_base":
            z0 = encoded_base.detach().clone()
            if restart > 0 or latent_init_noise > 0:
                noise = torch.as_tensor(
                    rng.normal(0.0, latent_init_noise, size=tuple(z0.shape)),
                    dtype=z0.dtype,
                    device=optim_device,
                )
                z0 = z0 + noise
        elif latent_init == "random":
            z0 = torch.as_tensor(
                rng.normal(0.0, 1.0, size=(autoencoder.latent_dim,)),
                dtype=torch.float32,
                device=optim_device,
            )
        else:
            raise ValueError(f"Unknown latent_init: {latent_init}")

        z_reference = z0.detach().clone()
        z = z0.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=learning_rate)

        for iteration in range(iterations):
            optimizer.zero_grad()
            decoded = autoencoder.decode(z).squeeze(0)

            surrogate_loss = differentiable_objective_a6_torch(
                decoded,
                weights=weights,
                delta_t=delta_t,
                target_od=target_od,
                origins=origins,
                destinations=destinations,
                total_agents_by_t=total_agents_by_t,
                m=m,
                latent_z=z,
                latent_reference=z_reference if latent_l2_weight > 0 else None,
                latent_l2_weight=latent_l2_weight,
                genttp_model=model,
                genttp_device=device,
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
                soft_a6 = autoencoder.decode(z).squeeze(0).detach().cpu().numpy()
            candidate_a6 = project_a6_counts(
                soft_a6,
                m=m,
                total_agents_by_t=total_agents_by_t,
            )

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
                best_soft_a6 = soft_a6.copy()
                best_z = z.detach().cpu().numpy().copy()
                best_name = f"latent_restart_{restart:03d}_step_{iteration:04d}.npy"

            history.append({
                "search_space": "latent",
                "restart": restart,
                "iteration": iteration,
                "surrogate_loss": float(surrogate_loss.detach().cpu().item()),
                "objective": score.objective,
                "adttp": score.adttp,
                "g1_od": score.g1,
                "g2_admissibility": score.g2,
                "latent_norm": float(torch.linalg.norm(z.detach()).cpu().item()),
                "best_objective": best_score.objective,
                "improved": bool(improved),
            })

            print(
                f"Latent restart {restart:03d}, iter {iteration:04d}: "
                f"surrogate={float(surrogate_loss.detach().cpu().item()):.4f}, "
                f"objective={score.objective:.4f}, "
                f"adttp={score.adttp:.4f}, "
                f"g1={score.g1:.4f}, "
                f"g2={score.g2:.4f}, "
                f"best={best_score.objective:.4f}"
            )

    if best_a6 is None or best_soft_a6 is None or best_z is None:
        with torch.no_grad():
            z = encoded_base.detach().clone()
            best_soft_a6 = autoencoder.decode(z).squeeze(0).detach().cpu().numpy()
            best_z = z.detach().cpu().numpy()
        best_a6 = project_a6_counts(best_soft_a6, m=m, total_agents_by_t=total_agents_by_t)
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

    return best_a6, best_score, history, best_name, best_z, best_soft_a6

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


def generate_local_grid_a6(
    base: np.ndarray,
    nodes: list[int],
    timesteps: list[int],
    values: list[int],
    m: int | None,
    max_candidates: int,
):
    """Enumerate a local Cartesian grid by replacing selected A6 entries."""
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


def load_candidates_from_dir(
    candidate_dir: str | Path,
    num_nodes: int | None,
) -> list[tuple[str, np.ndarray]]:
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
    parser.add_argument(
        "--candidate_dir",
        type=str,
        default=None,
        help="Optional dir with candidate .npy assignments. Kept for compatibility; not used by gradient mode.",
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
        "--nodes",
        type=str,
        default="",
        help="Comma/range list, e.g. '0,3,10:20'. Kept for compatibility.",
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        default="",
        help="Comma/range list. Kept for compatibility.",
    )
    parser.add_argument(
        "--values",
        type=str,
        default="0,1",
        help="Grid values for selected entries. Kept for compatibility.",
    )
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

    # Search mode. "a1" keeps the repaired Step-2 relaxed-A1 optimizer.
    # "latent" implements the PDF idea: optimise z in R^k and decode with fd(z).
    parser.add_argument(
        "--search_space",
        type=str,
        default="a1",
        choices=["a1", "latent"],
        help="Optimisation space: repaired relaxed A1, or autoencoder latent z in R^k.",
    )

    # Assignment autoencoder arguments.
    parser.add_argument("--train_autoencoder", action="store_true")
    parser.add_argument("--only_train_autoencoder", action="store_true")
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument(
        "--ae_train_dir",
        type=str,
        default=None,
        help="Directory of .npy A6 assignments used to train the autoencoder.",
    )
    parser.add_argument(
        "--ae_manifest",
        type=str,
        default=None,
        help="CSV manifest with an assignment_path column used to train the autoencoder.",
    )
    parser.add_argument(
        "--ae_target_timesteps",
        type=int,
        default=None,
        help=(
            "Fixed T for AE input/output. If omitted, uses max T across AE training files, "
            "base_assignment, and q_seed when present."
        ),
    )
    parser.add_argument(
        "--ae_pad_value",
        type=float,
        default=0.0,
        help="Value used to pad shorter A6 matrices along the time axis.",
    )
    parser.add_argument(
        "--ae_allow_crop",
        action="store_true",
        help="Allow cropping assignments longer than --ae_target_timesteps. Disabled by default.",
    )
    parser.add_argument("--ae_epochs", type=int, default=100)
    parser.add_argument("--ae_batch_size", type=int, default=64)
    parser.add_argument("--ae_latent_dim", type=int, default=32)
    parser.add_argument("--ae_hidden_dims", type=str, default="512,256")
    parser.add_argument("--ae_learning_rate", type=float, default=1e-3)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--ae_output_activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "softplus", "relu", "identity"],
    )
    parser.add_argument(
        "--ae_output_scale",
        type=float,
        default=None,
        help="Scale for sigmoid decoder output. Defaults to m or max training/base value.",
    )
    parser.add_argument(
        "--ae_time_weight_power",
        type=float,
        default=1.0,
        help="Power for later-timestep weighted reconstruction loss.",
    )
    parser.add_argument(
        "--latent_init",
        type=str,
        default="encode_base",
        choices=["encode_base", "random"],
    )
    parser.add_argument("--latent_restarts", type=int, default=1)
    parser.add_argument("--latent_init_noise", type=float, default=0.1)
    parser.add_argument("--latent_l2_weight", type=float, default=0.0)
    parser.add_argument(
        "--latent_use_genttp_gradient",
        action="store_true",
        help="Backpropagate through GenTTP during latent optimisation. Otherwise GenTTP is evaluation-only.",
    )

    parser.add_argument("--od_weight", type=float, default=1_000.0)
    parser.add_argument("--admissibility_weight", type=float, default=1_000.0)
    parser.add_argument("--origins", type=str, default="")
    parser.add_argument("--destinations", type=str, default="")
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

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup base configuration.
    base_a6 = load_assignment_as_a6(args.base_assignment, num_nodes=args.num_nodes)
    if args.num_nodes is None:
        args.num_nodes = int(base_a6.shape[0])

    weights = ObjectiveWeights(
        od=args.od_weight,
        admissibility=args.admissibility_weight,
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
    ae_history: list[dict] | None = None

    # 3. Optional train/load autoencoder for latent mode.
    if args.train_autoencoder:
        ae_files = list_autoencoder_training_files(
            ae_manifest=args.ae_manifest,
            ae_train_dir=args.ae_train_dir,
            candidate_dir=args.candidate_dir,
        )
        print(f"Loading {len(ae_files)} assignments for autoencoder training...", flush=True)
        min_ae_timesteps = int(base_a6.shape[1])
        if q_seed_a6 is not None:
            min_ae_timesteps = max(min_ae_timesteps, int(q_seed_a6.shape[1]))
        training_a6 = load_a6_stack(
            ae_files,
            num_nodes=args.num_nodes,
            target_timesteps=args.ae_target_timesteps,
            min_timesteps=min_ae_timesteps,
            pad_value=args.ae_pad_value,
            allow_crop=args.ae_allow_crop,
        )

        ae_input_shape = tuple(training_a6.shape[1:])
        if int(ae_input_shape[0]) != int(base_a6.shape[0]):
            raise ValueError(
                f"AE training N={ae_input_shape[0]} does not match base N={base_a6.shape[0]}"
            )
        if int(base_a6.shape[1]) != int(ae_input_shape[1]):
            print(
                f"Padding base assignment from T={base_a6.shape[1]} to AE T={ae_input_shape[1]}.",
                flush=True,
            )
            base_a6 = pad_or_crop_a6_time(
                base_a6,
                target_timesteps=int(ae_input_shape[1]),
                pad_value=args.ae_pad_value,
                name="base_assignment",
                allow_crop=args.ae_allow_crop,
            )

        ae_output_scale = args.ae_output_scale
        if ae_output_scale is None:
            ae_output_scale = float(args.m or max(np.nanmax(training_a6), np.nanmax(base_a6), 1.0))

        autoencoder_path = Path(args.ae_checkpoint) if args.ae_checkpoint else out_dir / "assignment_autoencoder.pt"
        ae_history_path = out_dir / "autoencoder_training_history.csv"
        autoencoder, ae_history = train_assignment_autoencoder(
            training_a6=training_a6,
            output_path=autoencoder_path,
            latent_dim=args.ae_latent_dim,
            hidden_dims=parse_hidden_dims(args.ae_hidden_dims),
            output_activation=args.ae_output_activation,
            output_scale=ae_output_scale,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            learning_rate=args.ae_learning_rate,
            weight_decay=args.ae_weight_decay,
            time_weight_power=args.ae_time_weight_power,
            device=device,
            seed=args.seed,
            history_csv_path=ae_history_path,
        )
        pd.DataFrame(ae_history).to_csv(ae_history_path, index=False)
        print(f"Autoencoder saved to: {autoencoder_path}", flush=True)
        print(f"Autoencoder history saved to: {ae_history_path}", flush=True)

        if args.only_train_autoencoder:
            return

    if args.search_space == "latent" and autoencoder is None:
        if args.ae_checkpoint is None:
            raise ValueError(
                "--search_space latent requires --ae_checkpoint or --train_autoencoder."
            )
        autoencoder_path = Path(args.ae_checkpoint)
        autoencoder = load_autoencoder_checkpoint(autoencoder_path, device=device)

        if int(autoencoder.input_shape[0]) != int(base_a6.shape[0]):
            raise ValueError(
                f"AE checkpoint N={autoencoder.input_shape[0]} does not match "
                f"base assignment N={base_a6.shape[0]}"
            )
        if int(autoencoder.input_shape[1]) != int(base_a6.shape[1]):
            print(
                f"Padding base assignment from T={base_a6.shape[1]} "
                f"to AE checkpoint T={autoencoder.input_shape[1]}."
            )
            base_a6 = pad_or_crop_a6_time(
                base_a6,
                target_timesteps=int(autoencoder.input_shape[1]),
                pad_value=args.ae_pad_value,
                name="base_assignment",
                allow_crop=args.ae_allow_crop,
            )

    if q_seed_a6 is not None and int(q_seed_a6.shape[1]) != int(base_a6.shape[1]):
        print(f"Padding q_seed from T={q_seed_a6.shape[1]} to T={base_a6.shape[1]}.")
        q_seed_a6 = pad_or_crop_a6_time(
            q_seed_a6,
            target_timesteps=int(base_a6.shape[1]),
            pad_value=args.ae_pad_value,
            name="q_seed",
            allow_crop=args.ae_allow_crop,
        )
        q_seed_tn = q_seed_a6.T.astype(np.float32)

    total_by_t = np.sum(base_a6, axis=0) if args.preserve_total_by_t else None

    # 4. Optimise either repaired relaxed A1 or autoencoder latent z.
    best_z = None
    best_soft_a6 = None

    if args.search_space == "latent":
        best_a6, best_score, history, best_name, best_z, best_soft_a6 = perform_latent_search(
            autoencoder=autoencoder,
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
            log_every=args.log_every,
            latent_init=args.latent_init,
            latent_restarts=args.latent_restarts,
            latent_init_noise=args.latent_init_noise,
            latent_l2_weight=args.latent_l2_weight,
            use_genttp_gradient=args.latent_use_genttp_gradient,
            rng=rng,
        )
    else:
        initial_p = initialize_search_a1_from_a6(
            base_a6,
            m=m_latent,
            threshold=args.projection_threshold,
            rng=rng,
        )

        best_a6, best_score, history, best_name = perform_gradient_search_a1(
            initial_a1=initial_p,
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
            projection_threshold=args.projection_threshold,
            temperature=args.temperature,
            log_every=args.log_every,
        )

    # 5. Save outputs.
    best_path = out_dir / "best_assignment_a6.npy"
    save_a6(best_path, best_a6)

    if best_z is not None:
        np.save(out_dir / "best_latent_z.npy", np.asarray(best_z, dtype=np.float32))
    if best_soft_a6 is not None:
        np.save(out_dir / "best_decoded_soft_a6.npy", np.asarray(best_soft_a6, dtype=np.float32))

    history_path = out_dir / "optimization_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        "search_space": args.search_space,
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
        "ae_pad_value": float(args.ae_pad_value),
        "ae_allow_crop": bool(args.ae_allow_crop),
        "ae_checkpoint": str(autoencoder_path) if autoencoder_path is not None else None,
        "latent_restarts": args.latent_restarts if args.search_space == "latent" else None,
        "latent_init": args.latent_init if args.search_space == "latent" else None,
        "latent_use_genttp_gradient": bool(args.latent_use_genttp_gradient),
    }

    summary_path = out_dir / "optimization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Gradient search complete.")
    print(f"Search space: {args.search_space}")
    print(f"Best objective: {best_score.objective:.4f}")
    print(f"Best assignment saved to: {best_path}")
    if int(base_a6.shape[1]) != original_base_timesteps:
        print(f"Effective AE/optimization horizon: T={base_a6.shape[1]} (base was T={original_base_timesteps})")
    if best_z is not None:
        print(f"Best latent vector saved to: {out_dir / 'best_latent_z.npy'}")
        print(f"Best soft decoded A6 saved to: {out_dir / 'best_decoded_soft_a6.npy'}")
    print(f"History saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
