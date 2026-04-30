#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Split plików — taka sama logika jak w inference.py
# ============================================================

def split_file_names(q_dir, a_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    q_dir = Path(q_dir)
    a_dir = Path(a_dir)

    q_files = {p.name for p in q_dir.glob("*.npy")}
    a_files = {p.name for p in a_dir.glob("*.npy")}
    common_files = sorted(q_files & a_files)

    if not common_files:
        raise RuntimeError(f"Brak wspólnych plików .npy między {q_dir} i {a_dir}")

    rng = random.Random(seed)
    rng.shuffle(common_files)

    n = len(common_files)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_files = common_files[:n_train]
    val_files = common_files[n_train:n_train + n_val]
    test_files = common_files[n_train + n_val:]

    return train_files, val_files, test_files


# ============================================================
# Helpery
# ============================================================

def safe_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def parse_indices(text: Optional[str]) -> list[int]:
    """
    Obsługuje:
      "0,1,5"
      "0:10"
      "0:20:2"
      "0,3:8,20"
    """
    if text is None or str(text).strip() == "":
        return []

    out = []

    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"Niepoprawny zakres indeksów: {part}")

            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1

            out.extend(list(range(start, stop, step)))
        else:
            out.append(int(part))

    seen = set()
    unique = []

    for i in out:
        if i not in seen:
            unique.append(i)
            seen.add(i)

    return unique


def read_file_names_file(path: str | Path) -> list[str]:
    names = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            names.append(Path(line).name)

    return names


def find_single(pattern: str, run_dir: Path) -> Path:
    matches = sorted(run_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"Nie znaleziono pliku pasującego do {pattern} w {run_dir}")

    if len(matches) > 1:
        print(f"[INFO] Dla wzorca {pattern} znaleziono kilka plików, biorę pierwszy: {matches[0].name}")

    return matches[0]


def load_seed_steps(run_dir: Path) -> Optional[int]:
    matches = sorted(run_dir.glob("*_metrics.csv"))

    if not matches:
        return None

    df = pd.read_csv(matches[0])

    if df.empty:
        return None

    if "seed_steps" not in df.columns:
        return None

    value = df.iloc[0]["seed_steps"]

    if pd.isna(value):
        return None

    return int(value)


# ============================================================
# TT i błędy
# ============================================================

def eval_slice(pred: np.ndarray, real: np.ndarray, seed_steps: Optional[int]):
    if seed_steps is None:
        return pred, real

    return pred[seed_steps:], real[seed_steps:]


def compute_tt_stats(
    pred: np.ndarray,
    real: np.ndarray,
    delta_t: float = 10.0,
    seed_steps: Optional[int] = None,
):
    """
    Zgodne z inference_visual.py:

    TT total:
        delta_t * sum(q) po całym eval horizon i wszystkich nodach

    TT per timestep:
        delta_t * sum(abs(q_t)) po nodach

    Jeśli q >= 0, total i suma per_t są równoważne.
    """
    pred_eval, real_eval = eval_slice(pred, real, seed_steps)

    tt_real = float(delta_t * np.sum(real_eval))
    tt_pred = float(delta_t * np.sum(pred_eval))

    tt_signed_diff = float(tt_pred - tt_real)
    tt_abs_diff = float(abs(tt_signed_diff))
    tt_rel_diff = float(tt_abs_diff / tt_real) if tt_real != 0 else np.nan

    tt_real_per_t = delta_t * np.sum(np.abs(real_eval), axis=1)
    tt_pred_per_t = delta_t * np.sum(np.abs(pred_eval), axis=1)

    tt_real_cum = np.cumsum(tt_real_per_t)
    tt_pred_cum = np.cumsum(tt_pred_per_t)

    return {
        "tt_real": tt_real,
        "tt_pred": tt_pred,
        "tt_signed_diff": tt_signed_diff,
        "tt_abs_diff": tt_abs_diff,
        "tt_rel_diff": tt_rel_diff,
        "tt_real_per_t": tt_real_per_t,
        "tt_pred_per_t": tt_pred_per_t,
        "tt_real_cum": tt_real_cum,
        "tt_pred_cum": tt_pred_cum,
    }


def compute_eval_errors(pred: np.ndarray, real: np.ndarray, seed_steps: Optional[int]):
    pred_eval, real_eval = eval_slice(pred, real, seed_steps)

    err = pred_eval - real_eval
    abs_err = np.abs(err)
    sq_err = err ** 2

    return {
        "n_values": int(abs_err.size),
        "sum_abs_err": float(np.sum(abs_err)),
        "sum_sq_err": float(np.sum(sq_err)),
        "mae": float(np.mean(abs_err)) if abs_err.size else np.nan,
        "rmse": float(np.sqrt(np.mean(sq_err))) if sq_err.size else np.nan,
        "per_node_sum_abs_err": np.sum(abs_err, axis=0),
        "per_node_sum_sq_err": np.sum(sq_err, axis=0),
        "per_node_count": np.full(pred_eval.shape[1], pred_eval.shape[0], dtype=np.int64),
        "per_node_real_sum": np.sum(real_eval, axis=0),
        "per_node_pred_sum": np.sum(pred_eval, axis=0),
        "per_node_has_signal": np.any(np.abs(real_eval) > 1e-12, axis=0),
    }


@dataclass
class AggregateState:
    delta_t: float

    files_ok: int = 0
    files_failed: int = 0

    total_values: int = 0
    total_abs_err: float = 0.0
    total_sq_err: float = 0.0

    total_tt_real: float = 0.0
    total_tt_pred: float = 0.0
    total_tt_abs_diff_filewise: float = 0.0

    max_eval_horizon: int = 0

    tt_real_per_t_sum: Optional[np.ndarray] = None
    tt_pred_per_t_sum: Optional[np.ndarray] = None
    tt_per_t_count: Optional[np.ndarray] = None

    per_node_sum_abs_err: Optional[np.ndarray] = None
    per_node_sum_sq_err: Optional[np.ndarray] = None
    per_node_count: Optional[np.ndarray] = None
    per_node_real_sum: Optional[np.ndarray] = None
    per_node_pred_sum: Optional[np.ndarray] = None
    per_node_has_signal: Optional[np.ndarray] = None

    def _ensure_time_capacity(self, horizon: int):
        if self.tt_real_per_t_sum is None:
            self.tt_real_per_t_sum = np.zeros(horizon, dtype=np.float64)
            self.tt_pred_per_t_sum = np.zeros(horizon, dtype=np.float64)
            self.tt_per_t_count = np.zeros(horizon, dtype=np.int64)
            self.max_eval_horizon = horizon
            return

        if horizon <= self.max_eval_horizon:
            return

        pad = horizon - self.max_eval_horizon

        self.tt_real_per_t_sum = np.pad(self.tt_real_per_t_sum, (0, pad))
        self.tt_pred_per_t_sum = np.pad(self.tt_pred_per_t_sum, (0, pad))
        self.tt_per_t_count = np.pad(self.tt_per_t_count, (0, pad))

        self.max_eval_horizon = horizon

    def _ensure_node_capacity(self, n_nodes: int):
        if self.per_node_sum_abs_err is None:
            self.per_node_sum_abs_err = np.zeros(n_nodes, dtype=np.float64)
            self.per_node_sum_sq_err = np.zeros(n_nodes, dtype=np.float64)
            self.per_node_count = np.zeros(n_nodes, dtype=np.int64)
            self.per_node_real_sum = np.zeros(n_nodes, dtype=np.float64)
            self.per_node_pred_sum = np.zeros(n_nodes, dtype=np.float64)
            self.per_node_has_signal = np.zeros(n_nodes, dtype=bool)
            return

        if n_nodes != len(self.per_node_sum_abs_err):
            raise ValueError(
                f"Różna liczba node'ów między plikami: "
                f"było {len(self.per_node_sum_abs_err)}, teraz {n_nodes}"
            )

    def update(self, tt_stats: dict, err_stats: dict):
        self.files_ok += 1

        self.total_values += err_stats["n_values"]
        self.total_abs_err += err_stats["sum_abs_err"]
        self.total_sq_err += err_stats["sum_sq_err"]

        self.total_tt_real += tt_stats["tt_real"]
        self.total_tt_pred += tt_stats["tt_pred"]
        self.total_tt_abs_diff_filewise += tt_stats["tt_abs_diff"]

        horizon = len(tt_stats["tt_real_per_t"])
        self._ensure_time_capacity(horizon)

        self.tt_real_per_t_sum[:horizon] += tt_stats["tt_real_per_t"]
        self.tt_pred_per_t_sum[:horizon] += tt_stats["tt_pred_per_t"]
        self.tt_per_t_count[:horizon] += 1

        n_nodes = len(err_stats["per_node_sum_abs_err"])
        self._ensure_node_capacity(n_nodes)

        self.per_node_sum_abs_err += err_stats["per_node_sum_abs_err"]
        self.per_node_sum_sq_err += err_stats["per_node_sum_sq_err"]
        self.per_node_count += err_stats["per_node_count"]
        self.per_node_real_sum += err_stats["per_node_real_sum"]
        self.per_node_pred_sum += err_stats["per_node_pred_sum"]
        self.per_node_has_signal |= err_stats["per_node_has_signal"]

    def summary_dict(self):
        tt_signed_diff_total = float(self.total_tt_pred - self.total_tt_real)
        tt_abs_diff_total = float(abs(tt_signed_diff_total))
        tt_rel_diff_total = (
            float(tt_abs_diff_total / self.total_tt_real)
            if self.total_tt_real != 0
            else np.nan
        )

        return {
            "files_ok": self.files_ok,
            "files_failed": self.files_failed,
            "delta_t": self.delta_t,

            "mae_weighted_eval": (
                float(self.total_abs_err / self.total_values)
                if self.total_values
                else np.nan
            ),
            "rmse_weighted_eval": (
                float(math.sqrt(self.total_sq_err / self.total_values))
                if self.total_values
                else np.nan
            ),

            "tt_real_total": float(self.total_tt_real),
            "tt_pred_total": float(self.total_tt_pred),
            "tt_signed_diff_total": tt_signed_diff_total,
            "tt_abs_diff_total": tt_abs_diff_total,
            "tt_rel_diff_total": tt_rel_diff_total,

            "tt_abs_diff_mean_per_file": (
                float(self.total_tt_abs_diff_filewise / self.files_ok)
                if self.files_ok
                else np.nan
            ),

            "n_eval_values_total": int(self.total_values),
        }


# ============================================================
# Wybór plików
# ============================================================

def select_test_files(args):
    _, _, test_files = split_file_names(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    original_test_files = list(test_files)

    if args.file_pattern:
        pattern = re.compile(args.file_pattern)
        test_files = [f for f in test_files if pattern.search(f)]

    if args.all_test:
        selected = test_files

    elif args.file_names:
        selected = [Path(x).name for x in args.file_names]

    elif args.file_names_file:
        selected = read_file_names_file(args.file_names_file)

    elif args.test_indices:
        indices = parse_indices(args.test_indices)
        selected = []

        for i in indices:
            if i < 0 or i >= len(test_files):
                raise IndexError(f"test index {i} poza zakresem [0, {len(test_files) - 1}]")
            selected.append(test_files[i])

    else:
        if args.num_files <= 0:
            raise ValueError(
                "--num_files musi być > 0 albo użyj "
                "--all_test / --test_indices / --file_names"
            )

        start = args.start_index
        stop = min(start + args.num_files, len(test_files))
        selected = test_files[start:stop]

    selected = list(dict.fromkeys(selected))

    test_set = set(original_test_files)
    missing = [f for f in selected if f not in test_set]

    if missing:
        raise ValueError(
            "Te pliki nie należą do test splitu dla podanych seed/train_ratio/val_ratio: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    return selected, original_test_files


# ============================================================
# Wywoływanie inference.py
# ============================================================

def build_inference_cmd(args, file_name: str, run_dir: Path, passthrough_args: list[str]):
    python_bin = args.python_bin or sys.executable

    cmd = [
        python_bin,
        str(args.inference_script),

        "--q_dir", str(args.q_dir),
        "--a_dir", str(args.a_dir),
        "--adjdata", str(args.adjdata),
        "--checkpoint", str(args.checkpoint),
        "--output_dir", str(run_dir),

        "--file_name", file_name,

        "--seed", str(args.seed),
        "--train_ratio", str(args.train_ratio),
        "--val_ratio", str(args.val_ratio),

        "--seq_length_q", str(args.seq_length_q),
        "--seq_length_a", str(args.seq_length_a),

        "--device", str(args.device),
        "--num_nodes", str(args.num_nodes),
        "--nhid", str(args.nhid),
        "--dropout", str(args.dropout),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),

        "--kernel_size", str(args.kernel_size),
        "--blocks", str(args.blocks),
        "--layers", str(args.layers),

        "--sequence_model", str(args.sequence_model),
        "--fuse_method", str(args.fuse_method),

        "--a_embedding_size", str(args.a_embedding_size),
        "--a_hidden_size", str(args.a_hidden_size),
        "--q_rep_dim", str(args.q_rep_dim),
        "--fused_dim", str(args.fused_dim),
        "--mlp_hidden_dim", str(args.mlp_hidden_dim),
        "--attention_num_heads", str(args.attention_num_heads),
        "--attention_ff_dim", str(args.attention_ff_dim),
    ]

    if args.gcn_bool:
        cmd.append("--gcn_bool")

    if args.addaptadj:
        cmd.append("--addaptadj")

    if args.randomadj:
        cmd.append("--randomadj")

    cmd.extend(passthrough_args)

    return cmd


def run_inference_for_file(args, file_name: str, run_dir: Path, passthrough_args: list[str]):
    pred_existing = list(run_dir.glob("*_pred_q.npy"))
    real_existing = list(run_dir.glob("*_real_q.npy"))

    if args.skip_existing and pred_existing and real_existing:
        return True, "SKIPPED_EXISTING"

    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_inference_cmd(
        args=args,
        file_name=file_name,
        run_dir=run_dir,
        passthrough_args=passthrough_args,
    )

    with open(run_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, cmd)) + "\n")

    print(f"[RUN] {file_name} -> {run_dir}", flush=True)

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with open(run_dir / "stdout.log", "w", encoding="utf-8") as f:
        f.write(result.stdout)

    with open(run_dir / "stderr.log", "w", encoding="utf-8") as f:
        f.write(result.stderr)

    if args.print_subprocess_output:
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        msg = f"FAILED returncode={result.returncode}; stderr: {run_dir / 'stderr.log'}"
        return False, msg

    return True, "OK"


# ============================================================
# Analiza jednego runu
# ============================================================

def analyze_run(run_dir: Path, file_name: str, delta_t: float):
    stem = safe_stem(file_name)

    pred_path = run_dir / f"{stem}_pred_q.npy"
    real_path = run_dir / f"{stem}_real_q.npy"

    if not pred_path.exists():
        pred_path = find_single("*_pred_q.npy", run_dir)

    if not real_path.exists():
        real_path = find_single("*_real_q.npy", run_dir)

    pred = np.load(pred_path)
    real = np.load(real_path)

    if pred.shape != real.shape:
        raise ValueError(
            f"{file_name}: pred i real mają różne shape: "
            f"{pred.shape} vs {real.shape}"
        )

    if pred.ndim != 2:
        raise ValueError(f"{file_name}: oczekiwany shape (T, N), dostałem {pred.shape}")

    seed_steps = load_seed_steps(run_dir)

    tt_stats = compute_tt_stats(
        pred=pred,
        real=real,
        delta_t=delta_t,
        seed_steps=seed_steps,
    )

    err_stats = compute_eval_errors(
        pred=pred,
        real=real,
        seed_steps=seed_steps,
    )

    row = {
        "file_name": file_name,
        "run_dir": str(run_dir),

        "seed_steps": seed_steps,
        "timesteps_total": int(pred.shape[0]),
        "timesteps_eval": int(pred.shape[0] - (seed_steps or 0)),
        "nodes": int(pred.shape[1]),

        "mae_eval": err_stats["mae"],
        "rmse_eval": err_stats["rmse"],

        "tt_real": tt_stats["tt_real"],
        "tt_pred": tt_stats["tt_pred"],
        "tt_signed_diff": tt_stats["tt_signed_diff"],
        "tt_abs_diff": tt_stats["tt_abs_diff"],
        "tt_rel_diff": tt_stats["tt_rel_diff"],
    }

    return row, tt_stats, err_stats


# ============================================================
# Zapisy i wykresy
# ============================================================

def save_per_node_metrics(batch_dir: Path, agg: AggregateState):
    if agg.per_node_sum_abs_err is None:
        return

    count = np.maximum(agg.per_node_count, 1)

    mae = agg.per_node_sum_abs_err / count
    rmse = np.sqrt(agg.per_node_sum_sq_err / count)

    tt_real_node = agg.delta_t * agg.per_node_real_sum
    tt_pred_node = agg.delta_t * agg.per_node_pred_sum
    tt_signed_diff_node = tt_pred_node - tt_real_node
    tt_abs_diff_node = np.abs(tt_signed_diff_node)

    tt_rel_diff_node = np.where(
        tt_real_node != 0,
        tt_abs_diff_node / tt_real_node,
        np.nan,
    )

    df = pd.DataFrame({
        "node": np.arange(len(mae)),
        "has_real_signal": agg.per_node_has_signal,
        "n_values": agg.per_node_count,

        "mae_eval": mae,
        "rmse_eval": rmse,

        "tt_real": tt_real_node,
        "tt_pred": tt_pred_node,
        "tt_signed_diff": tt_signed_diff_node,
        "tt_abs_diff": tt_abs_diff_node,
        "tt_rel_diff": tt_rel_diff_node,
    }).sort_values("mae_eval", ascending=False)

    df.to_csv(batch_dir / "node_metrics_aggregate.csv", index=False)

    rows = []

    for label, mask in [
        ("zero_nodes_global", ~agg.per_node_has_signal),
        ("nonzero_nodes_global", agg.per_node_has_signal),
    ]:
        n_nodes = int(np.sum(mask))

        if n_nodes == 0:
            rows.append({
                "group": label,
                "n_nodes": 0,
                "n_values": 0,
                "mae_eval": np.nan,
                "rmse_eval": np.nan,
                "tt_real": np.nan,
                "tt_pred": np.nan,
                "tt_signed_diff": np.nan,
                "tt_abs_diff": np.nan,
                "tt_rel_diff": np.nan,
            })
            continue

        values = int(np.sum(agg.per_node_count[mask]))
        sum_abs = float(np.sum(agg.per_node_sum_abs_err[mask]))
        sum_sq = float(np.sum(agg.per_node_sum_sq_err[mask]))

        tt_real = float(agg.delta_t * np.sum(agg.per_node_real_sum[mask]))
        tt_pred = float(agg.delta_t * np.sum(agg.per_node_pred_sum[mask]))

        signed = float(tt_pred - tt_real)
        abs_diff = float(abs(signed))
        rel = float(abs_diff / tt_real) if tt_real != 0 else np.nan

        rows.append({
            "group": label,
            "n_nodes": n_nodes,
            "n_values": values,

            "mae_eval": float(sum_abs / values) if values else np.nan,
            "rmse_eval": float(math.sqrt(sum_sq / values)) if values else np.nan,

            "tt_real": tt_real,
            "tt_pred": tt_pred,
            "tt_signed_diff": signed,
            "tt_abs_diff": abs_diff,
            "tt_rel_diff": rel,
        })

    pd.DataFrame(rows).to_csv(batch_dir / "node_groups_aggregate.csv", index=False)


def save_tt_timeseries(batch_dir: Path, agg: AggregateState):
    if agg.tt_real_per_t_sum is None:
        return

    count = np.maximum(agg.tt_per_t_count, 1)

    df = pd.DataFrame({
        "t_eval": np.arange(agg.max_eval_horizon),
        "n_files_available": agg.tt_per_t_count,

        "tt_real_per_t_sum": agg.tt_real_per_t_sum,
        "tt_pred_per_t_sum": agg.tt_pred_per_t_sum,

        "tt_real_per_t_mean": agg.tt_real_per_t_sum / count,
        "tt_pred_per_t_mean": agg.tt_pred_per_t_sum / count,

        "tt_real_cum_sum": np.cumsum(agg.tt_real_per_t_sum),
        "tt_pred_cum_sum": np.cumsum(agg.tt_pred_per_t_sum),
    })

    df.to_csv(batch_dir / "tt_timeseries_aggregate.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(
        df["t_eval"],
        df["tt_real_per_t_sum"],
        label="real TT per timestep, suma po plikach",
    )
    plt.plot(
        df["t_eval"],
        df["tt_pred_per_t_sum"],
        label="pred TT per timestep, suma po plikach",
    )
    plt.title(f"TT per timestep — agregacja zbioru, Δt={agg.delta_t}")
    plt.xlabel("t eval")
    plt.ylabel("Σ plików Δt * ||q_t||_1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_per_timestep_aggregate.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(
        df["t_eval"],
        df["tt_real_cum_sum"],
        label="real cumulative TT, suma po plikach",
    )
    plt.plot(
        df["t_eval"],
        df["tt_pred_cum_sum"],
        label="pred cumulative TT, suma po plikach",
    )
    plt.title(f"Cumulative TT — agregacja zbioru, Δt={agg.delta_t}")
    plt.xlabel("t eval")
    plt.ylabel("cumulative TT")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_cumulative_aggregate.png", dpi=220)
    plt.close()


def plot_real_vs_pred_tt(batch_dir: Path, per_file_df: pd.DataFrame, summary: dict):
    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if ok_df.empty:
        return

    x = ok_df["tt_real"].to_numpy(dtype=float)
    y = ok_df["tt_pred"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]
    y = y[finite]

    if len(x) == 0:
        return

    min_v = float(min(np.min(x), np.min(y)))
    max_v = float(max(np.max(x), np.max(y)))

    pad = 0.03 * (max_v - min_v) if max_v > min_v else 1.0
    lo = min_v - pad
    hi = max_v + pad

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, alpha=0.75)
    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1,
        label="idealnie: pred = real",
    )

    total_rel = summary.get("tt_rel_diff_total", np.nan)

    if np.isfinite(total_rel):
        text = (
            f"Suma zbioru:\n"
            f"real TT = {summary.get('tt_real_total', np.nan):.3g}\n"
            f"pred TT = {summary.get('tt_pred_total', np.nan):.3g}\n"
            f"rel diff = {total_rel:.2%}"
        )
    else:
        text = "Suma zbioru: rel diff = nan"

    plt.text(
        0.03,
        0.97,
        text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )

    plt.title("Real TT vs predicted TT — każdy punkt = jeden plik")
    plt.xlabel("Ground truth TT")
    plt.ylabel("Predicted TT")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "real_tt_vs_predicted_tt.png", dpi=240)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(
        ["real TT", "predicted TT"],
        [summary["tt_real_total"], summary["tt_pred_total"]],
    )
    plt.title("Suma TT dla całego zbioru")
    plt.ylabel("TT")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(batch_dir / "global_real_tt_vs_predicted_tt.png", dpi=220)
    plt.close()

    plot_df = ok_df.sort_values("tt_rel_diff", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(max(10, 0.25 * len(plot_df)), 5))
    plt.bar(
        np.arange(len(plot_df)),
        100.0 * plot_df["tt_rel_diff"].to_numpy(dtype=float),
    )
    plt.title("TT relative error per file")
    plt.xlabel("plik posortowany malejąco po błędzie względnym")
    plt.ylabel("|pred-real| / real [%]")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_relative_error_per_file.png", dpi=220)
    plt.close()


def save_run_manifest(
    batch_dir: Path,
    args,
    selected_files: list[str],
    all_test_files: list[str],
    passthrough_args: list[str],
):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_files_count": len(selected_files),
        "test_files_count": len(all_test_files),
        "selected_files": selected_files,
        "passthrough_args": passthrough_args,
        "args": vars(args),
    }

    with open(batch_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Uruchamia inference.py dla wielu plików i liczy zbiorcze statystyki, "
            "zwłaszcza TT."
        )
    )

    # Ścieżki
    p.add_argument("--inference_script", type=Path, default=Path("inference.py"))
    p.add_argument(
        "--python_bin",
        type=str,
        default=None,
        help="np. /home/.../env/bin/python; domyślnie bieżący Python",
    )

    p.add_argument("--q_dir", type=str, required=True)
    p.add_argument("--a_dir", type=str, required=True)
    p.add_argument("--adjdata", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)

    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--batch_name", type=str, default=None)

    # Wybór plików
    p.add_argument("--all_test", action="store_true")
    p.add_argument("--num_files", type=int, default=10)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--test_indices", type=str, default=None)
    p.add_argument("--file_names", nargs="*", default=None)
    p.add_argument("--file_names_file", type=str, default=None)
    p.add_argument("--file_pattern", type=str, default=None)

    # Split
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    # Sekwencje
    p.add_argument("--seq_length_q", type=int, default=15)
    p.add_argument("--seq_length_a", type=int, default=30)

    # Model — muszą odpowiadać checkpointowi
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_nodes", type=int, default=195)

    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)

    p.add_argument("--gcn_bool", action="store_true")
    p.add_argument("--addaptadj", action="store_true")
    p.add_argument("--randomadj", action="store_true")

    p.add_argument("--kernel_size", type=int, default=2)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)

    p.add_argument(
        "--sequence_model",
        type=str,
        default="lstm",
        choices=["lstm", "gru", "attention"],
    )

    p.add_argument(
        "--fuse_method",
        type=str,
        default="Attention",
        choices=["concatenate", "Attention", "wavenet_only", "assignment_only"],
    )

    p.add_argument("--a_embedding_size", type=int, default=32)
    p.add_argument("--a_hidden_size", type=int, default=64)
    p.add_argument("--q_rep_dim", type=int, default=32)
    p.add_argument("--fused_dim", type=int, default=64)
    p.add_argument("--mlp_hidden_dim", type=int, default=128)
    p.add_argument("--attention_num_heads", type=int, default=4)
    p.add_argument("--attention_ff_dim", type=int, default=128)

    # Statystyki i wykonanie
    p.add_argument("--delta_t", type=float, default=10.0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--print_subprocess_output", action="store_true")

    return p


def main():
    parser = build_parser()
    args, passthrough_args = parser.parse_known_args()

    selected_files, all_test_files = select_test_files(args)

    if not selected_files:
        raise RuntimeError("Nie wybrano żadnych plików do inferencji.")

    batch_name = args.batch_name or f"batch_infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = args.output_root / batch_name
    per_file_root = batch_dir / "per_file"

    batch_dir.mkdir(parents=True, exist_ok=True)
    per_file_root.mkdir(parents=True, exist_ok=True)

    save_run_manifest(
        batch_dir=batch_dir,
        args=args,
        selected_files=selected_files,
        all_test_files=all_test_files,
        passthrough_args=passthrough_args,
    )

    print(f"Batch dir: {batch_dir}", flush=True)
    print(f"Liczba plików testowych w splicie: {len(all_test_files)}", flush=True)
    print(f"Wybrano plików: {len(selected_files)}", flush=True)

    print("Pierwsze wybrane pliki:", flush=True)
    for f in selected_files[:10]:
        print(f"  - {f}", flush=True)

    if len(selected_files) > 10:
        print(f"  ... +{len(selected_files) - 10}", flush=True)

    rows = []
    agg = AggregateState(delta_t=args.delta_t)

    for idx, file_name in enumerate(selected_files):
        run_dir = per_file_root / f"{idx:04d}_{safe_stem(file_name)}"

        if args.dry_run:
            cmd = build_inference_cmd(
                args=args,
                file_name=file_name,
                run_dir=run_dir,
                passthrough_args=passthrough_args,
            )
            print("[DRY RUN]", " ".join(map(str, cmd)), flush=True)
            continue

        ok, status_msg = run_inference_for_file(
            args=args,
            file_name=file_name,
            run_dir=run_dir,
            passthrough_args=passthrough_args,
        )

        if not ok:
            agg.files_failed += 1

            rows.append({
                "file_name": file_name,
                "run_dir": str(run_dir),
                "status": "FAILED",
                "status_msg": status_msg,
            })

            print(f"[FAILED] {file_name}: {status_msg}", flush=True)

            if not args.continue_on_error:
                pd.DataFrame(rows).to_csv(
                    batch_dir / "per_file_metrics_partial.csv",
                    index=False,
                )
                raise RuntimeError(f"Inference failed for {file_name}: {status_msg}")

            continue

        try:
            row, tt_stats, err_stats = analyze_run(
                run_dir=run_dir,
                file_name=file_name,
                delta_t=args.delta_t,
            )

            row["status"] = "OK"
            row["status_msg"] = status_msg

            rows.append(row)
            agg.update(tt_stats=tt_stats, err_stats=err_stats)

            print(
                f"[OK] {file_name}: "
                f"TT real={row['tt_real']:.6f}, "
                f"TT pred={row['tt_pred']:.6f}, "
                f"rel diff={row['tt_rel_diff']:.3%}, "
                f"MAE={row['mae_eval']:.6f}",
                flush=True,
            )

        except Exception as e:
            agg.files_failed += 1

            rows.append({
                "file_name": file_name,
                "run_dir": str(run_dir),
                "status": "ANALYSIS_FAILED",
                "status_msg": repr(e),
            })

            print(f"[ANALYSIS_FAILED] {file_name}: {e}", flush=True)

            if not args.continue_on_error:
                pd.DataFrame(rows).to_csv(
                    batch_dir / "per_file_metrics_partial.csv",
                    index=False,
                )
                raise

    if args.dry_run:
        print("Dry run zakończony — nie zapisuję metryk ani wykresów.", flush=True)
        return

    per_file_df = pd.DataFrame(rows)
    per_file_df.to_csv(batch_dir / "per_file_metrics.csv", index=False)

    summary = agg.summary_dict()
    summary["batch_dir"] = str(batch_dir)
    summary["checkpoint"] = str(args.checkpoint)
    summary["selected_files_count"] = len(selected_files)

    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if not ok_df.empty:
        summary["mae_mean_per_file"] = float(ok_df["mae_eval"].mean())
        summary["rmse_mean_per_file"] = float(ok_df["rmse_eval"].mean())
        summary["tt_rel_diff_mean_per_file"] = float(ok_df["tt_rel_diff"].mean())
        summary["tt_rel_diff_median_per_file"] = float(ok_df["tt_rel_diff"].median())

    with open(batch_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(batch_dir / "summary.csv", index=False)

    save_per_node_metrics(batch_dir, agg)
    save_tt_timeseries(batch_dir, agg)
    plot_real_vs_pred_tt(batch_dir, per_file_df, summary)

    print("\n=== PODSUMOWANIE ZBIORCZE ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    print(f"\nZapisano wyniki w: {batch_dir}", flush=True)
    print("Najważniejszy wykres: real_tt_vs_predicted_tt.png", flush=True)


if __name__ == "__main__":
    main()