#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
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
import torch

import scienceplots

# ============================================================
# Import z istniejącego inference.py
# ============================================================
# Skrypt zakłada, że batch_inference_eval_gpu.py leży w tym samym katalogu
# co inference.py, czyli np. /home/drozd/OptimalAssignment.
#
# Używamy:
#   - split_file_names
#   - load_flow_TN
#   - load_assign_TN
#   - build_trainer
#
# Dzięki temu architektura modelu i split są spójne z inference.py.

from inference import (
    split_file_names,
    load_flow_TN,
    load_assign_TN,
    build_trainer,
)


# ============================================================
# Helpery wyboru plików
# ============================================================

try:
    import scienceplots  # noqa: F401  # rejestruje style `science` w matplotlib
    SCIENCEPLOTS_AVAILABLE = True
except Exception:
    scienceplots = None
    SCIENCEPLOTS_AVAILABLE = False


def configure_plot_style(style_names: list[str] | None = None) -> None:
    """
    Konfiguruje globalny styl wykresów.

    Domyślnie używa SciencePlots z wyłączonym LaTeX-em, bo etykiety
    zawierają polskie znaki i symbole Unicode.
    """
    if style_names is None or len(style_names) == 0:
        style_names = ["science", "no-latex"]

    if SCIENCEPLOTS_AVAILABLE:
        try:
            plt.style.use(style_names)
            print(f"Plot style: SciencePlots {style_names}", flush=True)
        except Exception as e:
            print(
                f"[WARN] Nie udało się ustawić stylu SciencePlots {style_names}: {repr(e)}. "
                "Używam domyślnego stylu matplotlib.",
                flush=True,
            )
    else:
        print(
            "[WARN] Pakiet scienceplots nie jest zainstalowany. "
            "Zainstaluj: pip install SciencePlots. Używam domyślnego stylu matplotlib.",
            flush=True,
        )

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 240,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "legend.frameon": True,
        "axes.unicode_minus": False,
    })


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
                raise IndexError(
                    f"test index {i} poza zakresem [0, {len(test_files) - 1}]"
                )

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
# Dane
# ============================================================

@dataclass
class LoadedItem:
    selected_index: int
    file_name: str
    real_q_TN: np.ndarray
    assign_TN: np.ndarray
    current_nodes: int
    nodes_to_copy: int


def load_one_item(args, selected_index: int, file_name: str) -> LoadedItem:
    flow_path = Path(args.q_dir) / file_name
    assign_path = Path(args.a_dir) / file_name

    real_q_TN, current_nodes, nodes_to_copy = load_flow_TN(
        flow_path,
        target_nodes=args.num_nodes,
    )

    assign_TN = load_assign_TN(
        assign_path,
        current_nodes=current_nodes,
        target_nodes=args.num_nodes,
    )

    if assign_TN.shape[0] != real_q_TN.shape[0]:
        raise ValueError(
            f"Niezgodna liczba timestepów dla {file_name}: "
            f"flow={real_q_TN.shape[0]}, assign={assign_TN.shape[0]}"
        )

    if real_q_TN.ndim != 2:
        raise ValueError(f"{file_name}: real_q_TN powinien mieć shape (T, N), dostałem {real_q_TN.shape}")

    if assign_TN.ndim != 2:
        raise ValueError(f"{file_name}: assign_TN powinien mieć shape (T, N), dostałem {assign_TN.shape}")

    if real_q_TN.shape[1] != args.num_nodes:
        raise ValueError(
            f"{file_name}: real_q_TN ma {real_q_TN.shape[1]} node'ów, "
            f"oczekiwano {args.num_nodes}"
        )

    if assign_TN.shape[1] != args.num_nodes:
        raise ValueError(
            f"{file_name}: assign_TN ma {assign_TN.shape[1]} node'ów, "
            f"oczekiwano {args.num_nodes}"
        )

    return LoadedItem(
        selected_index=selected_index,
        file_name=file_name,
        real_q_TN=np.asarray(real_q_TN, dtype=np.float32),
        assign_TN=np.asarray(assign_TN, dtype=np.float32),
        current_nodes=int(current_nodes),
        nodes_to_copy=int(nodes_to_copy),
    )


# ============================================================
# Batch rollout na GPU
# ============================================================

@torch.no_grad()
def rollout_many_sequences_gpu(
    model,
    device,
    real_q_BTN: np.ndarray,
    assign_BTN: np.ndarray,
    seq_length_q: int,
    seq_length_a: int,
):
    """
    real_q_BTN:  (B, T, N)
    assign_BTN: (B, T, N)

    Zwraca:
        generated_q_BTN: (B, T, N)
        seed_steps: int

    To jest zbatched odpowiednik rollout_one_sequence z inference.py.
    Model dostaje naraz:
        q_tensor: (B, seq_length_q, N)
        a_tensor: (B, seq_length_a + 1, N)
    zgodnie z oryginalnym slicingiem:
        assign_TN[t - seq_length_a:t + 1]
    """

    seed_steps = max(seq_length_q, seq_length_a)

    if real_q_BTN.ndim != 3:
        raise ValueError(f"real_q_BTN powinien mieć shape (B, T, N), dostałem {real_q_BTN.shape}")

    if assign_BTN.ndim != 3:
        raise ValueError(f"assign_BTN powinien mieć shape (B, T, N), dostałem {assign_BTN.shape}")

    if real_q_BTN.shape != assign_BTN.shape:
        raise ValueError(f"real_q_BTN i assign_BTN mają różne shape: {real_q_BTN.shape} vs {assign_BTN.shape}")

    B, T, N = real_q_BTN.shape

    if T <= seed_steps:
        raise ValueError(
            f"Za krótka sekwencja: T={T}, potrzeba > {seed_steps}"
        )

    real_tensor = torch.from_numpy(real_q_BTN).to(device=device, dtype=torch.float32)
    assign_tensor = torch.from_numpy(assign_BTN).to(device=device, dtype=torch.float32)

    generated = torch.zeros_like(real_tensor)

    # Seed: pierwsze max(Lq, La) kroków prawdziwego q.
    generated[:, :seed_steps, :] = real_tensor[:, :seed_steps, :]

    for t in range(seed_steps, T):
        q_window = generated[:, t - seq_length_q:t, :]          # (B, Lq, N)
        a_window = assign_tensor[:, t - seq_length_a:t + 1, :]  # (B, La+1, N)

        pred = model(q_window, a_window)

        if pred.dim() == 2:
            pred_step = pred                 # (B, N)
        elif pred.dim() == 3 and pred.shape[1] == 1:
            pred_step = pred[:, 0, :]        # (B, N)
        else:
            raise ValueError(f"Nieoczekiwany shape predykcji: {tuple(pred.shape)}")

        generated[:, t, :] = pred_step

    return generated.detach().cpu().numpy(), seed_steps


# ============================================================
# Metryki i TT
# ============================================================

def compute_one_file_metrics(
    pred_TN: np.ndarray,
    real_TN: np.ndarray,
    file_name: str,
    selected_index: int,
    seed_steps: int,
    nodes_to_copy: int,
    delta_t: float,
):
    pred_eval = pred_TN[seed_steps:, :nodes_to_copy]
    real_eval = real_TN[seed_steps:, :nodes_to_copy]

    err = pred_eval - real_eval
    abs_err = np.abs(err)
    sq_err = err ** 2

    mae = float(np.mean(abs_err)) if abs_err.size else np.nan
    rmse = float(np.sqrt(np.mean(sq_err))) if sq_err.size else np.nan

    # Zgodnie z inference_visual.py:
    # TT total liczone jako delta_t * sum(q)
    tt_real = float(delta_t * np.sum(real_eval))
    tt_pred = float(delta_t * np.sum(pred_eval))

    tt_signed_diff = float(tt_pred - tt_real)
    tt_abs_diff = float(abs(tt_signed_diff))
    tt_rel_diff = float(tt_abs_diff / tt_real) if tt_real != 0 else np.nan

    # Profil czasowy: delta_t * sum(abs(q_t)) po nodach.
    tt_real_per_t = delta_t * np.sum(np.abs(real_eval), axis=1)
    tt_pred_per_t = delta_t * np.sum(np.abs(pred_eval), axis=1)

    per_node_sum_abs_err = np.sum(abs_err, axis=0)
    per_node_sum_sq_err = np.sum(sq_err, axis=0)
    per_node_real_sum = np.sum(real_eval, axis=0)
    per_node_pred_sum = np.sum(pred_eval, axis=0)
    per_node_has_signal = np.any(np.abs(real_eval) > 1e-12, axis=0)

    row = {
        "selected_index": int(selected_index),
        "file_name": file_name,
        "status": "OK",
        "status_msg": "OK",

        "seed_steps": int(seed_steps),
        "timesteps_total": int(pred_TN.shape[0]),
        "timesteps_eval": int(pred_eval.shape[0]),
        "nodes": int(nodes_to_copy),

        "mae_eval": mae,
        "rmse_eval": rmse,

        "tt_real": tt_real,
        "tt_pred": tt_pred,
        "tt_signed_diff": tt_signed_diff,
        "tt_abs_diff": tt_abs_diff,
        "tt_rel_diff": tt_rel_diff,
    }

    details = {
        "n_values": int(abs_err.size),
        "sum_abs_err": float(np.sum(abs_err)),
        "sum_sq_err": float(np.sum(sq_err)),

        "tt_real_per_t": tt_real_per_t,
        "tt_pred_per_t": tt_pred_per_t,

        "per_node_sum_abs_err": per_node_sum_abs_err,
        "per_node_sum_sq_err": per_node_sum_sq_err,
        "per_node_count": np.full(nodes_to_copy, pred_eval.shape[0], dtype=np.int64),
        "per_node_real_sum": per_node_real_sum,
        "per_node_pred_sum": per_node_pred_sum,
        "per_node_has_signal": per_node_has_signal,
    }

    return row, details


class AggregateState:
    def __init__(self, delta_t: float):
        self.delta_t = float(delta_t)

        self.files_ok = 0
        self.files_failed = 0

        self.total_values = 0
        self.total_abs_err = 0.0
        self.total_sq_err = 0.0

        self.total_tt_real = 0.0
        self.total_tt_pred = 0.0
        self.total_tt_abs_diff_filewise = 0.0

        self.tt_real_per_t_sum = None
        self.tt_pred_per_t_sum = None
        self.tt_per_t_count = None
        self.max_eval_horizon = 0

        self.per_node_sum_abs_err = None
        self.per_node_sum_sq_err = None
        self.per_node_count = None
        self.per_node_real_sum = None
        self.per_node_pred_sum = None
        self.per_node_has_signal = None

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
                f"było {len(self.per_node_sum_abs_err)}, teraz {n_nodes}. "
                f"Jeśli to normalne, trzymaj --num_nodes stałe i upewnij się, że nodes_to_copy jest stałe."
            )

    def update_ok(self, row: dict, details: dict):
        self.files_ok += 1

        self.total_values += details["n_values"]
        self.total_abs_err += details["sum_abs_err"]
        self.total_sq_err += details["sum_sq_err"]

        self.total_tt_real += row["tt_real"]
        self.total_tt_pred += row["tt_pred"]
        self.total_tt_abs_diff_filewise += row["tt_abs_diff"]

        tt_real_per_t = details["tt_real_per_t"]
        tt_pred_per_t = details["tt_pred_per_t"]

        horizon = len(tt_real_per_t)
        self._ensure_time_capacity(horizon)

        self.tt_real_per_t_sum[:horizon] += tt_real_per_t
        self.tt_pred_per_t_sum[:horizon] += tt_pred_per_t
        self.tt_per_t_count[:horizon] += 1

        n_nodes = len(details["per_node_sum_abs_err"])
        self._ensure_node_capacity(n_nodes)

        self.per_node_sum_abs_err += details["per_node_sum_abs_err"]
        self.per_node_sum_sq_err += details["per_node_sum_sq_err"]
        self.per_node_count += details["per_node_count"]
        self.per_node_real_sum += details["per_node_real_sum"]
        self.per_node_pred_sum += details["per_node_pred_sum"]
        self.per_node_has_signal |= details["per_node_has_signal"]

    def update_failed(self):
        self.files_failed += 1

    def summary_dict(self):
        tt_signed_diff_total = float(self.total_tt_pred - self.total_tt_real)
        tt_abs_diff_total = float(abs(tt_signed_diff_total))
        tt_rel_diff_total = (
            float(tt_abs_diff_total / self.total_tt_real)
            if self.total_tt_real != 0
            else np.nan
        )

        return {
            "files_ok": int(self.files_ok),
            "files_failed": int(self.files_failed),
            "delta_t": float(self.delta_t),

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
# Zapisy i wykresy
# ============================================================

def save_manifest(batch_dir: Path, args, selected_files: list[str], all_test_files: list[str]):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_files_count": len(selected_files),
        "test_files_count": len(all_test_files),
        "selected_files": selected_files,
        "args": vars(args),
    }

    with open(batch_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def save_arrays_if_needed(
    args,
    batch_dir: Path,
    item: LoadedItem,
    pred_TN: np.ndarray,
):
    if not args.save_arrays:
        return

    out_dir = batch_dir / "per_file_arrays" / f"{item.selected_index:06d}_{safe_stem(item.file_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{safe_stem(item.file_name)}_pred_q.npy", pred_TN[:, :item.nodes_to_copy])
    np.save(out_dir / f"{safe_stem(item.file_name)}_real_q.npy", item.real_q_TN[:, :item.nodes_to_copy])
    np.save(out_dir / f"{safe_stem(item.file_name)}_assign.npy", item.assign_TN[:, :item.nodes_to_copy])


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
    plt.plot(df["t_eval"], df["tt_real_per_t_sum"], label="real TT per timestep, suma po plikach")
    plt.plot(df["t_eval"], df["tt_pred_per_t_sum"], label="pred TT per timestep, suma po plikach")
    plt.title(f"TT per timestep — agregacja zbioru, Δt={agg.delta_t}")
    plt.xlabel("t eval")
    plt.ylabel("Σ plików Δt * ||q_t||_1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_per_timestep_aggregate.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["t_eval"], df["tt_real_cum_sum"], label="real cumulative TT, suma po plikach")
    plt.plot(df["t_eval"], df["tt_pred_cum_sum"], label="pred cumulative TT, suma po plikach")
    plt.title(f"Cumulative TT — agregacja zbioru, Δt={agg.delta_t}")
    plt.xlabel("t eval")
    plt.ylabel("cumulative TT")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_cumulative_aggregate.png", dpi=220)
    plt.close()


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

    group_rows = []

    for group_name, mask in [
        ("zero_nodes_global", ~agg.per_node_has_signal),
        ("nonzero_nodes_global", agg.per_node_has_signal),
    ]:
        n_nodes = int(np.sum(mask))

        if n_nodes == 0:
            group_rows.append({
                "group": group_name,
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

        group_rows.append({
            "group": group_name,
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

    pd.DataFrame(group_rows).to_csv(batch_dir / "node_groups_aggregate.csv", index=False)

def _json_float(value):
    """Bezpieczna konwersja numpy scalar / NaN / inf do wartości JSON-friendly."""
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _save_histogram_with_bins(
    out_dir: Path,
    metric: str,
    values: pd.Series,
    bins: int,
    title: str,
    xlabel: str,
):
    """Zapisuje histogram PNG oraz tabelę koszyków CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    s = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if s.empty:
        return

    effective_bins = min(int(bins), max(1, int(s.nunique())))
    effective_bins = max(effective_bins, 1)

    counts, edges = np.histogram(s.to_numpy(dtype=float), bins=effective_bins)

    hist_df = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
    })

    hist_df["density"] = (
        hist_df["count"] / hist_df["count"].sum()
        if hist_df["count"].sum()
        else 0.0
    )

    hist_df.to_csv(out_dir / f"{metric}_histogram_bins.csv", index=False)

    q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])
    mean = s.mean()

    plt.figure(figsize=(9, 5.5))
    plt.hist(s.to_numpy(dtype=float), bins=effective_bins, alpha=0.85)
    plt.axvline(mean, linestyle="-", linewidth=1.2, label=f"mean = {mean:.4g}")
    plt.axvline(q50, linestyle="--", linewidth=1.2, label=f"median = {q50:.4g}")
    plt.axvline(q25, linestyle=":", linewidth=1.2, label=f"Q1 = {q25:.4g}")
    plt.axvline(q75, linestyle=":", linewidth=1.2, label=f"Q3 = {q75:.4g}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("liczba plików")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_histogram.png", dpi=240)
    plt.close()


def _quartile_error_summary(
    group_name: str,
    subset: pd.DataFrame,
    tt_q25: float,
    tt_q75: float,
) -> dict:
    """Błąd modelu dla dolnego albo górnego kwartylu real TT."""
    row = {
        "quartile_group": group_name,
        "tt_real_q25_threshold_global": _json_float(tt_q25),
        "tt_real_q75_threshold_global": _json_float(tt_q75),
        "n_files": int(len(subset)),
    }

    if subset.empty:
        keys = [
            "tt_real_min",
            "tt_real_mean",
            "tt_real_median",
            "tt_real_max",
            "tt_pred_mean",
            "tt_pred_median",
            "tt_signed_diff_mean",
            "tt_signed_diff_median",
            "tt_abs_diff_mean",
            "tt_abs_diff_median",
            "tt_rel_diff_mean",
            "tt_rel_diff_median",
            "tt_rel_diff_percent_mean",
            "tt_rel_diff_percent_median",
            "mae_eval_mean",
            "rmse_eval_mean",
            "tt_real_total",
            "tt_pred_total",
            "tt_signed_diff_total",
            "tt_abs_diff_total",
            "tt_rel_diff_total",
            "underprediction_fraction",
            "overprediction_fraction",
        ]

        for key in keys:
            row[key] = np.nan

        return row

    tt_real_total = float(subset["tt_real"].sum())
    tt_pred_total = float(subset["tt_pred"].sum())
    signed_total = tt_pred_total - tt_real_total
    abs_total = abs(signed_total)

    row.update({
        "tt_real_min": _json_float(subset["tt_real"].min()),
        "tt_real_mean": _json_float(subset["tt_real"].mean()),
        "tt_real_median": _json_float(subset["tt_real"].median()),
        "tt_real_max": _json_float(subset["tt_real"].max()),

        "tt_pred_mean": _json_float(subset["tt_pred"].mean()),
        "tt_pred_median": _json_float(subset["tt_pred"].median()),

        "tt_signed_diff_mean": _json_float(subset["tt_signed_diff"].mean()),
        "tt_signed_diff_median": _json_float(subset["tt_signed_diff"].median()),

        "tt_abs_diff_mean": _json_float(subset["tt_abs_diff"].mean()),
        "tt_abs_diff_median": _json_float(subset["tt_abs_diff"].median()),

        "tt_rel_diff_mean": _json_float(subset["tt_rel_diff"].mean()),
        "tt_rel_diff_median": _json_float(subset["tt_rel_diff"].median()),
        "tt_rel_diff_percent_mean": _json_float(100.0 * subset["tt_rel_diff"].mean()),
        "tt_rel_diff_percent_median": _json_float(100.0 * subset["tt_rel_diff"].median()),

        "mae_eval_mean": _json_float(subset["mae_eval"].mean()) if "mae_eval" in subset.columns else np.nan,
        "rmse_eval_mean": _json_float(subset["rmse_eval"].mean()) if "rmse_eval" in subset.columns else np.nan,

        "tt_real_total": _json_float(tt_real_total),
        "tt_pred_total": _json_float(tt_pred_total),
        "tt_signed_diff_total": _json_float(signed_total),
        "tt_abs_diff_total": _json_float(abs_total),
        "tt_rel_diff_total": _json_float(abs_total / tt_real_total) if tt_real_total != 0 else np.nan,

        "underprediction_fraction": _json_float((subset["tt_signed_diff"] < 0).mean()),
        "overprediction_fraction": _json_float((subset["tt_signed_diff"] > 0).mean()),
    })

    return row


def _json_float(value):
    """Bezpieczna konwersja numpy scalar / NaN / inf do wartości JSON-friendly."""
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _save_histogram_with_bins(
    out_dir: Path,
    metric: str,
    values: pd.Series,
    bins: int,
    title: str,
    xlabel: str,
):
    """Zapisuje histogram PNG oraz tabelę koszyków CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    s = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if s.empty:
        return

    effective_bins = min(int(bins), max(1, int(s.nunique())))
    effective_bins = max(effective_bins, 1)

    counts, edges = np.histogram(s.to_numpy(dtype=float), bins=effective_bins)

    hist_df = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
    })

    hist_df["density"] = (
        hist_df["count"] / hist_df["count"].sum()
        if hist_df["count"].sum()
        else 0.0
    )

    hist_df.to_csv(out_dir / f"{metric}_histogram_bins.csv", index=False)

    q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])
    mean = s.mean()

    plt.figure(figsize=(9, 5.5))
    plt.hist(s.to_numpy(dtype=float), bins=effective_bins, alpha=0.85)
    plt.axvline(mean, linestyle="-", linewidth=1.2, label=f"mean = {mean:.4g}")
    plt.axvline(q50, linestyle="--", linewidth=1.2, label=f"median = {q50:.4g}")
    plt.axvline(q25, linestyle=":", linewidth=1.2, label=f"Q1 = {q25:.4g}")
    plt.axvline(q75, linestyle=":", linewidth=1.2, label=f"Q3 = {q75:.4g}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("liczba plików")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_histogram.png", dpi=240)
    plt.close()


def _quartile_error_summary(
    group_name: str,
    subset: pd.DataFrame,
    tt_q25: float,
    tt_q75: float,
) -> dict:
    """Błąd modelu dla dolnego albo górnego kwartylu real TT."""
    row = {
        "quartile_group": group_name,
        "tt_real_q25_threshold_global": _json_float(tt_q25),
        "tt_real_q75_threshold_global": _json_float(tt_q75),
        "n_files": int(len(subset)),
    }

    if subset.empty:
        keys = [
            "tt_real_min",
            "tt_real_mean",
            "tt_real_median",
            "tt_real_max",
            "tt_pred_mean",
            "tt_pred_median",
            "tt_signed_diff_mean",
            "tt_signed_diff_median",
            "tt_abs_diff_mean",
            "tt_abs_diff_median",
            "tt_rel_diff_mean",
            "tt_rel_diff_median",
            "tt_rel_diff_percent_mean",
            "tt_rel_diff_percent_median",
            "mae_eval_mean",
            "rmse_eval_mean",
            "tt_real_total",
            "tt_pred_total",
            "tt_signed_diff_total",
            "tt_abs_diff_total",
            "tt_rel_diff_total",
            "underprediction_fraction",
            "overprediction_fraction",
        ]

        for key in keys:
            row[key] = np.nan

        return row

    tt_real_total = float(subset["tt_real"].sum())
    tt_pred_total = float(subset["tt_pred"].sum())
    signed_total = tt_pred_total - tt_real_total
    abs_total = abs(signed_total)

    row.update({
        "tt_real_min": _json_float(subset["tt_real"].min()),
        "tt_real_mean": _json_float(subset["tt_real"].mean()),
        "tt_real_median": _json_float(subset["tt_real"].median()),
        "tt_real_max": _json_float(subset["tt_real"].max()),

        "tt_pred_mean": _json_float(subset["tt_pred"].mean()),
        "tt_pred_median": _json_float(subset["tt_pred"].median()),

        "tt_signed_diff_mean": _json_float(subset["tt_signed_diff"].mean()),
        "tt_signed_diff_median": _json_float(subset["tt_signed_diff"].median()),

        "tt_abs_diff_mean": _json_float(subset["tt_abs_diff"].mean()),
        "tt_abs_diff_median": _json_float(subset["tt_abs_diff"].median()),

        "tt_rel_diff_mean": _json_float(subset["tt_rel_diff"].mean()),
        "tt_rel_diff_median": _json_float(subset["tt_rel_diff"].median()),
        "tt_rel_diff_percent_mean": _json_float(100.0 * subset["tt_rel_diff"].mean()),
        "tt_rel_diff_percent_median": _json_float(100.0 * subset["tt_rel_diff"].median()),

        "mae_eval_mean": _json_float(subset["mae_eval"].mean()) if "mae_eval" in subset.columns else np.nan,
        "rmse_eval_mean": _json_float(subset["rmse_eval"].mean()) if "rmse_eval" in subset.columns else np.nan,

        "tt_real_total": _json_float(tt_real_total),
        "tt_pred_total": _json_float(tt_pred_total),
        "tt_signed_diff_total": _json_float(signed_total),
        "tt_abs_diff_total": _json_float(abs_total),
        "tt_rel_diff_total": _json_float(abs_total / tt_real_total) if tt_real_total != 0 else np.nan,

        "underprediction_fraction": _json_float((subset["tt_signed_diff"] < 0).mean()),
        "overprediction_fraction": _json_float((subset["tt_signed_diff"] > 0).mean()),
    })

    return row


def save_detailed_tt_descriptive_stats(
    batch_dir: Path,
    per_file_df: pd.DataFrame,
    hist_bins: int = 40,
    top_n: int = 10,
) -> dict:
    """
    Zapisuje:
      - statystyki opisowe TT,
      - odchylenia, wariancje, kwantyle,
      - histogramy,
      - błąd modelu w dolnym i górnym kwartylu TT_real,
      - nazwy plików z najniższym i najwyższym TT.
    """

    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if ok_df.empty:
        print("[WARN] Brak poprawnych plików do statystyk opisowych TT.")
        return {}

    report_dir = batch_dir / "tt_descriptive_report"
    hist_dir = report_dir / "histograms"

    report_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Kolumny pomocnicze.
    ok_df["tt_pred_over_real"] = np.where(
        ok_df["tt_real"] != 0,
        ok_df["tt_pred"] / ok_df["tt_real"],
        np.nan,
    )

    ok_df["tt_signed_rel_diff"] = np.where(
        ok_df["tt_real"] != 0,
        ok_df["tt_signed_diff"] / ok_df["tt_real"],
        np.nan,
    )

    ok_df["tt_abs_rel_diff_percent"] = 100.0 * ok_df["tt_rel_diff"]
    ok_df["tt_signed_rel_diff_percent"] = 100.0 * ok_df["tt_signed_rel_diff"]

    numeric_cols = [
        "tt_real",
        "tt_pred",
        "tt_signed_diff",
        "tt_abs_diff",
        "tt_rel_diff",
        "tt_abs_rel_diff_percent",
        "tt_signed_rel_diff",
        "tt_signed_rel_diff_percent",
        "tt_pred_over_real",
        "mae_eval",
        "rmse_eval",
    ]

    numeric_cols = [c for c in numeric_cols if c in ok_df.columns]

    # --------------------------------------------------------
    # Statystyki opisowe + kwantyle
    # --------------------------------------------------------
    stat_rows = []
    quantile_rows = []

    quantiles = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    for col in numeric_cols:
        s = (
            pd.to_numeric(ok_df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if s.empty:
            continue

        q = s.quantile(quantiles)

        mean = s.mean()
        std_sample = s.std(ddof=1) if s.count() > 1 else np.nan
        std_population = s.std(ddof=0) if s.count() > 0 else np.nan
        variance_sample = s.var(ddof=1) if s.count() > 1 else np.nan

        stat_rows.append({
            "metric": col,
            "count": int(s.count()),
            "missing": int(len(ok_df[col]) - s.count()),

            "mean": _json_float(mean),
            "std": _json_float(std_sample),
            "std_sample": _json_float(std_sample),
            "std_population": _json_float(std_population),
            "variance_sample": _json_float(variance_sample),
            "sem": _json_float(std_sample / math.sqrt(s.count())) if s.count() > 1 else np.nan,

            "min": _json_float(q.loc[0.00]),
            "p01": _json_float(q.loc[0.01]),
            "p05": _json_float(q.loc[0.05]),
            "p10": _json_float(q.loc[0.10]),
            "p25": _json_float(q.loc[0.25]),
            "median": _json_float(q.loc[0.50]),
            "p75": _json_float(q.loc[0.75]),
            "p90": _json_float(q.loc[0.90]),
            "p95": _json_float(q.loc[0.95]),
            "p99": _json_float(q.loc[0.99]),
            "max": _json_float(q.loc[1.00]),

            "range": _json_float(q.loc[1.00] - q.loc[0.00]),
            "iqr": _json_float(q.loc[0.75] - q.loc[0.25]),
            "mad_from_mean": _json_float(np.mean(np.abs(s - mean))),
            "mad_from_median": _json_float(np.median(np.abs(s - s.median()))),

            "cv": _json_float(std_sample / mean) if s.count() > 1 and mean != 0 else np.nan,
            "skew": _json_float(s.skew()) if s.count() > 2 else np.nan,
            "kurtosis": _json_float(s.kurtosis()) if s.count() > 3 else np.nan,
        })

        for level, value in q.items():
            quantile_rows.append({
                "metric": col,
                "quantile": float(level),
                "value": _json_float(value),
            })

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(batch_dir / "tt_descriptive_stats.csv", index=False)
    stats_df.to_csv(report_dir / "tt_descriptive_stats.csv", index=False)

    quantile_df = pd.DataFrame(quantile_rows)
    quantile_df.to_csv(report_dir / "tt_quantiles_long.csv", index=False)

    if not quantile_df.empty:
        (
            quantile_df
            .pivot(index="metric", columns="quantile", values="value")
            .reset_index()
            .to_csv(report_dir / "tt_quantiles_wide.csv", index=False)
        )

    # --------------------------------------------------------
    # Histogramy
    # --------------------------------------------------------
    hist_specs = {
        "tt_real": ("Histogram real TT", "Ground truth TT"),
        "tt_pred": ("Histogram predicted TT", "Predicted TT"),
        "tt_signed_diff": ("Histogram signed TT error", "Predicted TT - Ground truth TT"),
        "tt_abs_diff": ("Histogram absolute TT error", "|Predicted TT - Ground truth TT|"),
        "tt_abs_rel_diff_percent": ("Histogram relative TT error", "|pred-real| / real [%]"),
        "mae_eval": ("Histogram MAE per file", "MAE eval"),
        "rmse_eval": ("Histogram RMSE per file", "RMSE eval"),
    }

    for metric, (title, xlabel) in hist_specs.items():
        if metric in ok_df.columns:
            _save_histogram_with_bins(
                out_dir=hist_dir,
                metric=metric,
                values=ok_df[metric],
                bins=hist_bins,
                title=title,
                xlabel=xlabel,
            )

    # --------------------------------------------------------
    # Korelacje i bias
    # --------------------------------------------------------
    corr_pearson = ok_df[["tt_real", "tt_pred"]].corr(method="pearson").iloc[0, 1]
    corr_spearman = ok_df[["tt_real", "tt_pred"]].corr(method="spearman").iloc[0, 1]

    under_mask = ok_df["tt_signed_diff"] < 0
    over_mask = ok_df["tt_signed_diff"] > 0

    # --------------------------------------------------------
    # Błąd modelu w najniższym i najwyższym kwartylu TT_real
    # --------------------------------------------------------
    tt_q25 = float(ok_df["tt_real"].quantile(0.25))
    tt_q75 = float(ok_df["tt_real"].quantile(0.75))

    low_quartile_df = ok_df[ok_df["tt_real"] <= tt_q25].copy()
    high_quartile_df = ok_df[ok_df["tt_real"] >= tt_q75].copy()

    quartile_df = pd.DataFrame([
        _quartile_error_summary(
            "lowest_25pct_by_tt_real",
            low_quartile_df,
            tt_q25,
            tt_q75,
        ),
        _quartile_error_summary(
            "highest_25pct_by_tt_real",
            high_quartile_df,
            tt_q25,
            tt_q75,
        ),
    ])

    quartile_df.to_csv(report_dir / "tt_error_low_high_real_tt_quartiles.csv", index=False)
    quartile_df.to_csv(batch_dir / "tt_error_low_high_real_tt_quartiles.csv", index=False)

    # Pełna analiza po kwartylach.
    try:
        ok_df["tt_real_quartile"] = pd.qcut(
            ok_df["tt_real"],
            q=4,
            duplicates="drop",
        )

        by_quartile_df = (
            ok_df
            .groupby("tt_real_quartile", observed=True)
            .agg(
                n_files=("file_name", "count"),
                tt_real_min=("tt_real", "min"),
                tt_real_mean=("tt_real", "mean"),
                tt_real_median=("tt_real", "median"),
                tt_real_max=("tt_real", "max"),
                tt_pred_mean=("tt_pred", "mean"),
                tt_signed_diff_mean=("tt_signed_diff", "mean"),
                tt_abs_diff_mean=("tt_abs_diff", "mean"),
                tt_rel_diff_mean=("tt_rel_diff", "mean"),
                tt_rel_diff_median=("tt_rel_diff", "median"),
                mae_eval_mean=("mae_eval", "mean"),
                rmse_eval_mean=("rmse_eval", "mean"),
            )
            .reset_index()
        )

        by_quartile_df["tt_real_quartile"] = by_quartile_df["tt_real_quartile"].astype(str)
        by_quartile_df["tt_rel_diff_percent_mean"] = 100.0 * by_quartile_df["tt_rel_diff_mean"]
        by_quartile_df["tt_rel_diff_percent_median"] = 100.0 * by_quartile_df["tt_rel_diff_median"]

        by_quartile_df.to_csv(report_dir / "tt_error_by_real_tt_quartile.csv", index=False)

    except Exception as e:
        print(f"[WARN] Nie udało się policzyć tt_error_by_real_tt_quartile.csv: {repr(e)}")

    # Zachowanie poprzedniej analizy decylowej.
    try:
        ok_df["tt_real_bin"] = pd.qcut(
            ok_df["tt_real"],
            q=10,
            duplicates="drop",
        )

        bin_df = (
            ok_df
            .groupby("tt_real_bin", observed=True)
            .agg(
                n_files=("file_name", "count"),
                tt_real_mean=("tt_real", "mean"),
                tt_pred_mean=("tt_pred", "mean"),
                tt_signed_diff_mean=("tt_signed_diff", "mean"),
                tt_abs_diff_mean=("tt_abs_diff", "mean"),
                tt_rel_diff_mean=("tt_rel_diff", "mean"),
                tt_rel_diff_median=("tt_rel_diff", "median"),
                mae_eval_mean=("mae_eval", "mean"),
                rmse_eval_mean=("rmse_eval", "mean"),
            )
            .reset_index()
        )

        bin_df["tt_real_bin"] = bin_df["tt_real_bin"].astype(str)
        bin_df.to_csv(batch_dir / "tt_error_by_real_tt_quantile.csv", index=False)
        bin_df.to_csv(report_dir / "tt_error_by_real_tt_decile.csv", index=False)

    except Exception as e:
        print(f"[WARN] Nie udało się policzyć tt_error_by_real_tt_quantile.csv: {repr(e)}")

    # --------------------------------------------------------
    # Pliki z najniższym i najwyższym TT
    # --------------------------------------------------------
    extreme_cols = [
        "selected_index",
        "file_name",
        "tt_real",
        "tt_pred",
        "tt_signed_diff",
        "tt_abs_diff",
        "tt_rel_diff",
        "mae_eval",
        "rmse_eval",
    ]

    extreme_cols = [c for c in extreme_cols if c in ok_df.columns]

    lowest_tt_df = (
        ok_df
        .sort_values("tt_real", ascending=True)
        .head(top_n)[extreme_cols]
        .copy()
    )

    highest_tt_df = (
        ok_df
        .sort_values("tt_real", ascending=False)
        .head(top_n)[extreme_cols]
        .copy()
    )

    lowest_tt_df.to_csv(report_dir / "tt_files_lowest_real_tt.csv", index=False)
    highest_tt_df.to_csv(report_dir / "tt_files_highest_real_tt.csv", index=False)

    lowest_tt_df.to_csv(batch_dir / "tt_files_lowest_real_tt.csv", index=False)
    highest_tt_df.to_csv(batch_dir / "tt_files_highest_real_tt.csv", index=False)

    with open(report_dir / "tt_extreme_files.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "lowest_real_tt": lowest_tt_df.to_dict(orient="records"),
                "highest_real_tt": highest_tt_df.to_dict(orient="records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary JSON/CSV
    # --------------------------------------------------------
    summary = {
        "n_files": int(len(ok_df)),
        "hist_bins_requested": int(hist_bins),
        "extreme_files_n": int(top_n),

        "tt_real_q25": _json_float(tt_q25),
        "tt_real_q75": _json_float(tt_q75),

        "tt_real_mean": _json_float(ok_df["tt_real"].mean()),
        "tt_pred_mean": _json_float(ok_df["tt_pred"].mean()),

        "tt_real_median": _json_float(ok_df["tt_real"].median()),
        "tt_pred_median": _json_float(ok_df["tt_pred"].median()),

        "tt_signed_error_mean": _json_float(ok_df["tt_signed_diff"].mean()),
        "tt_signed_error_median": _json_float(ok_df["tt_signed_diff"].median()),

        "tt_abs_error_mean": _json_float(ok_df["tt_abs_diff"].mean()),
        "tt_abs_error_median": _json_float(ok_df["tt_abs_diff"].median()),

        "tt_relative_error_mean": _json_float(ok_df["tt_rel_diff"].mean()),
        "tt_relative_error_median": _json_float(ok_df["tt_rel_diff"].median()),

        "tt_relative_error_percent_mean": _json_float(100.0 * ok_df["tt_rel_diff"].mean()),
        "tt_relative_error_percent_median": _json_float(100.0 * ok_df["tt_rel_diff"].median()),

        "tt_signed_relative_error_mean": _json_float(ok_df["tt_signed_rel_diff"].mean()),
        "tt_signed_relative_error_median": _json_float(ok_df["tt_signed_rel_diff"].median()),

        "tt_signed_relative_error_percent_mean": _json_float(100.0 * ok_df["tt_signed_rel_diff"].mean()),
        "tt_signed_relative_error_percent_median": _json_float(100.0 * ok_df["tt_signed_rel_diff"].median()),

        "tt_pred_over_real_mean": _json_float(ok_df["tt_pred_over_real"].mean()),
        "tt_pred_over_real_median": _json_float(ok_df["tt_pred_over_real"].median()),

        "pearson_corr_real_pred_tt": _json_float(corr_pearson),
        "spearman_corr_real_pred_tt": _json_float(corr_spearman),

        "underprediction_count": int(under_mask.sum()),
        "overprediction_count": int(over_mask.sum()),
        "exact_count": int((ok_df["tt_signed_diff"] == 0).sum()),

        "underprediction_fraction": _json_float(under_mask.mean()),
        "overprediction_fraction": _json_float(over_mask.mean()),

        "lowest_tt_files": lowest_tt_df["file_name"].astype(str).tolist(),
        "highest_tt_files": highest_tt_df["file_name"].astype(str).tolist(),

        "report_dir": str(report_dir),
    }

    for _, qrow in quartile_df.iterrows():
        prefix = (
            "low_quartile"
            if qrow["quartile_group"] == "lowest_25pct_by_tt_real"
            else "high_quartile"
        )

        for key in [
            "n_files",
            "tt_real_mean",
            "tt_pred_mean",
            "tt_abs_diff_mean",
            "tt_rel_diff_percent_mean",
            "mae_eval_mean",
            "rmse_eval_mean",
            "tt_rel_diff_total",
        ]:
            summary[f"{prefix}_{key}"] = (
                int(qrow[key])
                if key == "n_files"
                else _json_float(qrow[key])
            )

    with open(batch_dir / "tt_descriptive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(report_dir / "tt_descriptive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_for_csv = dict(summary)
    summary_for_csv["lowest_tt_files"] = " | ".join(summary["lowest_tt_files"])
    summary_for_csv["highest_tt_files"] = " | ".join(summary["highest_tt_files"])

    pd.DataFrame([summary_for_csv]).to_csv(batch_dir / "tt_descriptive_summary.csv", index=False)
    pd.DataFrame([summary_for_csv]).to_csv(report_dir / "tt_descriptive_summary.csv", index=False)

    print("\n=== TT descriptive summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nPliki z najniższym real TT:")
    for file_name in summary["lowest_tt_files"]:
        print(f"  - {file_name}")

    print("\nPliki z najwyższym real TT:")
    for file_name in summary["highest_tt_files"]:
        print(f"  - {file_name}")

    print(f"Zapisano raport TT w: {report_dir}")

    return summary


def plot_real_vs_pred_tt_yzoom(batch_dir: Path, per_file_df: pd.DataFrame):
    """
    Drugi scatterplot: ta sama relacja real TT vs pred TT,
    ale oś Y jest skalowana do zakresu predicted TT,
    a nie do wspólnej skali z linią y=x.

    Dzięki temu widać zmienność predykcji.
    """

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

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

    x_lo = x_min - x_pad
    x_hi = x_max + x_pad
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    # Prosta regresja liniowa pred ~ real
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x_lo, x_hi, 200)
        y_line = slope * x_line + intercept
    else:
        slope, intercept = np.nan, np.nan
        x_line = np.array([])
        y_line = np.array([])

    plt.figure(figsize=(9, 6))

    plt.scatter(x, y, alpha=0.55, label="assignments")

    # Linia trendu
    if len(x_line) > 0:
        plt.plot(
            x_line,
            y_line,
            linewidth=1.5,
            label=f"trend: pred = {slope:.3f} * real + {intercept:.2g}",
        )

    # Linia idealna też zostaje, ale nie wymusza skali osi
    ideal_y = np.array([x_lo, x_hi])
    plt.plot(
        [x_lo, x_hi],
        ideal_y,
        linestyle="--",
        linewidth=1,
        label="real = predicted",
    )

    plt.axhline(
        np.mean(y),
        linestyle=":",
        linewidth=1,
        label=f"mean pred TT = {np.mean(y):.3g}",
    )

    plt.title("Real TT vs predicted TT — zoom osi Y")
    plt.xlabel("Ground truth TT")
    plt.ylabel("Predicted TT")

    plt.xlim(x_lo, x_hi)
    plt.ylim(y_lo, y_hi)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "real_tt_vs_predicted_tt_yzoom_dirichlet.png", dpi=240)
    plt.close()


def plot_tt_signed_error_vs_real_tt(batch_dir: Path, per_file_df: pd.DataFrame):
    """
    Bardzo pomocny wykres diagnostyczny:
    pokazuje, czy model zaniża/zawyża TT zależnie od wielkości real TT.
    """

    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if ok_df.empty:
        return

    x = ok_df["tt_real"].to_numpy(dtype=float)
    y = ok_df["tt_signed_diff"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) == 0:
        return

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x_min - x_pad, x_max + x_pad, 200)
        y_line = slope * x_line + intercept
    else:
        slope, intercept = np.nan, np.nan
        x_line = np.array([])
        y_line = np.array([])

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.75, label="pliki")
    plt.axhline(0.0, linestyle="--", linewidth=1, label="brak biasu")

    if len(x_line) > 0:
        plt.plot(
            x_line,
            y_line,
            linewidth=1.5,
            label=f"trend błędu: err = {slope:.3f} * real + {intercept:.2g}",
        )

    plt.title("Signed TT error vs real TT")
    plt.xlabel("Ground truth TT")
    plt.ylabel("Predicted TT - Ground truth TT")
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_signed_error_vs_real_tt.png", dpi=240)
    plt.close()


def plot_tt_relative_error_vs_real_tt(batch_dir: Path, per_file_df: pd.DataFrame):
    """
    Wykres błędu względnego TT [%] względem real TT.
    """

    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if ok_df.empty:
        return

    x = ok_df["tt_real"].to_numpy(dtype=float)
    y = 100.0 * ok_df["tt_rel_diff"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) == 0:
        return

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.75)
    plt.axhline(np.mean(y), linestyle=":", linewidth=1, label=f"mean = {np.mean(y):.2f}%")
    plt.axhline(np.median(y), linestyle="--", linewidth=1, label=f"median = {np.median(y):.2f}%")

    plt.title("Relative TT error vs real TT")
    plt.xlabel("Ground truth TT")
    plt.ylabel("|Predicted TT - Ground truth TT| / Ground truth TT [%]")
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_relative_error_vs_real_tt.png", dpi=240)
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
    plt.scatter(x, y, alpha=0.55)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="real = pred")

    total_rel = summary.get("tt_rel_diff_total", np.nan)

    if np.isfinite(total_rel):
        text = (
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

    plt.title("Real TT vs predicted TT")
    plt.xlabel("Ground truth TT")
    plt.ylabel("Predicted TT")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "real_tt_vs_predicted_tt_dirichlet.png", dpi=240)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(["real TT", "predicted TT"], [summary["tt_real_total"], summary["tt_pred_total"]])
    plt.title("Suma TT dla całego zbioru")
    plt.ylabel("TT")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(batch_dir / "global_real_tt_vs_predicted_tt.png", dpi=220)
    plt.close()

    plot_df = ok_df.sort_values("tt_rel_diff", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(max(10, 0.25 * len(plot_df)), 5))
    plt.bar(np.arange(len(plot_df)), 100.0 * plot_df["tt_rel_diff"].to_numpy(dtype=float))
    plt.title("TT relative error per file")
    plt.xlabel("plik posortowany malejąco po błędzie względnym")
    plt.ylabel("|pred-real| / real [%]")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(batch_dir / "tt_relative_error_per_file.png", dpi=220)
    plt.close()


def write_error_log(batch_dir: Path, file_name: str, error: Exception):
    err_dir = batch_dir / "errors"
    err_dir.mkdir(exist_ok=True)

    path = err_dir / f"{safe_stem(file_name)}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(repr(error))
        f.write("\n")


# ============================================================
# Batch processing
# ============================================================

def process_items_batch(
    args,
    batch_dir: Path,
    model,
    device,
    items: list[LoadedItem],
    agg: AggregateState,
    rows: list[dict],
):
    """
    Przetwarza grupę plików o takim samym T.
    Wewnątrz idzie pełny batch na GPU.
    """

    if not items:
        return

    # Dla bezpieczeństwa sprawdzamy T.
    t_values = {item.real_q_TN.shape[0] for item in items}

    if len(t_values) != 1:
        raise ValueError(f"process_items_batch dostał różne T: {sorted(t_values)}")

    T = items[0].real_q_TN.shape[0]
    seed_steps = max(args.seq_length_q, args.seq_length_a)

    if T <= seed_steps:
        for item in items:
            msg = f"Za krótka sekwencja: T={T}, seed_steps={seed_steps}"
            row = {
                "selected_index": int(item.selected_index),
                "file_name": item.file_name,
                "status": "FAILED",
                "status_msg": msg,
            }
            rows.append(row)
            agg.update_failed()
        return

    real_q_BTN = np.stack([item.real_q_TN for item in items], axis=0).astype(np.float32)
    assign_BTN = np.stack([item.assign_TN for item in items], axis=0).astype(np.float32)

    generated_BTN, seed_steps = rollout_many_sequences_gpu(
        model=model,
        device=device,
        real_q_BTN=real_q_BTN,
        assign_BTN=assign_BTN,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
    )

    for b, item in enumerate(items):
        pred_TN = generated_BTN[b]
        real_TN = item.real_q_TN

        row, details = compute_one_file_metrics(
            pred_TN=pred_TN,
            real_TN=real_TN,
            file_name=item.file_name,
            selected_index=item.selected_index,
            seed_steps=seed_steps,
            nodes_to_copy=item.nodes_to_copy,
            delta_t=args.delta_t,
        )

        add_two_band_debug_columns(
            row=row,
            pred_TN=pred_TN,
            real_TN=real_TN,
            assign_TN=item.assign_TN,
            seed_steps=seed_steps,
            current_nodes=item.current_nodes,
            nodes_to_copy=item.nodes_to_copy,
            delta_t=args.delta_t,
        )

        rows.append(row)
        agg.update_ok(row, details)

        save_arrays_if_needed(
            args=args,
            batch_dir=batch_dir,
            item=item,
            pred_TN=pred_TN,
        )

        print(
            f"[OK] idx={item.selected_index} {item.file_name}: "
            f"TT real={row['tt_real']:.6f}, "
            f"TT pred={row['tt_pred']:.6f}, "
            f"rel diff={row['tt_rel_diff']:.3%}, "
            f"MAE={row['mae_eval']:.6f}",
            flush=True,
        )

    # Zwolnienie pamięci po batchu.
    del real_q_BTN
    del assign_BTN
    del generated_BTN

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def flush_bucket(
    args,
    batch_dir: Path,
    model,
    device,
    bucket_items: list[LoadedItem],
    agg: AggregateState,
    rows: list[dict],
):
    """
    Jeśli batch na GPU padnie przez OOM, zmniejszamy batch rekurencyjnie.
    """
    if not bucket_items:
        return

    try:
        process_items_batch(
            args=args,
            batch_dir=batch_dir,
            model=model,
            device=device,
            items=bucket_items,
            agg=agg,
            rows=rows,
        )

    except RuntimeError as e:
        msg = str(e).lower()

        if "out of memory" in msg and len(bucket_items) > 1:
            print(
                f"[OOM] Batch size {len(bucket_items)} za duży. "
                f"Dzielę na pół i próbuję dalej.",
                flush=True,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mid = len(bucket_items) // 2

            flush_bucket(
                args=args,
                batch_dir=batch_dir,
                model=model,
                device=device,
                bucket_items=bucket_items[:mid],
                agg=agg,
                rows=rows,
            )

            flush_bucket(
                args=args,
                batch_dir=batch_dir,
                model=model,
                device=device,
                bucket_items=bucket_items[mid:],
                agg=agg,
                rows=rows,
            )

            return

        raise


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Batch inference na jednej GPU: ładuje model raz i liczy wiele plików "
            "równolegle w batchach (B, T, N)."
        )
    )

    # Pliki
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

    # Batch GPU
    p.add_argument(
        "--batch_size_files",
        type=int,
        default=16,
        help="Ile plików/symulacji liczyć jednocześnie na GPU.",
    )

    # Statystyki
    p.add_argument("--delta_t", type=float, default=10.0)

    # Debug/output
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--save_arrays", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument(
        "--hist_bins",
        type=int,
        default=40,
        help="Liczba koszyków histogramów w raporcie TT.",
    )

    p.add_argument(
        "--extreme_files_n",
        type=int,
        default=10,
        help="Ile nazw plików z najniższym i najwyższym TT zapisać/wypisać.",
    )

    p.add_argument(
        "--plot_style",
        nargs="*",
        default=["science", "no-latex"],
        help=(
            "Style matplotlib/SciencePlots, np. --plot_style science no-latex "
            "albo --plot_style science ieee no-latex."
        ),
    )

    return p

# ============================================================
# Diagnostyka collapse / dwóch poziomów TT_pred
# ============================================================

def _add_flat_stats(row: dict, prefix: str, arr: np.ndarray, eps: float = 1e-9):
    """
    Dodaje do row statystyki rozkładu tablicy.
    Używane dla pred_eval, real_eval, assign_eval, sum po czasie itd.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    row[f"{prefix}_n"] = int(a.size)

    if a.size == 0:
        for key in [
            "finite_n", "sum", "abs_sum", "mean", "std", "min", "p01",
            "p05", "p25", "median", "p75", "p95", "p99", "max",
            "range", "iqr", "cv_abs_mean", "nonzero_count",
            "positive_count", "negative_count", "zero_fraction",
        ]:
            row[f"{prefix}_{key}"] = np.nan
        return

    finite = np.isfinite(a)
    v = a[finite]
    row[f"{prefix}_finite_n"] = int(v.size)

    if v.size == 0:
        return

    q01, q05, q25, q50, q75, q95, q99 = np.quantile(
        v, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0

    row[f"{prefix}_sum"] = float(np.sum(v))
    row[f"{prefix}_abs_sum"] = float(np.sum(np.abs(v)))
    row[f"{prefix}_mean"] = mean
    row[f"{prefix}_std"] = std
    row[f"{prefix}_min"] = float(np.min(v))
    row[f"{prefix}_p01"] = float(q01)
    row[f"{prefix}_p05"] = float(q05)
    row[f"{prefix}_p25"] = float(q25)
    row[f"{prefix}_median"] = float(q50)
    row[f"{prefix}_p75"] = float(q75)
    row[f"{prefix}_p95"] = float(q95)
    row[f"{prefix}_p99"] = float(q99)
    row[f"{prefix}_max"] = float(np.max(v))
    row[f"{prefix}_range"] = float(np.max(v) - np.min(v))
    row[f"{prefix}_iqr"] = float(q75 - q25)
    row[f"{prefix}_cv_abs_mean"] = float(std / abs(mean)) if abs(mean) > eps else np.nan

    row[f"{prefix}_nonzero_count"] = int(np.count_nonzero(np.abs(v) > eps))
    row[f"{prefix}_positive_count"] = int(np.count_nonzero(v > eps))
    row[f"{prefix}_negative_count"] = int(np.count_nonzero(v < -eps))
    row[f"{prefix}_zero_fraction"] = float(np.mean(np.abs(v) <= eps))


def add_two_band_debug_columns(
    row: dict,
    pred_TN: np.ndarray,
    real_TN: np.ndarray,
    assign_TN: np.ndarray,
    seed_steps: int,
    current_nodes: int,
    nodes_to_copy: int,
    delta_t: float,
):
    """
    Rozszerza pojedynczy row o statystyki, które pozwalają ustalić,
    skąd biorą się dwa poziomy TT_pred.

    Kluczowe pytania:
      1. Czy pred_eval ma prawie stałą średnią?
      2. Czy TT_pred = średnia_pred * liczba_elementów * delta_t?
      3. Czy poziomy korelują z T / liczbą wartości / assignment maską?
      4. Czy assignmenty mają dwie rodziny?
    """

    pred_eval = np.asarray(pred_TN[seed_steps:, :nodes_to_copy], dtype=np.float64)
    real_eval = np.asarray(real_TN[seed_steps:, :nodes_to_copy], dtype=np.float64)
    assign_eval = np.asarray(assign_TN[seed_steps:, :nodes_to_copy], dtype=np.float64)

    err_eval = pred_eval - real_eval

    row["current_nodes"] = int(current_nodes)
    row["nodes_to_copy"] = int(nodes_to_copy)
    row["eval_values"] = int(pred_eval.size)

    # Rekonstrukcja TT z mean * liczba elementów.
    # Jeśli dwa poziomy wynikają głównie z pred_eval_mean, to tutaj będzie to widać.
    if pred_eval.size > 0:
        row["tt_pred_from_pred_mean_check"] = float(delta_t * np.mean(pred_eval) * pred_eval.size)
        row["tt_real_from_real_mean_check"] = float(delta_t * np.mean(real_eval) * real_eval.size)

        row["tt_pred_per_eval_value"] = float(row["tt_pred"] / (delta_t * pred_eval.size))
        row["tt_real_per_eval_value"] = float(row["tt_real"] / (delta_t * real_eval.size))

        row["tt_pred_minus_mean_reconstruction"] = float(
            row["tt_pred"] - row["tt_pred_from_pred_mean_check"]
        )
        row["tt_real_minus_mean_reconstruction"] = float(
            row["tt_real"] - row["tt_real_from_real_mean_check"]
        )
    else:
        row["tt_pred_from_pred_mean_check"] = np.nan
        row["tt_real_from_real_mean_check"] = np.nan
        row["tt_pred_per_eval_value"] = np.nan
        row["tt_real_per_eval_value"] = np.nan
        row["tt_pred_minus_mean_reconstruction"] = np.nan
        row["tt_real_minus_mean_reconstruction"] = np.nan

    # Statystyki wartości q.
    _add_flat_stats(row, "pred_eval", pred_eval)
    _add_flat_stats(row, "real_eval", real_eval)
    _add_flat_stats(row, "err_eval", err_eval)

    # Statystyki assignmentów.
    _add_flat_stats(row, "assign_eval", assign_eval)

    # Ile różnych wartości assignmentu — przydatne, jeśli jest one-hot / binarne / thresholdowane.
    if assign_eval.size > 0:
        row["assign_eval_unique_count_rounded_6"] = int(
            len(np.unique(np.round(assign_eval.ravel(), 6)))
        )
    else:
        row["assign_eval_unique_count_rounded_6"] = 0

    # Sumy po czasie: jeśli poziomy TT_pred wynikają ze stałego profilu per timestep,
    # będzie to widoczne w pred_per_t_sum_mean/std.
    pred_per_t_sum = np.sum(pred_eval, axis=1)
    real_per_t_sum = np.sum(real_eval, axis=1)
    assign_per_t_sum = np.sum(assign_eval, axis=1)

    pred_per_t_abs_sum = np.sum(np.abs(pred_eval), axis=1)
    real_per_t_abs_sum = np.sum(np.abs(real_eval), axis=1)
    assign_per_t_abs_sum = np.sum(np.abs(assign_eval), axis=1)

    _add_flat_stats(row, "pred_per_t_sum", pred_per_t_sum)
    _add_flat_stats(row, "real_per_t_sum", real_per_t_sum)
    _add_flat_stats(row, "assign_per_t_sum", assign_per_t_sum)

    _add_flat_stats(row, "pred_per_t_abs_sum", pred_per_t_abs_sum)
    _add_flat_stats(row, "real_per_t_abs_sum", real_per_t_abs_sum)
    _add_flat_stats(row, "assign_per_t_abs_sum", assign_per_t_abs_sum)

    # Sumy po node'ach: jeśli tylko wybrane node'y generują poziomy,
    # będzie to widoczne w pred_per_node_sum/std/range.
    pred_per_node_sum = np.sum(pred_eval, axis=0)
    real_per_node_sum = np.sum(real_eval, axis=0)
    assign_per_node_sum = np.sum(assign_eval, axis=0)

    _add_flat_stats(row, "pred_per_node_sum", pred_per_node_sum)
    _add_flat_stats(row, "real_per_node_sum", real_per_node_sum)
    _add_flat_stats(row, "assign_per_node_sum", assign_per_node_sum)

    # Bardzo bezpośrednia miara "collapse":
    # ile zmienności ma predykcja względem reala.
    pred_std = row.get("pred_eval_std", np.nan)
    real_std = row.get("real_eval_std", np.nan)

    row["pred_real_std_ratio_eval"] = (
        float(pred_std / real_std)
        if np.isfinite(pred_std) and np.isfinite(real_std) and real_std != 0
        else np.nan
    )

    pred_range = row.get("pred_eval_range", np.nan)
    real_range = row.get("real_eval_range", np.nan)

    row["pred_real_range_ratio_eval"] = (
        float(pred_range / real_range)
        if np.isfinite(pred_range) and np.isfinite(real_range) and real_range != 0
        else np.nan
    )


def _kmeans_1d_two_clusters(values: np.ndarray, max_iter: int = 100):
    """
    Prosty 1D k-means bez sklearn.
    Zwraca:
      labels: 0 dla low, 1 dla high, -1 dla NaN
      centers_sorted: [center_low, center_high]
      threshold: środek między centrami
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    v = values[finite]

    labels_full = np.full(values.shape, -1, dtype=int)

    if v.size < 2 or np.unique(v).size < 2:
        center = float(np.nanmean(v)) if v.size else np.nan
        return labels_full, np.array([center, center], dtype=np.float64), np.nan

    centers = np.array(
        [np.quantile(v, 0.25), np.quantile(v, 0.75)],
        dtype=np.float64,
    )

    if centers[0] == centers[1]:
        centers = np.array([np.min(v), np.max(v)], dtype=np.float64)

    labels = np.zeros(v.shape, dtype=int)

    for _ in range(max_iter):
        dist = np.abs(v[:, None] - centers[None, :])
        new_labels = np.argmin(dist, axis=1)

        new_centers = centers.copy()
        for k in [0, 1]:
            if np.any(new_labels == k):
                new_centers[k] = np.mean(v[new_labels == k])

        if np.allclose(new_centers, centers, rtol=1e-10, atol=1e-10):
            labels = new_labels
            centers = new_centers
            break

        labels = new_labels
        centers = new_centers

    order = np.argsort(centers)
    low_cluster = order[0]
    high_cluster = order[1]

    labels_low_high = np.where(labels == low_cluster, 0, 1)
    labels_full[finite] = labels_low_high

    centers_sorted = centers[order]
    threshold = float(np.mean(centers_sorted))

    return labels_full, centers_sorted, threshold


def _linear_fit_summary(name: str, df: pd.DataFrame) -> dict:
    d = df[["tt_real", "tt_pred"]].replace([np.inf, -np.inf], np.nan).dropna()

    out = {
        "group": name,
        "n": int(len(d)),
        "slope_pred_vs_real": np.nan,
        "intercept_pred_vs_real": np.nan,
        "pearson_real_pred": np.nan,
        "spearman_real_pred": np.nan,
    }

    if len(d) >= 2 and d["tt_real"].nunique() > 1:
        slope, intercept = np.polyfit(
            d["tt_real"].to_numpy(dtype=float),
            d["tt_pred"].to_numpy(dtype=float),
            deg=1,
        )

        out["slope_pred_vs_real"] = float(slope)
        out["intercept_pred_vs_real"] = float(intercept)
        out["pearson_real_pred"] = float(d["tt_real"].corr(d["tt_pred"], method="pearson"))
        out["spearman_real_pred"] = float(d["tt_real"].corr(d["tt_pred"], method="spearman"))

    return out


def save_two_band_collapse_diagnostics(batch_dir: Path, per_file_df: pd.DataFrame):
    """
    Główna diagnostyka dwóch poziomów TT_pred.

    Tworzy katalog:
        two_band_diagnostics/

    Najważniejsze pliki:
        per_file_metrics_with_pred_bands.csv
        tt_pred_band_summary.csv
        feature_separation_low_vs_high.csv
        correlations_with_tt_pred.csv
        tt_pred_sorted_with_gaps.csv
        collapse_summary.json
        tt_pred_histogram_two_bands.png
        real_vs_pred_tt_colored_by_band.png
        tt_pred_vs_T_colored_by_band.png
    """

    diag_dir = batch_dir / "two_band_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if ok_df.empty:
        print("[WARN] Brak OK rows dla two_band_collapse_diagnostics.")
        return

    if "tt_pred" not in ok_df.columns or "tt_real" not in ok_df.columns:
        print("[WARN] Brak kolumn tt_pred/tt_real.")
        return

    y = ok_df["tt_pred"].to_numpy(dtype=np.float64)
    x = ok_df["tt_real"].to_numpy(dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    ok_df = ok_df.loc[finite].copy()

    if ok_df.empty:
        print("[WARN] Brak skończonych wartości tt_real/tt_pred.")
        return

    y = ok_df["tt_pred"].to_numpy(dtype=np.float64)
    x = ok_df["tt_real"].to_numpy(dtype=np.float64)

    labels, centers, threshold = _kmeans_1d_two_clusters(y)

    ok_df["tt_pred_band_id"] = labels
    ok_df["tt_pred_band"] = np.where(ok_df["tt_pred_band_id"] == 0, "low", "high")
    ok_df.loc[ok_df["tt_pred_band_id"] < 0, "tt_pred_band"] = "nan"

    ok_df["tt_pred_band_threshold"] = threshold
    ok_df["tt_pred_distance_to_band_center"] = np.where(
        ok_df["tt_pred_band_id"] == 0,
        np.abs(ok_df["tt_pred"] - centers[0]),
        np.abs(ok_df["tt_pred"] - centers[1]),
    )

    ok_df.to_csv(diag_dir / "per_file_metrics_with_pred_bands.csv", index=False)

    # --------------------------------------------------------
    # Posortowane TT_pred i największe przerwy
    # --------------------------------------------------------
    sorted_df = ok_df.sort_values("tt_pred").reset_index(drop=True).copy()
    sorted_df["rank_by_tt_pred"] = np.arange(len(sorted_df))

    if len(sorted_df) >= 2:
        gaps = np.diff(sorted_df["tt_pred"].to_numpy(dtype=float))
        sorted_df["gap_to_next_tt_pred"] = np.r_[gaps, np.nan]
    else:
        sorted_df["gap_to_next_tt_pred"] = np.nan

    sorted_df.to_csv(diag_dir / "tt_pred_sorted_with_gaps.csv", index=False)

    largest_gaps = (
        sorted_df
        .dropna(subset=["gap_to_next_tt_pred"])
        .sort_values("gap_to_next_tt_pred", ascending=False)
        .head(20)
        .copy()
    )

    largest_gaps.to_csv(diag_dir / "tt_pred_largest_gaps_top20.csv", index=False)

    # --------------------------------------------------------
    # Podsumowanie pasm
    # --------------------------------------------------------
    agg_spec = {
        "n_files": ("file_name", "count"),

        "tt_pred_mean": ("tt_pred", "mean"),
        "tt_pred_std": ("tt_pred", "std"),
        "tt_pred_min": ("tt_pred", "min"),
        "tt_pred_median": ("tt_pred", "median"),
        "tt_pred_max": ("tt_pred", "max"),

        "tt_real_mean": ("tt_real", "mean"),
        "tt_real_std": ("tt_real", "std"),
        "tt_real_min": ("tt_real", "min"),
        "tt_real_median": ("tt_real", "median"),
        "tt_real_max": ("tt_real", "max"),

        "tt_rel_diff_mean": ("tt_rel_diff", "mean"),
        "tt_rel_diff_median": ("tt_rel_diff", "median"),

        "mae_eval_mean": ("mae_eval", "mean"),
        "mae_eval_median": ("mae_eval", "median"),
        "rmse_eval_mean": ("rmse_eval", "mean"),
    }

    extra_cols = [
        "timesteps_total",
        "timesteps_eval",
        "nodes",
        "current_nodes",
        "nodes_to_copy",
        "eval_values",

        "tt_pred_per_eval_value",
        "tt_real_per_eval_value",

        "pred_eval_mean",
        "pred_eval_std",
        "pred_eval_range",
        "pred_eval_cv_abs_mean",

        "real_eval_mean",
        "real_eval_std",
        "real_eval_range",
        "real_eval_cv_abs_mean",

        "pred_real_std_ratio_eval",
        "pred_real_range_ratio_eval",

        "assign_eval_sum",
        "assign_eval_abs_sum",
        "assign_eval_mean",
        "assign_eval_std",
        "assign_eval_nonzero_count",
        "assign_eval_unique_count_rounded_6",

        "assign_per_t_sum_mean",
        "assign_per_t_sum_std",
        "assign_per_t_sum_min",
        "assign_per_t_sum_max",

        "pred_per_t_sum_mean",
        "pred_per_t_sum_std",
        "pred_per_t_sum_min",
        "pred_per_t_sum_max",

        "real_per_t_sum_mean",
        "real_per_t_sum_std",
        "real_per_t_sum_min",
        "real_per_t_sum_max",
    ]

    for c in extra_cols:
        if c in ok_df.columns:
            for func in ["mean", "std", "min", "median", "max"]:
                agg_spec[f"{c}_{func}"] = (c, func)

    band_summary = (
        ok_df
        .groupby("tt_pred_band", observed=True)
        .agg(**agg_spec)
        .reset_index()
    )

    band_summary.to_csv(diag_dir / "tt_pred_band_summary.csv", index=False)

    # --------------------------------------------------------
    # Crosstaby: czy band zależy od T / nodes / current_nodes?
    # --------------------------------------------------------
    for col in ["timesteps_total", "timesteps_eval", "nodes", "current_nodes", "nodes_to_copy", "eval_values"]:
        if col in ok_df.columns:
            ct = pd.crosstab(ok_df[col], ok_df["tt_pred_band"])
            ct.to_csv(diag_dir / f"band_crosstab_by_{col}.csv")

            group = (
                ok_df
                .groupby([col, "tt_pred_band"], observed=True)
                .agg(
                    n_files=("file_name", "count"),
                    tt_pred_mean=("tt_pred", "mean"),
                    tt_pred_std=("tt_pred", "std"),
                    tt_real_mean=("tt_real", "mean"),
                    tt_rel_diff_mean=("tt_rel_diff", "mean"),
                    mae_eval_mean=("mae_eval", "mean"),
                )
                .reset_index()
            )
            group.to_csv(diag_dir / f"band_summary_by_{col}.csv", index=False)

    # --------------------------------------------------------
    # Korelacje cech z TT_pred
    # --------------------------------------------------------
    numeric_cols = list(ok_df.select_dtypes(include=[np.number]).columns)

    corr_rows = []
    target = ok_df["tt_pred"]

    for c in numeric_cols:
        if c == "tt_pred":
            continue

        s = pd.to_numeric(ok_df[c], errors="coerce")
        tmp = pd.DataFrame({"x": s, "y": target}).replace([np.inf, -np.inf], np.nan).dropna()

        if len(tmp) < 3 or tmp["x"].nunique() < 2:
            continue

        corr_rows.append({
            "feature": c,
            "n": int(len(tmp)),
            "pearson_with_tt_pred": float(tmp["x"].corr(tmp["y"], method="pearson")),
            "spearman_with_tt_pred": float(tmp["x"].corr(tmp["y"], method="spearman")),
            "abs_spearman_with_tt_pred": float(abs(tmp["x"].corr(tmp["y"], method="spearman"))),
        })

    corr_df = pd.DataFrame(corr_rows)

    if not corr_df.empty:
        corr_df = corr_df.sort_values("abs_spearman_with_tt_pred", ascending=False)
        corr_df.to_csv(diag_dir / "correlations_with_tt_pred.csv", index=False)

    # --------------------------------------------------------
    # Separacja cech low vs high
    # --------------------------------------------------------
    sep_rows = []

    for c in numeric_cols:
        s = pd.to_numeric(ok_df[c], errors="coerce")
        tmp = pd.DataFrame({
            "feature_value": s,
            "band": ok_df["tt_pred_band"],
        }).replace([np.inf, -np.inf], np.nan).dropna()

        low = tmp.loc[tmp["band"] == "low", "feature_value"]
        high = tmp.loc[tmp["band"] == "high", "feature_value"]

        if len(low) == 0 or len(high) == 0:
            continue

        low_std = float(low.std(ddof=1)) if len(low) > 1 else 0.0
        high_std = float(high.std(ddof=1)) if len(high) > 1 else 0.0
        pooled_std = math.sqrt((low_std ** 2 + high_std ** 2) / 2.0)

        low_mean = float(low.mean())
        high_mean = float(high.mean())
        diff = high_mean - low_mean

        sep_rows.append({
            "feature": c,
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "low_mean": low_mean,
            "high_mean": high_mean,
            "high_minus_low": float(diff),
            "low_std": low_std,
            "high_std": high_std,
            "standardized_diff": float(diff / pooled_std) if pooled_std > 0 else np.nan,
            "low_min": float(low.min()),
            "low_max": float(low.max()),
            "high_min": float(high.min()),
            "high_max": float(high.max()),
        })

    sep_df = pd.DataFrame(sep_rows)

    if not sep_df.empty:
        sep_df["abs_standardized_diff"] = sep_df["standardized_diff"].abs()
        sep_df = sep_df.sort_values("abs_standardized_diff", ascending=False)
        sep_df.to_csv(diag_dir / "feature_separation_low_vs_high.csv", index=False)

    # --------------------------------------------------------
    # Regresja globalnie i osobno w pasmach
    # --------------------------------------------------------
    reg_rows = [_linear_fit_summary("all", ok_df)]

    for band_name, sub in ok_df.groupby("tt_pred_band", observed=True):
        reg_rows.append(_linear_fit_summary(str(band_name), sub))

    pd.DataFrame(reg_rows).to_csv(diag_dir / "linear_fit_pred_vs_real_by_band.csv", index=False)

    # --------------------------------------------------------
    # Zbiorczy JSON collapse
    # --------------------------------------------------------
    y_range = float(np.max(y) - np.min(y)) if len(y) else np.nan
    x_range = float(np.max(x) - np.min(x)) if len(x) else np.nan

    y_std = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    x_std = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan

    if len(sorted_df) >= 2:
        largest_gap = float(np.nanmax(sorted_df["gap_to_next_tt_pred"]))
    else:
        largest_gap = np.nan

    summary = {
        "n_files": int(len(ok_df)),

        "tt_pred_center_low": float(centers[0]),
        "tt_pred_center_high": float(centers[1]),
        "tt_pred_center_gap": float(centers[1] - centers[0]),
        "tt_pred_band_threshold": float(threshold) if np.isfinite(threshold) else np.nan,

        "tt_pred_min": float(np.min(y)),
        "tt_pred_max": float(np.max(y)),
        "tt_pred_range": y_range,
        "tt_pred_std": y_std,

        "tt_real_min": float(np.min(x)),
        "tt_real_max": float(np.max(x)),
        "tt_real_range": x_range,
        "tt_real_std": x_std,

        "pred_range_over_real_range": float(y_range / x_range) if x_range != 0 else np.nan,
        "pred_std_over_real_std": float(y_std / x_std) if x_std != 0 else np.nan,

        "largest_gap_between_sorted_tt_pred": largest_gap,
        "largest_gap_fraction_of_pred_range": (
            float(largest_gap / y_range)
            if np.isfinite(largest_gap) and y_range != 0
            else np.nan
        ),

        "low_band_count": int((ok_df["tt_pred_band"] == "low").sum()),
        "high_band_count": int((ok_df["tt_pred_band"] == "high").sum()),
    }

    with open(diag_dir / "collapse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Wykresy
    # --------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.hist(ok_df["tt_pred"].to_numpy(dtype=float), bins=min(60, max(10, len(ok_df) // 4)))
    plt.axvline(centers[0], linestyle="--", linewidth=1.5, label=f"center low = {centers[0]:.3g}")
    plt.axvline(centers[1], linestyle="--", linewidth=1.5, label=f"center high = {centers[1]:.3g}")

    if np.isfinite(threshold):
        plt.axvline(threshold, linestyle=":", linewidth=1.5, label=f"threshold = {threshold:.3g}")

    plt.title("Histogram TT_pred — diagnostyka dwóch poziomów")
    plt.xlabel("TT_pred")
    plt.ylabel("liczba plików")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(diag_dir / "tt_pred_histogram_two_bands.png", dpi=240)
    plt.close()

    plt.figure(figsize=(9, 6))
    for band_name, sub in ok_df.groupby("tt_pred_band", observed=True):
        plt.scatter(
            sub["tt_real"],
            sub["tt_pred"],
            alpha=0.65,
            label=f"{band_name}, n={len(sub)}",
        )

    lo = float(min(ok_df["tt_real"].min(), ok_df["tt_pred"].min()))
    hi = float(max(ok_df["tt_real"].max(), ok_df["tt_pred"].max()))
    pad = 0.03 * (hi - lo) if hi > lo else 1.0

    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1, label="real = pred")
    plt.title("Real TT vs predicted TT — kolor wg pasma TT_pred")
    plt.xlabel("Ground truth TT")
    plt.ylabel("Predicted TT")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(diag_dir / "real_vs_pred_tt_colored_by_band.png", dpi=240)
    plt.close()

    if "timesteps_total" in ok_df.columns:
        plt.figure(figsize=(10, 6))
        for band_name, sub in ok_df.groupby("tt_pred_band", observed=True):
            plt.scatter(
                sub["timesteps_total"],
                sub["tt_pred"],
                alpha=0.65,
                label=f"{band_name}, n={len(sub)}",
            )

        plt.title("TT_pred vs T — czy poziomy zależą od długości sekwencji?")
        plt.xlabel("timesteps_total / T")
        plt.ylabel("TT_pred")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(diag_dir / "tt_pred_vs_T_colored_by_band.png", dpi=240)
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_df["rank_by_tt_pred"], sorted_df["tt_pred"], marker=".", linewidth=1)
    plt.axhline(centers[0], linestyle="--", linewidth=1, label="center low")
    plt.axhline(centers[1], linestyle="--", linewidth=1, label="center high")

    if np.isfinite(threshold):
        plt.axhline(threshold, linestyle=":", linewidth=1, label="threshold")

    plt.title("TT_pred posortowane rosnąco — widoczność przerwy między poziomami")
    plt.xlabel("rank po TT_pred")
    plt.ylabel("TT_pred")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(diag_dir / "tt_pred_sorted_curve.png", dpi=240)
    plt.close()

    print("\n=== TWO-BAND / COLLAPSE DIAGNOSTICS ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Zapisano diagnostykę dwóch poziomów w: {diag_dir}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_plot_style(args.plot_style)

    if args.hist_bins <= 0:
        raise ValueError("--hist_bins musi być > 0")

    if args.extreme_files_n <= 0:
        raise ValueError("--extreme_files_n musi być > 0")

    if args.batch_size_files <= 0:
        raise ValueError("--batch_size_files musi być > 0")

    selected_files, all_test_files = select_test_files(args)

    if not selected_files:
        raise RuntimeError("Nie wybrano żadnych plików do inferencji.")

    batch_name = args.batch_name or f"batch_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = args.output_root / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    save_manifest(
        batch_dir=batch_dir,
        args=args,
        selected_files=selected_files,
        all_test_files=all_test_files,
    )

    print(f"Batch dir: {batch_dir}", flush=True)
    print(f"Liczba plików testowych w splicie: {len(all_test_files)}", flush=True)
    print(f"Wybrano plików: {len(selected_files)}", flush=True)
    print(f"batch_size_files: {args.batch_size_files}", flush=True)

    print("Pierwsze wybrane pliki:", flush=True)
    for f in selected_files[:10]:
        print(f"  - {f}", flush=True)

    if len(selected_files) > 10:
        print(f"  ... +{len(selected_files) - 10}", flush=True)

    if args.dry_run:
        print("Dry run — kończę bez ładowania modelu.", flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}", flush=True)
    print("Ładuję model/checkpoint tylko raz...", flush=True)

    trainer = build_trainer(args, device)
    model = trainer.model
    model.eval()

    print("Model załadowany.", flush=True)

    rows = []
    agg = AggregateState(delta_t=args.delta_t)

    # Buckety po T, bo batchowanie wymaga takiego samego T.
    # Jeśli wszystkie pliki mają tę samą długość, realnie będzie jeden bucket.
    buckets: dict[int, list[LoadedItem]] = {}

    def maybe_flush_bucket(T: int, force: bool = False):
        bucket = buckets.get(T, [])

        if not bucket:
            return

        while len(bucket) >= args.batch_size_files or (force and bucket):
            take = bucket[:args.batch_size_files]
            del bucket[:args.batch_size_files]

            print(
                f"[BATCH] T={T}, B={len(take)}, "
                f"indices={take[0].selected_index}..{take[-1].selected_index}",
                flush=True,
            )

            flush_bucket(
                args=args,
                batch_dir=batch_dir,
                model=model,
                device=device,
                bucket_items=take,
                agg=agg,
                rows=rows,
            )

            # Zapis częściowy po każdym batchu.
            pd.DataFrame(rows).to_csv(batch_dir / "per_file_metrics_partial.csv", index=False)

    for selected_index, file_name in enumerate(selected_files):
        try:
            item = load_one_item(
                args=args,
                selected_index=selected_index + args.start_index,
                file_name=file_name,
            )

            T = item.real_q_TN.shape[0]

            if T not in buckets:
                buckets[T] = []

            buckets[T].append(item)

            maybe_flush_bucket(T, force=False)

        except Exception as e:
            print(f"[FAILED_LOAD] {file_name}: {repr(e)}", flush=True)
            write_error_log(batch_dir, file_name, e)

            rows.append({
                "selected_index": int(selected_index + args.start_index),
                "file_name": file_name,
                "status": "FAILED_LOAD",
                "status_msg": repr(e),
            })

            agg.update_failed()

            if not args.continue_on_error:
                raise

    # Flush wszystkiego, co zostało w bucketach.
    for T in sorted(list(buckets.keys())):
        maybe_flush_bucket(T, force=True)

    per_file_df = pd.DataFrame(rows)
    per_file_df.to_csv(batch_dir / "per_file_metrics.csv", index=False)

    summary = agg.summary_dict()
    summary["batch_dir"] = str(batch_dir)
    summary["checkpoint"] = str(args.checkpoint)
    summary["selected_files_count"] = int(len(selected_files))
    summary["batch_size_files"] = int(args.batch_size_files)
    summary["device"] = str(device)

    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if not ok_df.empty:
        summary["mae_mean_per_file"] = float(ok_df["mae_eval"].mean())
        summary["rmse_mean_per_file"] = float(ok_df["rmse_eval"].mean())
        summary["tt_rel_diff_mean_per_file"] = float(ok_df["tt_rel_diff"].mean())
        summary["tt_rel_diff_median_per_file"] = float(ok_df["tt_rel_diff"].median())

    with open(batch_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(batch_dir / "summary.csv", index=False)

    save_tt_timeseries(batch_dir, agg)
    save_per_node_metrics(batch_dir, agg)

    # Oryginalny wykres z osią 1:1
    plot_real_vs_pred_tt(batch_dir, per_file_df, summary)

    # Nowe wykresy diagnostyczne
    plot_real_vs_pred_tt_yzoom(batch_dir, per_file_df)
    plot_tt_signed_error_vs_real_tt(batch_dir, per_file_df)
    plot_tt_relative_error_vs_real_tt(batch_dir, per_file_df)

    # Szczegółowe statystyki opisowe TT + histogramy + kwartyle + skrajne pliki.
    tt_report_summary = save_detailed_tt_descriptive_stats(
        batch_dir=batch_dir,
        per_file_df=per_file_df,
        hist_bins=args.hist_bins,
        top_n=args.extreme_files_n,
    )

    if tt_report_summary:
        for key in [
            "report_dir",
            "tt_real_q25",
            "tt_real_q75",
            "low_quartile_n_files",
            "low_quartile_tt_abs_diff_mean",
            "low_quartile_tt_rel_diff_percent_mean",
            "low_quartile_mae_eval_mean",
            "low_quartile_rmse_eval_mean",
            "high_quartile_n_files",
            "high_quartile_tt_abs_diff_mean",
            "high_quartile_tt_rel_diff_percent_mean",
            "high_quartile_mae_eval_mean",
            "high_quartile_rmse_eval_mean",
            "lowest_tt_files",
            "highest_tt_files",
        ]:
            if key in tt_report_summary:
                summary[f"tt_report_{key}"] = tt_report_summary[key]

        # Nadpisujemy summary po dodaniu raportu TT.
        with open(batch_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        summary_for_csv = dict(summary)

        for key in ["tt_report_lowest_tt_files", "tt_report_highest_tt_files"]:
            if isinstance(summary_for_csv.get(key), list):
                summary_for_csv[key] = " | ".join(map(str, summary_for_csv[key]))

        pd.DataFrame([summary_for_csv]).to_csv(batch_dir / "summary.csv", index=False)

    # Diagnostyka dwóch poziomów / collapse do baseline'u
    save_two_band_collapse_diagnostics(batch_dir, per_file_df)

    print("\n=== PODSUMOWANIE ZBIORCZE ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    print(f"\nZapisano wyniki w: {batch_dir}", flush=True)
    print("Najważniejszy wykres: real_tt_vs_predicted_tt_dirichlet.png", flush=True)
    print("Raport TT: tt_descriptive_report/", flush=True)


if __name__ == "__main__":
    main()