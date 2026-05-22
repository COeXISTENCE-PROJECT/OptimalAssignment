#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
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

from utils.inference import (
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

    # TT total liczone jako delta_t * sum(q)
    tt_real = float(delta_t * np.sum(real_eval))
    tt_pred = float(delta_t * np.sum(pred_eval))

    tt_signed_diff = float(tt_pred - tt_real)
    tt_abs_diff = float(abs(tt_signed_diff))

    tt_signed_pct_diff = (
        float(100.0 * tt_signed_diff / tt_real)
        if tt_real != 0
        else np.nan
    )

    tt_abs_pct_diff = (
        float(abs(tt_signed_pct_diff))
        if np.isfinite(tt_signed_pct_diff)
        else np.nan
    )

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
        "tt_signed_pct_diff": tt_signed_pct_diff,
        "tt_abs_pct_diff": tt_abs_pct_diff,
    }

    details = {
        "n_values": int(abs_err.size),
        "sum_abs_err": float(np.sum(abs_err)),
        "sum_sq_err": float(np.sum(sq_err)),
    }

    return row, details


class AggregateState:
    """Minimalny stan agregacji potrzebny do raportu TT i metryk eval."""

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

    def update_ok(self, row: dict, details: dict):
        self.files_ok += 1

        self.total_values += details["n_values"]
        self.total_abs_err += details["sum_abs_err"]
        self.total_sq_err += details["sum_sq_err"]

        self.total_tt_real += row["tt_real"]
        self.total_tt_pred += row["tt_pred"]
        self.total_tt_abs_diff_filewise += row["tt_abs_diff"]

    def update_failed(self):
        self.files_failed += 1



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

def _json_float(value):
    """Konwersja numpy/NaN/inf do wartości bezpiecznych dla JSON."""
    try:
        value = float(value)
    except Exception:
        return np.nan

    return value if np.isfinite(value) else np.nan


def _clean_numeric(values) -> pd.Series:
    return (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _common_tt_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    min_v = float(min(np.min(x), np.min(y)))
    max_v = float(max(np.max(x), np.max(y)))

    pad = 0.03 * (max_v - min_v) if max_v > min_v else 1.0

    return min_v - pad, max_v + pad


def compute_linear_regression_stats(x_values, y_values) -> dict:
    """
    Regresja liniowa:
        predicted TT = slope * real TT + intercept

    Bez scipy/statsmodels.
    """
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    n = int(len(x))

    out = {
        "n": n,
        "model": "tt_pred = slope * tt_real + intercept",
        "slope": np.nan,
        "intercept": np.nan,
        "r": np.nan,
        "r_squared": np.nan,
        "residual_mean": np.nan,
        "residual_std": np.nan,
        "residual_mae": np.nan,
        "residual_rmse": np.nan,
        "slope_std_error": np.nan,
        "intercept_std_error": np.nan,
    }

    if n < 2:
        return out

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    x_centered = x - x_mean
    y_centered = y - y_mean

    sxx = float(np.sum(x_centered ** 2))
    syy = float(np.sum(y_centered ** 2))
    sxy = float(np.sum(x_centered * y_centered))

    if sxx <= 0:
        return out

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    y_hat = slope * x + intercept
    residuals = y - y_hat

    sse = float(np.sum(residuals ** 2))

    r = sxy / math.sqrt(sxx * syy) if syy > 0 else np.nan
    r_squared = r ** 2 if np.isfinite(r) else np.nan

    out.update({
        "slope": _json_float(slope),
        "intercept": _json_float(intercept),
        "r": _json_float(r),
        "r_squared": _json_float(r_squared),
        "residual_mean": _json_float(np.mean(residuals)),
        "residual_std": _json_float(np.std(residuals, ddof=1)) if n > 1 else np.nan,
        "residual_mae": _json_float(np.mean(np.abs(residuals))),
        "residual_rmse": _json_float(math.sqrt(np.mean(residuals ** 2))),
    })

    if n > 2:
        mse = sse / (n - 2)
        out["slope_std_error"] = _json_float(math.sqrt(mse / sxx))
        out["intercept_std_error"] = _json_float(
            math.sqrt(mse * (1.0 / n + x_mean ** 2 / sxx))
        )

    return out


def describe_tt_series(values, metric_name: str = "tt_real") -> dict:
    s = _clean_numeric(values)

    out = {
        "metric": metric_name,
        "count": int(s.count()),
        "mean": np.nan,
        "sd": np.nan,
        "std_sample": np.nan,
        "std_population": np.nan,
        "min": np.nan,
        "q01": np.nan,
        "q05": np.nan,
        "q10": np.nan,
        "q25": np.nan,
        "median": np.nan,
        "q75": np.nan,
        "q90": np.nan,
        "q95": np.nan,
        "q99": np.nan,
        "max": np.nan,
        "iqr": np.nan,
    }

    if s.empty:
        return out

    quantiles = {
        "min": 0.00,
        "q01": 0.01,
        "q05": 0.05,
        "q10": 0.10,
        "q25": 0.25,
        "median": 0.50,
        "q75": 0.75,
        "q90": 0.90,
        "q95": 0.95,
        "q99": 0.99,
        "max": 1.00,
    }

    q = s.quantile(list(quantiles.values()))

    out.update({
        "mean": _json_float(s.mean()),
        "sd": _json_float(s.std(ddof=1)) if s.count() > 1 else np.nan,
        "std_sample": _json_float(s.std(ddof=1)) if s.count() > 1 else np.nan,
        "std_population": _json_float(s.std(ddof=0)),
    })

    for key, level in quantiles.items():
        out[key] = _json_float(q.loc[level])

    out["iqr"] = _json_float(out["q75"] - out["q25"])

    return out


def save_histogram_values(
    batch_dir: Path,
    values,
    metric_name: str = "tt_real",
    bins: int = 40,
) -> dict:
    """
    Zapisuje histogram jako CSV z koszykami.
    Nie tworzy dodatkowego wykresu PNG, żeby finalniertu były tylko dwa wykresy TT.
    """
    s = _clean_numeric(values)

    out = {
        "metric": metric_name,
        "n": int(s.count()),
        "bins": int(bins),
        "csv": str(batch_dir / f"{metric_name}_histogram_bins.csv"),
    }

    if s.empty:
        pd.DataFrame(
            columns=["bin_left", "bin_right", "count", "density"]
        ).to_csv(
            batch_dir / f"{metric_name}_histogram_bins.csv",
            index=False,
        )
        return out



    effective_bins = min(int(bins), max(1, int(s.nunique())))
    counts, edges = np.histogram(s.to_numpy(dtype=float), bins=effective_bins)

    hist_df = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
    })

    total = int(hist_df["count"].sum())
    hist_df["density"] = hist_df["count"] / total if total else 0.0

    hist_df.to_csv(batch_dir / f"{metric_name}_histogram_bins.csv", index=False)

    out["bins_effective"] = int(effective_bins)
    out["total_count"] = total

    return out

def select_representative_tt_peaks(ok_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wybiera 3 reprezentatywne przypadki według real TT:
      1) minimalny TT,
      2) maksymalny TT,
      3) przypadek ze środkowego kwartyla, najbliższy medianie.

    Środkowy kwartyl rozumiemy tutaj jako centralny zakres:
        Q25 <= tt_real <= Q75
    """
    if ok_df.empty or "tt_real" not in ok_df.columns:
        return pd.DataFrame()

    df = ok_df.copy()

    numeric_cols = [
        "tt_real",
        "tt_pred",
        "tt_signed_diff",
        "tt_abs_diff",
        "tt_signed_pct_diff",
        "tt_abs_pct_diff",
        "mae_eval",
        "rmse_eval",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[np.isfinite(df["tt_real"])].copy()

    if df.empty:
        return pd.DataFrame()

    q25 = float(df["tt_real"].quantile(0.25))
    median = float(df["tt_real"].quantile(0.50))
    q75 = float(df["tt_real"].quantile(0.75))

    selected = []
    used_indices = set()

    def _append_row(label: str, selection_rule: str, idx):
        row = df.loc[idx].copy()
        row["peak_label"] = label
        row["selection_rule"] = selection_rule
        row["tt_real_q25"] = q25
        row["tt_real_median"] = median
        row["tt_real_q75"] = q75
        selected.append(row)
        used_indices.add(idx)

    min_idx = df["tt_real"].idxmin()
    max_idx = df["tt_real"].idxmax()

    _append_row(
        "min_tt_real",
        "minimum tt_real over OK files",
        min_idx,
    )

    middle_df = df[
        (df["tt_real"] >= q25) &
        (df["tt_real"] <= q75)
    ].copy()

    # Staramy się nie wybrać drugi raz min/max jako punktu środkowego.
    middle_unique_df = middle_df.drop(
        index=list(used_indices | {max_idx}),
        errors="ignore",
    )

    if middle_unique_df.empty:
        middle_unique_df = middle_df

    if not middle_unique_df.empty:
        middle_idx = (middle_unique_df["tt_real"] - median).abs().idxmin()

        _append_row(
            "middle_quartile_tt_real",
            "closest to median tt_real among files with Q25 <= tt_real <= Q75",
            middle_idx,
        )

    _append_row(
        "max_tt_real",
        "maximum tt_real over OK files",
        max_idx,
    )

    peaks_df = pd.DataFrame(selected)

    preferred_cols = [
        "peak_label",
        "selection_rule",
        "selected_index",
        "file_name",
        "tt_real",
        "tt_pred",
        "tt_signed_diff",
        "tt_abs_diff",
        "tt_signed_pct_diff",
        "tt_abs_pct_diff",
        "mae_eval",
        "rmse_eval",
        "tt_real_q25",
        "tt_real_median",
        "tt_real_q75",
        "status",
        "seed_steps",
        "timesteps_total",
        "timesteps_eval",
        "nodes",
    ]

    cols = [c for c in preferred_cols if c in peaks_df.columns]
    remaining_cols = [c for c in peaks_df.columns if c not in cols]

    return peaks_df[cols + remaining_cols].reset_index(drop=True)


def plot_tt_histogram(
    batch_dir: Path,
    values,
    selected_peaks_df: pd.DataFrame | None = None,
    metric_name: str = "tt_real",
    bins: int = 40,
) -> dict:
    """
    Tworzy histogram TT jako PNG i opcjonalnie zaznacza 3 wybrane przypadki.
    """
    s = _clean_numeric(values)

    out_path = batch_dir / f"{metric_name}_histogram.png"

    out = {
        "metric": metric_name,
        "n": int(s.count()),
        "bins_requested": int(bins),
        "png": str(out_path),
    }

    if s.empty:
        return out

    effective_bins = min(int(bins), max(1, int(s.nunique())))
    out["bins_effective"] = int(effective_bins)

    plt.figure(figsize=(8, 5))

    plt.hist(
        s.to_numpy(dtype=float),
        bins=effective_bins,
        alpha=0.75,
    )

    plt.title("Histogram real TT")
    plt.xlabel("Real TT")
    plt.ylabel("Liczba plików")
    plt.grid(True)

    mean_v = float(s.mean())
    min_v = float(s.min())
    max_v = float(s.max())

    legend_labels = [
        f"Mean = {mean_v:.4g}",
        f"Min = {min_v:.4g}",
        f"Max = {max_v:.4g}",
    ]

    legend_handles = [
        plt.Line2D([], [], linestyle="none", label=label)
        for label in legend_labels
    ]

    plt.legend(
        handles=legend_handles,
        title="Real TT",
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(out_path, dpi=240)
    plt.close()

    return out


def save_tt_report(
    batch_dir: Path,
    per_file_df: pd.DataFrame,
    hist_bins: int = 40,
) -> dict:
    """
    Minimalny raport TT:
      - opis real TT: mean, sd, kwantyle,
      - histogram real TT jako CSV,
      - statystyki predykcji TT,
      - różnice w dolnym i górnym kwartylu real TT,
      - statystyki regresji tt_pred ~ tt_real.
    """
    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    report_dir = batch_dir / "tt_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    if ok_df.empty:
        print("[WARN] Brak poprawnych plików do raportu TT.")
        return {}

    for col in [
        "tt_real",
        "tt_pred",
        "tt_signed_diff",
        "tt_abs_diff",
        "mae_eval",
        "rmse_eval",
    ]:
        if col in ok_df.columns:
            ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")

    # --------------------------------------------------------
    # Statystyki opisowe real TT
    # --------------------------------------------------------
    tt_real_stats = describe_tt_series(ok_df["tt_real"], metric_name="tt_real")

    pd.DataFrame([tt_real_stats]).to_csv(
        report_dir / "tt_real_descriptive_stats.csv",
        index=False,
    )

    quantile_cols = [
        "min",
        "q01",
        "q05",
        "q10",
        "q25",
        "median",
        "q75",
        "q90",
        "q95",
        "q99",
        "max",
    ]

    pd.DataFrame([
        {
            "metric": "tt_real",
            "quantile": key,
            "value": tt_real_stats[key],
        }
        for key in quantile_cols
    ]).to_csv(
        report_dir / "tt_real_quantiles.csv",
        index=False,
    )

    hist_info = save_histogram_values(
        batch_dir=report_dir,
        values=ok_df["tt_real"],
        metric_name="tt_real",
        bins=hist_bins,
    )

    # --------------------------------------------------------
    # Histogram real TT jako PNG + 3 reprezentatywne przypadki
    # --------------------------------------------------------
    selected_peaks_df = select_representative_tt_peaks(ok_df)

    if not selected_peaks_df.empty:
        selected_peaks_df.to_csv(
            report_dir / "tt_selected_peaks.csv",
            index=False,
        )

        # Kopia w katalogu batcha, obok najważniejszych plików.
        selected_peaks_df.to_csv(
            batch_dir / "tt_selected_peaks.csv",
            index=False,
        )

        with open(report_dir / "tt_selected_peaks.json", "w", encoding="utf-8") as f:
            json.dump(
                selected_peaks_df.to_dict(orient="records"),
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    hist_plot_info = plot_tt_histogram(
        batch_dir=report_dir,
        values=ok_df["tt_real"],
        selected_peaks_df=selected_peaks_df,
        metric_name="tt_real",
        bins=hist_bins,
    )

    hist_info.update(hist_plot_info)

    # --------------------------------------------------------
    # Regresja predicted TT ~ real TT
    # --------------------------------------------------------
    regression = compute_linear_regression_stats(
        ok_df["tt_real"],
        ok_df["tt_pred"],
    )

    pd.DataFrame([regression]).to_csv(
        report_dir / "tt_regression_stats.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Statystyki predykcji
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Statystyki predykcji per file
    # --------------------------------------------------------

    # Jeśli uruchamiasz raport na starszym per_file_metrics.csv,
    # gdzie nie było jeszcze tych kolumn, policz je tutaj.
    if "tt_signed_pct_diff" not in ok_df.columns:
        ok_df["tt_signed_pct_diff"] = np.where(
            ok_df["tt_real"] != 0,
            100.0 * ok_df["tt_signed_diff"] / ok_df["tt_real"],
            np.nan,
        )

    if "tt_abs_pct_diff" not in ok_df.columns:
        ok_df["tt_abs_pct_diff"] = np.abs(ok_df["tt_signed_pct_diff"])

    for col in [
        "tt_signed_pct_diff",
        "tt_abs_pct_diff",
    ]:
        ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")

    def _series_stats(prefix: str, values) -> dict:
        s = _clean_numeric(values)

        out = {
            f"{prefix}_count": int(s.count()),
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

        if s.empty:
            return out

        q = s.quantile([0.25, 0.50, 0.75])

        out.update({
            f"{prefix}_mean": _json_float(s.mean()),
            f"{prefix}_sd": _json_float(s.std(ddof=1)) if s.count() > 1 else np.nan,
            f"{prefix}_median": _json_float(q.loc[0.50]),
            f"{prefix}_q25": _json_float(q.loc[0.25]),
            f"{prefix}_q75": _json_float(q.loc[0.75]),
            f"{prefix}_min": _json_float(s.min()),
            f"{prefix}_max": _json_float(s.max()),
        })

        return out

    tt_real_q25 = float(ok_df["tt_real"].quantile(0.25))
    tt_real_q75 = float(ok_df["tt_real"].quantile(0.75))

    low_real_tt_df = ok_df[ok_df["tt_real"] <= tt_real_q25].copy()
    high_real_tt_df = ok_df[ok_df["tt_real"] >= tt_real_q75].copy()

    prediction_stats = {
        "n_files": int(len(ok_df)),

        "definition_signed_pct_diff": "100 * (tt_pred - tt_real) / tt_real",
        "definition_abs_pct_diff": "abs(100 * (tt_pred - tt_real) / tt_real)",

        "tt_real_q25_threshold": _json_float(tt_real_q25),
        "tt_real_q75_threshold": _json_float(tt_real_q75),

        "low_quartile_n_files": int(len(low_real_tt_df)),
        "high_quartile_n_files": int(len(high_real_tt_df)),
    }

    # Globalnie po plikach, bez sumowania TT po całym zbiorze.
    prediction_stats.update(
        _series_stats(
            "tt_signed_diff_per_file",
            ok_df["tt_signed_diff"],
        )
    )

    prediction_stats.update(
        _series_stats(
            "tt_abs_diff_per_file",
            ok_df["tt_abs_diff"],
        )
    )

    prediction_stats.update(
        _series_stats(
            "tt_signed_pct_diff_per_file",
            ok_df["tt_signed_pct_diff"],
        )
    )

    prediction_stats.update(
        _series_stats(
            "tt_abs_pct_diff_per_file",
            ok_df["tt_abs_pct_diff"],
        )
    )

    # Dolny kwartyl real TT.
    prediction_stats.update(
        _series_stats(
            "low_quartile_signed_pct_diff_per_file",
            low_real_tt_df["tt_signed_pct_diff"],
        )
    )

    prediction_stats.update(
        _series_stats(
            "low_quartile_abs_pct_diff_per_file",
            low_real_tt_df["tt_abs_pct_diff"],
        )
    )

    # Górny kwartyl real TT.
    prediction_stats.update(
        _series_stats(
            "high_quartile_signed_pct_diff_per_file",
            high_real_tt_df["tt_signed_pct_diff"],
        )
    )

    prediction_stats.update(
        _series_stats(
            "high_quartile_abs_pct_diff_per_file",
            high_real_tt_df["tt_abs_pct_diff"],
        )
    )

    pd.DataFrame([prediction_stats]).to_csv(
        report_dir / "tt_prediction_stats.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Osobna tabela: dolny/górny kwartyl real TT
    # --------------------------------------------------------
    def _prediction_group_stats(group_name: str, subset: pd.DataFrame) -> dict:
        row = {
            "group": group_name,
            "n_files": int(len(subset)),

            "tt_real_min": np.nan,
            "tt_real_mean": np.nan,
            "tt_real_median": np.nan,
            "tt_real_max": np.nan,

            "tt_pred_mean": np.nan,
            "tt_pred_median": np.nan,

            "tt_signed_diff_mean_per_file": np.nan,
            "tt_signed_diff_median_per_file": np.nan,
            "tt_abs_diff_mean_per_file": np.nan,
            "tt_abs_diff_median_per_file": np.nan,

            "tt_signed_pct_diff_mean_per_file": np.nan,
            "tt_signed_pct_diff_sd_per_file": np.nan,
            "tt_signed_pct_diff_median_per_file": np.nan,
            "tt_signed_pct_diff_q25_per_file": np.nan,
            "tt_signed_pct_diff_q75_per_file": np.nan,

            "tt_abs_pct_diff_mean_per_file": np.nan,
            "tt_abs_pct_diff_sd_per_file": np.nan,
            "tt_abs_pct_diff_median_per_file": np.nan,
            "tt_abs_pct_diff_q25_per_file": np.nan,
            "tt_abs_pct_diff_q75_per_file": np.nan,
        }

        if subset.empty:
            return row

        signed_pct = _clean_numeric(subset["tt_signed_pct_diff"])
        abs_pct = _clean_numeric(subset["tt_abs_pct_diff"])

        signed_pct_q = signed_pct.quantile([0.25, 0.50, 0.75]) if not signed_pct.empty else None
        abs_pct_q = abs_pct.quantile([0.25, 0.50, 0.75]) if not abs_pct.empty else None

        row.update({
            "tt_real_min": _json_float(subset["tt_real"].min()),
            "tt_real_mean": _json_float(subset["tt_real"].mean()),
            "tt_real_median": _json_float(subset["tt_real"].median()),
            "tt_real_max": _json_float(subset["tt_real"].max()),

            "tt_pred_mean": _json_float(subset["tt_pred"].mean()),
            "tt_pred_median": _json_float(subset["tt_pred"].median()),

            "tt_signed_diff_mean_per_file": _json_float(subset["tt_signed_diff"].mean()),
            "tt_signed_diff_median_per_file": _json_float(subset["tt_signed_diff"].median()),
            "tt_abs_diff_mean_per_file": _json_float(subset["tt_abs_diff"].mean()),
            "tt_abs_diff_median_per_file": _json_float(subset["tt_abs_diff"].median()),
        })

        if signed_pct_q is not None:
            row.update({
                "tt_signed_pct_diff_mean_per_file": _json_float(signed_pct.mean()),
                "tt_signed_pct_diff_sd_per_file": _json_float(
                    signed_pct.std(ddof=1)) if signed_pct.count() > 1 else np.nan,
                "tt_signed_pct_diff_median_per_file": _json_float(signed_pct_q.loc[0.50]),
                "tt_signed_pct_diff_q25_per_file": _json_float(signed_pct_q.loc[0.25]),
                "tt_signed_pct_diff_q75_per_file": _json_float(signed_pct_q.loc[0.75]),
            })

        if abs_pct_q is not None:
            row.update({
                "tt_abs_pct_diff_mean_per_file": _json_float(abs_pct.mean()),
                "tt_abs_pct_diff_sd_per_file": _json_float(abs_pct.std(ddof=1)) if abs_pct.count() > 1 else np.nan,
                "tt_abs_pct_diff_median_per_file": _json_float(abs_pct_q.loc[0.50]),
                "tt_abs_pct_diff_q25_per_file": _json_float(abs_pct_q.loc[0.25]),
                "tt_abs_pct_diff_q75_per_file": _json_float(abs_pct_q.loc[0.75]),
            })

        return row

    prediction_groups = pd.DataFrame([
        _prediction_group_stats("all_files", ok_df),
        _prediction_group_stats("lowest_25pct_by_real_tt", low_real_tt_df),
        _prediction_group_stats("highest_25pct_by_real_tt", high_real_tt_df),
    ])

    prediction_groups.to_csv(
        report_dir / "tt_prediction_error_low_high_real_tt_quartiles.csv",
        index=False,
    )

    summary = {
        "report_dir": str(report_dir),
        "tt_real_descriptive_stats": tt_real_stats,
        "tt_real_histogram": hist_info,
        "tt_selected_peaks": (
            selected_peaks_df.to_dict(orient="records")
            if "selected_peaks_df" in locals() and not selected_peaks_df.empty
            else []
        ),
        "prediction_stats": prediction_stats,
        "regression_stats": regression,
    }

    with open(report_dir / "tt_report_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Kopie najważniejszych plików w katalogu batcha.
    pd.DataFrame([tt_real_stats]).to_csv(
        batch_dir / "tt_real_descriptive_stats.csv",
        index=False,
    )
    pd.DataFrame([prediction_stats]).to_csv(
        batch_dir / "tt_prediction_stats.csv",
        index=False,
    )
    pd.DataFrame([regression]).to_csv(
        batch_dir / "tt_regression_stats.csv",
        index=False,
    )
    prediction_groups.to_csv(
        batch_dir / "tt_prediction_error_low_high_real_tt_quartiles.csv",
        index=False,
    )

    print("\n=== TT report summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Zapisano raport TT w: {report_dir}")

    return summary


def _real_pred_arrays(per_file_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    x = ok_df["tt_real"].to_numpy(dtype=float)
    y = ok_df["tt_pred"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)

    return x[finite], y[finite]


def plot_real_vs_pred_tt(batch_dir: Path, per_file_df: pd.DataFrame):
    """
    Wykres 1:
    tylko real TT vs predicted TT.
    Bez rel diff, bez regresji, bez dodatkowych adnotacji.
    """
    x, y = _real_pred_arrays(per_file_df)

    if len(x) == 0:
        return

    lo, hi = _common_tt_limits(x, y)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, alpha=0.60, label="files")
    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1,
        label="real = predicted",
    )

    plt.title("Real TT vs predicted TT")
    plt.xlabel("Real TT")
    plt.ylabel("Predicted TT")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "real_tt_vs_predicted_tt_1.png", dpi=240)
    plt.close()


def plot_real_vs_pred_tt_with_regression(
        batch_dir: Path,
        per_file_df: pd.DataFrame,
        regression: dict | None = None,
):
    """
    Wykres 2:
    real TT vs predicted TT + krzywa regresji liniowej.
    Bez zoomowania osi. Zaktualizowana legenda z uwzględnieniem
    korelacji i uśrednionych procentowych błędów predykcji.
    """
    x, y = _real_pred_arrays(per_file_df)

    if len(x) == 0:
        return

    lo, hi = _common_tt_limits(x, y)

    regression = regression or compute_linear_regression_stats(x, y)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, alpha=0.60, label="files")

    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1,
        label="real = predicted",
    )

    slope = regression.get("slope", np.nan)
    intercept = regression.get("intercept", np.nan)
    r_squared = regression.get("r_squared", np.nan)
    r_val = regression.get("r", np.nan)

    # Obliczenie błędu względnego (procentowego) per plik: 100 * (y - x) / x
    pct_diffs = np.where(x != 0, 100.0 * (y - x) / x, np.nan)

    # Średnia różnica procentowa uwzględniająca znak (MPE)
    mean_signed_pct_diff = float(np.nanmean(pct_diffs))

    # Średnia absolutna różnica procentowa (MAPE)
    mean_abs_pct_diff = float(np.nanmean(np.abs(pct_diffs)))

    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.linspace(lo, hi, 200)
        y_line = slope * x_line + intercept

        # Konstrukcja wieloliniowego opisu, aby legenda nie była zbyt długa w poziomie
        legend_label = (
            f"Regression\n"
            f"Correlation = {r_val:.4g}\n"
            f"$R^2$ = {r_squared:.4g}\n"
            f"Mean \\% diff = {mean_signed_pct_diff:+.2f}\\%"
        )

        plt.plot(
            x_line,
            y_line,
            linewidth=1.8,
            color="tab:green",
            label=legend_label,
        )

    plt.title("Real TT vs predicted TT — regression")
    plt.xlabel("Real TT")
    plt.ylabel("Predicted TT")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(batch_dir / "lstm_att_reg.png", dpi=240)
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
            f"diff={row['tt_signed_diff']:.6f}, "
            f"pct diff={row['tt_signed_pct_diff']:.3f}%, "
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
        "--plot_style",
        nargs="*",
        default=["science", "no-latex"],
        help=(
            "Style matplotlib/SciencePlots, np. --plot_style science no-latex "
            "albo --plot_style science ieee no-latex."
        ),
    )

    return p




def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_plot_style(args.plot_style)

    if args.hist_bins <= 0:
        raise ValueError("--hist_bins musi być > 0")

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

    tt_signed_diff_total = float(agg.total_tt_pred - agg.total_tt_real)
    tt_abs_diff_total = float(abs(tt_signed_diff_total))

    summary = {
        "files_ok": int(agg.files_ok),
        "files_failed": int(agg.files_failed),
        "delta_t": float(agg.delta_t),

        "mae_weighted_eval": (
            float(agg.total_abs_err / agg.total_values)
            if agg.total_values
            else np.nan
        ),
        "rmse_weighted_eval": (
            float(math.sqrt(agg.total_sq_err / agg.total_values))
            if agg.total_values
            else np.nan
        ),


        "tt_abs_diff_mean_per_file": (
            float(agg.total_tt_abs_diff_filewise / agg.files_ok)
            if agg.files_ok
            else np.nan
        ),

        "n_eval_values_total": int(agg.total_values),
        "batch_dir": str(batch_dir),
        "checkpoint": str(args.checkpoint),
        "selected_files_count": int(len(selected_files)),
        "batch_size_files": int(args.batch_size_files),
        "device": str(device),
    }

    ok_df = per_file_df[per_file_df["status"] == "OK"].copy()

    if not ok_df.empty:
        summary["mae_mean_per_file"] = _json_float(ok_df["mae_eval"].mean())
        summary["rmse_mean_per_file"] = _json_float(ok_df["rmse_eval"].mean())
        summary["tt_signed_diff_mean_per_file"] = _json_float(ok_df["tt_signed_diff"].mean())
        summary["tt_abs_diff_mean_per_file"] = _json_float(ok_df["tt_abs_diff"].mean())

    tt_report_summary = save_tt_report(
        batch_dir=batch_dir,
        per_file_df=per_file_df,
        hist_bins=args.hist_bins,
    )

    regression = (
        tt_report_summary.get("regression_stats", {})
        if tt_report_summary
        else {}
    )

    plot_real_vs_pred_tt(batch_dir, per_file_df)
    plot_real_vs_pred_tt_with_regression(
        batch_dir,
        per_file_df,
        regression=regression,
    )

    if tt_report_summary:
        summary["tt_report_dir"] = tt_report_summary.get("report_dir")

        prediction_stats = tt_report_summary.get("prediction_stats", {})
        regression_stats = tt_report_summary.get("regression_stats", {})

        for key in [
            "tt_signed_pct_diff_per_file_mean",
            "tt_signed_pct_diff_per_file_sd",
            "tt_signed_pct_diff_per_file_median",
            "tt_abs_pct_diff_per_file_mean",
            "tt_abs_pct_diff_per_file_sd",
            "tt_abs_pct_diff_per_file_median",

            "low_quartile_signed_pct_diff_per_file_mean",
            "low_quartile_signed_pct_diff_per_file_median",
            "low_quartile_abs_pct_diff_per_file_mean",
            "low_quartile_abs_pct_diff_per_file_median",

            "high_quartile_signed_pct_diff_per_file_mean",
            "high_quartile_signed_pct_diff_per_file_median",
            "high_quartile_abs_pct_diff_per_file_mean",
            "high_quartile_abs_pct_diff_per_file_median",
        ]:
            if key in prediction_stats:
                summary[f"tt_prediction_{key}"] = prediction_stats[key]
            if key in prediction_stats:
                summary[f"tt_prediction_{key}"] = prediction_stats[key]

        for key in [
            "slope",
            "intercept",
            "r",
            "r_squared",
            "residual_rmse",
            "residual_mae",
        ]:
            if key in regression_stats:
                summary[f"tt_regression_{key}"] = regression_stats[key]

    with open(batch_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(batch_dir / "summary.csv", index=False)

    print(f"\nZapisano wyniki w: {batch_dir}", flush=True)
    print("Wykresy TT:", flush=True)
    print("  - real_tt_vs_predicted_tt_1.png", flush=True)
    print("  - lstm_att_reg.png", flush=True)
    print("Raport TT: tt_report/", flush=True)
    print("Histogram TT:", flush=True)
    print("  - tt_report/tt_real_histogram.png", flush=True)
    print("Wybrane przypadki TT:", flush=True)
    print("  - tt_selected_peaks.csv", flush=True)
    print("  - tt_report/tt_selected_peaks.csv", flush=True)
    print("  - tt_report/tt_selected_peaks.json", flush=True)


if __name__ == "__main__":
    main()