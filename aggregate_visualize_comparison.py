#!/usr/bin/env python3
"""
aggregate_visualize_comparison.py

Agreguje gotowy folder:

comparison_dir/
  runs/
    Model_A/
      pred_q.npy
      real_q.npy
      run_info.json
    Model_B/
      pred_q.npy
      real_q.npy
      run_info.json

Tworzy:
- all_model_metrics.csv
- per_node_metrics.csv
- node_activity.csv
- selected_nodes.csv
- 15 wykresów dokładnych
- 15 wykresów rolling mean
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "grid", "no-latex"])
except ImportError:
    print("[WARN] Brak SciencePlots. Zainstaluj: pip install SciencePlots. Używam stylu domyślnego.")


COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#882255",
    "#332288",
    "#999999",
    "#44AA99",
    "#AA4499",
]
GROUND_TRUTH_COLOR = "#111111"


@dataclass
class ModelData:
    name: str
    run_dir: Path
    pred_path: Path
    real_path: Path
    seed_steps: Optional[int]
    pred: np.ndarray
    real: np.ndarray


def slugify(text: str) -> str:
    out = []
    for ch in text.strip():
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_array_2d(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float64, copy=False)
    raise ValueError(f"{path} ma shape {arr.shape}; oczekuję (T, N) albo (T, N, 1).")


def scan_models(comparison_dir: Path) -> Dict[str, ModelData]:
    runs_dir = comparison_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Nie istnieje katalog: {runs_dir}")

    models: Dict[str, ModelData] = {}

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        pred_path = run_dir / "pred_q.npy"
        real_path = run_dir / "real_q.npy"

        if not pred_path.exists() or not real_path.exists():
            print(f"[WARN] Pomijam {run_dir}, bo brakuje pred_q.npy lub real_q.npy.")
            continue

        info_path = run_dir / "run_info.json"
        if info_path.exists():
            info = load_json(info_path)
            name = info.get("model_name", run_dir.name)
            seed_steps = info.get("seed_steps")
        else:
            name = run_dir.name
            seed_steps = None

        pred = load_array_2d(pred_path)
        real = load_array_2d(real_path)

        if pred.shape != real.shape:
            raise ValueError(f"{name}: pred i real mają różne shape: {pred.shape} vs {real.shape}")

        if name in models:
            raise ValueError(f"Duplikat nazwy modelu: {name}")

        models[name] = ModelData(
            name=name,
            run_dir=run_dir,
            pred_path=pred_path,
            real_path=real_path,
            seed_steps=None if seed_steps is None else int(seed_steps),
            pred=pred,
            real=real,
        )

    if not models:
        raise RuntimeError(f"Nie znaleziono żadnych modeli w {runs_dir}")

    return models


def reorder_models(models: Dict[str, ModelData], base_model: str) -> Dict[str, ModelData]:
    if base_model not in models:
        raise ValueError(f"base_model={base_model} nie istnieje. Dostępne: {list(models)}")

    ordered = {base_model: models[base_model]}
    for name, model in models.items():
        if name != base_model:
            ordered[name] = model
    return ordered


def get_eval_window(
    models: Dict[str, ModelData],
    eval_start_override: Optional[int],
    t_end_override: Optional[int],
    n_nodes_override: Optional[int],
) -> Tuple[int, int, int]:
    if eval_start_override is not None:
        eval_start = int(eval_start_override)
    else:
        seeds = [m.seed_steps for m in models.values() if m.seed_steps is not None]
        eval_start = max(seeds) if seeds else 30

    t_end = min(m.pred.shape[0] for m in models.values())
    n_nodes = min(m.pred.shape[1] for m in models.values())

    if t_end_override is not None:
        t_end = min(t_end, int(t_end_override))

    if n_nodes_override is not None:
        n_nodes = min(n_nodes, int(n_nodes_override))

    if eval_start >= t_end:
        raise ValueError(f"eval_start={eval_start} >= t_end={t_end}")

    return eval_start, t_end, n_nodes


def check_real_consistency(
    models: Dict[str, ModelData],
    eval_start: int,
    t_end: int,
    n_nodes: int,
    allow_different_real: bool,
) -> None:
    items = list(models.items())
    ref_name, ref_model = items[0]
    ref = ref_model.real[eval_start:t_end, :n_nodes]

    for name, model in items[1:]:
        real = model.real[eval_start:t_end, :n_nodes]
        same = np.allclose(ref, real, atol=1e-8, rtol=0.0)

        if not same:
            msg = f"Ground truth modelu {name} różni się od {ref_name}."
            if allow_different_real:
                print(f"[WARN] {msg} Kontynuuję przez --allow_different_real.")
            else:
                raise ValueError(
                    msg
                    + " Modele powinny być porównywane na tym samym test secie."
                )


def compute_metrics_and_rankings(
    models: Dict[str, ModelData],
    base_model: str,
    eval_start: int,
    t_end: int,
    n_nodes: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    per_node_rows = []

    for name, model in models.items():
        pred = model.pred[eval_start:t_end, :n_nodes]
        real = model.real[eval_start:t_end, :n_nodes]

        err = pred - real
        abs_err = np.abs(err)
        sq_err = err ** 2

        per_node_mae = abs_err.mean(axis=0)
        per_node_rmse = np.sqrt(sq_err.mean(axis=0))

        metrics_rows.append(
            {
                "model": name,
                "run_dir": str(model.run_dir),
                "pred_path": str(model.pred_path),
                "real_path": str(model.real_path),
                "seed_steps": model.seed_steps,
                "eval_start": eval_start,
                "t_end": t_end,
                "n_nodes": n_nodes,
                "timesteps_eval": t_end - eval_start,
                "global_mae": float(abs_err.mean()),
                "global_rmse": float(np.sqrt(sq_err.mean())),
                "global_median_abs_error": float(np.median(abs_err)),
                "global_max_abs_error": float(abs_err.max()),
            }
        )

        for node in range(n_nodes):
            per_node_rows.append(
                {
                    "model": name,
                    "node": node,
                    "mae": float(per_node_mae[node]),
                    "rmse": float(per_node_rmse[node]),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("global_mae", ascending=True)
    per_node_df = pd.DataFrame(per_node_rows)

    base = models[base_model]
    real_ref = base.real[eval_start:t_end, :n_nodes]
    pred_base = base.pred[eval_start:t_end, :n_nodes]

    activity_df = pd.DataFrame(
        {
            "node": np.arange(n_nodes),
            "activity_mean_abs_q": np.mean(np.abs(real_ref), axis=0),
            "activity_sum_abs_q": np.sum(np.abs(real_ref), axis=0),
            "activity_std_q": np.std(real_ref, axis=0),
        }
    )

    base_abs = np.abs(pred_base - real_ref)
    base_mae_df = pd.DataFrame(
        {
            "node": np.arange(n_nodes),
            f"{base_model}_mae": base_abs.mean(axis=0),
            f"{base_model}_rmse": np.sqrt(np.mean((pred_base - real_ref) ** 2, axis=0)),
            "activity_mean_abs_q": activity_df["activity_mean_abs_q"].to_numpy(),
        }
    )

    metrics_df.to_csv(output_dir / "all_model_metrics.csv", index=False)
    per_node_df.to_csv(output_dir / "per_node_metrics.csv", index=False)
    activity_df.sort_values("activity_mean_abs_q", ascending=False).to_csv(
        output_dir / "node_activity.csv",
        index=False,
    )
    base_mae_df.sort_values(f"{base_model}_mae", ascending=True).to_csv(
        output_dir / f"base_model_{slugify(base_model)}_mae_ranking.csv",
        index=False,
    )

    return metrics_df, per_node_df, activity_df, base_mae_df


def select_nodes(
    activity_df: pd.DataFrame,
    base_mae_df: pd.DataFrame,
    base_model: str,
    k: int,
) -> Dict[str, List[int]]:
    top_active = (
        activity_df.sort_values("activity_mean_abs_q", ascending=False)
        .head(k)["node"]
        .astype(int)
        .tolist()
    )

    least_active = (
        activity_df.sort_values("activity_mean_abs_q", ascending=True)
        .head(k)["node"]
        .astype(int)
        .tolist()
    )

    base_best = (
        base_mae_df.sort_values(f"{base_model}_mae", ascending=True)
        .head(k)["node"]
        .astype(int)
        .tolist()
    )

    return {
        "top_active": top_active,
        "least_active": least_active,
        "base_best_mae": base_best,
    }


def rolling_mean(y: np.ndarray, window: int, center: bool) -> np.ndarray:
    return (
        pd.Series(y)
        .rolling(window=window, min_periods=1, center=center)
        .mean()
        .to_numpy(dtype=float)
    )


def node_mae(model: ModelData, node: int, eval_start: int, t_end: int) -> float:
    return float(np.mean(np.abs(model.pred[eval_start:t_end, node] - model.real[eval_start:t_end, node])))


def plot_one_node(
    models: Dict[str, ModelData],
    base_model: str,
    node: int,
    category: str,
    rank: int,
    eval_start: int,
    t_end: int,
    output_path: Path,
    rolling_window: Optional[int],
    rolling_center: bool,
    dpi: int,
) -> None:
    x = np.arange(eval_start, t_end)

    # Ground truth bierzemy z base_model.
    gt = models[base_model].real[eval_start:t_end, node]
    if rolling_window is not None:
        gt = rolling_mean(gt, rolling_window, rolling_center)

    fig, ax = plt.subplots(figsize=(12.0, 4.8))

    ax.plot(
        x,
        gt,
        label="Ground truth",
        color=GROUND_TRUTH_COLOR,
        linewidth=2.8,
        zorder=20,
    )

    for idx, (name, model) in enumerate(models.items()):
        y = model.pred[eval_start:t_end, node]
        if rolling_window is not None:
            y = rolling_mean(y, rolling_window, rolling_center)

        mae = node_mae(model, node, eval_start, t_end)
        is_base = name == base_model

        ax.plot(
            x,
            y,
            label=f"{name} | MAE={mae:.4g}",
            color=COLORS[idx % len(COLORS)],
            linewidth=2.25 if is_base else 1.75,
            linestyle="-" if is_base else "--",
            alpha=0.98 if is_base else 0.86,
        )

    nice_category = category.replace("_", " ")
    if rolling_window is None:
        title = f"{nice_category} #{rank}: node {node} — exact"
    else:
        title = f"{nice_category} #{rank}: node {node} — rolling mean, window={rolling_window}"

    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("q")
    ax.margins(x=0.01)
    ax.legend(fontsize=8, loc="best", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_selected_nodes_table(
    selected: Dict[str, List[int]],
    models: Dict[str, ModelData],
    base_model: str,
    activity_df: pd.DataFrame,
    base_mae_df: pd.DataFrame,
    eval_start: int,
    t_end: int,
    output_dir: Path,
) -> None:
    activity_map = activity_df.set_index("node")["activity_mean_abs_q"].to_dict()
    base_mae_col = f"{base_model}_mae"
    base_mae_map = base_mae_df.set_index("node")[base_mae_col].to_dict()

    rows = []
    for category, nodes in selected.items():
        for rank, node in enumerate(nodes, start=1):
            node = int(node)
            row = {
                "category": category,
                "rank": rank,
                "node": node,
                "activity_mean_abs_q": float(activity_map[node]),
                base_mae_col: float(base_mae_map[node]),
            }
            for name, model in models.items():
                row[f"{name}_mae"] = node_mae(model, node, eval_start, t_end)
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "selected_nodes.csv", index=False)


def save_timeseries_csvs(
    selected: Dict[str, List[int]],
    models: Dict[str, ModelData],
    base_model: str,
    eval_start: int,
    t_end: int,
    output_dir: Path,
) -> None:
    ts_dir = output_dir / "timeseries_selected_nodes"
    ts_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(eval_start, t_end)

    for category, nodes in selected.items():
        for rank, node in enumerate(nodes, start=1):
            node = int(node)
            data = {
                "timestep": x,
                "node": np.full(len(x), node, dtype=int),
                "ground_truth": models[base_model].real[eval_start:t_end, node],
            }

            for name, model in models.items():
                pred = model.pred[eval_start:t_end, node]
                real = model.real[eval_start:t_end, node]
                data[f"pred_{name}"] = pred
                data[f"abs_error_{name}"] = np.abs(pred - real)

            pd.DataFrame(data).to_csv(
                ts_dir / f"{category}_rank{rank:02d}_node_{node:03d}.csv",
                index=False,
            )


def make_all_plots(
    selected: Dict[str, List[int]],
    models: Dict[str, ModelData],
    base_model: str,
    eval_start: int,
    t_end: int,
    output_dir: Path,
    rolling_window: int,
    rolling_center: bool,
    dpi: int,
) -> None:
    figures_dir = output_dir / "figures"

    for category, nodes in selected.items():
        for rank, node in enumerate(nodes, start=1):
            node = int(node)

            exact_path = (
                figures_dir
                / "exact"
                / category
                / f"{category}_rank{rank:02d}_node_{node:03d}_exact.png"
            )

            rolling_path = (
                figures_dir
                / "rolling"
                / category
                / f"{category}_rank{rank:02d}_node_{node:03d}_rolling_w{rolling_window}.png"
            )

            plot_one_node(
                models=models,
                base_model=base_model,
                node=node,
                category=category,
                rank=rank,
                eval_start=eval_start,
                t_end=t_end,
                output_path=exact_path,
                rolling_window=None,
                rolling_center=rolling_center,
                dpi=dpi,
            )

            plot_one_node(
                models=models,
                base_model=base_model,
                node=node,
                category=category,
                rank=rank,
                eval_start=eval_start,
                t_end=t_end,
                output_path=rolling_path,
                rolling_window=rolling_window,
                rolling_center=rolling_center,
                dpi=dpi,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--comparison_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)

    parser.add_argument("--output_subdir", type=str, default="visualizations")

    parser.add_argument(
        "--eval_start",
        type=int,
        default=None,
        help="Jeśli brak, używa max(seed_steps) z run_info.json albo 30.",
    )
    parser.add_argument("--t_end", type=int, default=None)
    parser.add_argument("--n_nodes", type=int, default=None)

    parser.add_argument("--top_k_each", type=int, default=5)
    parser.add_argument("--rolling_window", type=int, default=25)
    parser.add_argument("--rolling_center", action="store_true")
    parser.add_argument("--dpi", type=int, default=240)

    parser.add_argument("--allow_different_real", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comparison_dir = Path(args.comparison_dir).expanduser().resolve()
    output_dir = comparison_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = scan_models(comparison_dir)
    models = reorder_models(models, args.base_model)

    eval_start, t_end, n_nodes = get_eval_window(
        models=models,
        eval_start_override=args.eval_start,
        t_end_override=args.t_end,
        n_nodes_override=args.n_nodes,
    )

    check_real_consistency(
        models=models,
        eval_start=eval_start,
        t_end=t_end,
        n_nodes=n_nodes,
        allow_different_real=args.allow_different_real,
    )

    print(f"[INFO] comparison_dir: {comparison_dir}")
    print(f"[INFO] output_dir:     {output_dir}")
    print(f"[INFO] base_model:     {args.base_model}")
    print(f"[INFO] models:         {list(models)}")
    print(f"[INFO] eval window:    t=[{eval_start}, {t_end}), nodes=[0, {n_nodes})")

    metrics_df, per_node_df, activity_df, base_mae_df = compute_metrics_and_rankings(
        models=models,
        base_model=args.base_model,
        eval_start=eval_start,
        t_end=t_end,
        n_nodes=n_nodes,
        output_dir=output_dir,
    )

    selected = select_nodes(
        activity_df=activity_df,
        base_mae_df=base_mae_df,
        base_model=args.base_model,
        k=args.top_k_each,
    )

    save_selected_nodes_table(
        selected=selected,
        models=models,
        base_model=args.base_model,
        activity_df=activity_df,
        base_mae_df=base_mae_df,
        eval_start=eval_start,
        t_end=t_end,
        output_dir=output_dir,
    )

    save_timeseries_csvs(
        selected=selected,
        models=models,
        base_model=args.base_model,
        eval_start=eval_start,
        t_end=t_end,
        output_dir=output_dir,
    )

    make_all_plots(
        selected=selected,
        models=models,
        base_model=args.base_model,
        eval_start=eval_start,
        t_end=t_end,
        output_dir=output_dir,
        rolling_window=args.rolling_window,
        rolling_center=args.rolling_center,
        dpi=args.dpi,
    )

    manifest = {
        "comparison_dir": str(comparison_dir),
        "output_dir": str(output_dir),
        "base_model": args.base_model,
        "models": list(models),
        "eval_start": eval_start,
        "t_end": t_end,
        "n_nodes": n_nodes,
        "top_k_each": args.top_k_each,
        "rolling_window": args.rolling_window,
        "selected_nodes": selected,
        "n_exact_plots": sum(len(v) for v in selected.values()),
        "n_rolling_plots": sum(len(v) for v in selected.values()),
    }

    with (output_dir / "visualization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[OK] Gotowe.")
    print(f"[OK] Ranking modeli:")
    print(metrics_df[["model", "global_mae", "global_rmse", "seed_steps"]].to_string(index=False))
    print(f"\n[OK] Wykresy zapisane w: {output_dir / 'figures'}")
    print("[OK] Wybrane node'y:")
    for category, nodes in selected.items():
        print(f"  - {category}: {nodes}")


if __name__ == "__main__":
    main()