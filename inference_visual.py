import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_tt_stats(pred: np.ndarray, real: np.ndarray, delta_t: float = 10.0, seed_steps=None):
    if seed_steps is not None:
        pred_eval = pred[seed_steps:]
        real_eval = real[seed_steps:]
    else:
        pred_eval = pred
        real_eval = real

    tt_real = float(delta_t * np.sum(real_eval))
    tt_pred = float(delta_t * np.sum(pred_eval))

    tt_abs_diff = float(abs(tt_pred - tt_real))
    tt_signed_diff = float(tt_pred - tt_real)
    tt_rel_diff = float(tt_abs_diff / tt_real) if tt_real != 0 else np.nan

    # profil w czasie: TT na pojedynczy krok
    tt_real_per_t = delta_t * np.sum(np.abs(real_eval), axis=1)
    tt_pred_per_t = delta_t * np.sum(np.abs(pred_eval), axis=1)

    # skumulowany TT
    tt_real_cum = np.cumsum(tt_real_per_t)
    tt_pred_cum = np.cumsum(tt_pred_per_t)

    return {
        "tt_real": tt_real,
        "tt_pred": tt_pred,
        "tt_abs_diff": tt_abs_diff,
        "tt_signed_diff": tt_signed_diff,
        "tt_rel_diff": tt_rel_diff,
        "tt_real_per_t": tt_real_per_t,
        "tt_pred_per_t": tt_pred_per_t,
        "tt_real_cum": tt_real_cum,
        "tt_pred_cum": tt_pred_cum,
    }


def get_eval_arrays(pred: np.ndarray, real: np.ndarray, seed_steps=None):
    if seed_steps is not None:
        return pred[seed_steps:], real[seed_steps:]
    return pred, real


def classify_zero_nonzero_nodes(real: np.ndarray, seed_steps=None, zero_tol: float = 1e-12):
    _, real_eval = get_eval_arrays(real, real, seed_steps=seed_steps)
    per_node_has_signal = np.any(np.abs(real_eval) > zero_tol, axis=0)
    nonzero_mask = per_node_has_signal
    zero_mask = ~nonzero_mask
    return zero_mask, nonzero_mask


def compute_group_summaries(pred: np.ndarray, real: np.ndarray, seed_steps=None, delta_t: float = 10.0, zero_tol: float = 1e-12):
    pred_eval, real_eval = get_eval_arrays(pred, real, seed_steps=seed_steps)
    zero_mask, nonzero_mask = classify_zero_nonzero_nodes(real, seed_steps=seed_steps, zero_tol=zero_tol)

    def summarize_group(mask: np.ndarray, group_name: str):
        n_nodes = int(mask.sum())

        if n_nodes == 0:
            return {
                "group": group_name,
                "n_nodes": 0,
                "node_fraction": 0.0,
                "tt_real": np.nan,
                "tt_pred": np.nan,
                "tt_abs_diff": np.nan,
                "tt_signed_diff": np.nan,
                "tt_rel_diff": np.nan,
                "mae": np.nan,
            }

        pred_group = pred_eval[:, mask]
        real_group = real_eval[:, mask]
        abs_err = np.abs(pred_group - real_group)

        tt_real = float(delta_t * np.sum(real_group))
        tt_pred = float(delta_t * np.sum(pred_group))
        tt_abs_diff = float(abs(tt_pred - tt_real))
        tt_signed_diff = float(tt_pred - tt_real)
        tt_rel_diff = float(tt_abs_diff / tt_real) if tt_real != 0 else np.nan
        mae = float(abs_err.mean())

        return {
            "group": group_name,
            "n_nodes": n_nodes,
            "node_fraction": float(n_nodes / real.shape[1]),
            "tt_real": tt_real,
            "tt_pred": tt_pred,
            "tt_abs_diff": tt_abs_diff,
            "tt_signed_diff": tt_signed_diff,
            "tt_rel_diff": tt_rel_diff,
            "mae": mae,
        }

    return {
        "zero_nodes": summarize_group(zero_mask, "zero_nodes"),
        "nonzero_nodes": summarize_group(nonzero_mask, "nonzero_nodes"),
        "zero_node_indices": np.flatnonzero(zero_mask).astype(int),
        "nonzero_node_indices": np.flatnonzero(nonzero_mask).astype(int),
    }

def find_single(pattern: str, run_dir: Path) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Nie znaleziono pliku pasującego do: {pattern} w {run_dir}")
    if len(matches) > 1:
        print(f"[INFO] Dla wzorca {pattern} znaleziono kilka plików, biorę pierwszy: {matches[0].name}")
    return matches[0]


def save_activity_ranking(real: np.ndarray, out_dir: Path):
    # aktywność liczona z prawdziwego q
    activity_mean_abs = np.mean(np.abs(real), axis=0)
    activity_sum_abs = np.sum(np.abs(real), axis=0)
    activity_std = np.std(real, axis=0)

    df = pd.DataFrame({
        "node": np.arange(real.shape[1]),
        "activity_mean_abs_q": activity_mean_abs,
        "activity_sum_abs_q": activity_sum_abs,
        "activity_std_q": activity_std,
    }).sort_values("activity_mean_abs_q", ascending=False)

    df.to_csv(out_dir / "activity_ranking.csv", index=False)

    print("\nTop 10 najaktywniejszych node'ów:")
    print(df.head(10).to_string(index=False))

    return df


def plot_top_active_nodes_detailed(pred: np.ndarray, real: np.ndarray, out_dir: Path, seed_steps=None, top_k: int = 10):
    activity_df = save_activity_ranking(real, out_dir)
    top_nodes = activity_df.head(top_k)["node"].astype(int).tolist()

    detailed_dir = out_dir / "top_active_nodes"
    detailed_dir.mkdir(exist_ok=True)

    for node in top_nodes:
        signed_err = pred[:, node] - real[:, node]
        abs_err = np.abs(signed_err)

        # zmiany w czasie
        delta_real = np.diff(real[:, node], prepend=real[0, node])
        delta_pred = np.diff(pred[:, node], prepend=pred[0, node])
        delta_err = delta_pred - delta_real

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

        # 1. real vs pred
        pred_vis = pred[:, node].copy()
        if seed_steps is not None:
            pred_vis[:seed_steps] = np.nan

        axes[0].plot(real[:, node], label="real q")
        axes[0].plot(pred_vis, label="pred q")
        if seed_steps is not None:
            axes[0].axvline(seed_steps, linestyle="--", linewidth=1, label="seed_steps")
        axes[0].set_title(f"Node {node}: real vs pred")
        axes[0].set_ylabel("q")
        axes[0].grid(True)
        axes[0].legend()

        # 2. signed error
        axes[1].plot(signed_err, label="pred - real")
        axes[1].axhline(0.0, linestyle="--", linewidth=1)
        if seed_steps is not None:
            axes[1].axvline(seed_steps, linestyle="--", linewidth=1)
        axes[1].set_title(f"Node {node}: signed error over time")
        axes[1].set_ylabel("pred-real")
        axes[1].grid(True)
        axes[1].legend()

        # 3. absolute error
        axes[2].plot(abs_err, label="|pred - real|")
        if seed_steps is not None:
            axes[2].axvline(seed_steps, linestyle="--", linewidth=1)
        axes[2].set_title(f"Node {node}: absolute error over time")
        axes[2].set_ylabel("|pred-real|")
        axes[2].grid(True)
        axes[2].legend()

        # 4. zmiana w czasie
        axes[3].plot(delta_real, label="delta real")
        axes[3].plot(delta_pred, label="delta pred")
        if seed_steps is not None:
            axes[3].axvline(seed_steps, linestyle="--", linewidth=1)
        axes[3].set_title(f"Node {node}: temporal change")
        axes[3].set_xlabel("t")
        axes[3].set_ylabel("delta q")
        axes[3].grid(True)
        axes[3].legend()

        plt.tight_layout()
        plt.savefig(detailed_dir / f"node_{node:03d}_detailed_analysis.png", dpi=220)
        plt.close()

        # zapis csv dla każdego noda
        df_node = pd.DataFrame({
            "t": np.arange(real.shape[0]),
            "real_q": real[:, node],
            "pred_q": pred[:, node],
            "signed_error": signed_err,
            "absolute_error": abs_err,
            "delta_real": delta_real,
            "delta_pred": delta_pred,
            "delta_error": delta_err,
        })
        df_node.to_csv(detailed_dir / f"node_{node:03d}_timeseries.csv", index=False)

    return top_nodes

def plot_tt_stats(tt_stats: dict, out_dir: Path, delta_t: float):
    plt.figure(figsize=(12, 5))
    plt.plot(tt_stats["tt_real_per_t"], label="real TT per step")
    plt.plot(tt_stats["tt_pred_per_t"], label="pred TT per step")
    plt.title(f"Travel time per timestep (Δt={delta_t})")
    plt.xlabel("t (eval horizon)")
    plt.ylabel("Δt * ||q_t||_1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tt_per_timestep.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(tt_stats["tt_real_cum"], label="real cumulative TT")
    plt.plot(tt_stats["tt_pred_cum"], label="pred cumulative TT")
    plt.title(f"Cumulative travel time (Δt={delta_t})")
    plt.xlabel("t (eval horizon)")
    plt.ylabel("cumulative TT")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tt_cumulative.png", dpi=200)
    plt.close()

def compute_metrics(pred: np.ndarray, real: np.ndarray):
    err = pred - real
    abs_err = np.abs(err)
    sq_err = err ** 2

    mae_global = float(abs_err.mean())
    rmse_global = float(np.sqrt(sq_err.mean()))

    per_node_mae = abs_err.mean(axis=0)
    per_node_rmse = np.sqrt(sq_err.mean(axis=0))

    per_time_mae = abs_err.mean(axis=1)
    per_time_rmse = np.sqrt(sq_err.mean(axis=1))

    return {
        "err": err,
        "abs_err": abs_err,
        "mae_global": mae_global,
        "rmse_global": rmse_global,
        "per_node_mae": per_node_mae,
        "per_node_rmse": per_node_rmse,
        "per_time_mae": per_time_mae,
        "per_time_rmse": per_time_rmse,
    }

def save_tt_stats(tt_stats: dict, out_dir: Path, delta_t: float):
    df = pd.DataFrame([{
        "delta_t": delta_t,
        "tt_real": tt_stats["tt_real"],
        "tt_pred": tt_stats["tt_pred"],
        "tt_signed_diff": tt_stats["tt_signed_diff"],
        "tt_abs_diff": tt_stats["tt_abs_diff"],
        "tt_rel_diff": tt_stats["tt_rel_diff"],
    }])
    df.to_csv(out_dir / "tt_stats.csv", index=False)

    df_time = pd.DataFrame({
        "t_eval": np.arange(len(tt_stats["tt_real_per_t"])),
        "tt_real_per_t": tt_stats["tt_real_per_t"],
        "tt_pred_per_t": tt_stats["tt_pred_per_t"],
        "tt_real_cum": tt_stats["tt_real_cum"],
        "tt_pred_cum": tt_stats["tt_pred_cum"],
    })
    df_time.to_csv(out_dir / "tt_timeseries.csv", index=False)


def save_group_summaries(group_summaries: dict, out_dir: Path, n_nodes_total: int):
    group_df = pd.DataFrame([
        group_summaries["zero_nodes"],
        group_summaries["nonzero_nodes"],
    ])
    group_df.to_csv(out_dir / "group_summary.csv", index=False)

    node_groups_df = pd.DataFrame({
        "node": np.arange(n_nodes_total),
        "is_zero_node": False,
        "is_nonzero_node": False,
    })
    node_groups_df.loc[group_summaries["zero_node_indices"], "is_zero_node"] = True
    node_groups_df.loc[group_summaries["nonzero_node_indices"], "is_nonzero_node"] = True
    node_groups_df.to_csv(out_dir / "node_groups.csv", index=False)


def save_summary(
    run_dir: Path,
    out_dir: Path,
    pred: np.ndarray,
    real: np.ndarray,
    assign: np.ndarray,
    metrics: dict,
    seed_steps,
    tt_stats: dict,
    group_summaries: dict,
):
    summary = {
        "run_dir": str(run_dir),
        "pred_shape": list(pred.shape),
        "real_shape": list(real.shape),
        "assign_shape": list(assign.shape),
        "timesteps": int(pred.shape[0]),
        "nodes": int(pred.shape[1]),
        "mae_global": metrics["mae_global"],
        "rmse_global": metrics["rmse_global"],
        "seed_steps": None if seed_steps is None else int(seed_steps),
        "tt_summary": {
            "tt_real": tt_stats["tt_real"],
            "tt_pred": tt_stats["tt_pred"],
            "tt_signed_diff": tt_stats["tt_signed_diff"],
            "tt_abs_diff": tt_stats["tt_abs_diff"],
            "tt_rel_diff": tt_stats["tt_rel_diff"],
        },
        "node_groups": {
            "zero_nodes_count": group_summaries["zero_nodes"]["n_nodes"],
            "nonzero_nodes_count": group_summaries["nonzero_nodes"]["n_nodes"],
            "zero_nodes": group_summaries["zero_nodes"],
            "nonzero_nodes": group_summaries["nonzero_nodes"],
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def save_rankings(out_dir: Path, metrics: dict):
    node_df = pd.DataFrame({
        "node": np.arange(len(metrics["per_node_mae"])),
        "mae": metrics["per_node_mae"],
        "rmse": metrics["per_node_rmse"],
    }).sort_values("mae", ascending=False)

    time_df = pd.DataFrame({
        "t": np.arange(len(metrics["per_time_mae"])),
        "mae": metrics["per_time_mae"],
        "rmse": metrics["per_time_rmse"],
    }).sort_values("mae", ascending=False)

    node_df.to_csv(out_dir / "node_ranking.csv", index=False)
    time_df.to_csv(out_dir / "time_ranking.csv", index=False)

    print("\nTop 10 node'ów z największym MAE:")
    print(node_df.head(10).to_string(index=False))

    print("\nTop 10 timestepów z największym MAE:")
    print(time_df.head(10).to_string(index=False))

    return node_df, time_df


def maybe_load_metrics_csv(run_dir: Path):
    matches = sorted(run_dir.glob("*_metrics.csv"))
    if not matches:
        return None, None

    path = matches[0]
    df = pd.read_csv(path)
    if df.empty:
        return path, None

    row = df.iloc[0].to_dict()
    return path, row


def add_seed_line(seed_steps):
    if seed_steps is not None:
        plt.axvline(seed_steps, linestyle="--", linewidth=1, label="seed_steps")


def plot_mean_signal(pred: np.ndarray, real: np.ndarray, out_dir: Path, seed_steps):
    pred_mean = pred.mean(axis=1)
    real_mean = real.mean(axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(real_mean, label="real mean over nodes")
    plt.plot(pred_mean, label="pred mean over nodes")
    add_seed_line(seed_steps)
    plt.title("Średni sygnał po wszystkich node'ach")
    plt.xlabel("t")
    plt.ylabel("mean q")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_signal_over_time.png", dpi=200)
    plt.close()


def plot_error_over_time(metrics: dict, out_dir: Path, seed_steps):
    plt.figure(figsize=(12, 5))
    plt.plot(metrics["per_time_mae"], label="MAE over time")
    plt.plot(metrics["per_time_rmse"], label="RMSE over time")
    add_seed_line(seed_steps)
    plt.title("Błąd w czasie")
    plt.xlabel("t")
    plt.ylabel("error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "error_over_time.png", dpi=200)
    plt.close()


def plot_error_heatmap(abs_err: np.ndarray, out_dir: Path, seed_steps):
    plt.figure(figsize=(14, 6))
    plt.imshow(abs_err.T, aspect="auto", origin="lower")
    plt.colorbar(label="|pred-real|")
    if seed_steps is not None:
        plt.axvline(seed_steps, linestyle="--", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("node")
    plt.title("Heatmapa błędu bezwzględnego")
    plt.tight_layout()
    plt.savefig(out_dir / "absolute_error_heatmap.png", dpi=200)
    plt.close()


def plot_error_histogram(abs_err: np.ndarray, out_dir: Path):
    plt.figure(figsize=(10, 5))
    plt.hist(abs_err.ravel(), bins=100)
    plt.title("Histogram |pred-real|")
    plt.xlabel("|pred-real|")
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "absolute_error_histogram.png", dpi=200)
    plt.close()


def plot_top_nodes_bar(node_df: pd.DataFrame, out_dir: Path, top_k: int = 20):
    top_df = node_df.head(top_k).sort_values("mae", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["node"].astype(str), top_df["mae"])
    plt.title(f"Top {top_k} node'ów z największym MAE")
    plt.xlabel("MAE")
    plt.ylabel("node")
    plt.tight_layout()
    plt.savefig(out_dir / f"top_{top_k}_nodes_by_mae.png", dpi=200)
    plt.close()


def plot_selected_nodes(pred: np.ndarray, real: np.ndarray, assign: np.ndarray, out_dir: Path, nodes, seed_steps):
    nodes_dir = out_dir / "node_plots"
    nodes_dir.mkdir(exist_ok=True)

    for node in nodes:
        plt.figure(figsize=(12, 5))
        pred_vis = pred[:, node].copy()
        if seed_steps is not None:
            pred_vis[:seed_steps] = np.nan

        plt.plot(real[:, node], label="real q")
        plt.plot(pred_vis, label="pred q")
        add_seed_line(seed_steps)
        plt.title(f"Node {node}: real vs pred")
        plt.xlabel("t")
        plt.ylabel("q")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(nodes_dir / f"node_{node:03d}_real_vs_pred.png", dpi=200)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(np.abs(pred[:, node] - real[:, node]))
        add_seed_line(seed_steps)
        plt.title(f"Node {node}: |pred-real|")
        plt.xlabel("t")
        plt.ylabel("absolute error")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(nodes_dir / f"node_{node:03d}_abs_error.png", dpi=200)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(assign[:, node])
        add_seed_line(seed_steps)
        plt.title(f"Node {node}: assignment input")
        plt.xlabel("t")
        plt.ylabel("a")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(nodes_dir / f"node_{node:03d}_assign.png", dpi=200)
        plt.close()


def choose_nodes(node_df: pd.DataFrame, n_nodes_total: int, num_best=3, num_middle=2, num_worst=3):
    worst_nodes = node_df.head(num_worst)["node"].tolist()

    middle_start = max(0, len(node_df) // 2 - num_middle // 2)
    middle_nodes = node_df.iloc[middle_start:middle_start + num_middle]["node"].tolist()

    best_nodes = node_df.tail(num_best)["node"].tolist()

    nodes = []
    for n in worst_nodes + middle_nodes + best_nodes:
        n = int(n)
        if 0 <= n < n_nodes_total and n not in nodes:
            nodes.append(n)
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Katalog jednego runu inferencji, np. ./infer_20260414_151403")
    parser.add_argument("--output_subdir", type=str, default="analysis", help="Podkatalog na wyniki analizy")
    parser.add_argument("--top_k_nodes", type=int, default=10)
    parser.add_argument("--num_best_nodes", type=int, default=10)
    parser.add_argument("--num_middle_nodes", type=int, default=5)
    parser.add_argument("--num_worst_nodes", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Nie istnieje run_dir: {run_dir}")

    out_dir = run_dir / args.output_subdir
    out_dir.mkdir(exist_ok=True)

    pred_path = find_single("*_pred_q.npy", run_dir)
    real_path = find_single("*_real_q.npy", run_dir)
    assign_path = find_single("*_assign.npy", run_dir)

    pred = np.load(pred_path)
    real = np.load(real_path)
    assign = np.load(assign_path)

    if pred.shape != real.shape:
        raise ValueError(f"pred i real mają różne shape: {pred.shape} vs {real.shape}")
    if pred.ndim != 2:
        raise ValueError(f"Skrypt zakłada shape (T, N), dostałem {pred.shape}")
    if assign.shape != pred.shape:
        print(f"[INFO] assign ma inny shape niż pred/real: {assign.shape} vs {pred.shape}. Analiza dalej będzie działać, jeśli pierwsze 2 wymiary są zgodne.")
        if assign.ndim != 2 or assign.shape[0] != pred.shape[0] or assign.shape[1] != pred.shape[1]:
            raise ValueError("assign nie ma zgodnego shape (T, N)")

    metrics_csv_path, metrics_csv_row = maybe_load_metrics_csv(run_dir)
    seed_steps = None
    if metrics_csv_row is not None and "seed_steps" in metrics_csv_row and pd.notna(metrics_csv_row["seed_steps"]):
        seed_steps = int(metrics_csv_row["seed_steps"])

    print(f"run_dir:   {run_dir}")
    print(f"pred_path: {pred_path.name}")
    print(f"real_path: {real_path.name}")
    print(f"assign_path: {assign_path.name}")
    if metrics_csv_path is not None:
        print(f"metrics_csv: {metrics_csv_path.name}")
    print(f"pred shape:   {pred.shape}")
    print(f"real shape:   {real.shape}")
    print(f"assign shape: {assign.shape}")
    print(f"seed_steps:   {seed_steps}")

    metrics = compute_metrics(pred, real)

    delta_t = 10.0  # albo 1.0, jeśli chcesz jednostkę "na krok"
    tt_stats = compute_tt_stats(pred, real, delta_t=delta_t, seed_steps=seed_steps)
    group_summaries = compute_group_summaries(
        pred,
        real,
        seed_steps=seed_steps,
        delta_t=delta_t,
    )

    save_summary(
        run_dir,
        out_dir,
        pred,
        real,
        assign,
        metrics,
        seed_steps,
        tt_stats,
        group_summaries,
    )
    node_df, time_df = save_rankings(out_dir, metrics)

    top_active_nodes = plot_top_active_nodes_detailed(
        pred=pred,
        real=real,
        out_dir=out_dir,
        seed_steps=seed_steps,
        top_k=10,
    )

    print("\nTop 5 najaktywniejszych node'ów:", top_active_nodes)

    print(f"\nTT real:       {tt_stats['tt_real']:.6f}")
    print(f"TT pred:       {tt_stats['tt_pred']:.6f}")
    print(f"TT signed diff:{tt_stats['tt_signed_diff']:.6f}")
    print(f"TT abs diff:   {tt_stats['tt_abs_diff']:.6f}")
    print(f"TT rel diff:   {tt_stats['tt_rel_diff']:.6%}")

    print("\nPodsumowanie grup node'ów:")
    for group_key, group_label in (("zero_nodes", "zerowe"), ("nonzero_nodes", "niezerowe")):
        group = group_summaries[group_key]
        print(f"- nody {group_label}: {group['n_nodes']}")
        print(f"  MAE:           {group['mae']:.6f}" if pd.notna(group['mae']) else "  MAE:           nan")
        print(f"  TT real:       {group['tt_real']:.6f}" if pd.notna(group['tt_real']) else "  TT real:       nan")
        print(f"  TT pred:       {group['tt_pred']:.6f}" if pd.notna(group['tt_pred']) else "  TT pred:       nan")
        print(f"  TT signed diff:{group['tt_signed_diff']:.6f}" if pd.notna(group['tt_signed_diff']) else "  TT signed diff:nan")
        print(f"  TT abs diff:   {group['tt_abs_diff']:.6f}" if pd.notna(group['tt_abs_diff']) else "  TT abs diff:   nan")
        print(f"  TT rel diff:   {group['tt_rel_diff']:.6%}" if pd.notna(group['tt_rel_diff']) else "  TT rel diff:   nan")

    save_tt_stats(tt_stats, out_dir, delta_t=delta_t)
    save_group_summaries(group_summaries, out_dir, n_nodes_total=pred.shape[1])
    plot_tt_stats(tt_stats, out_dir, delta_t=delta_t)

    selected_nodes = choose_nodes(
        node_df=node_df,
        n_nodes_total=pred.shape[1],
        num_best=args.num_best_nodes,
        num_middle=args.num_middle_nodes,
        num_worst=args.num_worst_nodes,
    )

    print("\nWybrane node'y do wykresów:", selected_nodes)

    plot_mean_signal(pred, real, out_dir, seed_steps)
    plot_error_over_time(metrics, out_dir, seed_steps)
    plot_error_heatmap(metrics["abs_err"], out_dir, seed_steps)
    plot_error_histogram(metrics["abs_err"], out_dir)
    plot_top_nodes_bar(node_df, out_dir, top_k=args.top_k_nodes)
    plot_selected_nodes(pred, real, assign, out_dir, selected_nodes, seed_steps)

    print(f"\nAnaliza zakończona. Wyniki zapisane w: {out_dir}")


if __name__ == "__main__":
    main()