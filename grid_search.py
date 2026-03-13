import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd


def receptive_field(kernel_size: int, blocks: int, layers: int) -> int:
    """
    Approximate receptive field used in Graph WaveNet-like temporal stack:
    R = 1 + b * (k - 1) * (2^l - 1)

    where:
      b = blocks
      k = kernel_size
      l = layers
    """
    return 1 + blocks * (kernel_size - 1) * (2 ** layers - 1)


def build_command(
    base_dir: str,
    save_prefix: str,
    config: Dict[str, Any],
) -> List[str]:
    """
    Build command line for train.py based on config.
    """
    command = [
        sys.executable,
        os.path.join(base_dir, "train.py"),
        "--device", str(config["device"]),
        "--data", str(config["data"]),
        "--adjdata", str(config["adjdata"]),
        "--adjtype", str(config["adjtype"]),
        "--num_nodes", str(config["num_nodes"]),
        "--in_dim", str(config["in_dim"]),
        "--seq_length", str(config["seq_length"]),
        "--nhid", str(config["nhid"]),
        "--epochs", str(config["epochs"]),
        "--print_every", str(config["print_every"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--dropout", str(config["dropout"]),
        "--weight_decay", str(config["weight_decay"]),
        "--save", save_prefix,
        "--kernel_size", str(config["kernel_size"]),
        "--blocks", str(config["blocks"]),
        "--layers", str(config["layers"]),
    ]

    # Boolean flags
    if config.get("gcn_bool", False):
        command.append("--gcn_bool")
    if config.get("aptonly", False):
        command.append("--aptonly")
    if config.get("addaptadj", False):
        command.append("--addaptadj")
    if config.get("randomadj", False):
        command.append("--randomadj")

    return command


def summarize_training_csv(csv_path: str) -> Optional[Dict[str, Any]]:
    """
    Read training_metrics.csv and extract the best metrics.
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": f"csv_read_error: {e}"}

    required_columns = [
        "epoch", "train_loss", "valid_loss",
        "valid_rmse", "valid_mape", "train_time", "val_time"
    ]
    if not all(col in df.columns for col in required_columns):
        return {"status": "csv_missing_columns"}

    best_loss_idx = df["valid_loss"].idxmin()
    best_rmse_idx = df["valid_rmse"].idxmin()
    best_mape_idx = df["valid_mape"].idxmin()

    best_valid_loss = float(df.loc[best_loss_idx, "valid_loss"])
    best_valid_rmse = float(df.loc[best_rmse_idx, "valid_rmse"])
    best_valid_mape = float(df.loc[best_mape_idx, "valid_mape"])
    best_epoch_loss = int(df.loc[best_loss_idx, "epoch"])
    best_epoch_rmse = int(df.loc[best_rmse_idx, "epoch"])
    best_epoch_mape = int(df.loc[best_mape_idx, "epoch"])

    train_loss_at_best = float(df.loc[best_loss_idx, "train_loss"])
    generalization_gap = best_valid_loss - train_loss_at_best

    total_train_time = float(df["train_time"].sum())
    total_val_time = float(df["val_time"].sum())
    mean_epoch_time = float(df["train_time"].mean())

    return {
        "best_valid_loss": best_valid_loss,
        "best_valid_rmse": best_valid_rmse,
        "best_valid_mape": best_valid_mape,
        "best_epoch_loss": best_epoch_loss,
        "best_epoch_rmse": best_epoch_rmse,
        "best_epoch_mape": best_epoch_mape,
        "train_loss_at_best": train_loss_at_best,
        "generalization_gap": generalization_gap,
        "total_train_time": total_train_time,
        "total_val_time": total_val_time,
        "mean_epoch_time": mean_epoch_time,
        "epochs_recorded": len(df),
    }


def generate_grid() -> List[Dict[str, Any]]:
    """
    Define a hyperparameter grid.
    """

    grid = {
        "learning_rate": [1e-3],
        "batch_size": [32],
        "dropout": [0.2],
        "weight_decay": [1e-4],
        "nhid": [32],
        "kernel_size": [2, 3],
        "blocks": [3, 4],
        "layers": [2, 3],
        "gcn_bool": [True],
        "addaptadj": [True, False],
        "randomadj": [False, True],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))

        # Sensible constraints:
        #randomadj only matters if addaptadj=True
        if cfg["randomadj"] and not cfg["addaptadj"]:
            continue

        configs.append(cfg)

    return configs


def run_single_experiment(
    base_dir: str,
    experiment_root: str,
    run_idx: int,
    fixed_config: Dict[str, Any],
    variable_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run one experiment and return collected result dict.
    """
    config = {**fixed_config, **variable_config}

    rf = receptive_field(
        kernel_size=config["kernel_size"],
        blocks=config["blocks"],
        layers=config["layers"],
    )

    # Skip invalid configs where temporal receptive field is too short
    if rf < config["seq_length"]:
        return {
            **config,
            "run_idx": run_idx,
            "receptive_field": rf,
            "status": f"skipped_rf<{config['seq_length']}",
        }

    run_name = (
        f"run_{run_idx:03d}"
        f"_lr{config['learning_rate']}"
        f"_bs{config['batch_size']}"
        f"_do{config['dropout']}"
        f"_wd{config['weight_decay']}"
        f"_h{config['nhid']}"
        f"_k{config['kernel_size']}"
        f"_b{config['blocks']}"
        f"_l{config['layers']}"
        f"_gcn{int(config['gcn_bool'])}"
        f"_adapt{int(config['addaptadj'])}"
        f"_rand{int(config['randomadj'])}"
    )

    save_dir = os.path.join(experiment_root, run_name)
    os.makedirs(save_dir, exist_ok=True)

    save_prefix = os.path.join(save_dir, "model")

    command = build_command(
        base_dir=base_dir,
        save_prefix=save_prefix,
        config=config,
    )

    print("=" * 100)
    print(f"[RUN {run_idx}] {run_name}")
    print(f"Save dir: {save_dir}")
    print(f"Receptive field: {rf}")
    print("Command:")
    print(" ".join(command))
    print("=" * 100)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    full_log = []
    for line in process.stdout:
        print(line, end="")
        full_log.append(line)

    process.wait()

    # Save raw log
    log_path = os.path.join(save_dir, "train_stdout.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(full_log)

    result = {
        **config,
        "run_idx": run_idx,
        "run_name": run_name,
        "save_dir": save_dir,
        "receptive_field": rf,
        "return_code": process.returncode,
    }

    if process.returncode != 0:
        result["status"] = f"train_failed_code_{process.returncode}"
        return result

    # Expected metrics file
    metrics_csv = os.path.join(save_dir, "training_metrics.csv")
    summary = summarize_training_csv(metrics_csv)

    if summary is None:
        result["status"] = "missing_training_metrics_csv"
        return result

    result.update(summary)
    result["metrics_csv"] = metrics_csv
    return result


def run_grid_search():
    """
    Main grid-search routine.
    """
    base_dir = os.path.abspath(os.path.dirname(__file__))

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

    experiment_root = os.path.join(base_dir, "garage", f"gridsearch_{job_id}")
    os.makedirs(experiment_root, exist_ok=True)

    # Fixed configuration: dataset / system / general training params
    fixed_config = {
        "device": "cpu",  # change to e.g. cuda:0 if available
        "data": os.path.join(base_dir, "data", "WAVENET_READY"),
        "adjdata": os.path.join(base_dir, "data", "adjacency_matrix.csv"),
        "adjtype": "doubletransition",
        "num_nodes": 1430,
        "in_dim": 1,
        "seq_length": 12,
        "epochs": 20,
        "print_every": 10,
        "aptonly": False,
    }

    configs = generate_grid()
    print(f"Total candidate configurations: {len(configs)}")
    print(f"Experiment root: {experiment_root}")

    all_results = []

    for run_idx, variable_config in enumerate(configs, start=1):
        result = run_single_experiment(
            base_dir=base_dir,
            experiment_root=experiment_root,
            run_idx=run_idx,
            fixed_config=fixed_config,
            variable_config=variable_config,
        )
        all_results.append(result)

        results_df = pd.DataFrame(all_results)
        results_csv = os.path.join(experiment_root, "grid_results.csv")
        results_df.to_csv(results_csv, index=False)

        print(f"\n[INFO] Partial results saved to: {results_csv}\n")

    # Final summary
    results_df = pd.DataFrame(all_results)

    if "best_valid_loss" in results_df.columns:
        results_df = results_df.sort_values(
            by=["status", "best_valid_loss"],
            ascending=[True, True],
            na_position="last"
        )

    final_csv = os.path.join(experiment_root, "grid_results_sorted.csv")
    results_df.to_csv(final_csv, index=False)

    print("=" * 100)
    print("GRID SEARCH FINISHED")
    print(f"Results: {final_csv}")

    successful = results_df[results_df["status"] == "ok"] if "status" in results_df.columns else pd.DataFrame()

    if not successful.empty:
        best = successful.sort_values("best_valid_loss").iloc[0]
        print("\nBest configuration:")
        print(best.to_string())
    else:
        print("\nNo successful runs found.")
    print("=" * 100)


if __name__ == "__main__":
    run_grid_search()