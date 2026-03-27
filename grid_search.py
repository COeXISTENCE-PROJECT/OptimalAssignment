import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def receptive_field(kernel_size: int, blocks: int, layers: int) -> int:
    """
    Approximate receptive field used in Graph WaveNet-like temporal stack:
    R = 1 + b * (k - 1) * (2^l - 1)
    """
    return 1 + blocks * (kernel_size - 1) * (2 ** layers - 1)


def find_available_datasets(processed_root: Path) -> List[str]:
    """
    Return dataset names that contain the required Graph WaveNet files.
    """
    if not processed_root.exists():
        return []

    datasets = []

    for dataset_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        train_path = dataset_dir / "train.npz"
        val_path = dataset_dir / "val.npz"
        test_path = dataset_dir / "test.npz"
        adj_path = dataset_dir / "adjacency_matrices" / "adjacency_spatial.csv"

        if train_path.exists() and val_path.exists() and test_path.exists() and adj_path.exists():
            datasets.append(dataset_dir.name)

    return datasets


def load_dataset_metadata(dataset_dir: Path) -> Dict[str, Any]:
    """
    Read dataset dimensions directly from train.npz.
    Expected x shape: (samples, seq_len, num_nodes, in_dim)
    """
    train_path = dataset_dir / "train.npz"
    adj_path = dataset_dir / "adjacency_matrices" / "adjacency_spatial.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")

    if not adj_path.exists():
        raise FileNotFoundError(f"Missing file: {adj_path}")

    with np.load(train_path) as data:
        x = data["x"]

        if x.ndim != 4:
            raise ValueError(
                f"Unexpected x shape in {train_path}: {x.shape}. "
                "Expected (samples, seq_len, num_nodes, in_dim)."
            )

        seq_length = int(x.shape[1])
        num_nodes = int(x.shape[2])
        in_dim = int(x.shape[3])

    return {
        "data_dir": dataset_dir,
        "adj_path": adj_path,
        "seq_length": seq_length,
        "num_nodes": num_nodes,
        "in_dim": in_dim,
    }


def build_command(
    base_dir: Path,
    save_prefix: Path,
    config: Dict[str, Any],
) -> List[str]:
    """
    Build command line for train.py based on config.
    """
    command = [
        sys.executable,
        str(base_dir / "train.py"),
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
        "--save", str(save_prefix),
        "--kernel_size", str(config["kernel_size"]),
        "--blocks", str(config["blocks"]),
        "--layers", str(config["layers"]),
    ]

    if config.get("gcn_bool", False):
        command.append("--gcn_bool")
    if config.get("aptonly", False):
        command.append("--aptonly")
    if config.get("addaptadj", False):
        command.append("--addaptadj")
    if config.get("randomadj", False):
        command.append("--randomadj")

    return command


def summarize_training_csv(csv_path: Path) -> Optional[Dict[str, Any]]:
    """
    Read training_metrics.csv and extract best metrics.
    """
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": f"csv_read_error: {e}"}

    required_columns = [
        "epoch",
        "train_loss",
        "valid_loss",
        "valid_rmse",
        "valid_mape",
        "train_time",
        "val_time",
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
        "status": "ok",
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
    Define hyperparameter grid.
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

        # randomadj only makes sense if addaptadj=True
        if cfg["randomadj"] and not cfg["addaptadj"]:
            continue

        configs.append(cfg)

    return configs


def run_single_experiment(
    base_dir: Path,
    dataset_name: str,
    experiment_root: Path,
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

    if rf < config["seq_length"]:
        return {
            **config,
            "dataset": dataset_name,
            "run_idx": run_idx,
            "receptive_field": rf,
            "status": f"skipped_rf_lt_seq_length_{config['seq_length']}",
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

    save_dir = experiment_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_prefix = save_dir / "model"

    command = build_command(
        base_dir=base_dir,
        save_prefix=save_prefix,
        config=config,
    )

    print("=" * 100)
    print(f"[{dataset_name}] RUN {run_idx}: {run_name}")
    print(f"Save dir: {save_dir}")
    print(f"Receptive field: {rf}")
    print("Command:")
    print(" ".join(command))
    print("=" * 100)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(base_dir),
    )

    full_log = []
    for line in process.stdout:
        print(line, end="")
        full_log.append(line)

    process.wait()

    log_path = save_dir / "train_stdout.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(full_log)

    result = {
        **config,
        "dataset": dataset_name,
        "run_idx": run_idx,
        "run_name": run_name,
        "save_dir": str(save_dir),
        "receptive_field": rf,
        "return_code": process.returncode,
    }

    if process.returncode != 0:
        result["status"] = f"train_failed_code_{process.returncode}"
        return result

    metrics_csv = save_dir / "training_metrics.csv"
    summary = summarize_training_csv(metrics_csv)

    if summary is None:
        result["status"] = "missing_training_metrics_csv"
        return result

    result.update(summary)
    result["metrics_csv"] = str(metrics_csv)
    return result


def run_grid_search_for_dataset(
    dataset_name: str,
    base_dir: Path,
    processed_root: Path,
    garage_root: Path,
    device: str = "cpu",
    epochs: int = 20,
    print_every: int = 10,
    adjtype: str = "doubletransition",
    aptonly: bool = False,
) -> None:
    """
    Run grid search for one dataset.
    """
    dataset_dir = processed_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    metadata = load_dataset_metadata(dataset_dir)

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    experiment_root = garage_root / dataset_name / f"gridsearch_{job_id}"
    experiment_root.mkdir(parents=True, exist_ok=True)

    fixed_config = {
        "device": device,
        "data": str(metadata["data_dir"]),
        "adjdata": str(metadata["adj_path"]),
        "adjtype": adjtype,
        "num_nodes": metadata["num_nodes"],
        "in_dim": metadata["in_dim"],
        "seq_length": metadata["seq_length"],
        "epochs": epochs,
        "print_every": print_every,
        "aptonly": aptonly,
    }

    configs = generate_grid()

    print(f"Dataset: {dataset_name}")
    print(f"Data dir: {metadata['data_dir']}")
    print(f"Adjacency: {metadata['adj_path']}")
    print(f"num_nodes={metadata['num_nodes']}, in_dim={metadata['in_dim']}, seq_length={metadata['seq_length']}")
    print(f"Total candidate configurations: {len(configs)}")
    print(f"Experiment root: {experiment_root}")

    all_results = []

    for run_idx, variable_config in enumerate(configs, start=1):
        result = run_single_experiment(
            base_dir=base_dir,
            dataset_name=dataset_name,
            experiment_root=experiment_root,
            run_idx=run_idx,
            fixed_config=fixed_config,
            variable_config=variable_config,
        )
        all_results.append(result)

        partial_df = pd.DataFrame(all_results)
        partial_csv = experiment_root / "grid_results.csv"
        partial_df.to_csv(partial_csv, index=False)
        print(f"\n[INFO] Partial results saved to: {partial_csv}\n")

    results_df = pd.DataFrame(all_results)

    if "best_valid_loss" in results_df.columns:
        results_df = results_df.sort_values(
            by=["status", "best_valid_loss"],
            ascending=[True, True],
            na_position="last",
        )

    final_csv = experiment_root / "grid_results_sorted.csv"
    results_df.to_csv(final_csv, index=False)

    print("=" * 100)
    print(f"GRID SEARCH FINISHED FOR DATASET: {dataset_name}")
    print(f"Results: {final_csv}")

    if "status" in results_df.columns:
        successful = results_df[results_df["status"] == "ok"]
    else:
        successful = pd.DataFrame()

    if not successful.empty:
        best = successful.sort_values("best_valid_loss").iloc[0]
        print("\nBest configuration:")
        print(best.to_string())
    else:
        print("\nNo successful runs found.")

    print("=" * 100)


def run_grid_search_for_all_datasets(
    base_dir: Path,
    processed_root: Path,
    garage_root: Path,
    device: str = "cpu",
    epochs: int = 20,
    print_every: int = 10,
    adjtype: str = "doubletransition",
    aptonly: bool = False,
) -> None:
    """
    Run grid search for all datasets found in processed_networks.
    """
    datasets = find_available_datasets(processed_root)

    if not datasets:
        raise ValueError(
            f"No valid datasets found in: {processed_root}\n"
            "Expected:\n"
            "processed_networks/<dataset>/train.npz\n"
            "processed_networks/<dataset>/val.npz\n"
            "processed_networks/<dataset>/test.npz\n"
            "processed_networks/<dataset>/adjacency_matrices/adjacency_spatial.csv"
        )

    print(f"Found datasets: {datasets}")

    for dataset_name in datasets:
        run_grid_search_for_dataset(
            dataset_name=dataset_name,
            base_dir=base_dir,
            processed_root=processed_root,
            garage_root=garage_root,
            device=device,
            epochs=epochs,
            print_every=print_every,
            adjtype=adjtype,
            aptonly=aptonly,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph WaveNet grid search for processed datasets")

    parser.add_argument("--dataset", type=str, default=None, help="Dataset name, e.g. ingolstadt_770")
    parser.add_argument("--all", action="store_true", help="Run grid search for all datasets")
    parser.add_argument("--processed-root", type=str, default="data/processed_networks")
    parser.add_argument("--garage-root", type=str, default="garage")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--aptonly", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.all and args.dataset is not None:
        parser.error("Use either --dataset <name> or --all, not both.")

    if not args.all and args.dataset is None:
        parser.error("Provide either --dataset <name> or --all.")

    base_dir = Path(__file__).resolve().parent
    processed_root = (base_dir / args.processed_root).resolve()
    garage_root = (base_dir / args.garage_root).resolve()

    if args.all:
        run_grid_search_for_all_datasets(
            base_dir=base_dir,
            processed_root=processed_root,
            garage_root=garage_root,
            device=args.device,
            epochs=args.epochs,
            print_every=args.print_every,
            adjtype=args.adjtype,
            aptonly=args.aptonly,
        )
    else:
        run_grid_search_for_dataset(
            dataset_name=args.dataset,
            base_dir=base_dir,
            processed_root=processed_root,
            garage_root=garage_root,
            device=args.device,
            epochs=args.epochs,
            print_every=args.print_every,
            adjtype=args.adjtype,
            aptonly=args.aptonly,
        )


if __name__ == "__main__":
    main()