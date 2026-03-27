import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def find_available_datasets(processed_root: Path) -> list[str]:
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


def load_dataset_metadata(dataset_dir: Path) -> dict:
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
                "Expected 4 dimensions: (samples, seq_len, num_nodes, in_dim)."
            )

        seq_length = int(x.shape[1])
        num_nodes = int(x.shape[2])
        in_dim = int(x.shape[3])

    return {
        "seq_length": seq_length,
        "num_nodes": num_nodes,
        "in_dim": in_dim,
        "adj_path": adj_path,
        "data_dir": dataset_dir,
    }


def run_single_wavenet_training(
    dataset_name: str,
    base_dir: Path,
    processed_root: Path,
    garage_root: Path,
    device: str = "cpu",
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    nhid: int = 32,
    kernel_size: int = 2,
    blocks: int = 4,
    layers: int = 2,
    print_every: int = 10,
    use_gcn: bool = True,
    addaptadj: bool = False,
) -> None:
    """
    Run Graph WaveNet training for one dataset.
    """
    dataset_dir = processed_root / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    metadata = load_dataset_metadata(dataset_dir)

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_dir = garage_root / dataset_name / f"experiment_{job_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_save_prefix = save_dir / "model"
    train_script = base_dir / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found: {train_script}")

    command = [
        sys.executable,
        str(train_script),
        "--device", str(device),
        "--data", str(metadata["data_dir"]),
        "--adjdata", str(metadata["adj_path"]),
        "--adjtype", "doubletransition",
        "--num_nodes", str(metadata["num_nodes"]),
        "--in_dim", str(metadata["in_dim"]),
        "--seq_length", str(metadata["seq_length"]),
        "--nhid", str(nhid),
        "--epochs", str(epochs),
        "--print_every", str(print_every),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--save", str(model_save_prefix),
        "--kernel_size", str(kernel_size),
        "--blocks", str(blocks),
        "--layers", str(layers),
    ]

    if use_gcn:
        command.append("--gcn_bool")

    if addaptadj:
        command.append("--addaptadj")

    print("=" * 90)
    print(f"Starting training for dataset: {dataset_name}")
    print(f"Data directory: {metadata['data_dir']}")
    print(f"Adjacency path: {metadata['adj_path']}")
    print(f"num_nodes={metadata['num_nodes']}, in_dim={metadata['in_dim']}, seq_length={metadata['seq_length']}")
    print(f"Save directory: {save_dir}")
    print("=" * 90)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(base_dir),
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode == 0:
        print(f"Training completed successfully for dataset: {dataset_name}")
    else:
        print(f"Error while training dataset {dataset_name}. Exit code: {process.returncode}")


def run_wavenet_training_for_all(
    base_dir: Path,
    processed_root: Path,
    garage_root: Path,
    device: str = "cpu",
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    nhid: int = 32,
    kernel_size: int = 2,
    blocks: int = 4,
    layers: int = 2,
    print_every: int = 10,
    use_gcn: bool = True,
    addaptadj: bool = False,
) -> None:
    """
    Run Graph WaveNet training for every dataset found in processed_networks.
    """
    datasets = find_available_datasets(processed_root)

    if not datasets:
        raise ValueError(
            f"No valid datasets found in: {processed_root}\n"
            "Expected structure:\n"
            "processed_networks/<dataset>/train.npz\n"
            "processed_networks/<dataset>/val.npz\n"
            "processed_networks/<dataset>/test.npz\n"
            "processed_networks/<dataset>/adjacency_matrices/adjacency_spatial.csv"
        )

    print(f"Found {len(datasets)} dataset(s): {datasets}")

    for dataset_name in datasets:
        run_single_wavenet_training(
            dataset_name=dataset_name,
            base_dir=base_dir,
            processed_root=processed_root,
            garage_root=garage_root,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            nhid=nhid,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
            print_every=print_every,
            use_gcn=use_gcn,
            addaptadj=addaptadj,
        )


def analyze_training_csv(csv_path: str) -> None:
    """
    Analyze training_metrics.csv.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: File {csv_path} does not exist.")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

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
        print("Error: CSV file does not contain all required columns.")
        print(f"Required columns: {required_columns}")
        sys.exit(1)

    best_loss_idx = df["valid_loss"].idxmin()
    best_rmse_idx = df["valid_rmse"].idxmin()
    best_mape_idx = df["valid_mape"].idxmin()

    best_val_loss = df.loc[best_loss_idx, "valid_loss"]
    corresponding_train_loss = df.loc[best_loss_idx, "train_loss"]
    gen_gap = best_val_loss - corresponding_train_loss

    total_train_time = df["train_time"].sum()
    total_val_time = df["val_time"].sum()
    mean_epoch_time = df["train_time"].mean()

    print("=" * 60)
    print(f"training report: {csv_path.parent.name}")
    print("=" * 60)

    print("\n[1] time summary")
    print(f"  • Recorded epochs:       {len(df)}")
    print(f"  • Total training time:   {total_train_time / 60:.2f} min")
    print(f"  • Total validation time: {total_val_time / 60:.2f} min")
    print(f"  • Average epoch time:    {mean_epoch_time:.2f} s")

    print("\n[2] optimal training stopping points")
    print(f"  • Minimum Loss (MAE): {best_val_loss:.4f} (epoch {df.loc[best_loss_idx, 'epoch']})")
    print(f"  • Minimum RMSE:       {df.loc[best_rmse_idx, 'valid_rmse']:.4f} (epoch {df.loc[best_rmse_idx, 'epoch']})")
    print(f"  • Minimum MAPE:       {df.loc[best_mape_idx, 'valid_mape']:.4f} (epoch {df.loc[best_mape_idx, 'epoch']})")

    print(f"\n[3] overfitting analysis for epoch {df.loc[best_loss_idx, 'epoch']}")
    print(f"  • Train Loss (MAE):   {corresponding_train_loss:.4f}")
    print(f"  • Valid Loss (MAE):   {best_val_loss:.4f}")
    print(f"  • Generalization Gap: {gen_gap:.4f}")

    if corresponding_train_loss != 0 and gen_gap > (0.2 * corresponding_train_loss):
        print("    Significant difference between training and validation error (>20%).")
        print("    This may indicate early overfitting.")

    print("\n[4] last 5 epochs")
    recent_df = df.tail(5)[["epoch", "train_loss", "valid_loss", "valid_rmse"]].copy()
    print(recent_df.to_string(index=False))
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph WaveNet training and metrics analysis")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training for one dataset or all datasets")
    train_parser.add_argument("--dataset", type=str, default=None, help="Dataset name, e.g. ingolstadt_770")
    train_parser.add_argument("--all", action="store_true", help="Train on all datasets in processed_networks")
    train_parser.add_argument("--processed-root", type=str, default="data/processed_networks")
    train_parser.add_argument("--garage-root", type=str, default="garage")
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--nhid", type=int, default=32)
    train_parser.add_argument("--kernel-size", type=int, default=2)
    train_parser.add_argument("--blocks", type=int, default=4)
    train_parser.add_argument("--layers", type=int, default=2)
    train_parser.add_argument("--print-every", type=int, default=10)
    train_parser.add_argument("--no-gcn", action="store_true", help="Disable --gcn_bool")
    train_parser.add_argument("--addaptadj", action="store_true", help="Enable adaptive adjacency")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze training_metrics.csv")
    analyze_parser.add_argument("csv_path", type=str, help="Path to training_metrics.csv")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    processed_root = (base_dir / args.processed_root).resolve() if hasattr(args, "processed_root") else None
    garage_root = (base_dir / args.garage_root).resolve() if hasattr(args, "garage_root") else None

    if args.command == "analyze":
        analyze_training_csv(args.csv_path)
        return

    if args.command == "train":
        if args.all and args.dataset is not None:
            parser.error("Use either --dataset <name> or --all, not both.")

        if not args.all and args.dataset is None:
            parser.error("For training, provide either --dataset <name> or --all.")

        use_gcn = not args.no_gcn

        if args.all:
            run_wavenet_training_for_all(
                base_dir=base_dir,
                processed_root=processed_root,
                garage_root=garage_root,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                nhid=args.nhid,
                kernel_size=args.kernel_size,
                blocks=args.blocks,
                layers=args.layers,
                print_every=args.print_every,
                use_gcn=use_gcn,
                addaptadj=args.addaptadj,
            )
        else:
            run_single_wavenet_training(
                dataset_name=args.dataset,
                base_dir=base_dir,
                processed_root=processed_root,
                garage_root=garage_root,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                nhid=args.nhid,
                kernel_size=args.kernel_size,
                blocks=args.blocks,
                layers=args.layers,
                print_every=args.print_every,
                use_gcn=use_gcn,
                addaptadj=args.addaptadj,
            )


if __name__ == "__main__":
    main()