import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from engine import TrainerGenTTP
from utilities import (
    load_csv_adj,
    split_file_names,
    make_subset_loader,
    infer_target_dim_from_batch,
    init_metric_acc,
    update_metric_acc,
    finalize_metric_acc,
    evaluate_loader,
    set_seed,
    save_learning_curves
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train GenTTP model.")

    # Data paths
    parser.add_argument("--q_dir", type=str, required=True, help="Directory with flow files.")
    parser.add_argument("--a_dir", type=str, required=True, help="Directory with assignment files.")
    parser.add_argument("--adjdata", type=str, required=True, help="Path to adjacency matrix CSV.")

    # Experiment setup
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str, default="GenTTP")

    # Data split
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    # Sequence lengths
    parser.add_argument("--seq_length_q", type=int, default=15)
    parser.add_argument("--seq_length_a", type=int, default=30)
    parser.add_argument("--seq_length_y", type=int, default=1)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)

    # Graph settings
    parser.add_argument("--num_nodes", type=int, default=195)
    parser.add_argument("--gcn_bool", action="store_true")
    parser.add_argument("--addaptadj", action="store_true")
    parser.add_argument("--randomadj", action="store_true")

    # Model architecture
    parser.add_argument("--sequence_model", type=str, default="gru", choices=["lstm", "gru", "attention"])
    parser.add_argument("--fuse_method", type=str, default="attention", choices=["concatenate", "attention", "wavenet_only", "assignment_only"],)

    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--fused_dim", type=int, default=64)
    parser.add_argument("--a_embedding_size", type=int, default=32)
    parser.add_argument("--a_hidden_size", type=int, default=64)
    parser.add_argument("--q_rep_dim", type=int, default=32)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)
    parser.add_argument("--attention_num_heads", type=int, default=4)
    parser.add_argument("--attention_ff_dim", type=int, default=128)

    # Temporal convolution
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)

    return parser


def main():
    args = build_arg_parser().parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    output_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, "best_model.pth")
    metrics_path = os.path.join(output_dir, "training_metrics.csv")
    curves_path = os.path.join(output_dir, "learning_curves.png")

    print(args)

    # Load adjacency matrix
    supports = load_csv_adj(args.adjdata, args.num_nodes, device)
    print(f"Loaded adjacency matrix from: {args.adjdata}")

    adjinit = None if args.randomadj else supports[0]

    # Split files to avoid data leakage
    train_files, val_files, test_files = split_file_names(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"train files: {len(train_files)}")
    print(f"val files: {len(val_files)}")
    print(f"test files: {len(test_files)}")

    train_loader = make_subset_loader(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        selected_files=train_files,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
        seq_length_y=args.seq_length_y,
        target_nodes=args.num_nodes,
    )

    val_loader = make_subset_loader(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        selected_files=val_files,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
        seq_length_y=args.seq_length_y,
        target_nodes=args.num_nodes,
    )

    test_loader = make_subset_loader(
        q_dir=args.q_dir,
        a_dir=args.a_dir,
        selected_files=test_files,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seq_length_q=args.seq_length_q,
        seq_length_a=args.seq_length_a,
        seq_length_y=args.seq_length_y,
        target_nodes=args.num_nodes,
    )

    print("train dataset size:", len(train_loader.dataset))
    print("val dataset size:", len(val_loader.dataset))
    print("test dataset size:", len(test_loader.dataset))
    print("batches per epoch:", len(train_loader))

    # Infer output dimension from the first batch
    first_batch = next(iter(train_loader))
    target_dim = infer_target_dim_from_batch(first_batch)

    engine = TrainerGenTTP(
        scaler=None,
        in_dim=1,
        num_nodes=args.num_nodes,
        nhid=args.nhid,
        dropout=args.dropout,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        supports=supports,
        gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj,
        aptinit=adjinit,
        kernel_size=args.kernel_size,
        blocks=args.blocks,
        layers=args.layers,
        target_dim=target_dim,
        sequence_model=args.sequence_model,
        fuse_method=args.fuse_method,
        a_embedding_size=args.a_embedding_size,
        a_hidden_size=args.a_hidden_size,
        q_rep_dim=args.q_rep_dim,
        fused_dim=args.fused_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        attention_num_heads=args.attention_num_heads,
        attention_ff_dim=args.attention_ff_dim,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_mae": [],
        "train_rmse": [],
        "valid_loss": [],
        "valid_mae": [],
        "valid_rmse": [],
        "train_time": [],
        "val_time": [],
    }

    best_val_mae = float("inf")
    best_epoch = -1

    print("Start training", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_acc = init_metric_acc()

        train_start = time.time()

        for batch in train_loader:
            metrics = engine.train(batch)
            update_metric_acc(train_acc, metrics, batch)

        train_time = time.time() - train_start
        train_metrics = finalize_metric_acc(train_acc)

        val_start = time.time()
        valid_metrics = evaluate_loader(engine, val_loader)
        val_time = time.time() - val_start

        # Select best checkpoint using validation MAE
        if valid_metrics["mae"] < best_val_mae:
            best_val_mae = valid_metrics["mae"]
            best_epoch = epoch
            torch.save(engine.model.state_dict(), best_model_path)

        history["epoch"].append(epoch)

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_rmse"].append(train_metrics["rmse"])

        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_mae"].append(valid_metrics["mae"])
        history["valid_rmse"].append(valid_metrics["rmse"])

        history["train_time"].append(train_time)
        history["val_time"].append(val_time)

        print(
            f"Epoch: {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train MAE: {train_metrics['mae']:.4f}, "
            f"Train RMSE: {train_metrics['rmse']:.4f} | "
            f"Valid Loss: {valid_metrics['loss']:.4f}, "
            f"Valid MAE: {valid_metrics['mae']:.4f}, "
            f"Valid RMSE: {valid_metrics['rmse']:.4f} | "
            f"Time: {train_time:.2f}s"
        )

    print("Training finished")
    print("Average training time: {:.4f} secs/epoch".format(np.mean(history["train_time"])))
    print("Average validation time: {:.4f} secs".format(np.mean(history["val_time"])))

    df_metrics = pd.DataFrame(history)
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Training metrics saved to: {metrics_path}")

    save_learning_curves(history, curves_path)
    print(f"Learning curves saved to: {curves_path}")

    # Evaluate the best checkpoint on the test set
    engine.model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_loader(engine, test_loader)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(
        "Test MAE: {:.4f}, Test RMSE: {:.4f}".format(
            test_metrics["mae"],
            test_metrics["rmse"],
        )
    )

    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(engine.model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()