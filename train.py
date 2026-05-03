import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import TrainerADTTP
import pandas as pd
import os
from dataset_utils.DataLoader import make_qA_loader
from dataset_utils.DataLoader import SumoFolderDataset
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import random
import pandas as pd
import csv
import json
import wandb

# from your_dataset_module import make_qA_loader

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='/scratch/tmp', help='data root path')
parser.add_argument('--q_dir', type=str, default='/scratch/tmp/vec_flows_10s', help='flow directory')
parser.add_argument('--a_dir', type=str, default='/scratch/tmp/vec_assignments_10s', help='assignment directory')
parser.add_argument('--seed', type=int, default=42, help='random seed for split')
parser.add_argument('--train_ratio', type=float, default=0.7, help='train split ratio')
parser.add_argument('--val_ratio', type=float, default=0.15, help='validation split ratio')
parser.add_argument('--seq_length_q', type=int, default=15, help='q history length')
parser.add_argument('--seq_length_a', type=int, default=30, help='a history length')
parser.add_argument('--seq_length_y', type=int, default=1, help='prediction horizon length')

parser.add_argument('--loss', type=str, default='mae', choices=['mae', 'mape', 'rmse', 'adj_mape', 'flow_cons'], help='loss used for optimization')
parser.add_argument('--monitor', type=str, default='loss', choices=['loss', 'mae', 'mape', 'rmse', 'adj_mape', 'flow_cons'], help='metric used to select best checkpoint')
parser.add_argument('--alpha', type=float, default=1.0, help='weight for MAE in combined loss: loss = alpha*MAE + (1-alpha)*ADJ_MAPE')

parser.add_argument('--adjdata', type=str, default=None, help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')

parser.add_argument('--sequence_model', type=str, default='lstm', choices=['lstm', 'gru', 'attention'])
parser.add_argument('--fuse_method', type=str, default='Attention', choices=['concatenate', 'Attention', 'wavenet_only', 'assignment_only'])
parser.add_argument('--fused_dim', type=int, default=64)
parser.add_argument('--a_embedding_size', type=int, default=32)
parser.add_argument('--a_hidden_size', type=int, default=64)
parser.add_argument('--q_rep_dim', type=int, default=32)
parser.add_argument('--mlp_hidden_dim', type=int, default=128)
parser.add_argument('--attention_num_heads', type=int, default=4)
parser.add_argument('--attention_ff_dim', type=int, default=128)

parser.add_argument('--target_dim', type=int, default=1, help='output target dimension')
# dla ADTTP q_in_dim musi być 1
parser.add_argument('--in_dim', type=int, default=1, help='for ADTTP must be 1')
parser.add_argument('--num_nodes', type=int, default=195, help='number of nodes')
parser.add_argument('--nhid', type=int, default=32, help='hidden channels')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./garage/metr', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--kernel_size', type=int, default=2, help='convolution kernel size')
parser.add_argument('--blocks', type=int, default=4, help='number of ST blocks')
parser.add_argument('--layers', type=int, default=2, help='number of layers in one spatial or temporal network')
parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')

parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--wandb_project", type=str, default="OptimalAssignment")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)

args = parser.parse_args()


def load_csv_adj(csv_path, num_nodes, device):
    df = pd.read_csv(csv_path, index_col=0)

    A = df.to_numpy(dtype=np.float32)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got {A.shape}")

    if A.shape[0] != num_nodes:
        raise ValueError(
            f"Adjacency size {A.shape[0]} does not match num_nodes={num_nodes}"
        )

    A = torch.tensor(A, dtype=torch.float32, device=device)
    AT = A.transpose(0, 1).contiguous()
    I = torch.eye(num_nodes, dtype=torch.float32, device=device)

    return [A, AT, I]


def split_file_names(q_dir, a_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    q_dir = Path(q_dir)
    a_dir = Path(a_dir)

    q_files = {p.name for p in q_dir.glob("*.npy")}
    a_files = {p.name for p in a_dir.glob("*.npy")}
    common_files = sorted(q_files & a_files)

    if not common_files:
        raise RuntimeError(
            f"Brak wspólnych plików .npy między {q_dir} i {a_dir}"
        )

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


def make_subset_loader(
    q_dir,
    a_dir,
    selected_files,
    batch_size,
    shuffle,
    num_workers,
    seq_length_q,
    seq_length_a,
    seq_length_y,
    target_nodes,
):
    dataset = SumoFolderDataset(
        flow_dir=q_dir,
        assign_dir=a_dir,
        seq_length_q=seq_length_q,
        seq_length_a=seq_length_a,
        seq_length_y=seq_length_y,
        target_nodes=target_nodes,
    )

    selected_files = set(selected_files)
    indices = [i for i, (f_name, _) in enumerate(dataset.samples) if f_name in selected_files]

    if not indices:
        raise RuntimeError("Subset datasetu jest pusty.")

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


def infer_target_dim_from_batch(batch):
    y = batch["y"]
    if y.dim() == 2:
        # (B, N) -> one-step prediction
        return 1
    if y.dim() == 3:
        # zakładamy (B, H, N)
        return y.shape[1]
    raise ValueError(f"Unsupported y shape: {tuple(y.shape)}")


def maybe_inverse_transform(scaler, x):
    if scaler is None:
        return x
    return scaler.inverse_transform(x)

def adjusted_mape(pred, real, offset=1.0):
    return torch.mean(torch.abs(pred - real) / (torch.abs(real) + offset))


def flow_conservation(pred, real, offset=1.0):
    if pred.dim() == 2:
        # (B, N)
        pred_sum = pred.sum(dim=1)
        real_sum = real.sum(dim=1)
    elif pred.dim() == 3:
        # (B, H, N)
        pred_sum = pred.sum(dim=2)
        real_sum = real.sum(dim=2)
    else:
        raise ValueError(f"Unsupported prediction shape for flow_conservation: {tuple(pred.shape)}")

    return torch.mean(torch.abs(pred_sum - real_sum) / (torch.abs(real_sum) + offset))

def init_metric_acc():
    return {
        "loss": 0.0,
        "mae": 0.0,
        "mape": 0.0,
        "rmse": 0.0,
        "adj_mape": 0.0,
        "flow_cons": 0.0,
        "n": 0,
    }


def update_metric_acc(acc, metrics, batch):
    bs = batch["y"].size(0)

    acc["loss"] += metrics["loss"] * bs
    acc["mae"] += metrics["mae"] * bs
    acc["mape"] += metrics["mape"] * bs
    acc["rmse"] += metrics["rmse"] * bs
    acc["adj_mape"] += metrics["adj_mape"] * bs
    acc["flow_cons"] += metrics["flow_cons"] * bs
    acc["n"] += bs


def finalize_metric_acc(acc):
    n = max(acc["n"], 1)
    return {
        "loss": acc["loss"] / n,
        "mae": acc["mae"] / n,
        "mape": acc["mape"] / n,
        "rmse": acc["rmse"] / n,
        "adj_mape": acc["adj_mape"] / n,
        "flow_cons": acc["flow_cons"] / n,
    }

def evaluate_loader(engine, loader):
    acc = init_metric_acc()
    for batch in loader:
        metrics = engine.eval(batch)
        update_metric_acc(acc, metrics, batch)
    return finalize_metric_acc(acc)


def main():
    device = torch.device(args.device)
    wandb.init(
        project="adttp",
        entity="lime-pss-uniwersytet-jagiello-ski-w-krakowie",
        config=vars(args) if "args" in globals() else None,
    )

    if args.adjdata is None:
        raise ValueError("Podaj --adjdata z plikiem hex_adjacency_matrix.csv")

    supports = load_csv_adj(args.adjdata, args.num_nodes, device)
    print(f"Loaded CSV adjacency from {args.adjdata}")

    print(args)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

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

    scaler = None

    first_batch = next(iter(train_loader))
    target_dim = infer_target_dim_from_batch(first_batch)

    engine = TrainerADTTP(
        scaler=scaler,
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
        loss_name=args.loss,
        alpha=args.alpha
    )

    print("start training...", flush=True)

    history = {
        'epoch': [],
        'loss_name': [],
        'train_loss': [], 'train_mae': [], 'train_mape': [], 'train_rmse': [], 'train_adj_mape': [],
        'train_flow_cons': [],
        'valid_loss': [], 'valid_mae': [], 'valid_mape': [], 'valid_rmse': [], 'valid_adj_mape': [],
        'valid_flow_cons': [],
        'train_time': [], 'val_time': []
    }

    monitor_history = []

    for i in range(1, args.epochs + 1):
        history['loss_name'].append(args.loss)

        train_acc = init_metric_acc()
        t1 = time.time()

        for iter_idx, batch in enumerate(train_loader):
            metrics = engine.train(batch)
            update_metric_acc(train_acc, metrics, batch)
            if args.wandb:
                global_step = (i - 1) * len(train_loader) + iter_idx

                wandb.log(
                    {
                        "iter/train_loss": metrics["loss"],
                        "iter/train_mae": metrics["mae"],
                        "iter/train_mape": metrics["mape"],
                        "iter/train_rmse": metrics["rmse"],
                        "iter/train_adj_mape": metrics["adj_mape"],
                        "iter/train_flow_cons": metrics["flow_cons"],
                        "epoch": i,
                    },
                    step=global_step,
                )
            if iter_idx % args.print_every == 0:
                log = (
                    'Iter: {:03d}, '
                    'Train LOSS[{}]: {:.4f}, Train MAE: {:.4f}, '
                    'Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
                    'Train ADJ_MAPE: {:.4f}, Train FLOW_CONS: {:.4f}'
                )
                print(
                    log.format(
                        iter_idx,
                        args.loss.upper(),
                        metrics["loss"],
                        metrics["mae"],
                        metrics["mape"],
                        metrics["rmse"],
                        metrics["adj_mape"],
                        metrics["flow_cons"],
                    ),
                    flush=True
                )

        t2 = time.time()

        s1 = time.time()
        valid_metrics = evaluate_loader(engine, val_loader)
        s2 = time.time()

        print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(i, (s2 - s1)))


        train_metrics = finalize_metric_acc(train_acc)

        mtrain_loss = train_metrics["loss"]
        mtrain_mae = train_metrics["mae"]
        mtrain_mape = train_metrics["mape"]
        mtrain_rmse = train_metrics["rmse"]

        mvalid_loss = valid_metrics["loss"]
        mvalid_mae = valid_metrics["mae"]
        mvalid_mape = valid_metrics["mape"]
        mvalid_rmse = valid_metrics["rmse"]

        mtrain_adj_mape = train_metrics["adj_mape"]
        mtrain_flow_cons = train_metrics["flow_cons"]

        mvalid_adj_mape = valid_metrics["adj_mape"]
        mvalid_flow_cons = valid_metrics["flow_cons"]

        if args.monitor == 'loss':
            monitor_value = mvalid_loss
        elif args.monitor == 'mae':
            monitor_value = mvalid_mae
        elif args.monitor == 'mape':
            monitor_value = mvalid_mape
        elif args.monitor == 'rmse':
            monitor_value = mvalid_rmse
        elif args.monitor == 'adj_mape':
            monitor_value = mvalid_adj_mape
        elif args.monitor == 'flow_cons':
            monitor_value = mvalid_flow_cons
        else:
            raise ValueError(f"Unsupported monitor: {args.monitor}")

        monitor_history.append(monitor_value)
        if args.wandb:
            wandb.log(
                {
                    "epoch/train_loss": mtrain_loss,
                    "epoch/train_mae": mtrain_mae,
                    "epoch/train_mape": mtrain_mape,
                    "epoch/train_rmse": mtrain_rmse,
                    "epoch/train_adj_mape": mtrain_adj_mape,
                    "epoch/train_flow_cons": mtrain_flow_cons,

                    "epoch/valid_loss": mvalid_loss,
                    "epoch/valid_mae": mvalid_mae,
                    "epoch/valid_mape": mvalid_mape,
                    "epoch/valid_rmse": mvalid_rmse,
                    "epoch/valid_adj_mape": mvalid_adj_mape,
                    "epoch/valid_flow_cons": mvalid_flow_cons,

                    f"monitor/{args.monitor}": monitor_value,
                    "time/train_epoch": t2 - t1,
                    "time/val_epoch": s2 - s1,
                    "epoch": i,
                },
                step=i * len(train_loader),
            )
        history['epoch'].append(i)
        history['train_loss'].append(mtrain_loss)
        history['train_mae'].append(mtrain_mae)
        history['train_mape'].append(mtrain_mape)
        history['train_rmse'].append(mtrain_rmse)
        history['valid_loss'].append(mvalid_loss)
        history['valid_mae'].append(mvalid_mae)
        history['valid_mape'].append(mvalid_mape)
        history['valid_rmse'].append(mvalid_rmse)
        history['train_time'].append(t2 - t1)
        history['val_time'].append(s2 - s1)
        history['train_adj_mape'].append(mtrain_adj_mape)
        history['train_flow_cons'].append(mtrain_flow_cons)
        history['valid_adj_mape'].append(mvalid_adj_mape)
        history['valid_flow_cons'].append(mvalid_flow_cons)

        log = (
            'Epoch: {:03d}, '
            'Train LOSS[{}]: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train ADJ_MAPE: {:.4f}, Train FLOW_CONS: {:.4f}, '
            'Valid LOSS[{}]: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid ADJ_MAPE: {:.4f}, Valid FLOW_CONS: {:.4f}, '
            'Monitor[{}]: {:.4f}, Training Time: {:.4f}/epoch'
        )
        print(
            log.format(
                i,
                args.loss.upper(), mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_adj_mape, mtrain_flow_cons,
                args.loss.upper(), mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_adj_mape, mvalid_flow_cons,
                args.monitor.upper(), monitor_value,
                (t2 - t1)
            ),
            flush=True
        )

        torch.save(
            engine.model.state_dict(),
            args.save + "_epoch_" + str(i) + "_" + str(round(monitor_value, 4)) + ".pth"
        )

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(history['train_time'])))
    print("Average Inference Time: {:.4f} secs".format(np.mean(history['val_time'])))

    save_dir = os.path.dirname(os.path.abspath(args.save))
    data_out_dir = os.path.join(save_dir, "data")
    os.makedirs(data_out_dir, exist_ok=True)

    df_metrics = pd.DataFrame(history)
    csv_path = os.path.join(data_out_dir, "training_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Statistics saved to: {csv_path}")

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    epochs_range = history['epoch']

    axes[0].plot(epochs_range, history['train_loss'], label=f'Train LOSS[{args.loss.upper()}]')
    axes[0].plot(epochs_range, history['valid_loss'], label=f'Valid LOSS[{args.loss.upper()}]')
    axes[0].set_title(f'Optimization Loss ({args.loss.upper()})')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(epochs_range, history['train_mae'], label='Train MAE')
    axes[1].plot(epochs_range, history['valid_mae'], label='Valid MAE')
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[2].plot(epochs_range, history['train_mape'], label='Train MAPE')
    axes[2].plot(epochs_range, history['valid_mape'], label='Valid MAPE')
    axes[2].set_title('MAPE')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    axes[3].plot(epochs_range, history['train_rmse'], label='Train RMSE')
    axes[3].plot(epochs_range, history['valid_rmse'], label='Valid RMSE')
    axes[3].set_title('RMSE')
    axes[3].set_xlabel('Epoch')
    axes[3].legend()
    axes[3].grid(True, linestyle='--', alpha=0.7)

    axes[4].plot(epochs_range, history['train_adj_mape'], label='Train ADJ_MAPE')
    axes[4].plot(epochs_range, history['valid_adj_mape'], label='Valid ADJ_MAPE')
    axes[4].set_title('ADJ_MAPE')
    axes[4].set_xlabel('Epoch')
    axes[4].legend()
    axes[4].grid(True, linestyle='--', alpha=0.7)

    axes[5].plot(epochs_range, history['train_flow_cons'], label='Train FLOW_CONS')
    axes[5].plot(epochs_range, history['valid_flow_cons'], label='Valid FLOW_CONS')
    axes[5].set_title('FLOW_CONS')
    axes[5].set_xlabel('Epoch')
    axes[5].legend()
    axes[5].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(data_out_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"saved learning curves to: {plot_path}")

    if args.wandb:
        wandb.log({
            "artifacts/learning_curves": wandb.Image(plot_path),
            "artifacts/training_metrics": wandb.Table(dataframe=df_metrics),
        })

    bestid = np.argmin(monitor_history)
    best_path = args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(monitor_history[bestid], 4)) + ".pth"
    engine.model.load_state_dict(torch.load(best_path, map_location=device))

    print("Training finished")
    print("Best checkpoint selected by VALID {}: {:.4f}".format(args.monitor.upper(), monitor_history[bestid]))
    print("Optimization loss used during training: {}".format(args.loss.upper()))

    test_metrics = evaluate_loader(engine, test_loader)
    if args.wandb:
        wandb.log({
            "test/mae": test_metrics["mae"],
            "test/mape": test_metrics["mape"],
            "test/rmse": test_metrics["rmse"],
            "test/adj_mape": test_metrics["adj_mape"],
            "test/flow_cons": test_metrics["flow_cons"],
            "best/epoch": bestid + 1,
            f"best/valid_{args.monitor}": monitor_history[bestid],
        })
    print(
        'Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test ADJ_MAPE: {:.4f}, Test FLOW_CONS: {:.4f}'.format(
            test_metrics["mae"],
            test_metrics["mape"],
            test_metrics["rmse"],
            test_metrics["adj_mape"],
            test_metrics["flow_cons"],
        )
    )

    torch.save(
        engine.model.state_dict(),
        args.save + "_exp" + str(args.expid) + "_best_" + str(round(monitor_history[bestid], 4)) + ".pth"
    )
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
