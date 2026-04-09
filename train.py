# import torch
# import numpy as np
# import argparse
# import time
# import util
# import matplotlib.pyplot as plt
# from engine import TrainerADTTP
# import pandas as pd
# import os
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:3', help='')
# parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
# parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
# parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
# parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
# parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
# parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
# parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
# parser.add_argument('--seq_length', type=int, default=12, help='')
# parser.add_argument('--nhid', type=int, default=32, help='')
# parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
# parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
# parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
# parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
# parser.add_argument('--epochs', type=int, default=100, help='')
# parser.add_argument('--print_every', type=int, default=50, help='')
# # parser.add_argument('--seed',type=int,default=99,help='random seed')
# parser.add_argument('--save', type=str, default='./garage/metr', help='save path')
# parser.add_argument('--expid', type=int, default=1, help='experiment id')
# parser.add_argument('--kernel_size', type=int, default=2, help='convolution kernel size')
# parser.add_argument('--blocks', type=int, default=4, help='number of ST blocks')
# parser.add_argument('--layers', type=int, default=2, help='number of layers in one spatial or tempolar network')
#
#
#
# args = parser.parse_args()
#
#
# def main():
#     # set seed
#     # torch.manual_seed(args.seed)
#     # np.random.seed(args.seed)
#     # load data
#     device = torch.device(args.device)
#     sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
#     dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
#     scaler = dataloader['scaler']
#     supports = [torch.tensor(i).to(device) for i in adj_mx]
#
#
#     print(args)
#
#     if args.randomadj:
#         adjinit = None
#     else:
#         adjinit = supports[0]
#
#     if args.aptonly:
#         supports = None
#
#     engine = TrainerADTTP(
#         scaler=scaler,
#         in_dim=args.in_dim,
#         num_nodes=args.num_nodes,
#         nhid=args.nhid,
#         dropout=args.dropout,
#         lrate=args.learning_rate,
#         wdecay=args.weight_decay,
#         device=device,
#         supports=supports,
#         gcn_bool=args.gcn_bool,
#         addaptadj=args.addaptadj,
#         aptinit=adjinit,
#         kernel_size=args.kernel_size,
#         blocks=args.blocks,
#         layers=args.layers,
#         target_dim=args.num_nodes, #przewidywanie wektora nodów
#     )
#
#
#     print("start training...", flush=True)
#
#     # initialization of dict for statistical analysis
#     history = {
#         'epoch': [],
#         'train_loss': [], 'train_mape': [], 'train_rmse': [],
#         'valid_loss': [], 'valid_mape': [], 'valid_rmse': [],
#         'train_time': [], 'val_time': []
#     }
#     his_loss = []
#
#     for i in range(1, args.epochs + 1):
#         # if i % 10 == 0:
#         # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
#         # for g in engine.optimizer.param_groups:
#         # g['lr'] = lr
#         train_loss = []
#         train_mape = []
#         train_rmse = []
#         t1 = time.time()
#         dataloader['train_loader'].shuffle()
#         for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
#             trainx = torch.Tensor(x).to(device)
#             trainx = trainx.transpose(1, 3)
#             trainy = torch.Tensor(y).to(device)
#             trainy = trainy.transpose(1, 3)
#             metrics = engine.train(trainx, trainy[:, 0, :, :])
#             train_loss.append(metrics[0])
#             train_mape.append(metrics[1])
#             train_rmse.append(metrics[2])
#             if iter % args.print_every == 0:
#                 log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
#                 print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
#         t2 = time.time()
#
#         # validation
#         valid_loss = []
#         valid_mape = []
#         valid_rmse = []
#
#         s1 = time.time()
#         for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
#             testx = torch.Tensor(x).to(device)
#             testx = testx.transpose(1, 3)
#             testy = torch.Tensor(y).to(device)
#             testy = testy.transpose(1, 3)
#             metrics = engine.eval(testx, testy[:, 0, :, :])
#             valid_loss.append(metrics[0])
#             valid_mape.append(metrics[1])
#             valid_rmse.append(metrics[2])
#         s2 = time.time()
#
#         log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
#         print(log.format(i, (s2 - s1)))
#
#         mtrain_loss = np.mean(train_loss)
#         mtrain_mape = np.mean(train_mape)
#         mtrain_rmse = np.mean(train_rmse)
#
#         mvalid_loss = np.mean(valid_loss)
#         mvalid_mape = np.mean(valid_mape)
#         mvalid_rmse = np.mean(valid_rmse)
#         his_loss.append(mvalid_loss)
#
#         # AKTUALIZACJA SŁOWNIKA W KAŻDEJ EPOCE
#         history['epoch'].append(i)
#         history['train_loss'].append(mtrain_loss)
#         history['train_mape'].append(mtrain_mape)
#         history['train_rmse'].append(mtrain_rmse)
#         history['valid_loss'].append(mvalid_loss)
#         history['valid_mape'].append(mvalid_mape)
#         history['valid_rmse'].append(mvalid_rmse)
#         history['train_time'].append(t2 - t1)
#         history['val_time'].append(s2 - s1)
#
#         log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
#         print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
#               flush=True)
#         torch.save(engine.model.state_dict(),
#                    args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
#
#     print("Average Training Time: {:.4f} secs/epoch".format(np.mean(history['train_time'])))
#     print("Average Inference Time: {:.4f} secs".format(np.mean(history['val_time'])))
#
#     #collecting and saving training data
#
#     save_dir = os.path.dirname(os.path.abspath(args.save))
#     data_out_dir = os.path.join(save_dir, "data")
#     os.makedirs(data_out_dir, exist_ok=True)
#
#     # save to csv
#     df_metrics = pd.DataFrame(history)
#     csv_path = os.path.join(data_out_dir, "training_metrics.csv")
#     df_metrics.to_csv(csv_path, index=False)
#     print(f"Statistics saved to: {csv_path}")
#
#     # learning curves
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     epochs_range = history['epoch']
#
#     # plot (MAE)
#     axes[0].plot(epochs_range, history['train_loss'], label='Train Loss (MAE)', color='blue')
#     axes[0].plot(epochs_range, history['valid_loss'], label='Valid Loss (MAE)', color='orange')
#     axes[0].set_title('Loss (MAE)')
#     axes[0].set_xlabel('Epoch')
#     axes[0].legend()
#     axes[0].grid(True, linestyle='--', alpha=0.7)
#
#     # plot MAPE
#     axes[1].plot(epochs_range, history['train_mape'], label='Train MAPE', color='blue')
#     axes[1].plot(epochs_range, history['valid_mape'], label='Valid MAPE', color='orange')
#     axes[1].set_title('MAPE')
#     axes[1].set_xlabel('Epoch')
#     axes[1].legend()
#     axes[1].grid(True, linestyle='--', alpha=0.7)
#
#     # plot RMSE
#     axes[2].plot(epochs_range, history['train_rmse'], label='Train RMSE', color='blue')
#     axes[2].plot(epochs_range, history['valid_rmse'], label='Valid RMSE', color='orange')
#     axes[2].set_title('RMSE')
#     axes[2].set_xlabel('Epoch')
#     axes[2].legend()
#     axes[2].grid(True, linestyle='--', alpha=0.7)
#
#     plt.tight_layout()
#     plot_path = os.path.join(data_out_dir, "learning_curves.png")
#     plt.savefig(plot_path, dpi=300)
#     plt.close()
#     print(f"saved learning curves to: {plot_path}")
#
#     # testing
#     bestid = np.argmin(his_loss)
#     engine.model.load_state_dict(
#         torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
#
#     outputs = []
#     realy = torch.Tensor(dataloader['y_test']).to(device)
#     realy = realy.transpose(1, 3)[:, 0, :, :]
#
#     for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
#         testx = torch.Tensor(x).to(device)
#         testx = testx.transpose(1, 3)
#         with torch.no_grad():
#             preds = engine.model(testx).transpose(1, 3)
#         outputs.append(preds.squeeze())
#
#     yhat = torch.cat(outputs, dim=0)
#     yhat = yhat[:realy.size(0), ...]
#
#     print("Training finished")
#     print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
#
#     amae = []
#     amape = []
#     armse = []
#     for i in range(12):
#         pred = scaler.inverse_transform(yhat[:, :, i])
#         real = realy[:, :, i]
#         metrics = util.metric(pred, real)
#         log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#         print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
#         amae.append(metrics[0])
#         amape.append(metrics[1])
#         armse.append(metrics[2])
#
#     log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
#     print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
#     torch.save(engine.model.state_dict(),
#                args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")
#
#
# if __name__ == "__main__":
#     main()


#new (better) train function

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

# przykładowo:
# from your_dataset_module import make_qA_loader

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data root path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')

# dla ADTTP q_in_dim musi być 1
parser.add_argument('--in_dim', type=int, default=1, help='for ADTTP must be 1')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
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

args = parser.parse_args()


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


def main():
    device = torch.device(args.device)

    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # ----------------------------------------------------------
    # NOWE LOADERY POD ADTTP
    # Zakładam strukturę:
    #   args.data/train
    #   args.data/val
    #   args.data/test
    # oraz batch:
    #   {"x": {"q": ..., "a": ...}, "y": ...}
    # ----------------------------------------------------------
    train_loader = make_qA_loader(
        root_dir=os.path.join(args.data, "train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_qA_loader(
        root_dir=os.path.join(args.data, "val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_qA_loader(
        root_dir=os.path.join(args.data, "test"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    scaler = None  # ustaw swój scaler tutaj tylko jeśli rzeczywiście go masz

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
    )

    print("start training...", flush=True)

    history = {
        'epoch': [],
        'train_loss': [], 'train_mape': [], 'train_rmse': [],
        'valid_loss': [], 'valid_mape': [], 'valid_rmse': [],
        'train_time': [], 'val_time': []
    }
    his_loss = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []

        t1 = time.time()

        for iter_idx, batch in enumerate(train_loader):
            metrics = engine.train(batch)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter_idx % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter_idx, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for batch in val_loader:
            metrics = engine.eval(batch)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()

        print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(i, (s2 - s1)))

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        history['epoch'].append(i)
        history['train_loss'].append(mtrain_loss)
        history['train_mape'].append(mtrain_mape)
        history['train_rmse'].append(mtrain_rmse)
        history['valid_loss'].append(mvalid_loss)
        history['valid_mape'].append(mvalid_mape)
        history['valid_rmse'].append(mvalid_rmse)
        history['train_time'].append(t2 - t1)
        history['val_time'].append(s2 - s1)

        log = (
            'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
            'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        )
        print(
            log.format(
                i, mtrain_loss, mtrain_mape, mtrain_rmse,
                mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)
            ),
            flush=True
        )

        torch.save(
            engine.model.state_dict(),
            args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth"
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = history['epoch']

    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss (MAE)', color='blue')
    axes[0].plot(epochs_range, history['valid_loss'], label='Valid Loss (MAE)', color='orange')
    axes[0].set_title('Loss (MAE)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(epochs_range, history['train_mape'], label='Train MAPE', color='blue')
    axes[1].plot(epochs_range, history['valid_mape'], label='Valid MAPE', color='orange')
    axes[1].set_title('MAPE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[2].plot(epochs_range, history['train_rmse'], label='Train RMSE', color='blue')
    axes[2].plot(epochs_range, history['valid_rmse'], label='Valid RMSE', color='orange')
    axes[2].set_title('RMSE')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(data_out_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"saved learning curves to: {plot_path}")

    # -------------------------
    # TEST
    # -------------------------
    bestid = np.argmin(his_loss)
    best_path = args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"
    engine.model.load_state_dict(torch.load(best_path, map_location=device))

    outputs = []
    targets = []

    engine.model.eval()
    for batch in test_loader:
        x = batch["x"]
        y = batch["y"].to(device)

        x = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in x.items()
        }

        with torch.no_grad():
            preds = engine.model(x)

        preds = maybe_inverse_transform(scaler, preds)

        outputs.append(preds.detach().cpu())
        targets.append(y.detach().cpu())

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(targets, dim=0)

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # one-step: (B, N)
    if yhat.dim() == 2:
        pred = yhat
        real = realy
        metrics = util.metric(pred, real)
        print(
            'Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                metrics[0], metrics[1], metrics[2]
            )
        )

    # multi-step: (B, H, N)
    elif yhat.dim() == 3:
        amae = []
        amape = []
        armse = []

        for h in range(yhat.size(1)):
            pred = yhat[:, h, :]
            real = realy[:, h, :]
            metrics = util.metric(pred, real)

            print(
                'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                    h + 1, metrics[0], metrics[1], metrics[2]
                )
            )

            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        print(
            'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(
                yhat.size(1), np.mean(amae), np.mean(amape), np.mean(armse)
            )
        )

    else:
        raise ValueError(f"Unsupported prediction shape: {tuple(yhat.shape)}")

    torch.save(
        engine.model.state_dict(),
        args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth"
    )


if __name__ == "__main__":
    main()