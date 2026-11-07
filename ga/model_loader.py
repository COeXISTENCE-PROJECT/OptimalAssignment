import torch
from engine import TrainerGenTTP
from utilities import load_csv_adj

RUN_DIR = "outputs/checkpoints/"
CHECKPOINT = f"{RUN_DIR}/checkpoint_best.pth"
ADJ_FILE = f"{RUN_DIR}/21k_hex_adjacency_matrix.csv"
NUM_NODES = 195


def load_model(checkpoint=CHECKPOINT, adj_file=ADJ_FILE, device="cpu"):
    device = torch.device(device)
    supports = load_csv_adj(adj_file, NUM_NODES, device)

    trainer = TrainerGenTTP(
        scaler=None, in_dim=1, num_nodes=NUM_NODES,
        nhid=128, dropout=0.1, lrate=7e-4, wdecay=1e-4,
        device=device, supports=supports,
        gcn_bool=True, addaptadj=False, aptinit=supports[0],
        kernel_size=2, blocks=5, layers=2, target_dim=1,
        sequence_model="lstm", fuse_method="attention",
        a_embedding_size=96, a_hidden_size=128,
        q_rep_dim=128, fused_dim=128, mlp_hidden_dim=256,
        attention_num_heads=2, attention_ff_dim=64,
        fuse_attention_num_heads=2, fuse_attention_ff_dim=128,
        fuse_gated_update=True,
    )
    state = torch.load(checkpoint, map_location=device)
    trainer.model.load_state_dict(state)
    trainer.model.eval()
    return trainer.model


if __name__ == "__main__":
    model = load_model()
    print("model loaded,", sum(p.numel() for p in model.parameters()), "parameters")