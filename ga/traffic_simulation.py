import numpy as np
import torch

WINDOW_LENGTH_Q = 15
WINDOW_LENGTH_A = 30

@torch.no_grad()
def simulate_traffic(model, A, start_q=None, device="cpu"):
    num_steps, num_cells = A.shape
    history_len = max(WINDOW_LENGTH_Q, WINDOW_LENGTH_A)

    q_hist = np.zeros((history_len + num_steps, num_cells), dtype=np.float32)
    a_hist = np.zeros((history_len + num_steps, num_cells), dtype=np.float32)
    a_hist[history_len:] = A

    first_step = 0
    if start_q is not None:
        first_step = len(start_q)
        q_hist[history_len:history_len + first_step] = start_q

    for t in range(first_step, num_steps):
        i = history_len + t
        q_window = torch.from_numpy(q_hist[i - WINDOW_LENGTH_Q:i]).unsqueeze(0).to(device)
        a_window = torch.from_numpy(a_hist[i - WINDOW_LENGTH_A:i]).unsqueeze(0).to(device)
        pred = model(q_window, a_window)
        q_hist[i] = pred.reshape(-1).cpu().numpy()

    Q_hat = q_hist[history_len:] 
    return Q_hat, float(Q_hat.sum())   

@torch.no_grad()
def simulate_traffic_batch(model, A_batch, device="cpu"):
    """A_batch: (B, T, 195), czyli przydziały B osobników naraz."""
    batch_size, num_steps, num_cells = A_batch.shape
    history_len = max(WINDOW_LENGTH_Q, WINDOW_LENGTH_A)

    q_hist = np.zeros((batch_size, history_len + num_steps, num_cells), dtype=np.float32)
    a_hist = np.zeros((batch_size, history_len + num_steps, num_cells), dtype=np.float32)
    a_hist[:, history_len:] = A_batch

    for t in range(num_steps):
        i = history_len + t
        q_window = torch.from_numpy(q_hist[:, i - WINDOW_LENGTH_Q:i]).to(device)
        a_window = torch.from_numpy(a_hist[:, i - WINDOW_LENGTH_A:i]).to(device)
        pred = model(q_window, a_window)
        q_hist[:, i] = pred.reshape(batch_size, num_cells).cpu().numpy()

    Q_hat = q_hist[:, history_len:]
    return Q_hat, Q_hat.sum(axis=(1, 2))