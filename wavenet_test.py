import subprocess
import sys
import os
from datetime import datetime
import pandas as pd
import argparse


def run_wavenet_training():
    """
    Initializes the Graph WaveNet training process with support for absolute paths
    """
    # Determine the base directory (where this script is located)
    base_dir = os.path.abspath(os.path.dirname(__file__))

    # Experiment identification (SLURM_JOB_ID if available, otherwise a timestamp)
    job_id = os.environ.get("slurm job id", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Define absolute paths
    data_dir = os.path.join(base_dir, "data", "WAVENET_READY")
    adj_path = os.path.join(base_dir, "data", "adjacency_matrix.csv")

    # Create a unique directory for this specific job
    save_dir = os.path.join(base_dir, "garage", f"experiment_{job_id}")
    os.makedirs(save_dir, exist_ok=True)

    # The --save parameter in train.py is used as a save prefix (e.g., save_dir/model_epoch_1.pth)
    model_save_prefix = os.path.join(save_dir, "model")

    # Input tensor and topology parameters
    num_nodes = "1430"
    in_dim = "1"
    seq_length = "12"
    epochs = "2"
    batch_size = "64"
    learning_rate = "0.001"
    nhid = "32"  #number of hidded dimentions

    # b, k, l have to satisfy the equation for time window size and receptive field
    # T =< R = 1 + b * (k-1)*(2^l -1)
    kernel = "2"
    blocks = "4"
    layers = "2"

    # Define arguments as a list of strings
    command = [
        sys.executable, os.path.join(base_dir, "train.py"),
        "--device", "cpu",
        "--data", data_dir,
        "--adjdata", adj_path,
        "--adjtype", "doubletransition",
        "--num_nodes", num_nodes,
        "--in_dim", in_dim,
        "--seq_length", seq_length,    #to check
        "--nhid", nhid,
        "--gcn_bool",
        #"--addaptadj",
        "--epochs", epochs,
        "--print_every", "10",
        "--batch_size", batch_size,
        "--learning_rate", learning_rate,
        "--save", model_save_prefix,
        "--kernel_size", kernel,
        "--blocks", blocks,
        "--layers", layers
    ]

    print(f"Starting experiment (ID: {job_id})")
    print(f"Save directory: {save_dir}")

    # Execute the process
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Read stdout line by line to display training progress in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode == 0:
        print("Training completed successfully")
    else:
        print(f"Error! Process exited with code {process.returncode}")


def main():
    parser = argparse.ArgumentParser(description="training statistics analysis")
    parser.add_argument('csv_path', type=str, help='Path to the training_metrics.csv file')
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: File {args.csv_path} does not exist.")
        sys.exit(1)

    # Load data using Pandas
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    # Verify required columns are present
    required_columns = ['epoch', 'train_loss', 'valid_loss', 'valid_rmse', 'valid_mape', 'train_time', 'val_time']
    if not all(col in df.columns for col in required_columns):
        print("Error: CSV file does not contain all required columns.")
        sys.exit(1)

    # Extremes analysis (best epochs)
    best_loss_idx = df['valid_loss'].idxmin()
    best_rmse_idx = df['valid_rmse'].idxmin()
    best_mape_idx = df['valid_mape'].idxmin()

    # Calculate Generalization Gap for Loss (MAE) at the best epoch
    # This helps assess how much the model is overfitting to the training set compared to the validation set
    best_val_loss = df.loc[best_loss_idx, 'valid_loss']
    corresponding_train_loss = df.loc[best_loss_idx, 'train_loss']
    gen_gap = best_val_loss - corresponding_train_loss

    # 3. Time analysis
    total_train_time = df['train_time'].sum()
    total_val_time = df['val_time'].sum()
    mean_epoch_time = df['train_time'].mean()

    # formaring and displaying
    print("=" * 60)
    print(f" training report: {os.path.basename(os.path.dirname(args.csv_path))}")
    print("=" * 60)

    print(f"\n[1] time summary")
    print(f"  • Recorded epochs:      {len(df)}")
    print(f"  • Total training time:  {total_train_time / 60:.2f} min")
    print(f"  • Total validation time: {total_val_time / 60:.2f} min")
    print(f"  • Average epoch time:   {mean_epoch_time:.2f} s")

    print(f"\n[2] optimal training stopping points (Relative to the validation set)")
    print(f"  • Minimum Loss (MAE): {best_val_loss:.4f} (Achieved at epoch: {df.loc[best_loss_idx, 'epoch']})")
    print(
        f"  • Minimum RMSE:       {df.loc[best_rmse_idx, 'valid_rmse']:.4f} (Achieved at epoch: {df.loc[best_rmse_idx, 'epoch']})")
    print(
        f"  • Minimum MAPE:       {df.loc[best_mape_idx, 'valid_mape']:.4f} (Achieved at epoch: {df.loc[best_mape_idx, 'epoch']})")

    print(f"\n[3] overfitting analysis (For optimal epoch {df.loc[best_loss_idx, 'epoch']})")
    print(f"  • Train Loss (MAE):   {corresponding_train_loss:.4f}")
    print(f"  • Valid Loss (MAE):   {best_val_loss:.4f}")
    print(f"  • Generalization Gap: {gen_gap:.4f}")

    if gen_gap > (0.2 * corresponding_train_loss):
        print("    Significant difference between training and validation error (over 20%).")
        print("    This may suggest an early stage of model overfitting.")

    # 4. Preview of the last 5 epochs (trend)
    print(f"\n[4] last 5 epochs")
    recent_df = df.tail(5)[['epoch', 'train_loss', 'valid_loss', 'valid_rmse']].copy()
    print(recent_df.to_string(index=False))
    print("=" * 60)


# training execution
if __name__ == "__main__":
    # job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))

    if len(sys.argv) > 1:
        # statistical analysis if csv path
        main()
    else:
        run_wavenet_training()