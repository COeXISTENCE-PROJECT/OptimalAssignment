import subprocess
import sys
import os
from datetime import datetime
import pandas as pd
import argparse
import os
import sys


def run_wavenet_training():
    """
    Inicjuje proces treningowy Graph WaveNet z obsługą ścieżek bezwzględnych
    i zmiennych środowiskowych SLURM.
    """

    base_dir = os.path.abspath(os.path.dirname(__file__))
    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    data_dir = os.path.join(base_dir, "data", "WAVENET_READY")
    adj_path = os.path.join(base_dir, "data", "adjacency_matrix.csv")
    save_dir = os.path.join(base_dir, "garage", f"experiment_{job_id}")
    os.makedirs(save_dir, exist_ok=True)

    model_save_prefix = os.path.join(save_dir, "model")

    # ==========================================
    # Parametry tensora wejściowego i topologii
    # ==========================================
    num_nodes = "770"
    in_dim = "1"
    seq_length = "12"
    epochs = "2"
    batch_size = "64"
    learning_rate = "0.001"

    # Definicja argumentów jako lista stringów
    command = [
        sys.executable,
        os.path.join(base_dir, "train.py"),
        "--device",
        "cuda:0",  # Zmieniono na cuda:0 zakladajac dostep do GPU na slurm
        "--data",
        data_dir,
        "--adjdata",
        adj_path,
        "--adjtype",
        "doubletransition",
        "--num_nodes",
        num_nodes,
        "--in_dim",
        in_dim,
        "--seq_length",
        seq_length,
        "--addaptadj",
        "--epochs",
        epochs,
        "--print_every",
        "10",
        "--batch_size",
        batch_size,
        "--learning_rate",
        learning_rate,
        "--save",
        model_save_prefix,
    ]

    print(f"=== Rozpoczynam eksperyment (ID: {job_id}) ===")
    print(f"Katalog zapisu: {save_dir}")

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode == 0:
        print("=== Trening zakończony sukcesem ===")
    else:
        print(f"=== Błąd! Proces zakończył się kodem {process.returncode} ===")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza statystyk treningowych Graph WaveNet"
    )
    parser.add_argument(
        "csv_path", type=str, help="Ścieżka do pliku training_metrics.csv"
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Błąd: Plik {args.csv_path} nie istnieje.")
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku CSV: {e}")
        sys.exit(1)

    # Weryfikacja wymaganych kolumn
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
        print("Błąd: Plik CSV nie zawiera wszystkich wymaganych kolumn.")
        sys.exit(1)

    # 1. Analiza ekstremów (najlepsze epoki)
    best_loss_idx = df["valid_loss"].idxmin()
    best_rmse_idx = df["valid_rmse"].idxmin()
    best_mape_idx = df["valid_mape"].idxmin()

    # 2. Obliczenie Generalization Gap dla Loss (MAE) w najlepszej epoce
    best_val_loss = df.loc[best_loss_idx, "valid_loss"]
    corresponding_train_loss = df.loc[best_loss_idx, "train_loss"]
    gen_gap = best_val_loss - corresponding_train_loss

    # 3. Analiza czasowa
    total_train_time = df["train_time"].sum()
    total_val_time = df["val_time"].sum()
    mean_epoch_time = df["train_time"].mean()

    print("=" * 60)
    print(f" RAPORT Z TRENINGU: {os.path.basename(os.path.dirname(args.csv_path))}")
    print("=" * 60)

    print(f"\n[1] PODSUMOWANIE CZASOWE")
    print(f"  • Zarejestrowane epoki:     {len(df)}")
    print(f"  • Całkowity czas treningu:  {total_train_time / 60:.2f} min")
    print(f"  • Całkowity czas walidacji: {total_val_time / 60:.2f} min")
    print(f"  • Średni czas na epokę:     {mean_epoch_time:.2f} s")

    print(f"\n[2] OPTYMALNE PUNKTY ZATRZYMANIA (Względem zbioru walidacyjnego)")
    print(
        f"  • Minimum Loss (MAE): {best_val_loss:.4f} (Osiągnięte w epoce: {df.loc[best_loss_idx, 'epoch']})"
    )
    print(
        f"  • Minimum RMSE:       {df.loc[best_rmse_idx, 'valid_rmse']:.4f} (Osiągnięte w epoce: {df.loc[best_rmse_idx, 'epoch']})"
    )
    print(
        f"  • Minimum MAPE:       {df.loc[best_mape_idx, 'valid_mape']:.4f} (Osiągnięte w epoce: {df.loc[best_mape_idx, 'epoch']})"
    )

    print(
        f"\n[3] ANALIZA PRZEUCZENIA (Dla optymalnej epoki {df.loc[best_loss_idx, 'epoch']})"
    )
    print(f"  • Train Loss (MAE):   {corresponding_train_loss:.4f}")
    print(f"  • Valid Loss (MAE):   {best_val_loss:.4f}")
    print(f"  • Generalization Gap: {gen_gap:.4f}")
    if gen_gap > (0.2 * corresponding_train_loss):
        print(
            "  ! UWAGA: Znaczna różnica między błędem treningowym a walidacyjnym (powyżej 20%)."
        )
        print("    Może to sugerować wczesne stadium przeuczenia modelu (overfitting).")

    print(f"\n[4] TREND Z OSTATNICH 5 EPOK")
    recent_df = df.tail(5)[["epoch", "train_loss", "valid_loss", "valid_rmse"]].copy()
    print(recent_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    run_wavenet_training()

if __name__ == "__main__":
    main()
