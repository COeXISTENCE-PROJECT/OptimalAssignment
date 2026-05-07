#!/usr/bin/env python3
"""
save_inference_run_for_comparison.py

Skrypt do wielokrotnego odpalania pojedynczych inferencji do jednego porównania.

Każde uruchomienie tworzy/uzupełnia:

comparison_dir/
  runs/
    Model_Name/
      pred_q.npy
      real_q.npy
      assign.npy              opcjonalnie
      metrics.csv             opcjonalnie
      run_info.json
      command.sh
      stdout.log
      stderr.log

Możesz go odpalać wiele razy z różnymi modelami, parametrami i checkpointami.
Potem odpalasz aggregate_visualize_comparison.py na całym comparison_dir.

Przykład z komendą inline:

python save_inference_run_for_comparison.py \
  --comparison_dir /scratch/tmp/model_comparisons/exp_001 \
  --model_name Base_Att_LSTM \
  --cmd "/home/drozd/miniconda/envs/wavenet_env/bin/python inference.py --checkpoint /path/best.pth --output_dir {run_dir} ..." \
  --cwd /home/drozd/OptimalAssignment

Przykład z plikiem komendy:

python save_inference_run_for_comparison.py \
  --comparison_dir /scratch/tmp/model_comparisons/exp_001 \
  --model_name Base_Att_LSTM \
  --cmd_file cmds/base_att_lstm.cmd \
  --cwd /home/drozd/OptimalAssignment

Przykład tylko rejestracji istniejącego folderu inferencji:

python save_inference_run_for_comparison.py \
  --comparison_dir /scratch/tmp/model_comparisons/exp_001 \
  --model_name Already_Run_Model \
  --source_dir /scratch/tmp/some_existing_inference_dir
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


def slugify(text: str) -> str:
    out = []
    for ch in text.strip():
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "model"


def load_array_2d(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"{path} ma shape {arr.shape}; oczekuję (T, N) albo (T, N, 1).")


def find_single_any(root: Path, patterns: List[str], label: str) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(sorted(root.rglob(pattern)))

    # usuń duplikaty, zachowując kolejność
    unique = []
    seen = set()
    for p in matches:
        rp = str(p.resolve())
        if rp not in seen:
            unique.append(p)
            seen.add(rp)

    if not unique:
        raise FileNotFoundError(
            f"Nie znaleziono pliku {label} w {root}. Szukane wzorce: {patterns}"
        )

    if len(unique) > 1:
        print(f"[WARN] Znaleziono kilka plików {label}, biorę pierwszy: {unique[0]}")

    return unique[0]


def maybe_find_single_any(root: Path, patterns: List[str], label: str) -> Optional[Path]:
    try:
        return find_single_any(root, patterns, label)
    except FileNotFoundError:
        return None


def read_seed_steps(metrics_path: Optional[Path]) -> Optional[int]:
    if metrics_path is None:
        return None

    try:
        df = pd.read_csv(metrics_path)
    except Exception as exc:
        print(f"[WARN] Nie mogę odczytać metrics CSV {metrics_path}: {exc}")
        return None

    if df.empty or "seed_steps" not in df.columns:
        return None

    value = df.iloc[0]["seed_steps"]
    if pd.isna(value):
        return None

    return int(value)


def transfer_file(src: Path, dst: Path, mode: str, overwrite: bool) -> Path:
    src = src.resolve()
    dst = dst.resolve()

    if src == dst:
        return dst

    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            raise FileExistsError(f"Plik już istnieje: {dst}. Użyj --overwrite.")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Nieznany copy_mode: {mode}")

    return dst


def read_command(args: argparse.Namespace) -> Optional[str]:
    if args.cmd and args.cmd_file:
        raise ValueError("Podaj albo --cmd, albo --cmd_file, nie oba.")

    if args.cmd:
        return args.cmd.strip()

    if args.cmd_file:
        path = Path(args.cmd_file).expanduser().resolve()
        return path.read_text(encoding="utf-8").strip()

    return None


def render_command(command: str, *, run_dir: Path, model_name: str, comparison_dir: Path) -> str:
    return command.format(
        run_dir=str(run_dir),
        output_dir=str(run_dir),
        model_name=model_name,
        name=model_name,
        comparison_dir=str(comparison_dir),
        comparison_root=str(comparison_dir),
    )


def run_command(
    *,
    command: str,
    cwd: Path,
    run_dir: Path,
    model_name: str,
    comparison_dir: Path,
    dry_run: bool,
) -> None:
    print("\n[INFO] Komenda inferencji:")
    print(command)

    command_path = run_dir / "command.sh"
    command_path.write_text("#!/bin/bash\n" + command + "\n", encoding="utf-8")

    if dry_run:
        print("[DRY-RUN] Pomijam wykonanie komendy.")
        return

    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["MODEL_RUN_DIR"] = str(run_dir)
    env["COMPARISON_DIR"] = str(comparison_dir)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

    if completed.returncode != 0:
        stderr_tail = ""
        if stderr_path.exists():
            stderr_tail = "\n".join(
                stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            )

        raise RuntimeError(
            f"Inferencja modelu '{model_name}' zakończyła się kodem "
            f"{completed.returncode}.\n"
            f"stderr: {stderr_path}\n\n"
            f"Ostatnie linie stderr:\n{stderr_tail}"
        )


def standardize_outputs(
    *,
    search_dir: Path,
    target_dir: Path,
    pred_pattern: str,
    real_pattern: str,
    assign_pattern: str,
    metrics_pattern: str,
    copy_mode: str,
    overwrite: bool,
    forced_seed_steps: Optional[int],
) -> dict:
    pred_src = find_single_any(
        search_dir,
        [pred_pattern, "pred_q.npy", "pred.npy"],
        label="pred_q",
    )
    real_src = find_single_any(
        search_dir,
        [real_pattern, "real_q.npy", "real.npy", "ground_truth.npy"],
        label="real_q",
    )

    assign_src = maybe_find_single_any(
        search_dir,
        [assign_pattern, "assign.npy", "assignment.npy"],
        label="assign",
    )

    metrics_src = maybe_find_single_any(
        search_dir,
        [metrics_pattern, "metrics.csv"],
        label="metrics",
    )

    pred_dst = transfer_file(pred_src, target_dir / "pred_q.npy", copy_mode, overwrite)
    real_dst = transfer_file(real_src, target_dir / "real_q.npy", copy_mode, overwrite)

    assign_dst = None
    if assign_src is not None:
        assign_dst = transfer_file(assign_src, target_dir / "assign.npy", copy_mode, overwrite)

    metrics_dst = None
    if metrics_src is not None:
        metrics_dst = transfer_file(metrics_src, target_dir / "metrics.csv", copy_mode, overwrite)

    pred = load_array_2d(pred_dst)
    real = load_array_2d(real_dst)

    if pred.shape != real.shape:
        raise ValueError(f"pred i real mają różne shape: {pred.shape} vs {real.shape}")

    seed_steps = forced_seed_steps
    if seed_steps is None:
        seed_steps = read_seed_steps(metrics_dst)

    return {
        "pred_path": str(pred_dst),
        "real_path": str(real_dst),
        "assign_path": None if assign_dst is None else str(assign_dst),
        "metrics_path": None if metrics_dst is None else str(metrics_dst),
        "seed_steps": seed_steps,
        "pred_shape": list(pred.shape),
        "real_shape": list(real.shape),
        "source_pred_path": str(pred_src),
        "source_real_path": str(real_src),
        "source_assign_path": None if assign_src is None else str(assign_src),
        "source_metrics_path": None if metrics_src is None else str(metrics_src),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--comparison_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument(
        "--cmd",
        type=str,
        default=None,
        help="Komenda inferencji. Może używać placeholdera {run_dir}.",
    )

    parser.add_argument(
        "--cmd_file",
        type=str,
        default=None,
        help="Plik tekstowy z komendą inferencji. Może używać placeholdera {run_dir}.",
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help=(
            "Folder, w którym szukać wyników pred/real. "
            "Jeżeli brak, używa run_dir modelu. Przydatne do rejestracji istniejących inferencji."
        ),
    )

    parser.add_argument(
        "--cwd",
        type=str,
        default=".",
        help="Katalog roboczy do wykonania komendy inferencji.",
    )

    parser.add_argument("--pred_pattern", type=str, default="*_pred_q.npy")
    parser.add_argument("--real_pattern", type=str, default="*_real_q.npy")
    parser.add_argument("--assign_pattern", type=str, default="*_assign.npy")
    parser.add_argument("--metrics_pattern", type=str, default="*_metrics.csv")

    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="Jak przenieść wyniki do ustandaryzowanego folderu.",
    )

    parser.add_argument("--seed_steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comparison_dir = Path(args.comparison_dir).expanduser().resolve()
    model_name = args.model_name
    model_slug = slugify(model_name)

    runs_dir = comparison_dir / "runs"
    target_dir = runs_dir / model_slug

    comparison_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    command = read_command(args)

    if command is None and args.source_dir is None:
        raise ValueError("Podaj --cmd/--cmd_file albo --source_dir.")

    print(f"[INFO] comparison_dir: {comparison_dir}")
    print(f"[INFO] model_name:     {model_name}")
    print(f"[INFO] target_dir:     {target_dir}")

    existing_standard = (target_dir / "pred_q.npy").exists() and (target_dir / "real_q.npy").exists()

    if existing_standard and args.reuse_existing:
        print("[INFO] Wyniki już istnieją i podano --reuse_existing. Pomijam inferencję i standaryzację.")
    else:
        if command is not None:
            rendered = render_command(
                command,
                run_dir=target_dir,
                model_name=model_name,
                comparison_dir=comparison_dir,
            )
            run_command(
                command=rendered,
                cwd=Path(args.cwd).expanduser().resolve(),
                run_dir=target_dir,
                model_name=model_name,
                comparison_dir=comparison_dir,
                dry_run=args.dry_run,
            )
        else:
            rendered = None

        if args.dry_run:
            print("[DRY-RUN] Nie szukam wyników.")
            return

        search_dir = Path(args.source_dir).expanduser().resolve() if args.source_dir else target_dir

        outputs = standardize_outputs(
            search_dir=search_dir,
            target_dir=target_dir,
            pred_pattern=args.pred_pattern,
            real_pattern=args.real_pattern,
            assign_pattern=args.assign_pattern,
            metrics_pattern=args.metrics_pattern,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            forced_seed_steps=args.seed_steps,
        )

        run_info = {
            "model_name": model_name,
            "model_slug": model_slug,
            "comparison_dir": str(comparison_dir),
            "target_dir": str(target_dir),
            "source_dir": str(search_dir),
            "command": rendered,
            "copy_mode": args.copy_mode,
            **outputs,
        }

        with (target_dir / "run_info.json").open("w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)

        print("\n[OK] Zapisano ustandaryzowany run.")
        print(f"[OK] run_info: {target_dir / 'run_info.json'}")
        print(f"[OK] pred_q:   {target_dir / 'pred_q.npy'}")
        print(f"[OK] real_q:   {target_dir / 'real_q.npy'}")


if __name__ == "__main__":
    main()