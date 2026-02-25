import os
import shutil
from typing import Iterable


def clean_recorder(env) -> None:
    """
    Remove all files from the recorder folders before the next evaluation.

    Args:
        env: Environment containing a recorder with episode and detector folders.
    """
    folders: Iterable[str] = (
        env.recorder.episodes_folder,
        env.recorder.detector_folder,
    )

    for folder in folders:
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    print("[INFO] Recorder cleaned.")


def copy_best_individual(env, records_folder: str, generation: int) -> None:
    """
    Copy the best (elite) individual's episode and detector files
    into the best_individuals folder.

    The elite individual is assumed to be the lowest-numbered CSV file.

    Args:
        env: Environment containing a recorder.
        records_folder: Base folder where results are stored.
        generation: Current generation index (0-based).
    """
    best_folder = os.path.join(records_folder, "best_individuals")
    episodes_dst = os.path.join(best_folder, "episodes")
    detector_dst = os.path.join(best_folder, "detector")

    os.makedirs(episodes_dst, exist_ok=True)
    os.makedirs(detector_dst, exist_ok=True)

    def _next_index(dst_folder: str, prefix: str) -> int:
        """Return the next available index for files with a given prefix."""
        existing = [
            f for f in os.listdir(dst_folder)
            if f.startswith(prefix) and f.endswith(".csv")
        ]
        if not existing:
            return 1

        numbers = [
            int(f.replace(prefix, "").replace(".csv", ""))
            for f in existing
        ]
        return max(numbers) + 1

    def _extract_index(file_name: str, prefix: str) -> int:
        """Extract numeric index from a file name."""
        return int(file_name.replace(prefix, "").replace(".csv", ""))

    def _copy_first_file(src_folder: str, dst_folder: str, prefix: str) -> None:
        """Copy the lowest-numbered CSV file from src to dst."""
        csv_files = [
            f for f in os.listdir(src_folder) if f.endswith(".csv")
        ]
        if not csv_files:
            return

        csv_files.sort(key=lambda f: _extract_index(f, prefix))
        src_file = os.path.join(src_folder, csv_files[0])

        next_idx = _next_index(dst_folder, prefix)
        dst_file = os.path.join(dst_folder, f"{prefix}{next_idx}.csv")

        shutil.copy2(src_file, dst_file)

    _copy_first_file(
        env.recorder.episodes_folder,
        episodes_dst,
        prefix="ep",
    )
    _copy_first_file(
        env.recorder.detector_folder,
        detector_dst,
        prefix="detector_ep",
    )

    print(
        f"[INFO] Best individual (gen {generation + 1}) copied. "
        f"Total saved episodes: {len(os.listdir(episodes_dst))}"
    )


def copy_human_learning_episodes(env, records_folder: str) -> None:
    """
    Copy all human-learning episodes (before GA) into best_individuals.

    Args:
        env: Environment containing a recorder.
        records_folder: Base folder where results are stored.
    """
    best_folder = os.path.join(records_folder, "best_individuals")
    episodes_dst = os.path.join(best_folder, "episodes")
    detector_dst = os.path.join(best_folder, "detector")

    os.makedirs(episodes_dst, exist_ok=True)
    os.makedirs(detector_dst, exist_ok=True)

    def _copy_all_csv(src_folder: str, dst_folder: str) -> None:
        """Copy all CSV files from src folder to dst folder."""
        for file_name in os.listdir(src_folder):
            if file_name.endswith(".csv"):
                shutil.copy2(
                    os.path.join(src_folder, file_name),
                    os.path.join(dst_folder, file_name),
                )

    _copy_all_csv(env.recorder.episodes_folder, episodes_dst)
    _copy_all_csv(env.recorder.detector_folder, detector_dst)

    print(
        "[INFO] Human learning episodes copied. "
        f"Total saved episodes: {len(os.listdir(episodes_dst))}"
    )
