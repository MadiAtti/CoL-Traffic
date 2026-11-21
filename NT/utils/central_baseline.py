"""Measure centralized classifier accuracy at 5% dataset-size increments.

For each seed, the dataset is shuffled and evaluated in cumulative 5% chunks.
At each increment a fresh model is trained and evaluated on a stratified
holdout set. Per-seed curves and an averaged appendix figure are saved under
``results/baseline/``.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from utils.seed import set_seed
from data.custom_dataset import CustomDataset
from models.neural_network import TrafficLightningModule

FEATURE_COLUMNS = [
    "protocol",
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "src2dst_stddev_ps",
    "src2dst_max_ps",
    "dst2src_min_ps",
    "dst2src_mean_ps",
    "dst2src_stddev_ps",
    "dst2src_max_ps",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",
    "bidirectional_rst_packets",
]
TARGET_COLUMN = "application_name"
INPUT_DIM     = len(FEATURE_COLUMNS)
BATCH_SIZE    = 256
LR            = 1e-3
NUM_EPOCHS    = 40
TEST_SPLIT    = 0.2
NUM_CHUNKS    = 20
NUM_WORKERS   = 2


def load_dataset(seed: int) -> tuple:
    """Shuffle and split the source data into twenty equal-sized chunks.

    Args:
        seed: Random seed controlling the chunk order.

    Returns:
        A mapping of chunk names to data frames and the number of classes.
    """
    df = pd.read_parquet("dataset/dataset.parquet")

    le = LabelEncoder()
    df[TARGET_COLUMN] = le.fit_transform(df[TARGET_COLUMN])
    num_classes = len(le.classes_)
    print(f"  Number of classes: {num_classes} | Examples: {list(le.classes_[:5])}")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunks = np.array_split(df, NUM_CHUNKS)
    return {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}, num_classes

def load_seed_curves(seeds=None, results_dir="results/baseline"):
    """Load per-seed learning curves from their CSV files.

    Args:
        seeds: Seed identifiers to load; defaults to seeds 0 through 10.
        results_dir: Directory containing the per-seed curve CSV files.

    Returns:
        A mapping from seed to ``(percentages, accuracies)``.
    """
    if seeds is None:
        seeds = range(11)

    curves = {}
    for seed in seeds:
        file_path = os.path.join(results_dir, f"seed_{seed}_learning_curve.csv")
        if os.path.exists(file_path):
            d = pd.read_csv(file_path).sort_values("percentage").set_index("percentage")["accuracy"]
            curves[seed] = (d.index.tolist(), d.values.tolist())
        else:
            print(f"Warning: File not found for seed {seed}: {file_path}")
    return curves


def train_step(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    num_classes: int,
    seed: int,
) -> float:
    """Train one centralized classifier and return holdout accuracy.

    Args:
        X_train: Training feature matrix.
        y_train: Integer-encoded training labels.
        X_test: Holdout feature matrix.
        y_test: Integer-encoded holdout labels.
        num_classes: Number of target classes.
        seed: Random seed for model initialization and batching.

    Returns:
        Accuracy on the holdout set.
    """
    set_seed(seed)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    train_loader = DataLoader(
        CustomDataset(X_train.astype(np.float32), y_train.astype(np.int32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        CustomDataset(X_test.astype(np.float32), y_test.astype(np.int32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TrafficLightningModule(
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        lr=LR,
    )

    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, train_loader)

    metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)[0]
    return float(metrics["val_acc"])


def train_and_evaluate(dataset_chunks: dict, num_classes: int, seed: int):
    """Retrain a model after each additional 5% of shuffled data.

    Args:
        dataset_chunks: Ordered 5% data chunks from :func:`load_dataset`.
        num_classes: Number of target classes.
        seed: Random seed used for the stratified train/holdout split.

    Returns:
        Lists of cumulative dataset percentages and corresponding accuracies.
    """
    percentages = []
    accuracies  = []
    accumulated = pd.DataFrame()

    for i in range(NUM_CHUNKS):
        accumulated = pd.concat(
            [accumulated, dataset_chunks[f"chunk_{i}"]], ignore_index=True
        )
        current_pct = (i + 1) * 5

        X_all = accumulated[FEATURE_COLUMNS].values.astype(np.float32)
        y_all = accumulated[TARGET_COLUMN].values.astype(np.int32)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=TEST_SPLIT,
            stratify=y_all,
            random_state=seed,
        )

        print(f"  [seed={seed}] {current_pct:3d}% ({len(X_train):,} train / "
              f"{len(X_test):,} test) - training...")

        acc = train_step(X_train, y_train, X_test, y_test, num_classes, seed)

        print(f"  [seed={seed}] {current_pct:3d}% -> accuracy = {acc:.4f}")
        percentages.append(current_pct)
        accuracies.append(acc)

    return percentages, accuracies


def run_seed(seed: int):
    """Run and save the complete 5%-increment curve for one seed."""
    print(f"\n{'#' * 70}\n  SEED: {seed}\n{'#' * 70}")
    set_seed(seed)
    chunks, num_classes = load_dataset(seed=seed)
    percentages, accuracies = train_and_evaluate(chunks, num_classes, seed=seed)

    output_dir = "results/baseline"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"seed_{seed}_learning_curve.csv")
    pd.DataFrame(
        {"percentage": percentages, "accuracy": accuracies}
    ).to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return seed, percentages, accuracies


def plot_results(avg_accuracies: dict):
    """Save the averaged centralized accuracy-versus-data-size figure."""
    output_dir = "results/baseline"
    os.makedirs(output_dir, exist_ok=True)

    percentages = list(avg_accuracies.keys())
    accuracies  = list(avg_accuracies.values())

    print(f"\nPercentages : {percentages}")
    print(f"Avg accuracy: {[round(a, 4) for a in accuracies]}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        percentages, accuracies,
        marker="o", color="blue", linewidth=2.5, label="Mean accuracy",
    )
    plt.xlabel("Dataset size (%)")
    plt.ylabel("Accuracy")
    plt.title("Centralized accuracy by dataset size")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, "averaged_learning_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")


def main(argv=None):
    """Run seeds or aggregate existing seed curves into the appendix figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "plot"), default="plot")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(11)),
        help="Seed IDs to run or load (default: 0 through 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Parallel seed workers in run mode (default: {NUM_WORKERS}).",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least one")

    if args.mode == "run":
        os.makedirs("results/baseline", exist_ok=True)
        if args.workers == 1:
            results = [run_seed(seed) for seed in args.seeds]
        else:
            with Pool(processes=args.workers) as pool:
                results = pool.map(run_seed, args.seeds)
        curves = {
            seed: (percentages, accuracies)
            for seed, percentages, accuracies in results
        }
    else:
        curves = load_seed_curves(seeds=args.seeds)
        if not curves:
            raise FileNotFoundError(
                "No per-seed learning curves were found; run with --mode run first."
            )

    sum_accuracy = {}
    counts = {}
    for percentages, accuracies in curves.values():
        if len(percentages) != len(accuracies):
            raise ValueError("A per-seed learning curve has mismatched columns.")
        for percentage, accuracy in zip(percentages, accuracies):
            sum_accuracy[percentage] = sum_accuracy.get(percentage, 0.0) + accuracy
            counts[percentage] = counts.get(percentage, 0) + 1

    mean_accuracy = {
        percentage: sum_accuracy[percentage] / counts[percentage]
        for percentage in sorted(sum_accuracy)
    }
    results_df = pd.DataFrame(
        list(mean_accuracy.items()),
        columns=["percentage", "mean_accuracy"],
    )
    os.makedirs("results/baseline", exist_ok=True)
    csv_path = "results/baseline/averaged_learning_curve.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Averaged curve saved: {csv_path}")
    plot_results(mean_accuracy)


if __name__ == "__main__":
    main()