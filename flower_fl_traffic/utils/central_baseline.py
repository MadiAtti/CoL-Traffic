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
from multiprocessing import Pool, cpu_count
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from seed import set_seed
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
    df = pd.read_parquet("dataset/dataset.parquet")

    le = LabelEncoder()
    df[TARGET_COLUMN] = le.fit_transform(df[TARGET_COLUMN])
    num_classes = len(le.classes_)
    print(f"  Osztályok száma: {num_classes}  |  Példa: {list(le.classes_[:5])}")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunks = np.array_split(df, NUM_CHUNKS)
    return {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}, num_classes


def train_step(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    num_classes: int,
    seed: int,
) -> float:
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
              f"{len(X_test):,} test) – tanítás...")

        acc = train_step(X_train, y_train, X_test, y_test, num_classes, seed)

        print(f"  [seed={seed}] {current_pct:3d}% → val_acc = {acc:.4f}")
        percentages.append(current_pct)
        accuracies.append(acc)

    return percentages, accuracies


def run_seed(seed: int):
    try:
        print(f"\n{'#'*70}")
        print(f"  SEED: {seed} 🚀")
        print(f"{'#'*70}")

        set_seed(seed)
        chunks, num_classes = load_dataset(seed=seed)
        pcts, accs = train_and_evaluate(chunks, num_classes, seed=seed)

        save_path = f"results/baseline/seed_{seed}_learning_curve.csv"
        pd.DataFrame({"percentage": pcts, "accuracy": accs}).to_csv(save_path, index=False)
        print(f"💾 Eredmények elmentve: {save_path}")

        return seed, pcts, accs

    except Exception as e:
        import traceback
        print(f"❌ Hiba seed={seed} esetén: {e}")
        traceback.print_exc()
        return seed, None, None


def plot_results(avg_accuracies: dict):
    output_dir = "results/baseline"
    os.makedirs(output_dir, exist_ok=True)

    percentages = list(avg_accuracies.keys())
    accuracies  = list(avg_accuracies.values())

    print(f"\nPercentages : {percentages}")
    print(f"Avg accuracy: {[round(a, 4) for a in accuracies]}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        percentages, accuracies,
        marker="o", color="blue", linewidth=2.5, label="Átlagolt Accuracy",
    )
    plt.xlabel("Adat mennyisége (%)")
    plt.ylabel("Accuracy")
    plt.title("Modell teljesítmény az adatmennyiség függvényében")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_dir, "averaged_learning_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Diagram elmentve: {plot_path}")


if __name__ == "__main__":
    START_SEED = 0
    END_SEED   = 9

    os.makedirs("results/baseline", exist_ok=True)

    print(f"🔧 Párhuzamos workerek száma: {NUM_WORKERS}")

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(run_seed, range(START_SEED, END_SEED + 1))

    all_results = {
        seed: (pcts, accs)
        for seed, pcts, accs in results
        if pcts is not None
    }

    if not all_results:
        print("Nem sikerült egyetlen futást sem végrehajtani.")
    else:
        sum_acc = {}
        counts  = {}
        for seed, (pcts, accs) in all_results.items():
            for pct, acc in zip(pcts, accs):
                sum_acc[pct] = sum_acc.get(pct, 0.0) + acc
                counts[pct]  = counts.get(pct, 0) + 1

        avg_accuracies = {pct: sum_acc[pct] / counts[pct] for pct in sorted(sum_acc)}

        results_df = pd.DataFrame(
            list(avg_accuracies.items()),
            columns=["percentage", "mean_accuracy"],
        )
        csv_path = "results/baseline/averaged_learning_curve.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n💾 CSV elmentve: {csv_path}")

        plot_results(avg_accuracies)