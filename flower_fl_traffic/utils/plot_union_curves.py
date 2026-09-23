import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RATIO_ORDER = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RATIO_LABELS = ["1/8", "1/4", "1/2", "1/1", "2/1", "4/1", "8/1"]


def load_curve_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"dataset_key", "ratio", "local_p1_mean", "local_p2_mean", "fed_p1_mean", "fed_p2_mean"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_cols)}")

    return df


def plot_union(input_file: Path, output_file: Path, metric: str = "fed_p1_mean") -> None:
    df = load_curve_data(input_file)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 36,
            "xtick.labelsize": 32,
            "ytick.labelsize": 32,
            "legend.fontsize": 32,
        }
    )
    plt.figure(figsize=(13, 8))

    dataset_labels = {1: "1x", 5: "5x", 10: "10x"}

    for key in sorted(df["dataset_key"].unique()):
        label = dataset_labels.get(key, str(key))
        subset = df[df["dataset_key"] == key].copy()
        subset = subset.set_index("ratio").reindex(RATIO_ORDER).reset_index()

        y = (subset["fed_p1_mean"] - subset["local_p1_mean"]) / (1 - subset["local_p1_mean"])

        plt.plot(
            RATIO_LABELS,
            y,
            marker="o",
            markersize=16,
            markeredgewidth=2.5,
            linewidth=10,
            label=label,
        )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Datasize Ratio")
    plt.ylabel("Normalized Accuracy\nImprovement")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def main() -> None:
    input_path = Path("results/multi-split/averaged_results.csv")
    output_path = Path("results/multi-split/averaged_rcurves.png")

    plot_union(input_path, output_path, metric="fed_p1_mean")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()