import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for academic plotting
sns.set_theme(style="whitegrid", context="talk")

RESULTS_DIR = "results/baseline"


def load_seed_curves(results_dir=RESULTS_DIR):
    """Load the per-seed learning curves (seed_<n>_learning_curve.csv)."""
    files = sorted(glob.glob(os.path.join(results_dir, "seed_*_learning_curve.csv")))
    curves = []
    for f in files:
        d = pd.read_csv(f).sort_values("percentage").set_index("percentage")["accuracy"]
        curves.append(d)
    if not curves:
        return None
    return pd.concat(curves, axis=1)  # rows: percentage, columns: seeds


def generate_plots(csv_file="data_saturation_drug_discovery.csv",
                   output_path=os.path.join(RESULTS_DIR, "plot_accuracy.png")):
    # Load the data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    # Sort values by percentage ascending (5 -> 100)
    df = df.sort_values(by="percentage")

    seeds_df = load_seed_curves()

    # ==========================================
    # Plot: Accuracy with seed-to-seed spread
    # ==========================================
    plt.figure(figsize=(10, 6))

    if seeds_df is not None:
        n_seeds = seeds_df.shape[1]
        # Faint band: min-max across all seeds
        plt.fill_between(
            seeds_df.index, seeds_df.min(axis=1), seeds_df.max(axis=1),
            color="tab:blue", alpha=0.15, linewidth=0,
            label=f"Seed spread (min–max, {n_seeds} seeds)",
        )
        # Very faint individual seed curves
        for col in seeds_df.columns:
            plt.plot(seeds_df.index, seeds_df[col],
                     color="tab:blue", alpha=0.15, linewidth=0.8)
    else:
        print("Warning: per-seed CSVs not found, falling back to std band.")
        if "std_accuracy" in df.columns:
            plt.fill_between(
                df["percentage"],
                df["mean_accuracy"] - df["std_accuracy"],
                df["mean_accuracy"] + df["std_accuracy"],
                color="tab:blue", alpha=0.15, linewidth=0, label="±1 Std. Dev.",
            )

    plt.plot(df["percentage"], df["mean_accuracy"], color="tab:blue",
             marker="o", linewidth=2.5, label="Mean Accuracy")

    plt.xlabel("Dataset Size (%)", fontweight="bold")
    plt.ylabel("Global Accuracy", fontweight="bold")
    plt.title("Model Capacity Saturation:\nAccuracy over Dataset Size",
              pad=20, fontweight="bold")
    plt.ylim(0.8, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    generate_plots(os.path.join(RESULTS_DIR, "averaged_learning_curve.csv"))