import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RATIO_ORDER = ["1/8", "1/4", "1/2", "1/1", "2/1", "4/1", "8/1"]
CURVE_ORDER = ["1x", "5x", "10x"]


def load_curve_data(curve_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(curve_csv)

    required_cols = {"Ratio_Str", "Gain_Acc"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {curve_csv}: {sorted(missing_cols)}")

    if "Curve" in df.columns:
        grouped = (
            df.groupby(["Curve", "Ratio_Str"], as_index=False)["Gain_Acc"]
            .mean()
            .rename(columns={"Gain_Acc": "Mean_Gain"})
        )
        grouped["Ratio_Str"] = pd.Categorical(
            grouped["Ratio_Str"], categories=RATIO_ORDER, ordered=True
        )
        grouped = grouped.sort_values(["Curve", "Ratio_Str"]).reset_index(drop=True)
        return grouped

    grouped = (
        df.groupby("Ratio_Str", as_index=False)["Gain_Acc"]
        .mean()
        .rename(columns={"Gain_Acc": "Mean_Gain"})
    )

    grouped["Ratio_Str"] = pd.Categorical(
        grouped["Ratio_Str"], categories=RATIO_ORDER, ordered=True
    )
    grouped = grouped.sort_values("Ratio_Str").reset_index(drop=True)

    return grouped


def plot_union(root_dir: Path, output_file: Path) -> None:
    csv_candidates = [
        root_dir / "CSV" / "rs_ratio_results.csv",
        root_dir / "rs_ratio_results.csv",
        root_dir / "Curve_1" / "CSV" / "dd_ratio_results.csv",
        root_dir / "Curve_5" / "CSV" / "dd_ratio_results.csv",
        root_dir / "Curve_10" / "CSV" / "dd_ratio_results.csv",
    ]
    csv_path = next((candidate for candidate in csv_candidates if candidate.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Could not find any ratio CSV under the expected locations: "
            + ", ".join(str(path) for path in csv_candidates)
        )

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

    curve_df = load_curve_data(csv_path)
    if "Curve" in curve_df.columns:
        for label in CURVE_ORDER:
            subset = curve_df[curve_df["Curve"] == label].sort_values("Ratio_Str")
            if subset.empty:
                continue
            plt.plot(
                subset["Ratio_Str"].astype(str),
                subset["Mean_Gain"],
                marker="o",
                markersize=16,
                markeredgewidth=2.5,
                linewidth=10,
                label=label,
            )
    else:
        plt.plot(
            curve_df["Ratio_Str"].astype(str),
            curve_df["Mean_Gain"],
            marker="o",
            markersize=16,
            markeredgewidth=2.5,
            linewidth=10,
            label="curve",
        )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    # plt.title("Normalized Accuracy Improvement vs. Data Size Ratio (All Curves)")
    plt.xlabel("Datasize Ratio")
    plt.ylabel("Normalized Accuracy\nImprovement")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Union Curve_1, Curve_5, and Curve_10 into one plot."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing Curve_1, Curve_5, Curve_10 folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "combined_curves_plot4.png",
        help="Output image file path.",
    )
    args = parser.parse_args()

    plot_union(args.root, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
