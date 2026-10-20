import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RATIO_ORDER = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RATIO_LABELS = ["1/8", "1/4", "1/2", "1/1", "2/1", "4/1", "8/1"]

REQUIRED_COLS = {"dataset_key", "ratio", "local_p1_mean", "local_p2_mean", "fed_p1_mean", "fed_p2_mean"}


def load_curve_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_cols)}")

    return df


def normalized_improvement(df: pd.DataFrame) -> pd.Series:
    return (df["fed_p1_mean"] - df["local_p1_mean"]) / (1 - df["local_p1_mean"])


def describe_structure(obj, depth: int = 0, max_depth: int = 3) -> str:
    """Short structural summary of a JSON object (for error messages)."""
    pad = "  " * depth
    if depth >= max_depth:
        return f"{pad}...\n"
    if isinstance(obj, dict):
        out = f"{pad}dict with keys: {list(obj.keys())[:10]}\n"
        for k in list(obj.keys())[:2]:
            out += f"{pad} [{k!r}] ->\n" + describe_structure(obj[k], depth + 2, max_depth)
        return out
    if isinstance(obj, list):
        out = f"{pad}list of {len(obj)} items\n"
        if obj:
            out += describe_structure(obj[0], depth + 1, max_depth)
        return out
    return f"{pad}{type(obj).__name__}: {obj!r}\n"


def load_seed_json(path: str) -> pd.DataFrame:
    """Load one seed_<n>.json into a flat DataFrame with the required columns.

    Expected structure:
    {"seed": 0, "results": {"1x_r0.125": {"dataset_key": 1, "ratio": 0.125,
        "local": {"player1": ..., "player2": ...},
        "federated": {"player1": ..., "player2": ...}}, ...}}
    """
    with open(path) as f:
        data = json.load(f)

    rows = []
    try:
        for entry in data["results"].values():
            rows.append(
                {
                    "dataset_key": entry["dataset_key"],
                    "ratio": entry["ratio"],
                    "local_p1_mean": entry["local"]["player1"],
                    "local_p2_mean": entry["local"]["player2"],
                    "fed_p1_mean": entry["federated"]["player1"],
                    "fed_p2_mean": entry["federated"]["player2"],
                }
            )
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"{path}: unexpected JSON structure ({e!r})\n{describe_structure(data)}"
        )

    return pd.DataFrame(rows)


def plot_union(input_file: Path, output_file: Path) -> None:
    input_file = Path(input_file)
    output_file = Path(output_file)

    df = load_curve_data(input_file)

    font_cfg = {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 36,
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 32,
    }

    dataset_labels = {1: "1x", 5: "5x", 10: "10x"}
    x = list(range(len(RATIO_ORDER)))

    with plt.rc_context(font_cfg):
        plt.figure(figsize=(13, 8))

        for key in sorted(df["dataset_key"].unique()):
            label = dataset_labels.get(key, str(key))
            subset = df[df["dataset_key"] == key].copy()
            subset = subset.set_index("ratio").reindex(RATIO_ORDER).reset_index()

            y = normalized_improvement(subset)

            plt.plot(
                x,
                y,
                marker="o",
                markersize=16,
                markeredgewidth=2.5,
                linewidth=10,
                label=label,
                zorder=3,
            )

        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xticks(x, RATIO_LABELS)
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
    output_path = Path("results/final/tg_vs_all_NT.png")

    plot_union(input_path, output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()