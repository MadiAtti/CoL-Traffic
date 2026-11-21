"""Plot Difference values from CSV files as heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {"P_own", "P_other", "Difference"}
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "MagentaWhiteBlue", ["#c03a9f", "#ffffff", "#3155a6"]
)


def make_norm(values: pd.Series) -> TwoSlopeNorm:
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum >= 0:
        minimum = -0.01
    if maximum <= 0:
        maximum = 0.01
    return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)


def plot_csv(csv_path: Path) -> Path:
    data = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    matrix = data.pivot_table(
        index="P_own", columns="P_other", values="Difference", aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    plot_path = csv_path.with_suffix(".png")

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap=RESIDUAL_CMAP,
        norm=make_norm(data["Difference"]),
        cbar_kws={"label": "Gain difference"},
        ax=axis,
    )
    axis.set_title(f"Gain Difference: {csv_path.stem}")
    axis.set_xlabel("P_other")
    axis.set_ylabel("P_own")
    axis.invert_yaxis()  # p_own increases upwards, as in the heatmaps
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def find_csvs(paths: list[Path], script_directory: Path) -> list[Path]:
    if not paths:
        return sorted(script_directory.glob("*.csv"))

    csv_paths = []
    for path in paths:
        if path.is_dir():
            csv_paths.extend(path.glob("*.csv"))
        elif path.suffix.lower() == ".csv":
            csv_paths.append(path)
        else:
            raise FileNotFoundError(f"Not a CSV file or directory: {path}")
    return sorted(set(csv_paths))


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="CSV files or directories; defaults to CSVs beside this script.",
    )
    args = parser.parse_args()
    csv_paths = find_csvs(args.paths, script_directory)
    if not csv_paths:
        raise FileNotFoundError("No CSV files found")

    for csv_path in csv_paths:
        print(f"Wrote {plot_csv(csv_path)}")


if __name__ == "__main__":
    main()