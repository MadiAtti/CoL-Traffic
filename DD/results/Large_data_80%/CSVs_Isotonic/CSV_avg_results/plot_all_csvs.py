"""Create accuracy-gain heatmaps for every CSV in this directory.

The script supports both result layouts currently used in this folder:

* real-data files with Scenario/Client columns and smoothed gains;
* simulated aggregate files with Mean_Gain_Acc and SD_Gain_Acc columns.

Run from any directory with:
    python plot_all_csvs.py

Plots are written to a ``Plots`` folder beside this script. Use ``--output-dir``
to choose a different destination.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import pandas as pd
import seaborn as sns


INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = INPUT_DIR / "Plots"

GAIN_COLUMNS = ("Gain_Acc_Smoothed", "Mean_Gain_Acc")
STD_COLUMNS = ("Gain_Acc_Std_Original", "SD_Gain_Acc")


def first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate present in a DataFrame's columns."""
    return next((column for column in candidates if column in columns), None)


def safe_name(value: object) -> str:
    """Make a CSV value suitable for use in a filename."""
    return str(value).strip().replace(" ", "_").replace("/", "-")


def make_norm(values: pd.DataFrame) -> TwoSlopeNorm:
    """Build a zero-centered scale, including for all-positive/all-negative data."""
    minimum = float(values.min().min())
    maximum = float(values.max().max())
    if minimum >= 0:
        minimum = -0.01
    if maximum <= 0:
        maximum = 0.01
    return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)


def plot_matrix(
    data: pd.DataFrame,
    output_path: Path,
    clean_scenario: str,
    client: str,
    mechanism: str,
    gain_column: str,
    std_column: str,
) -> None:
    """Pivot one result group and save its annotated heatmap."""
    mean_matrix = data.pivot_table(
        index="P_own",
        columns="P_other",
        values=gain_column,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    std_matrix = data.pivot_table(
        index="P_own",
        columns="P_other",
        values=std_column,
        aggfunc="mean",
    ).reindex(index=mean_matrix.index, columns=mean_matrix.columns).fillna(0.0)

    annotations = mean_matrix.astype(object).copy()
    for column in mean_matrix.columns:
        annotations[column] = [
            f"{mean:.3f}\n±{std:.3f}"
            for mean, std in zip(mean_matrix[column], std_matrix[column])
        ]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        mean_matrix,
        annot=annotations,
        fmt="",
        cmap=RWG_CMAP,
        norm=make_norm(mean_matrix),
        annot_kws={"size": 10},
        cbar_kws={"label": "Empirical Accuracy Gain"},
    )
    plt.title(
        f"Accuracy Gain - {clean_scenario.upper()} - {client} ({mechanism.upper()})\n"
        "(Mean ± STD over 10 Seeds)"
    )
    plt.xlabel(f"P2 Privacy Parameter ({mechanism.upper()})")
    plt.ylabel(f"P1 Privacy Parameter ({mechanism.upper()})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_csv(csv_path: Path, output_dir: Path) -> int:
    """Create all required plots for one CSV and return the plot count."""
    data = pd.read_csv(csv_path)
    required = {"P_own", "P_other"}
    missing = required - set(data.columns)
    if missing:
        print(f"Skipped {csv_path.name}: missing columns {sorted(missing)}")
        return 0

    gain_column = first_existing(data.columns, GAIN_COLUMNS)
    std_column = first_existing(data.columns, STD_COLUMNS)
    if gain_column is None or std_column is None:
        print(f"Skipped {csv_path.name}: gain/standard-deviation columns not found")
        return 0

    mechanism = (
        str(data["Mechanism"].iloc[0])
        if "Mechanism" in data
        else csv_path.stem.removeprefix("sim_avg_")
    )
    plot_count = 0
    if "Client" in data.columns:
        groups = data.groupby(["Scenario", "Client"], dropna=False) if "Scenario" in data else data.groupby("Client", dropna=False)
        for group_key, group in groups:
            key_parts = group_key if isinstance(group_key, tuple) else (group_key,)
            scenario = str(key_parts[0]) if "Scenario" in data.columns else "sim"
            client = safe_name(key_parts[-1])
            clean_scenario = {"full": "real", "p1": "sim_p1", "p2": "sim_p2"}.get(scenario, scenario)
            output_path = output_dir / f"{clean_scenario}_{client}_{mechanism}.png"
            plot_matrix(group, output_path, clean_scenario, client, mechanism, gain_column, std_column)
            plot_count += 1
    else:
        clean_scenario = "sim"
        client = "aggregate"
        output_path = output_dir / f"{clean_scenario}_{client}_{mechanism}.png"
        plot_matrix(data, output_path, clean_scenario, client, mechanism, gain_column, std_column)
        plot_count += 1
    return plot_count


RWG_CMAP = LinearSegmentedColormap.from_list(
    "RedWhiteGreen",
    ["#d73027", "#ffffff", "#1a9850"],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Folder containing the CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder where PNG heatmaps will be saved.",
    )
    args = parser.parse_args()

    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {args.input_dir}")

    total_plots = sum(plot_csv(csv_file, args.output_dir) for csv_file in csv_files)
    print(f"Created {total_plots} plot(s) from {len(csv_files)} CSV file(s).")


if __name__ == "__main__":
    main()