"""Create RMSE-gain heatmaps for every CSV in this directory.

The script handles both result layouts used here:

* real-data files with ``Scenario`` and ``Client`` columns;
* simulated aggregate files with ``Mean_Gain_RMSE`` and ``SD_Gain_RMSE``.

Run from any directory with::

    python plot_all_csvs.py

Plots are written to a ``Plots`` folder beside this script. Use
``--output-dir`` to choose a different destination.
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
RWG_CMAP = LinearSegmentedColormap.from_list(
    "RedWhiteGreen",
    ["#d73027", "#ffffff", "#1a9850"],
)


def safe_name(value: object) -> str:
    """Make a CSV value suitable for use in a filename."""
    return str(value).strip().replace(" ", "_").replace("/", "-")


def make_norm(values: pd.DataFrame) -> TwoSlopeNorm:
    """Build a zero-centered scale, including all-positive/all-negative data."""
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
    scenario: str,
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
        cbar_kws={"label": "RMSE Gain"},
    )
    plt.title(
        f"RMSE Gain - {scenario.upper()} - {client} ({mechanism.upper()})\n"
        "(Mean ± STD)"
    )
    plt.xlabel(f"P_other Privacy Parameter ({mechanism.upper()})")
    plt.ylabel(f"P_own Privacy Parameter ({mechanism.upper()})")
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

    if "Client" in data.columns:
        gain_column = "Isotonic_Gain_RMSE"
        std_column = "Raw_Gain_RMSE_Std"
    else:
        gain_column = "Mean_Gain_RMSE"
        std_column = "SD_Gain_RMSE"

    missing = {gain_column, std_column} - set(data.columns)
    if missing:
        print(f"Skipped {csv_path.name}: missing columns {sorted(missing)}")
        return 0

    mechanism = (
        str(data["Mechanism"].iloc[0])
        if "Mechanism" in data.columns
        else csv_path.stem.removeprefix("sim_avg_")
    )
    plot_count = 0

    if "Client" in data.columns:
        groups = data.groupby(["Scenario", "Client"], dropna=False)
        for (scenario, client), group in groups:
            clean_scenario = {"full": "real", "p1": "sim_p1", "p2": "sim_p2"}.get(
                str(scenario), str(scenario)
            )
            clean_client = safe_name(client)
            output_path = output_dir / f"{clean_scenario}_{clean_client}_{mechanism}.png"
            plot_matrix(
                group,
                output_path,
                clean_scenario,
                clean_client,
                mechanism,
                gain_column,
                std_column,
            )
            plot_count += 1
    else:
        output_path = output_dir / f"sim_aggregate_{mechanism}.png"
        plot_matrix(
            data,
            output_path,
            "sim",
            "aggregate",
            mechanism,
            gain_column,
            std_column,
        )
        plot_count += 1

    return plot_count


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