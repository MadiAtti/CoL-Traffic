import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

TARGET_FILES = ("dp_real.csv", "sup_real.csv")
INPUT_DIR_NAMES = {"isotonic_csv"}
SCENARIO_LABELS = {
    "full": "FULL",
    "p1": "P1 SIMULATION",
    "p2": "P2 SIMULATION",
}


def find_target_csvs(root_dir: str) -> List[str]:
    """Recursively discover real RS files inside isotonic_csv folders."""
    discovered = []
    for dirpath, _, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) not in INPUT_DIR_NAMES:
            continue

        for target_name in TARGET_FILES:
            if target_name in filenames:
                discovered.append(os.path.join(dirpath, target_name))

    return sorted(discovered)


def _resolve_group_columns(df: pd.DataFrame) -> List[str]:
    preferred = ["Seed", "Scenario", "Mechanism", "P1_Param", "P2_Param"]
    available = [c for c in preferred if c in df.columns]

    # These two columns are mandatory for averaging by identical privacy setting.
    if "P1_Param" not in available or "P2_Param" not in available:
        raise ValueError("Missing required columns: P1_Param and/or P2_Param")

    return available


def average_clients(df: pd.DataFrame) -> pd.DataFrame:
    """Average records of Client 0 and 1 correctly by aligning P_own and P_other perspectives."""
    if "Client" not in df.columns:
        raise ValueError("Missing required column: Client")

    work = df.copy()
    work["Client_num"] = pd.to_numeric(work["Client"], errors="coerce")
    work = work[work["Client_num"].isin([0, 1])]

    if work.empty:
        raise ValueError("No rows with Client equal to 0 or 1")

    # 1. Flip parameters to align player perspectives
    work["P_own"] = np.where(work["Client_num"] == 0, work["P1_Param"], work["P2_Param"])
    work["P_other"] = np.where(work["Client_num"] == 0, work["P2_Param"], work["P1_Param"])

    # 2. Group by the unified perspective rather than strict P1/P2 parameters.
    # Some Isotonic outputs omit the Seed column entirely, so only include it when present.
    group_cols = [
        c for c in ["Seed", "Scenario", "Mechanism", "P_own", "P_other"] if c in work.columns
    ]

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols if c not in set(group_cols + ["Client", "Client_num", "P1_Param", "P2_Param"])
    ]

    if not numeric_cols:
        raise ValueError("No numeric metric columns were found to average")

    # 3. Ensure we have valid (Client 0, Client 1) pairs for the identical P_own/P_other perspective
    pair_count = (
        work.groupby(group_cols, dropna=False)["Client_num"]
        .nunique()
        .rename("n_clients")
        .reset_index()
    )
    valid_groups = pair_count[pair_count["n_clients"] == 2].drop(columns=["n_clients"])

    if valid_groups.empty:
        raise ValueError("No pairs found for identical P_own/P_other parameters.")

    # 4. Perform the symmetric average
    averaged = (
        work.merge(valid_groups, on=group_cols, how="inner")
        .groupby(group_cols, dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )

    averaged["Client"] = "avg_01"

    ordered_cols = group_cols + ["Client"] + [c for c in averaged.columns if c not in group_cols + ["Client"]]
    return averaged[ordered_cols]


def _safe_two_slope_norm(values: np.ndarray) -> TwoSlopeNorm:
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))

    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2

    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


def _resolve_plot_metric_columns(avg_df: pd.DataFrame) -> Tuple[str, str, str, str]:
    if "Raw_Gain_RMSE" in avg_df.columns:
        return "Raw_Gain_RMSE", "Raw_Gain_RMSE", "std", "Empirical RMSE Gain"

    if "Isotonic_Gain_RMSE" in avg_df.columns:
        spread_col = "Raw_Gain_RMSE_Std" if "Raw_Gain_RMSE_Std" in avg_df.columns else "Isotonic_Gain_RMSE"
        return "Isotonic_Gain_RMSE", spread_col, "mean", "Isotonic RMSE Gain"

    raise ValueError("Missing required gain column for plotting: Raw_Gain_RMSE or Isotonic_Gain_RMSE")


def generate_avg_heatmap(avg_df: pd.DataFrame, input_csv_path: str) -> str:
    """Generate heatmap from perspective-averaged data, mapping P_own to Y-axis and P_other to X-axis."""
    value_col, spread_col, spread_agg, cbar_label = _resolve_plot_metric_columns(avg_df)

    # Pivot on the unified perspective
    pivot_mean = avg_df.pivot_table(
        index="P_own", columns="P_other", values=value_col, aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    pivot_std = avg_df.pivot_table(
        index="P_own", columns="P_other", values=spread_col, aggfunc=spread_agg
    ).fillna(0.0).sort_index().sort_index(axis=1)

    norm = _safe_two_slope_norm(pivot_mean.values)
    rwg_cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"])

    annot_matrix = np.empty(pivot_mean.shape, dtype=object)
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            annot_matrix[i, j] = f"{pivot_mean.iloc[i, j]:.3f}\n±{pivot_std.iloc[i, j]:.3f}"

    csv_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    parts = csv_name.split("_", 1)
    mech = parts[0] if parts else "unknown"
    scenario = parts[1] if len(parts) > 1 else "unknown"

    csv_dir = os.path.dirname(input_csv_path)
    run_root = os.path.dirname(csv_dir)
    csv_dir_name = os.path.basename(csv_dir)
    
    plot_root_name = os.path.join("plots", "Isotonic_avg") if csv_dir_name == "isotonic_csv" else os.path.join("plots", "client_avg")
    out_plot_dir = os.path.join(run_root, plot_root_name, mech)
    os.makedirs(out_plot_dir, exist_ok=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        pivot_mean,
        annot=annot_matrix,
        fmt="",
        cmap=rwg_cmap,
        norm=norm,
        annot_kws={"size": 10},
        cbar_kws={"label": cbar_label},
    )

    scenario_label = SCENARIO_LABELS.get(scenario, scenario.upper())
    plt.title(f"RMSE Gain - {scenario_label} - AVG(P_own, P_other) ({mech.upper()})\n(Mean ± STD over Seeds)")
    plt.xlabel(f"P_other Privacy Parameter ({mech.upper()})")
    plt.ylabel(f"P_own Privacy Parameter ({mech.upper()})")

    output_path = os.path.join(out_plot_dir, f"{scenario}_avg_clients_{mech}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def process_csv(input_csv_path: str) -> Tuple[str, str, str]:
    """Process one CSV and return (input, output_csv, output_plot)."""
    df = pd.read_csv(input_csv_path)
    avg_df = average_clients(df)

    csv_dir = os.path.dirname(input_csv_path)
    out_csv_dir = os.path.join(csv_dir, "client_avg")
    os.makedirs(out_csv_dir, exist_ok=True)

    csv_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    avg_csv_path = os.path.join(out_csv_dir, f"{csv_name}_avg_clients.csv")
    avg_df.to_csv(avg_csv_path, index=False)

    avg_plot_path = generate_avg_heatmap(avg_df, input_csv_path)
    return input_csv_path, avg_csv_path, avg_plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Average the two RS client perspectives in isotonic real-result CSVs and generate heatmaps."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Results directory to recursively scan (defaults to this workspace root).",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    targets = find_target_csvs(root_dir)

    if not targets:
        print("No target files found. Expected dp_real.csv/sup_real.csv under Results/**/isotonic_csv.")
        return

    print(f"Found {len(targets)} target CSV files under: {root_dir}")

    succeeded = 0
    failed = 0

    for input_csv in targets:
        try:
            src, out_csv, out_plot = process_csv(input_csv)
            print(f"OK: {src}")
            print(f"    Averaged CSV: {out_csv}")
            print(f"    New Plot:     {out_plot}")
            succeeded += 1
        except Exception as exc:
            print(f"FAIL: {input_csv}")
            print(f"    Reason: {exc}")
            failed += 1

    print("Summary")
    print(f"  Success: {succeeded}")
    print(f"  Failed:  {failed}")


if __name__ == "__main__":
    main()
