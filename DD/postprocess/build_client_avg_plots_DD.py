import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.isotonic import IsotonicRegression

TARGET_FILES = (
    "dp_real.csv",
    "dp_sim_p1.csv",
    "dp_sim_p2.csv",
    "sup_real.csv",
    "sup_sim_p1.csv",
    "sup_sim_p2.csv",
)
SCENARIO_LABELS = {
    "real": "REAL",
    "sim_p1": "SIM_P1",
    "sim_p2": "SIM_P2",
}


def find_target_csvs(root_dir: str) -> List[str]:
    """Recursively discover raw mechanism/scenario CSV files."""
    discovered = []
    for dirpath, _, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) != "CSV":
            continue

        for target_name in TARGET_FILES:
            if target_name in filenames:
                discovered.append(os.path.join(dirpath, target_name))

    return sorted(discovered)


def apply_2d_isotonic_regression(matrix: np.ndarray) -> np.ndarray:
    """Project a grid onto the non-increasing 2D isotonic constraint."""
    smoothed = matrix.astype(float, copy=True)
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")

    for _ in range(10):
        for row in range(smoothed.shape[0]):
            smoothed[row, :] = ir.fit_transform(
                np.arange(smoothed.shape[1]), smoothed[row, :]
            )
        for column in range(smoothed.shape[1]):
            smoothed[:, column] = ir.fit_transform(
                np.arange(smoothed.shape[0]), smoothed[:, column]
            )

    return smoothed


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


def add_perspective_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map each client's P1/P2 grid into a common P_own/P_other perspective."""
    work = df.copy()
    client_num = pd.to_numeric(work["Client"], errors="coerce")
    work["P_own"] = np.where(client_num == 0, work["P1_Param"], work["P2_Param"])
    work["P_other"] = np.where(client_num == 0, work["P2_Param"], work["P1_Param"])
    return work


def average_simulation_client_grids(
    df: pd.DataFrame, value_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Average sim_p1/sim_p2 per client, then transpose Client 1 into P_own/P_other."""
    simulations = df[df["Scenario"].isin(["p1", "p2", "sim_p1", "sim_p2"])]
    if simulations.empty:
        raise ValueError("Missing simulation data for client-grid averaging")

    client_means = []
    client_stds = []
    for client in (0, 1):
        client_data = simulations[
            pd.to_numeric(simulations["Client"], errors="coerce") == client
        ]
        if client_data.empty:
            raise ValueError(f"Missing simulation data for Client {client}")

        mean_grid = client_data.pivot_table(
            index="P1_Param",
            columns="P2_Param",
            values=value_column,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        std_grid = client_data.pivot_table(
            index="P1_Param",
            columns="P2_Param",
            values="Gain_Acc_Std_Original",
            aggfunc="mean",
        ).reindex_like(mean_grid).fillna(0.0)

        if client == 1:
            mean_grid = mean_grid.transpose()
            std_grid = std_grid.transpose()

        client_means.append(mean_grid)
        client_stds.append(std_grid)

    mean_grid = pd.concat(client_means).groupby(level=0).mean().sort_index()
    mean_grid = mean_grid.reindex(sorted(mean_grid.columns), axis=1)
    std_grid = pd.concat(client_stds).groupby(level=0).mean().sort_index()
    std_grid = std_grid.reindex(mean_grid.index).reindex(mean_grid.columns, axis=1)
    return mean_grid, std_grid


def build_isotonic_client_data(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth each client's mean grid and retain seed variation for annotations."""
    required = {"Client", "P1_Param", "P2_Param", "Gain_Acc"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for isotonic processing: {sorted(missing)}")

    records = []
    group_columns = [column for column in ["Scenario", "Mechanism", "Client"] if column in df]
    if not group_columns:
        raise ValueError("Missing required grouping columns for isotonic processing")

    for group_values, group_df in df.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        mean_grid = group_df.pivot_table(
            index="P1_Param", columns="P2_Param", values="Gain_Acc", aggfunc="mean"
        ).sort_index().sort_index(axis=1)
        std_grid = group_df.pivot_table(
            index="P1_Param", columns="P2_Param", values="Gain_Acc", aggfunc="std"
        ).reindex_like(mean_grid).fillna(0.0)

        if mean_grid.empty or mean_grid.isna().any().any():
            label = "/".join(str(value) for value in group_values)
            raise ValueError(f"Incomplete P1/P2 grid for {label}")

        isotonic_grid = pd.DataFrame(
            apply_2d_isotonic_regression(mean_grid.to_numpy()),
            index=mean_grid.index,
            columns=mean_grid.columns,
        )
        for p1 in isotonic_grid.index:
            for p2 in isotonic_grid.columns:
                record = dict(zip(group_columns, group_values))
                record.update(
                    {
                        "P1_Param": p1,
                        "P2_Param": p2,
                        "Gain_Acc_Smoothed": isotonic_grid.loc[p1, p2],
                        "Gain_Acc_Std_Original": std_grid.loc[p1, p2],
                    }
                )
                records.append(record)

    return pd.DataFrame(records)


def _safe_two_slope_norm(values: np.ndarray) -> TwoSlopeNorm:
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))

    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2

    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


def _resolve_plot_metric_columns(avg_df: pd.DataFrame) -> Tuple[str, str, str, str]:
    if "Gain_Acc" in avg_df.columns:
        return "Gain_Acc", "Gain_Acc", "std", "Empirical Accuracy Gain"

    if "Gain_Acc_Smoothed" in avg_df.columns:
        spread_col = "Gain_Acc_Std_Original" if "Gain_Acc_Std_Original" in avg_df.columns else "Gain_Acc_Smoothed"
        return "Gain_Acc_Smoothed", spread_col, "mean", "Smoothed Accuracy Gain"

    raise ValueError("Missing required gain column for plotting: Gain_Acc or Gain_Acc_Smoothed")


def _plot_heatmap(
    values: pd.DataFrame,
    annotations: np.ndarray,
    output_path: str,
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
    norm: TwoSlopeNorm,
) -> str:
    rwg_cmap = LinearSegmentedColormap.from_list(
        "RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"]
    )
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        values,
        annot=annotations,
        fmt="",
        cmap=rwg_cmap,
        norm=norm,
        annot_kws={"size": 10},
        cbar_kws={"label": cbar_label},
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().invert_yaxis()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_isotonic_heatmap(
    isotonic_df: pd.DataFrame, input_csv_path: str
) -> str:
    """Generate one isotonic client heatmap with SD included in its annotations."""
    if len(isotonic_df["Client"].unique()) != 1:
        raise ValueError("Isotonic heatmaps require one client at a time")

    pivot_mean = isotonic_df.pivot_table(
        index="P1_Param", columns="P2_Param", values="Gain_Acc_Smoothed", aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    pivot_std = isotonic_df.pivot_table(
        index="P1_Param",
        columns="P2_Param",
        values="Gain_Acc_Std_Original",
        aggfunc="mean",
    ).reindex_like(pivot_mean).fillna(0.0)

    client = str(isotonic_df["Client"].iloc[0])
    mech = str(isotonic_df["Mechanism"].iloc[0])
    scenario = os.path.splitext(os.path.basename(input_csv_path))[0].split("_", 1)[1]
    csv_dir = os.path.dirname(input_csv_path)
    run_root = os.path.dirname(csv_dir)
    output_dir = os.path.join(run_root, "Plots_Isotonic", mech, f"client_{client}")
    os.makedirs(output_dir, exist_ok=True)

    mean_annotations = np.empty(pivot_mean.shape, dtype=object)
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            mean_annotations[i, j] = (
                f"{pivot_mean.iloc[i, j]:.3f}\n±{pivot_std.iloc[i, j]:.3f}"
            )

    client_label = f"Client {client}"
    regular_path = os.path.join(output_dir, f"{scenario}_{client}_isotonic.png")
    _plot_heatmap(
        pivot_mean,
        mean_annotations,
        regular_path,
        f"ISOTONIC Accuracy Gain - {scenario.upper()} - {client_label} ({mech.upper()})\n"
        "(Mean ± STD over Seeds)",
        f"P2 Privacy Parameter ({mech.upper()})",
        f"P1 Privacy Parameter ({mech.upper()})",
        "Isotonic Accuracy Gain",
        _safe_two_slope_norm(pivot_mean.to_numpy()),
    )
    return regular_path


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
    plot_root_name = (
        "Plots_Isotonic_avg"
        if csv_dir_name == "CSVs_Isotonic"
        else "Plots_client_avg"
    )
    out_plot_dir = os.path.join(run_root, plot_root_name, mech)
    os.makedirs(out_plot_dir, exist_ok=True)
    scenario_label = SCENARIO_LABELS.get(scenario, scenario.upper())
    output_path = os.path.join(out_plot_dir, f"{scenario}_avg_clients_{mech}.png")

    return _plot_heatmap(
        pivot_mean,
        annot_matrix,
        output_path,
        f"Accuracy Gain - {scenario_label} - AVG(P_own, P_other) ({mech.upper()})\n"
        "(Mean ± STD over Seeds)",
        f"P_other Privacy Parameter ({mech.upper()})",
        f"P_own Privacy Parameter ({mech.upper()})",
        cbar_label,
        norm=norm,
    )


def generate_avg_sd_heatmap(avg_df: pd.DataFrame, input_csv_path: str) -> str:
    """Generate the perspective-aligned average standard-deviation heatmap."""
    if "Gain_Acc_Std_Original" not in avg_df.columns:
        raise ValueError("Missing required standard-deviation column for plotting")

    pivot_sd = avg_df.pivot_table(
        index="P_own",
        columns="P_other",
        values="Gain_Acc_Std_Original",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    annotations = pivot_sd.map(lambda value: f"{value:.3f}").to_numpy()

    csv_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    parts = csv_name.split("_", 1)
    mech = parts[0] if parts else "unknown"
    scenario = parts[1] if len(parts) > 1 else "unknown"
    run_root = os.path.dirname(os.path.dirname(input_csv_path))
    output_dir = os.path.join(run_root, "Plots_Isotonic_avg", mech)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario}_avg_clients_{mech}_sd.png")

    return _plot_heatmap(
        pivot_sd,
        annotations,
        output_path,
        f"Accuracy Gain SD - {SCENARIO_LABELS.get(scenario, scenario.upper())} - "
        f"AVG(P_own, P_other) ({mech.upper()})",
        f"P_other Privacy Parameter ({mech.upper()})",
        f"P_own Privacy Parameter ({mech.upper()})",
        "Average Accuracy Gain SD",
        _safe_two_slope_norm(pivot_sd.to_numpy()),
    )


def generate_final_average_plots(
    frames: List[pd.DataFrame],
    run_root: str,
    mechanism: str,
    output_root: str,
    value_column: str,
    label: str,
) -> Tuple[str, str]:
    """Create one real normal plot and one simulation SD plot per mechanism."""
    all_data = pd.concat(frames, ignore_index=True)
    real = add_perspective_columns(
        all_data[all_data["Scenario"].isin(["full", "real"])]
    )
    simulations = all_data[
        all_data["Scenario"].isin(["p1", "p2", "sim_p1", "sim_p2"])
    ]
    if real.empty or simulations.empty:
        raise ValueError(f"Missing real or simulation data for {mechanism}")

    normal_mean = real.pivot_table(
        index="P_own", columns="P_other", values=value_column, aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    normal_sd = real.pivot_table(
        index="P_own", columns="P_other", values="Gain_Acc_Std_Original", aggfunc="mean"
    ).reindex_like(normal_mean).fillna(0.0)
    simulation_mean, simulation_sd = average_simulation_client_grids(
        simulations, value_column
    )

    output_dir = os.path.join(run_root, output_root, mechanism)
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            os.remove(os.path.join(output_dir, filename))

    normal_path = os.path.join(output_dir, f"{mechanism}_normal.png")
    sd_path = os.path.join(output_dir, f"{mechanism}_sd.png")
    normal_plot = _plot_heatmap(
        normal_mean,
        np.array([
            [f"{normal_mean.iloc[i, j]:.3f}\n±{normal_sd.iloc[i, j]:.3f}"
             for j in range(normal_mean.shape[1])]
            for i in range(normal_mean.shape[0])
        ]),
        normal_path,
        f"{label} Average Accuracy Gain - {mechanism.upper()} (REAL)\n(Mean ± STD)",
        f"P_other Privacy Parameter ({mechanism.upper()})",
        f"P_own Privacy Parameter ({mechanism.upper()})",
        f"Average {label} Accuracy Gain",
        _safe_two_slope_norm(normal_mean.to_numpy()),
    )
    sd_plot = _plot_heatmap(
        simulation_mean,
        np.array([
            [
                f"{simulation_mean.iloc[i, j]:.3f}\n±{simulation_sd.iloc[i, j]:.3f}"
                for j in range(simulation_mean.shape[1])
            ]
            for i in range(simulation_mean.shape[0])
        ]),
        sd_path,
        f"{label} Average Accuracy Gain SD - {mechanism.upper()} (SIMULATIONS)",
        f"P_other Privacy Parameter ({mechanism.upper()})",
        f"P_own Privacy Parameter ({mechanism.upper()})",
        f"Average {label} Accuracy Gain SD",
        _safe_two_slope_norm(simulation_mean.to_numpy()),
    )
    return normal_plot, sd_plot


def build_raw_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce seed-level raw results to client grids and seed standard deviations."""
    return (
        df.groupby(
            ["Scenario", "Mechanism", "Client", "P1_Param", "P2_Param"],
            dropna=False,
        )["Gain_Acc"]
        .agg(Gain_Acc="mean", Gain_Acc_Std_Original="std")
        .reset_index()
        .fillna({"Gain_Acc_Std_Original": 0.0})
    )


def process_csv(input_csv_path: str) -> pd.DataFrame:
    """Isotonicize one CSV and generate one heatmap per client."""
    df = pd.read_csv(input_csv_path)
    csv_dir = os.path.dirname(input_csv_path)
    run_root = os.path.dirname(csv_dir)
    out_csv_dir = os.path.join(run_root, "CSVs_Isotonic")
    os.makedirs(out_csv_dir, exist_ok=True)

    csv_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    isotonic_df = build_isotonic_client_data(df)
    isotonic_csv_path = os.path.join(out_csv_dir, f"{csv_name}.csv")
    isotonic_df.to_csv(isotonic_csv_path, index=False)

    heatmap_paths = []
    for client in isotonic_df["Client"].unique():
        client_df = isotonic_df[isotonic_df["Client"] == client]
        heatmap_paths.append(generate_isotonic_heatmap(client_df, input_csv_path))

    print(f"    Isotonic heatmaps: {', '.join(heatmap_paths)}")
    return isotonic_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply 2D isotonic regression to all mechanism/scenario CSVs, generate regular and "
            "SD heatmaps, then average Client 0 and 1 and generate average plots."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root directory to recursively scan for folders named CSV (defaults to this workspace root).",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    targets = find_target_csvs(root_dir)

    if not targets:
        print("No target files found. Expected mechanism/scenario CSVs under folders named CSV.")
        return

    print(f"Found {len(targets)} target CSV files under: {root_dir}")

    succeeded = 0
    failed = 0

    grouped_targets = {}
    for input_csv in targets:
        csv_name = os.path.splitext(os.path.basename(input_csv))[0]
        mechanism = csv_name.split("_", 1)[0]
        run_root = os.path.dirname(os.path.dirname(input_csv))
        grouped_targets.setdefault((run_root, mechanism), []).append(input_csv)

    for (run_root, mechanism), mechanism_targets in grouped_targets.items():
        isotonic_frames = []
        raw_frames = []
        try:
            for input_csv in mechanism_targets:
                print(f"Processing: {input_csv}")
                raw_df = pd.read_csv(input_csv)
                raw_frames.append(build_raw_summary(raw_df))
                isotonic_frames.append(process_csv(input_csv))
            normal_plot, sd_plot = generate_final_average_plots(
                isotonic_frames,
                run_root,
                mechanism,
                "Plots_Isotonic_avg",
                "Gain_Acc_Smoothed",
                "ISOTONIC",
            )
            raw_normal_plot, raw_sd_plot = generate_final_average_plots(
                raw_frames,
                run_root,
                mechanism,
                "Plots_client_avg",
                "Gain_Acc",
                "RAW",
            )
            print(f"    Final normal average: {normal_plot}")
            print(f"    Final SD average: {sd_plot}")
            print(f"    Raw normal average: {raw_normal_plot}")
            print(f"    Raw SD average: {raw_sd_plot}")
            succeeded += 1
        except Exception as exc:
            print(f"FAIL: {run_root}/{mechanism}")
            print(f"    Reason: {exc}")
            failed += 1

    print("Summary")
    print(f"  Success: {succeeded}")
    print(f"  Failed:  {failed}")


if __name__ == "__main__":
    main()
