from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
CSV_ROOTS = [
    ROOT / "Results" / "NF_1" / "isotonic_csv",
    ROOT / "Results" / "NF_5" / "isotonic_csv",
]
PLOT_ROOTS = [
    ROOT / "Results" / "NF_1" / "plots" / "Isotonic_avg",
    ROOT / "Results" / "NF_5" / "plots" / "Isotonic_avg",
]


def _safe_two_slope_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Ensures TwoSlopeNorm satisfies vmin < vcenter < vmax."""
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))

    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2

    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


def extract_variant_matrix(
    df: pd.DataFrame, 
    client_id: int, 
    val_col: str
) -> pd.DataFrame:
    """
    Extracts a single participant partition and maps parameters:
    - Client 0 (P11 or P21): P_own = P1_Param, P_other = P2_Param
    - Client 1 (P12 or P22): P_own = P2_Param, P_other = P1_Param
    """
    sub_df = df[df["Client_num"] == client_id].copy()
    if sub_df.empty:
        raise ValueError(f"Client {client_id} not found in the provided DataFrame.")

    # Align perspective
    if client_id == 0:
        sub_df["P_own"] = sub_df["P1_Param"]
        sub_df["P_other"] = sub_df["P2_Param"]
    else:
        sub_df["P_own"] = sub_df["P2_Param"]
        sub_df["P_other"] = sub_df["P1_Param"]

    # Pivot to standard matrix: Index (Rows) = P_own, Columns = P_other
    pivot = (
        sub_df.pivot_table(index="P_own", columns="P_other", values=val_col, aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return pivot


def build_average_sd_plot(mech: str, csv_root: Path, plot_root: Path) -> Tuple[Path, Path]:
    """
    Computes the 4-variant symmetric SD average (P11, P12, P21, P22) 
    and saves the heatmap with (P_own=0, P_other=0) in the upper-left.
    """
    files = [
        csv_root / f"{mech}_sim_p1.csv",
        csv_root / f"{mech}_sim_p2.csv",
    ]

    matrices: List[pd.DataFrame] = []

    for csv_path in files:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required SD simulation file: {csv_path}")

        df = pd.read_csv(csv_path)

        # Detect the isotonic metric used by the NF datasets.
        if "Isotonic_Gain_RMSE" in df.columns:
            val_col = "Isotonic_Gain_RMSE"
        elif "Raw_Gain_RMSE" in df.columns:
            val_col = "Raw_Gain_RMSE"
        else:
            raise ValueError(
                f"Missing RMSE gain column in {csv_path}; "
                "expected Isotonic_Gain_RMSE or Raw_Gain_RMSE"
            )

        df["Client_num"] = pd.to_numeric(df["Client"], errors="coerce")
        df["P1_Param"] = pd.to_numeric(df["P1_Param"])
        df["P2_Param"] = pd.to_numeric(df["P2_Param"])

        # Extract Client 0 and Client 1 separately to yield the 4 distinct matrices
        for cid in [0, 1]:
            variant_matrix = extract_variant_matrix(df, client_id=cid, val_col=val_col)
            matrices.append(variant_matrix)

    if len(matrices) != 4:
        raise ValueError(f"Expected 4 SD variants across the two simulation files, found {len(matrices)}")

    # Standardize coordinate labels across all 4 matrices
    common_p_own = sorted(set().union(*[m.index for m in matrices]))
    common_p_other = sorted(set().union(*[m.columns for m in matrices]))

    aligned_matrices = [
        m.reindex(index=common_p_own, columns=common_p_other) for m in matrices
    ]

    # Compute element-wise mean and population standard deviation across the 4 variants
    stack = np.stack([m.to_numpy() for m in aligned_matrices], axis=0)
    mean_matrix = np.mean(stack, axis=0)
    std_matrix = np.std(stack, axis=0, ddof=0)

    mean_df = pd.DataFrame(mean_matrix, index=common_p_own, columns=common_p_other)
    std_df = pd.DataFrame(std_matrix, index=common_p_own, columns=common_p_other)

    out_dir = plot_root / mech
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the values used to create the heatmap so the plot can be recreated later.
    average_csv = pd.DataFrame(
        [
            {
                "P_own": common_p_own[i],
                "P_other": common_p_other[j],
                "Mean_Gain_RMSE": mean_df.iloc[i, j],
                "SD_Gain_RMSE": std_df.iloc[i, j],
            }
            for i in range(mean_df.shape[0])
            for j in range(mean_df.shape[1])
        ]
    )
    average_csv_path = out_dir / f"sim_avg_{mech}.csv"
    average_csv.to_csv(average_csv_path, index=False)

    # Format cell annotations: Mean ± STD
    annot_matrix = np.empty(mean_df.shape, dtype=object)
    for i in range(mean_df.shape[0]):
        for j in range(mean_df.shape[1]):
            annot_matrix[i, j] = f"{mean_df.iloc[i, j]:.3f}\n±{std_df.iloc[i, j]:.3f}"

    norm = _safe_two_slope_norm(mean_df.to_numpy())
    rwg_cmap = LinearSegmentedColormap.from_list(
        "RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"]
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    # Standard heatmap layout maintains index 0 at the top, column 0 at the left
    # -> Origin (P_own=0, P_other=0) is positioned at the UPPER-LEFT
    sns.heatmap(
        mean_df,
        annot=annot_matrix,
        fmt="",
        cmap=rwg_cmap,
        norm=norm,
        annot_kws={"size": 10},
        cbar_kws={"label": "Isotonic RMSE Gain"},
        ax=ax,
    )

    ax.set_title(
        f"Isotonic RMSE Gain - SIM AVG (4 SD variants: P11, P12, P21, P22) - {mech.upper()}\n"
        f"(Mean ± SD over 4 variants)",
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel(f"P_other Privacy Parameter ({mech.upper()})", fontsize=11, labelpad=8)
    ax.set_ylabel(f"P_own Privacy Parameter ({mech.upper()})", fontsize=11, labelpad=8)

    out_path = out_dir / f"sim_avg_{mech}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return average_csv_path, out_path


def main() -> None:
    for csv_root, plot_root in zip(CSV_ROOTS, PLOT_ROOTS):
        if not csv_root.exists():
            print(f"Skipping missing directory: {csv_root}")
            continue

        for mech in ["dp", "sup"]:
            try:
                csv_output, plot_output = build_average_sd_plot(mech, csv_root, plot_root)
                print(f"Successfully generated CSV: {csv_output}")
                print(f"Successfully generated plot: {plot_output}")
            except Exception as err:
                print(f"Failed to generate {mech} in {csv_root}: {err}")


if __name__ == "__main__":
    main()