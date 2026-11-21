import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.isotonic import IsotonicRegression

CLIENT_MAP = {
    "p1": {0: "P11", 1: "P12"},
    "p2": {0: "P21", 1: "P22"},
}

def apply_2d_isotonic_regression(matrix):
    """
    Applies 2D isotonic regression to enforce strict monotonic decreasing behavior.
    Higher privacy (indices) MUST result in lower or equal accuracy gain.
    """
    smoothed = matrix.copy()
    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')

    # Alternating projections to satisfy 2D monotonicity
    for _ in range(10):
        # Smooth rows (P2 privacy increases -> utility should non-increase)
        for i in range(smoothed.shape[0]):
            smoothed[i, :] = ir.fit_transform(np.arange(smoothed.shape[1]), smoothed[i, :])
        
        # Smooth columns (P1 privacy increases -> utility should non-increase)
        for j in range(smoothed.shape[1]):
            smoothed[:, j] = ir.fit_transform(np.arange(smoothed.shape[0]), smoothed[:, j])
            
    return smoothed

def find_target_csvs(root_dir):
    """Discover raw RS experiment CSVs inside Results/**/csv folders."""
    discovered = []
    for dirpath, _, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).lower() == "csv":
            for f in filenames:
                if f.endswith(".csv"):
                    discovered.append(os.path.join(dirpath, f))
    return sorted(discovered)

def process_historical_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    req_cols = ["Mechanism", "Scenario", "Client", "P1_Param", "P2_Param", "Raw_Gain_RMSE"]
    if not all(col in df.columns for col in req_cols):
        raise ValueError(f"Missing required columns in {csv_path}")

    # Determine safe output directories (Sibling to the original CSV folder)
    parent_dir = os.path.dirname(csv_path)
    base_dir = os.path.dirname(parent_dir)

    csv_iso_dir = os.path.join(base_dir, "isotonic_csv")
    plot_iso_dir = os.path.join(base_dir, "plots", "Isotonic")
    
    os.makedirs(csv_iso_dir, exist_ok=True)
    out_csv_path = os.path.join(csv_iso_dir, os.path.basename(csv_path))

    rwg_cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"])
    smoothed_records = []

    for mech in df["Mechanism"].unique():
        mech_plot_dir = os.path.join(plot_iso_dir, mech)
        os.makedirs(mech_plot_dir, exist_ok=True)

        for scenario in df["Scenario"].unique():
            for client in df["Client"].unique():
                sub_df = df[
                    (df["Mechanism"] == mech)
                    & (df["Scenario"] == scenario)
                    & (df["Client"] == client)
                ]
                if sub_df.empty:
                    continue

                pivot_mean_raw = sub_df.pivot_table(
                    index="P1_Param", columns="P2_Param", values="Raw_Gain_RMSE", aggfunc="mean"
                ).sort_index().sort_index(axis=1)
                pivot_std = sub_df.pivot_table(
                    index="P1_Param", columns="P2_Param", values="Raw_Gain_RMSE", aggfunc="std"
                ).reindex_like(pivot_mean_raw).fillna(0)

                if pivot_mean_raw.isna().any().any():
                    raise ValueError(f"Incomplete P1/P2 grid in {csv_path} for {mech}/{scenario}/{client}")

                # 2. Apply Game-Theoretic Smoothing
                smoothed_values = apply_2d_isotonic_regression(pivot_mean_raw.values)
                pivot_mean_iso = pd.DataFrame(smoothed_values, index=pivot_mean_raw.index, columns=pivot_mean_raw.columns)

                # 3. Save flattened smoothed records for the new CSV
                for p1 in pivot_mean_iso.index:
                    for p2 in pivot_mean_iso.columns:
                        smoothed_records.append({
                            "Mechanism": mech,
                            "Scenario": scenario,
                            "Client": client,
                            "P1_Param": p1,
                            "P2_Param": p2,
                            "Isotonic_Gain_RMSE": pivot_mean_iso.loc[p1, p2],
                            "Raw_Gain_RMSE_Std": pivot_std.loc[p1, p2]
                        })

                # 4. Generate the Heatmap
                v_min = pivot_mean_iso.values.min()
                v_max = pivot_mean_iso.values.max()
                if v_min >= 0: v_min = -1e-2
                if v_max <= 0: v_max = 1e-2
                norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)

                annot_iso = np.empty(pivot_mean_iso.shape, dtype=object)
                for i in range(pivot_mean_iso.shape[0]):
                    for j in range(pivot_mean_iso.shape[1]):
                        annot_iso[i, j] = f"{pivot_mean_iso.iloc[i, j]:.3f}\n±{pivot_std.iloc[i, j]:.3f}"

                plt.figure(figsize=(9, 7))
                sns.heatmap(pivot_mean_iso, annot=annot_iso, fmt="", cmap=rwg_cmap, norm=norm, cbar_kws={'label': 'Smoothed Accuracy Gain'})

                clean_scenario = str(scenario).upper()
                clean_client = CLIENT_MAP.get(str(scenario), {}).get(client, f"Client {client}")
                
                plt.title(f"ISOTONIC RMSE Gain - {clean_scenario} - {clean_client} ({mech.upper()})\n(Mean ± STD over Seeds)")
                plt.xlabel(f"P2 Privacy Parameter ({mech.upper()})")
                plt.ylabel(f"P1 Privacy Parameter ({mech.upper()})")

                plot_filename = os.path.join(mech_plot_dir, f"{clean_scenario}_{clean_client}_{mech}.png")
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()

    # Save the aggregated smoothed dataframe
    if smoothed_records:
        pd.DataFrame(smoothed_records).to_csv(out_csv_path, index=False)
        return out_csv_path, plot_iso_dir

def main():
    parser = argparse.ArgumentParser(description="Apply isotonic regression to RS experiment RMSE gains.")
    parser.add_argument("--root", type=str, default="Results", help="Results directory to scan for raw csv folders.")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    targets = find_target_csvs(root_dir)

    if not targets:
        print("No target files found. Expected CSVs inside Results/**/csv folders.")
        return

    print(f"===========================================================")
    print(f" 🚀 FOUND {len(targets)} HISTORICAL CSV FILES")
    print(f"===========================================================")

    succeeded = 0
    for input_csv in targets:
        print(f"\nProcessing: {input_csv}")
        try:
            out_csv, out_plot_dir = process_historical_csv(input_csv)
            print(f"  ✅ SUCCESS: Mathematical smoothing applied.")
            print(f"  📁 Data saved to: {out_csv}")
            print(f"  📊 Plots saved to: {out_plot_dir}")
            succeeded += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

    print(f"\n===========================================================")
    print(f" 🎉 OPERATION COMPLETE: {succeeded}/{len(targets)} files successfully processed.")
    print(f"===========================================================")

if __name__ == "__main__":
    main()