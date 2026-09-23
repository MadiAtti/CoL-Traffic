import json
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


analyze = "accuracy"  # "accuracy" or "loss"

# --- Shared visual style ---
RWG_CMAP = LinearSegmentedColormap.from_list("RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"])
HEATMAP_FONT = {
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}


def _make_norm(matrix: np.ndarray) -> TwoSlopeNorm:
    v_min = float(np.nanmin(matrix))
    v_max = float(np.nanmax(matrix))
    if v_min >= 0:
        v_min = -1e-2
    if v_max <= 0:
        v_max = 1e-2
    return TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)


def _save_heatmap(matrix: np.ndarray, title: str, xlabel: str, ylabel: str,
                  save_path: str, tick_labels=None) -> None:
    norm = _make_norm(matrix)
    with plt.rc_context(HEATMAP_FONT):
        plt.figure(figsize=(12, 9))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap=RWG_CMAP,
            norm=norm,
            annot_kws={"size": 12},
            xticklabels=tick_labels if tick_labels is not None else "auto",
            yticklabels=tick_labels if tick_labels is not None else "auto",
            cbar_kws={"label": get_title_metric_name()},
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Saved: {save_path}")


# ------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_metric_from_local(data, sub_path=""):
    eval_m1 = data["models"]["M1"]["evaluation"]
    eval_m2 = data["models"]["M2"]["evaluation"]

    if sub_path == "P1/":
        p1_metric = eval_m1["p11"][analyze]
        p2_metric = eval_m2["p12"][analyze]
    elif sub_path == "P2/":
        p1_metric = eval_m1["p21"][analyze]
        p2_metric = eval_m2["p22"][analyze]
    else:
        p1_metric = eval_m1["p1"][analyze]
        p2_metric = eval_m2["p2"][analyze]

    print(f"Local Baseline - P1: {p1_metric:.4f}, P2: {p2_metric:.4f}")
    return p1_metric, p2_metric


def get_title_metric_name():
    if analyze == "accuracy":
        return "Relative Accuracy Improvement"
    return "Relative Loss Change"


def save_matrices(m1_diff, m2_diff, params, method_name, sc_name, seed):
    os.makedirs(f"{output_base_path}games/seed{seed}", exist_ok=True)

    bimatrix = np.zeros((len(params), len(params), 2))
    bimatrix[:, :, 0] = m1_diff
    bimatrix[:, :, 1] = m2_diff

    save_path = f"{output_base_path}games/seed{seed}/{method_name}_{sc_name}_bimatrix.npy"
    np.save(save_path, bimatrix)
    print(f"Matrix saved: {save_path}")


def process_and_plot(seed):
    folders = [
        ("", "Full_FL"),
        ("P1/", "P1_Subnet"),
        ("P2/", "P2_Subnet"),
    ]
    methods = [
        ("2_suppression/", "Suppression"),
        ("3_noise/", "Noise"),
    ]

    for method_path, method_name in methods:
        for sub_path, sc_name in folders:
            loc_path = f"{input_base_path}1_local_baseline/{sub_path}{seed}.json"
            loc_data = load_json(loc_path)
            if not loc_data:
                continue

            b_p1, b_p2 = get_metric_from_local(loc_data, sub_path)

            fed_path = f"{input_base_path}{method_path}{sub_path}{seed}.json"
            fed_data = load_json(fed_path)
            if not fed_data:
                continue

            config_data = fed_data["parameters"]["config"]

            if method_name == "Noise":
                if mode == "full":
                    level = "full_noise_levels"
                elif mode == "half":
                    level = "half_noise_levels"
                elif mode == "80percent":
                    level = "80percent_noise_levels"
                elif mode == "30percent":
                    level = "30percent_noise_levels"
                
                params = config_data[level]
                k1, k2 = "noise_p1", "noise_p2"
            else:
                params = config_data["sup_levels"]
                k1, k2 = "features_p1", "features_p2"

            experiments = fed_data["experiments"]
            size = len(params)

            m1_diff = np.zeros((size, size))
            m2_diff = np.zeros((size, size))

            for exp in experiments:
                val1 = exp[k1]
                val2 = exp[k2]
                try:
                    idx1 = params.index(val1)
                    idx2 = params.index(val2)
                except ValueError:
                    continue

                p1_metric = exp["final_evaluation"]["P1"][analyze]
                p2_metric = exp["final_evaluation"]["P2"][analyze]

                if analyze == "accuracy":
                    m1_diff[idx1][idx2] = (p1_metric - b_p1) / (1 - b_p1)
                    m2_diff[idx1][idx2] = (p2_metric - b_p2) / (1 - b_p2)
                else:
                    m1_diff[idx1][idx2] = (p1_metric - b_p1) / b_p1
                    m2_diff[idx1][idx2] = (p2_metric - b_p2) / b_p2

            save_matrices(m1_diff, m2_diff, params, method_name, sc_name, seed)

            display_params = [0.0 if x is None else x for x in params]
            os.makedirs(f"{output_base_path}plots/seed{seed}", exist_ok=True)

            for p_tag, matrix in [("P1", m1_diff), ("P2", m2_diff)]:
                _save_heatmap(
                    matrix=matrix,
                    title=f"{method_name} ({sc_name}) - {p_tag} {get_title_metric_name()} (Seed {seed})",
                    xlabel="Client 2 Params",
                    ylabel="Client 1 Params",
                    save_path=f"{output_base_path}plots/seed{seed}/{method_name}_{sc_name}_{p_tag}.png",
                    tick_labels=display_params,
                )


def average_plot_Full():
    for method in ["Suppression", "Noise"]:
        all_P1, all_P2 = [], []

        for seed in range(0, 10):
            path = f"{output_base_path}games/seed{seed}/{method}_Full_FL_bimatrix.npy"
            if not os.path.exists(path):
                print(f"Missing bimatrix for {method} (Full_FL) - Seed {seed}, skipping...")
                continue
            bm = np.load(path)
            all_P1.append(bm[:, :, 0])
            all_P2.append(bm[:, :, 1])

        if not all_P1 or not all_P2:
            continue

        P1 = np.mean(all_P1, axis=0)
        P2 = np.mean(all_P2, axis=0)
        final_real = (P1 + P2.T) / 2

        os.makedirs(f"{output_base_path}games/average", exist_ok=True)
        np.save(f"{output_base_path}games/average/{method}_Final_Real.npy", final_real)
        print(f"Average Full FL final matrix saved for {method}")
        print(f"  range: [{final_real.min():.4f}, {final_real.max():.4f}]")

        os.makedirs(f"{output_base_path}plots/average", exist_ok=True)
        _save_heatmap(
            matrix=final_real,
            title=f"Average {method} (Full_FL) - {get_title_metric_name()}",
            xlabel="Client 2 Params",
            ylabel="Client 1 Params",
            save_path=f"{output_base_path}games/average/{method}_Final_Real.png",
        )


def average_plot_SD():
    for method in ["Suppression", "Noise"]:
        all_P11, all_P12 = [], []
        all_P21, all_P22 = [], []

        for seed in range(0, 10):
            p1_path = f"{output_base_path}games/seed{seed}/{method}_P1_Subnet_bimatrix.npy"
            p2_path = f"{output_base_path}games/seed{seed}/{method}_P2_Subnet_bimatrix.npy"

            if os.path.exists(p1_path):
                bm = np.load(p1_path)
                all_P11.append(bm[:, :, 0])
                all_P12.append(bm[:, :, 1])
            else:
                print(f"Missing: {p1_path}")

            if os.path.exists(p2_path):
                bm = np.load(p2_path)
                all_P21.append(bm[:, :, 0])
                all_P22.append(bm[:, :, 1])
            else:
                print(f"Missing: {p2_path}")

        if not all_P11 or not all_P12 or not all_P21 or not all_P22:
            print(f"Insufficient data for {method}, skipping...")
            continue

        P11 = np.mean(all_P11, axis=0)
        P12 = np.mean(all_P12, axis=0)
        P21 = np.mean(all_P21, axis=0)
        P22 = np.mean(all_P22, axis=0)

        P1_pred = (P11 + P12.T) / 2
        P2_pred = (P22 + P21.T) / 2
        final_pred = (P1_pred + P2_pred.T) / 2

        os.makedirs(f"{output_base_path}games/average", exist_ok=True)
        np.save(f"{output_base_path}games/average/{method}_Final_Pred.npy", final_pred)
        print(f"Average SD final matrix saved for {method}")
        print(f"  P1_pred range:    [{P1_pred.min():.4f}, {P1_pred.max():.4f}]")
        print(f"  P2_pred range:    [{P2_pred.min():.4f}, {P2_pred.max():.4f}]")
        print(f"  final_pred range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")

        os.makedirs(f"{output_base_path}plots/average", exist_ok=True)
        _save_heatmap(
            matrix=final_pred,
            title=f"Average {method} (SD) - {get_title_metric_name()}",
            xlabel="Client 2 Params",
            ylabel="Client 1 Params",
            save_path=f"{output_base_path}games/average/{method}_Final_Pred.png",
        )


if __name__ == "__main__":
    mode = "full"  # "full", "half", or "quarter"

    input_base_path = f"results/{mode}/"
    output_base_path = f"results/{mode}/{analyze}/"

    for seed in range(0, 10):
        local = f"{input_base_path}1_local_baseline/P1/{seed}.json"

        if not os.path.exists(local):
            print(f"Missing results for seed {seed}, skipping...")
            continue

        process_and_plot(seed)

    average_plot_Full()
    average_plot_SD()