import json
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from evaluate_game import plot_heatmap


analyze = "accuracy"  # "accuracy" or "loss"


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

    title_metric = get_title_metric_name()

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
                params = config_data["noise_levels"]
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
                    # Positive: federated is better than local.
                    # Negative: federated is worse than local.
                    m1_diff[idx1][idx2] = (p1_metric - b_p1) / (1 - b_p1)
                    m2_diff[idx1][idx2] = (p2_metric - b_p2) / (1 - b_p2)
                else:
                    # Positive: federated loss is higher than local loss.
                    # Negative: federated loss is lower than local loss.
                    m1_diff[idx1][idx2] = (p1_metric - b_p1) / b_p1
                    m2_diff[idx1][idx2] = (p2_metric - b_p2) / b_p2

            save_matrices(m1_diff, m2_diff, params, method_name, sc_name, seed)

            display_params = [0.0 if x is None else x for x in params]

            for p_tag, matrix in [("P1", m1_diff), ("P2", m2_diff)]:
                plt.figure(figsize=(10, 8))

                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlGn",
                    center=0,
                    xticklabels=display_params,
                    yticklabels=display_params,
                )

                plt.title(f"{method_name} ({sc_name}) - {p_tag} {title_metric} (Seed {seed})")
                plt.xlabel("Client 2 Params")
                plt.ylabel("Client 1 Params")

                os.makedirs(f"{output_base_path}plots/seed{seed}", exist_ok=True)
                save_path = f"{output_base_path}plots/seed{seed}/{method_name}_{sc_name}_{p_tag}.png"

                plt.savefig(save_path)
                plt.close()

                print(f"Saved: {save_path}")


def average_plot_SD():
    title_metric = get_title_metric_name()

    for method in ["Suppression", "Noise"]:
        for sc_name in ["P1_Subnet", "P2_Subnet"]:
            all_diff = []

            for seed in range(0, 10):
                bimatrix_path = f"{output_base_path}games/seed{seed}/{method}_{sc_name}_bimatrix.npy"

                if not os.path.exists(bimatrix_path):
                    print(f"Missing bimatrix for {method} ({sc_name}) - Seed {seed}, skipping...")
                    print(f"Expected path: {bimatrix_path}")
                    continue

                bimatrix = np.load(bimatrix_path)

                all_diff.append(bimatrix[:, :, 0])
                all_diff.append(bimatrix[:, :, 1].transpose())

            if all_diff:
                diff = np.mean(all_diff, axis=0)

                os.makedirs(f"{output_base_path}games/average", exist_ok=True)
                save_matrix_path = f"{output_base_path}games/average/{method}_{sc_name}_bimatrix.npy"
                np.save(save_matrix_path, diff)

                print(f"Average SD matrix saved: {save_matrix_path}")

                plt.figure(figsize=(10, 8))
                sns.heatmap(diff, annot=True, fmt=".3f", cmap="RdYlGn", center=0)

                plt.title(f"Average {method} ({sc_name}) - {title_metric}")
                plt.xlabel("Client 2 Params")
                plt.ylabel("Client 1 Params")

                os.makedirs(f"{output_base_path}plots/average", exist_ok=True)
                save_plot_path = f"{output_base_path}plots/average/{method}_{sc_name}.png"

                plt.savefig(save_plot_path)
                plt.close()

                print(f"Saved: {save_plot_path}")


def average_plot_Full():
    title_metric = get_title_metric_name()

    for method in ["Suppression", "Noise"]:
        avg_m1_diff = []
        avg_m2_diff = []

        for seed in range(0, 10):
            bimatrix_path = f"{output_base_path}games/seed{seed}/{method}_Full_FL_bimatrix.npy"

            if not os.path.exists(bimatrix_path):
                print(f"Missing bimatrix for {method} (Full_FL) - Seed {seed}, skipping...")
                print(f"Expected path: {bimatrix_path}")
                continue

            bimatrix = np.load(bimatrix_path)

            avg_m1_diff.append(bimatrix[:, :, 0])
            avg_m2_diff.append(bimatrix[:, :, 1])

        if avg_m1_diff and avg_m2_diff:
            avg_m1_diff = np.mean(avg_m1_diff, axis=0)
            avg_m2_diff = np.mean(avg_m2_diff, axis=0)

            os.makedirs(f"{output_base_path}games/average", exist_ok=True)

            p1_save_path = f"{output_base_path}games/average/{method}_P1_Full_FL_bimatrix.npy"
            p2_save_path = f"{output_base_path}games/average/{method}_P2_Full_FL_bimatrix.npy"

            np.save(p1_save_path, avg_m1_diff)
            np.save(p2_save_path, avg_m2_diff)

            print(f"Average Full FL matrix saved: {p1_save_path}")
            print(f"Average Full FL matrix saved: {p2_save_path}")

            for p_tag, matrix in [("P1", avg_m1_diff), ("P2", avg_m2_diff)]:
                plt.figure(figsize=(10, 8))
                sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn", center=0)

                plt.title(f"Average {method} (Full_FL) - {p_tag} {title_metric}")
                plt.xlabel("Client 2 Params")
                plt.ylabel("Client 1 Params")

                os.makedirs(f"{output_base_path}plots/average", exist_ok=True)
                save_plot_path = f"{output_base_path}plots/average/{method}_Full_FL_{p_tag}_avg.png"

                plt.savefig(save_plot_path)
                plt.close()

                print(f"Saved: {save_plot_path}")


def get_diff_matrices():
    for method in ["Suppression", "Noise"]:
        p1_real_path = f"{output_base_path}games/average/{method}_P1_Full_FL_bimatrix.npy"
        p1_simulated_path = f"{output_base_path}games/average/{method}_P1_Subnet_bimatrix.npy"

        p2_real_path = f"{output_base_path}games/average/{method}_P2_Full_FL_bimatrix.npy"
        p2_simulated_path = f"{output_base_path}games/average/{method}_P2_Subnet_bimatrix.npy"

        required_paths = [
            p1_real_path,
            p1_simulated_path,
            p2_real_path,
            p2_simulated_path,
        ]

        missing_paths = [path for path in required_paths if not os.path.exists(path)]

        if missing_paths:
            print(f"Skipping diff matrices for {method}, missing files:")
            for path in missing_paths:
                print(f"  {path}")
            continue

        p1_real = np.load(p1_real_path)
        p1_simulated = np.load(p1_simulated_path)

        p2_real = np.load(p2_real_path)
        p2_simulated = np.load(p2_simulated_path)

        p1_diff = (p1_real - p1_simulated)
        p2_diff = (p2_real - p2_simulated)

        os.makedirs(f"{output_base_path}games/diff", exist_ok=True)

        p1_diff_path = f"{output_base_path}games/diff/{method}_P1_diff.npy"
        p2_diff_path = f"{output_base_path}games/diff/{method}_P2_diff.npy"

        np.save(p1_diff_path, p1_diff)
        np.save(p2_diff_path, p2_diff)

        os.makedirs(f"{output_base_path}plots/diff", exist_ok=True)

        p1_plot_path = f"{output_base_path}plots/diff/{method}_P1_diff.png"
        p2_plot_path = f"{output_base_path}plots/diff/{method}_P2_diff.png"

        plot_heatmap(
            p1_diff,
            f"{method} - P1 Real vs Simulated {analyze} Difference",
            p1_plot_path,
        )

        plot_heatmap(
            p2_diff,
            f"{method} - P2 Real vs Simulated {analyze} Difference",
            p2_plot_path,
        )

        print(f"Difference matrices saved for {method}")
        print(f"Saved: {p1_diff_path}")
        print(f"Saved: {p2_diff_path}")
        print(f"Saved: {p1_plot_path}")
        print(f"Saved: {p2_plot_path}")


if __name__ == "__main__":
    mode = "quarter"  # "full", "half", or "quarter"

    input_base_path = f"results/{mode}/"
    output_base_path = f"results/{mode}/{analyze}/"

    # for seed in range(0, 10):
    #     local = f"{input_base_path}1_local_baseline/P1/{seed}.json"

    #     if not os.path.exists(local):
    #         print(f"Missing results for seed {seed}, skipping...")
    #         continue

    #     process_and_plot(seed)

    # average_plot_Full()
    # average_plot_SD()
    get_diff_matrices()