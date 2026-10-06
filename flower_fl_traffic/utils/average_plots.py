import json
import os

import numpy as np
from omegaconf import OmegaConf

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


def get_params_for_method(method_name, mode, config_data):
    if method_name == "Noise":
        level_map = {
            "full": "full_noise_levels",
            "half": "half_noise_levels",
            "80percent": "80percent_noise_levels",
            "30percent": "30percent_noise_levels",
        }
        level = level_map.get(mode, "full_noise_levels")
        return config_data[level], "noise_p1", "noise_p2"
    else:
        return config_data["sup_levels"], "features_p1", "features_p2"


def save_matrices(output_base_path, m1_diff, m2_diff, params, method_name, sc_name, seed):
    os.makedirs(f"{output_base_path}games/seed{seed}", exist_ok=True)

    bimatrix = np.zeros((len(params), len(params), 2))
    bimatrix[:, :, 0] = m1_diff
    bimatrix[:, :, 1] = m2_diff

    save_path = f"{output_base_path}games/seed{seed}/{method_name}_{sc_name}_bimatrix.npy"
    np.save(save_path, bimatrix)
    print(f"Matrix saved: {save_path}")


def process_and_save(seed, mode, input_base_path, output_base_path):
    folders = [
        ("", "Full_FL"),
        ("P1/", "P1_Subnet"),
        ("P2/", "P2_Subnet"),
    ]
    methods = [
        ("2_suppression/", "Suppression"),
        ("3_noise/", "Noise"),
    ]
    max_privacy_paths = {
        "Suppression": "4_max_privacy_suppression/",
        "Noise": "4_max_privacy_noise/",
    }

    base_cfg = OmegaConf.load("conf/base.yaml")
    config_data = base_cfg.config

    for method_path, method_name in methods:
        params, k1, k2 = get_params_for_method(method_name, mode, config_data)
        size = len(params)

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

            experiments = fed_data["experiments"]

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

            # --- Max privacy ---
            mp_path = f"{input_base_path}{max_privacy_paths[method_name]}{sub_path}{seed}.json"
            mp_data = load_json(mp_path)

            mp_row_m1 = np.full(size, np.nan)
            mp_col_m1 = np.full(size, np.nan)
            mp_row_m2 = np.full(size, np.nan)
            mp_col_m2 = np.full(size, np.nan)
            mp_corner_m1, mp_corner_m2 = np.nan, np.nan

            if mp_data:
                for exp in mp_data["experiments"]:
                    val1 = exp[k1]
                    val2 = exp[k2]
                    p1_metric = exp["final_evaluation"]["P1"][analyze]
                    p2_metric = exp["final_evaluation"]["P2"][analyze]

                    if analyze == "accuracy":
                        d1 = (p1_metric - b_p1) / (1 - b_p1)
                        d2 = (p2_metric - b_p2) / (1 - b_p2)
                    else:
                        d1 = (p1_metric - b_p1) / b_p1
                        d2 = (p2_metric - b_p2) / b_p2

                    if val1 == "echo" and val2 in params:
                        j = params.index(val2)
                        mp_row_m1[j] = d1
                        mp_row_m2[j] = d2
                    elif val2 == "echo" and val1 in params:
                        i = params.index(val1)
                        mp_col_m1[i] = d1
                        mp_col_m2[i] = d2
                    elif val1 == "echo" and val2 == "echo":
                        mp_corner_m1 = d1
                        mp_corner_m2 = d2

            m1_ext = np.full((size + 1, size + 1), np.nan)
            m1_ext[:size, :size] = m1_diff
            m1_ext[size,  :size] = mp_row_m1
            m1_ext[:size,  size] = mp_col_m1
            m1_ext[size,  size] = mp_corner_m1

            m2_ext = np.full((size + 1, size + 1), np.nan)
            m2_ext[:size, :size] = m2_diff
            m2_ext[size,  :size] = mp_row_m2
            m2_ext[:size,  size] = mp_col_m2
            m2_ext[size,  size] = mp_corner_m2

            save_matrices(output_base_path, m1_ext, m2_ext, params + [None], method_name, sc_name, seed)


def average_Full(output_base_path):
    for method in ["Suppression", "Noise"]:
        all_P1, all_P2 = [], []
        all_real_seeds = []

        for seed in range(0, 10):
            path = f"{output_base_path}games/seed{seed}/{method}_Full_FL_bimatrix.npy"
            if not os.path.exists(path):
                print(f"Missing bimatrix for {method} (Full_FL) - Seed {seed}, skipping...")
                continue
            bm = np.load(path)
            p1_seed = bm[:, :, 0]
            p2_seed = bm[:, :, 1]

            all_P1.append(p1_seed)
            all_P2.append(p2_seed)
            all_real_seeds.append((p1_seed + p2_seed.T) / 2)

        if not all_P1 or not all_P2:
            continue

        # Átlag és szórás kiszámítása a seedek között
        P1 = np.nanmean(all_P1, axis=0)
        P2 = np.nanmean(all_P2, axis=0)
        final_real = (P1 + P2.T) / 2
        final_real_std = np.nanstd(all_real_seeds, axis=0)

        os.makedirs(f"{output_base_path}games/average", exist_ok=True)
        np.save(f"{output_base_path}games/average/{method}_Final_Real.npy", final_real)
        np.save(f"{output_base_path}games/average/{method}_Final_Real_Std.npy", final_real_std)
        print(f"Saved Real matrix & STD: {method}")


def average_SD(output_base_path):
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

            if os.path.exists(p2_path):
                bm = np.load(p2_path)
                all_P21.append(bm[:, :, 0])
                all_P22.append(bm[:, :, 1])

        if not all_P11 or not all_P12 or not all_P21 or not all_P22:
            continue

        # Átlagolás a seedek felett
        P11 = np.nanmean(all_P11, axis=0)
        P12 = np.nanmean(all_P12, axis=0)
        P21 = np.nanmean(all_P21, axis=0)
        P22 = np.nanmean(all_P22, axis=0)

        # 4 variáns előállítása a szórás kinyeréséhez
        v1 = P11
        v2 = P12.T
        v3 = P22
        v4 = P21.T

        variants = np.stack([v1, v2, v3, v4], axis=0)

        final_pred = np.nanmean(variants, axis=0)
        final_pred_std = np.nanstd(variants, axis=0)

        os.makedirs(f"{output_base_path}games/average", exist_ok=True)
        np.save(f"{output_base_path}games/average/{method}_Final_Pred.npy", final_pred)
        np.save(f"{output_base_path}games/average/{method}_Final_Pred_Std.npy", final_pred_std)
        print(f"Saved Pred matrix & STD: {method}")


if __name__ == "__main__":
    for mode in ["full", "half"]: # "full", "half", "80percent", "30percent"

        input_base_path = f"results/{mode}/"
        output_base_path = f"results/{mode}/{analyze}/"

        for seed in range(0, 10):
            local = f"{input_base_path}1_local_baseline/P1/{seed}.json"

            if not os.path.exists(local):
                print(f"Missing results for seed {seed}, skipping...")
                continue

            process_and_save(seed, mode, input_base_path, output_base_path)

        average_Full(output_base_path)
        average_SD(output_base_path)