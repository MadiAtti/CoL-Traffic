"""Build payoff-gain bimatrices from per-seed federated experiment logs.

The full-participant game uses players P1 and P2. The two self-division games
use P11/P12 and P21/P22, respectively. Accuracy gain is normalized by the
local model's remaining accuracy headroom; loss gain is relative to the local
loss.

Each output bimatrix has shape ``(n + 1, n + 1, 2)``. Its row and column axes
represent the first and second player's configured privacy parameters, and
the final index represents the separately measured maximum-privacy echo
scenario. Layer zero contains the first player's gain and layer one contains
the second player's gain. Missing scenarios remain ``NaN``.

Inputs are the local-baseline, suppression, noise, and maximum-privacy JSON
logs under ``results/<mode>/``. The outputs include per-seed bimatrices,
seed-averaged game bimatrices, and real/predicted matrices with standard
deviations under ``results/<mode>/<analyze>/games/``.

Aggregation order (as in the paper): raw surfaces are first averaged across
seeds, each player's averaged surface is then monotonized independently in
its own [own, other] orientation, and finally the players' monotonized
surfaces are averaged.

Gains are stored signed, as in RS and DD. The benefit is capped at zero only
inside the game utility, u_n = max(Delta_n, 0) - C_n (1 - p_n) (paper Sec. 7),
by ``GT/evaluate_game.to_clipped_gain``.
"""

import json
import os
from itertools import product

import numpy as np
from omegaconf import OmegaConf
from sklearn.isotonic import IsotonicRegression

analyze = "accuracy"  # "accuracy" or "loss"

NUM_SEEDS = 10

# (sub-folder of the input files, scenario name)
SCENARIO_FOLDERS = [
    ("", "Full_FL"),
    ("P1/", "P1_Subnet"),
    ("P2/", "P2_Subnet"),
]

# (input folder of the federated runs, method name)
METHODS = [
    ("2_suppression/", "Suppression"),
    ("3_noise/", "Noise"),
]

# Folder of the "max privacy" experiments for each method
MAX_PRIVACY_PATHS = {
    "Suppression": "4_max_privacy_suppression/",
    "Noise": "4_max_privacy_noise/",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_json(path):
    """Load a JSON file; return None (and print a message) if it does not exist."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def compute_gain(metric, baseline):
    """Signed gain of a federated result relative to the local baseline (see module docstring)."""
    if not np.isfinite(metric) or not np.isfinite(baseline):
        raise ValueError("Gain requires finite federated and baseline metrics.")
    if analyze == "accuracy":
        denominator = 1 - baseline
    else:
        denominator = baseline
    if denominator == 0:
        raise ValueError(f"Gain is undefined for the {analyze} baseline {baseline}.")
    gain = (metric - baseline) / denominator
    return gain


def monotonize_matrix(matrix):
    """Make every row and column non-increasing with isotonic regression."""
    result = np.asarray(matrix, dtype=float).copy()
    ir = IsotonicRegression(increasing=False)
    for i in range(result.shape[0]):
        result[i, :] = ir.fit_transform(
            np.arange(result.shape[1]), result[i, :]
        )
    for j in range(result.shape[1]):
        result[:, j] = ir.fit_transform(
            np.arange(result.shape[0]), result[:, j]
        )
    return result


def get_metric_from_local(data, sub_path=""):
    """
    Read the local-baseline metric of the two players of the given scenario.

    Full_FL ("")  -> models M1/M2, players p1  / p2
    P1_Subnet     -> players p11 / p12
    P2_Subnet     -> players p21 / p22

    Returns:
        (metric of the first player, metric of the second player)
    """
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
    """
    Return the configured defence levels of a method / data size, together
    with the JSON keys that hold the two players' parameters.

    Returns:
        (levels, key of the first player's parameter, key of the second player's parameter)
    """
    if method_name == "Noise":
        level_map = {
            "full": "full_noise_levels",
            "half": "half_noise_levels",
            "80percent": "percent80_noise_levels",
            "30percent": "percent30_noise_levels",
        }
        if mode not in level_map:
            raise ValueError(f"Unsupported dataset mode for noise: {mode}")
        return config_data[level_map[mode]], "noise_p1", "noise_p2"
    if method_name == "Suppression":
        return config_data["sup_levels"], "features_p1", "features_p2"
    raise ValueError(f"Unsupported privacy method: {method_name}")


def save_matrices(output_base_path, m1_diff, m2_diff, params, method_name, sc_name, seed):
    """Stack the two players' gain matrices into a (n, n, 2) bimatrix and save it for one seed."""
    os.makedirs(f"{output_base_path}games/seed{seed}", exist_ok=True)

    bimatrix = np.zeros((len(params), len(params), 2))
    bimatrix[:, :, 0] = m1_diff   # first player's gain
    bimatrix[:, :, 1] = m2_diff   # second player's gain

    save_path = f"{output_base_path}games/seed{seed}/{method_name}_{sc_name}_bimatrix.npy"
    np.save(save_path, bimatrix)
    print(f"Matrix saved: {save_path}")


def extend_with_max_privacy(core, row, col, corner):
    """
    Embed the (n, n) gain matrix `core` into an (n + 1, n + 1) matrix and fill
    the extra last row / column / corner with the max-privacy results.

        last row    -> first player at max privacy, second player at level j
        last column -> second player at max privacy, first player at level i
        corner      -> both players at max privacy
    """
    n = core.shape[0]
    ext = np.full((n + 1, n + 1), np.nan)
    ext[:n, :n] = core
    ext[n, :n] = row
    ext[:n, n] = col
    ext[n, n] = corner
    return ext


# --------------------------------------------------------------------------- #
# Per-seed matrix construction
# --------------------------------------------------------------------------- #
def process_and_save(seed, mode, input_base_path, output_base_path):
    """
    For one seed, build and save the bimatrix of every (method, scenario)
    combination: the regular (n x n) grid from the federated runs, extended by
    the max-privacy row / column / corner.
    """
    cfg_data = OmegaConf.load("conf/base.yaml").config

    for method_path, method_name in METHODS:
        params, k1, k2 = get_params_for_method(method_name, mode, cfg_data)
        size = len(params)

        for sub_path, sc_name in SCENARIO_FOLDERS:
            # ---- Local baseline of the two players (reference for the gain) ----
            loc_data = load_json(f"{input_base_path}1_local_baseline/{sub_path}{seed}.json")
            if not loc_data:
                continue
            b_p1, b_p2 = get_metric_from_local(loc_data, sub_path)

            # ---- Federated runs on the regular parameter grid ----
            fed_data = load_json(f"{input_base_path}{method_path}{sub_path}{seed}.json")
            if not fed_data:
                continue

            m1_diff = np.full((size, size), np.nan)
            m2_diff = np.full((size, size), np.nan)

            for exp in fed_data["experiments"]:
                val1, val2 = exp[k1], exp[k2]
                try:
                    idx1 = params.index(val1)
                    idx2 = params.index(val2)
                except ValueError:
                    continue  # parameter pair is not on the configured grid

                m1_diff[idx1][idx2] = compute_gain(exp["final_evaluation"]["P1"][analyze], b_p1)
                m2_diff[idx1][idx2] = compute_gain(exp["final_evaluation"]["P2"][analyze], b_p2)

            # ---- Max-privacy experiments ("echo" = maximum privacy) ----
            mp_data = load_json(
                f"{input_base_path}{MAX_PRIVACY_PATHS[method_name]}{sub_path}{seed}.json"
            )

            # row = first player at max privacy, col = second player at max privacy
            mp_row_m1 = np.full(size, np.nan)
            mp_col_m1 = np.full(size, np.nan)
            mp_row_m2 = np.full(size, np.nan)
            mp_col_m2 = np.full(size, np.nan)
            mp_corner_m1, mp_corner_m2 = np.nan, np.nan

            if mp_data:
                for exp in mp_data["experiments"]:
                    val1, val2 = exp[k1], exp[k2]
                    d1 = compute_gain(exp["final_evaluation"]["P1"][analyze], b_p1)
                    d2 = compute_gain(exp["final_evaluation"]["P2"][analyze], b_p2)

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

            m1_ext = extend_with_max_privacy(m1_diff, mp_row_m1, mp_col_m1, mp_corner_m1)
            m2_ext = extend_with_max_privacy(m2_diff, mp_row_m2, mp_col_m2, mp_corner_m2)

            save_matrices(output_base_path, m1_ext, m2_ext, params + [None],
                          method_name, sc_name, seed)


# --------------------------------------------------------------------------- #
# Averaging over seeds
# --------------------------------------------------------------------------- #
def save_average_bimatrix(output_base_path, method, scenario, first, second):
    """
    Save the seed-averaged bimatrix of one game (game orientation, shape (n+1, n+1, 2)).
    This is what the Nash search loads.
    """
    out_dir = f"{output_base_path}games/average"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/{method}_{scenario}_Bimatrix_Avg.npy",
            np.stack([first, second], axis=-1))


def _seed_mean(stack, label):
    """Average raw per-seed surfaces; refuse to continue if any cell is missing."""
    mean = np.nanmean(np.stack(stack, axis=0), axis=0)
    if not np.isfinite(mean).all():
        raise ValueError(f"Seed-averaged {label} surface contains non-finite cells.")
    return mean


def average_Full(output_base_path, seeds=None):
    """
    Real game (Full_FL), following the paper's aggregation order:

      1. average the raw per-seed gain surfaces of each player,
      2. monotonize each player's averaged surface independently in its own
         [own, other] orientation,
      3. average the two players' monotonized surfaces.

    Saves
      * the averaged, player-wise monotonized bimatrix in game orientation
        (row = P1 choice, column = P2 choice) for the Nash search,
      * the symmetrised matrix (M(P1) + M(P2_own)) / 2 in [own, other]
        orientation for fitting and heatmaps, and
      * the standard deviation over seeds of the raw symmetrised surface
        (P1 + P2_own) / 2.
    """
    seeds = tuple(range(NUM_SEEDS) if seeds is None else seeds)
    for method in ["Suppression", "Noise"]:
        raw_P1, raw_P2 = [], []   # both in game orientation [P1 choice, P2 choice]

        for seed in seeds:
            path = f"{output_base_path}games/seed{seed}/{method}_Full_FL_bimatrix.npy"
            if not os.path.exists(path):
                print(f"Missing bimatrix for {method} (Full_FL) - Seed {seed}, skipping...")
                continue
            bm = np.load(path)
            raw_P1.append(bm[:, :, 0])
            raw_P2.append(bm[:, :, 1])

        if not raw_P1:
            continue

        # Step 1: average raw surfaces across seeds.
        P1_avg = _seed_mean(raw_P1, f"{method} Full_FL P1")
        P2_avg = _seed_mean(raw_P2, f"{method} Full_FL P2")

        # Step 2: monotonize each player in its own [own, other] orientation.
        P1_own = monotonize_matrix(P1_avg)
        P2_own = monotonize_matrix(P2_avg.T)

        # Step 3: average the two players' monotonized surfaces.
        final_real = (P1_own + P2_own) / 2
        raw_symmetrised = [(p1 + p2.T) / 2 for p1, p2 in zip(raw_P1, raw_P2)]
        final_real_std = np.nanstd(np.stack(raw_symmetrised, axis=0), axis=0)

        os.makedirs(f"{output_base_path}games/average", exist_ok=True)
        np.save(f"{output_base_path}games/average/{method}_Final_Real.npy", final_real)
        np.save(f"{output_base_path}games/average/{method}_Final_Real_Std.npy", final_real_std)
        save_average_bimatrix(output_base_path, method, "Full_FL", P1_own, P2_own.T)
        print(f"Saved Real matrix & STD: {method}")


def average_SD(output_base_path, seeds=None):
    """
    Simulated (self-division) games, following the paper's aggregation order:

      1. average the raw per-seed gain surfaces of p11, p12, p21 and p22,
      2. monotonize each player's averaged surface independently in its own
         [own, other] orientation,
      3. average the four monotonized player surfaces.

    Saves
      * the averaged bimatrices of the two games (P1_Subnet = (p11, p12),
        P2_Subnet = (p21, p22)) in game orientation for the Nash search, and
      * the simulated matrix (mean of the 4 monotonized player surfaces) and
        its standard deviation across those 4 surfaces for the heatmaps.
    """
    seeds = tuple(range(NUM_SEEDS) if seeds is None else seeds)
    for method in ["Suppression", "Noise"]:
        raw = {"P11": [], "P12": [], "P21": [], "P22": []}

        for seed in seeds:
            for scenario, first, second in (
                ("P1_Subnet", "P11", "P12"),
                ("P2_Subnet", "P21", "P22"),
            ):
                path = f"{output_base_path}games/seed{seed}/{method}_{scenario}_bimatrix.npy"
                if os.path.exists(path):
                    bm = np.load(path)
                    raw[first].append(bm[:, :, 0])
                    raw[second].append(bm[:, :, 1])

        if not all(raw.values()):
            continue

        # Step 1: average raw surfaces across seeds (game orientation).
        avg = {name: _seed_mean(stack, f"{method} {name}") for name, stack in raw.items()}

        # Step 2: monotonize each player in its own [own, other] orientation.
        v1 = monotonize_matrix(avg["P11"])
        v2 = monotonize_matrix(avg["P12"].T)
        v3 = monotonize_matrix(avg["P21"])
        v4 = monotonize_matrix(avg["P22"].T)

        # Step 3: average the monotonized player surfaces.
        variants = np.stack([v1, v2, v3, v4], axis=0)
        final_pred = np.nanmean(variants, axis=0)
        final_pred_std = np.nanstd(variants, axis=0)

        os.makedirs(f"{output_base_path}games/average", exist_ok=True)
        np.save(f"{output_base_path}games/average/{method}_Final_Pred.npy", final_pred)
        np.save(f"{output_base_path}games/average/{method}_Final_Pred_Std.npy", final_pred_std)
        save_average_bimatrix(output_base_path, method, "P1_Subnet", v1, v2.T)
        save_average_bimatrix(output_base_path, method, "P2_Subnet", v3, v4.T)
        print(f"Saved Pred matrix & STD: {method}")


def build_matrices_for_mode(mode, seeds=None):
    """Build per-seed and averaged game matrices for one completed dataset mode."""
    supported_modes = {"full", "half", "80percent", "30percent"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported dataset mode: {mode}")

    seeds = list(range(NUM_SEEDS) if seeds is None else seeds)
    if not seeds:
        raise ValueError("At least one seed is required to build game matrices.")

    input_base_path = f"results/{mode}/"
    output_base_path = f"results/{mode}/{analyze}/"
    required_paths = [
        f"{input_base_path}1_local_baseline/{sub_path}{seed}.json"
        for seed in seeds
        for sub_path, _ in SCENARIO_FOLDERS
    ]
    required_paths.extend(
        f"{input_base_path}{folder}{sub_path}{seed}.json"
        for seed in seeds
        for sub_path, _ in SCENARIO_FOLDERS
        for folder in (
            "2_suppression/",
            "3_noise/",
            MAX_PRIVACY_PATHS["Suppression"],
            MAX_PRIVACY_PATHS["Noise"],
        )
    )
    missing = [path for path in required_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Cannot build complete game matrices; required experiment logs are "
            f"missing: {missing[:10]}"
        )

    config_data = OmegaConf.load("conf/base.yaml").config
    for seed in seeds:
        for sub_path, _ in SCENARIO_FOLDERS:
            for _, method_name in METHODS:
                levels, key1, key2 = get_params_for_method(
                    method_name, mode, config_data
                )
                expected_grid = set(product(levels, repeat=2))
                expected_max_privacy = {
                    ("echo", level) for level in levels
                } | {
                    (level, "echo") for level in levels
                } | {("echo", "echo")}

                regular_folder = next(
                    folder for folder, name in METHODS if name == method_name
                )
                for folder, expected in (
                    (regular_folder, expected_grid),
                    (MAX_PRIVACY_PATHS[method_name], expected_max_privacy),
                ):
                    path = f"{input_base_path}{folder}{sub_path}{seed}.json"
                    with open(path, "r", encoding="utf-8") as f:
                        experiments = json.load(f).get("experiments")
                    if not isinstance(experiments, list):
                        raise ValueError(f"Invalid experiment log structure: {path}")
                    actual = {
                        (entry.get(key1), entry.get(key2))
                        for entry in experiments
                    }
                    if actual != expected or len(experiments) != len(expected):
                        raise ValueError(
                            f"Incomplete or duplicate scenario grid in {path}: "
                            f"expected {len(expected)}, found {len(experiments)}."
                        )

    for seed in seeds:
        process_and_save(seed, mode, input_base_path, output_base_path)

    average_Full(output_base_path, seeds=seeds)
    average_SD(output_base_path, seeds=seeds)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build per-seed and averaged game matrices.")
    parser.add_argument("--modes", nargs="+", default=["full", "half", "80percent", "30percent"])
    args = parser.parse_args()
    for mode in args.modes:
        build_matrices_for_mode(mode)