import re
from pathlib import Path

import numpy as np
import pandas as pd


# Replace only the value on this line when evaluating another set of weights.
WEIGHTS_STRING = "x=0.732183, y=-3.528741e-07, alpha=0.603868, beta=0.472478, gamma=0.472746"

DATA_DIRECTORIES = ("NF_1", "NF_5")
REPORT_PATH = Path(__file__).with_name("Weight_Evaluation_Report.txt")


def parse_weights(weights_string):
    names = ("x", "y", "alpha", "beta", "gamma")
    values = {}
    for name in names:
        match = re.search(
            rf"\b{name}\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            weights_string,
        )
        if match is None:
            raise ValueError(f"Missing weight: {name}")
        values[name] = float(match.group(1))
    return values


def load_dataset(directory, scenario, mechanism):
    path = Path(__file__).with_name(directory) / f"{scenario}_{mechanism}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def calculate_rmse(real_data, simulation_data, weights):
    merge_columns = ["Mechanism", "Samples", "P_own", "P_other"]
    merged = pd.merge(
        real_data,
        simulation_data,
        on=merge_columns,
        suffixes=("_real", "_sim"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matching real/simulation rows were found.")

    samples = merged["Samples"].to_numpy(dtype=float)
    p_own = merged["P_own"].to_numpy(dtype=float)
    p_other = merged["P_other"].to_numpy(dtype=float)
    real_gain = merged["Gain_Acc_real"].to_numpy(dtype=float)
    simulation_gain = merged["Gain_Acc_sim"].to_numpy(dtype=float)

    x = weights["x"]
    y = weights["y"]
    alpha = weights["alpha"]
    beta = weights["beta"]
    gamma = weights["gamma"]
    transformed_gain = (
        (x + y * samples)
        * (alpha + beta * p_own + gamma * p_other)
        * simulation_gain
    )

    raw_rmse = np.sqrt(np.mean((real_gain - simulation_gain) ** 2))
    transformed_rmse = np.sqrt(np.mean((real_gain - transformed_gain) ** 2))
    return raw_rmse, transformed_rmse


def evaluate_weights():
    weights = parse_weights(WEIGHTS_STRING)
    results = {}

    for directory in DATA_DIRECTORIES:
        real_data = []
        simulation_data = []
        for mechanism in ("dp", "sup"):
            real_data.append(load_dataset(directory, "real", mechanism))
            simulation_data.append(load_dataset(directory, "sim", mechanism))

        raw_rmse, transformed_rmse = calculate_rmse(
            pd.concat(real_data, ignore_index=True),
            pd.concat(simulation_data, ignore_index=True),
            weights,
        )
        results[directory] = {"raw": raw_rmse, "transformed": transformed_rmse}

    all_real = pd.concat(
        [load_dataset(directory, "real", mechanism)
         for directory in DATA_DIRECTORIES
         for mechanism in ("dp", "sup")],
        ignore_index=True,
    )
    all_simulation = pd.concat(
        [load_dataset(directory, "sim", mechanism)
         for directory in DATA_DIRECTORIES
         for mechanism in ("dp", "sup")],
        ignore_index=True,
    )
    overall_raw, overall_transformed = calculate_rmse(
        all_real, all_simulation, weights
    )

    lines = [
        "WEIGHTED RMSE EVALUATION",
        "=" * 60,
        f"Weights: {WEIGHTS_STRING}",
        "",
        "1. Overall RMSE (Without Function)",
        f"{overall_raw:.6f}",
        "",
        "2. Overall RMSE (With Weights)",
        f"{overall_transformed:.6f}",
        "",
        "3. RMSE by Dataset Size",
        "",
    ]
    for directory in DATA_DIRECTORIES:
        result = results[directory]
        lines.extend([
            f"{directory}:",
            f"  Raw RMSE (Without Function): {result['raw']:.6f}",
            f"  Transformed RMSE (With Weights): {result['transformed']:.6f}",
            "",
        ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_weights()
