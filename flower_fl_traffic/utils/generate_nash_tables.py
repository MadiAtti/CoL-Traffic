"""Generate original-format Nash comparison tables for measured and transformed games."""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import evaluate_game, nesh_visualisation
from utils.visualizer import apply_f, get_actual_grid, get_dataset_size_descriptor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
DATASET_SPLIT = "80-30"
CANDIDATES_PATH = Path(f"results/nash/c_candidates_{DATASET_SPLIT}.json")
SPLIT_MODES = {
    "100-50": ("half", "full"),
    "80-30": ("30percent", "80percent"),
}
METHODS = ("Noise", "Suppression")
SCENARIOS = ("P1_Subnet", "P2_Subnet")


def _load_selected_weights():
    """Load the selected symmetric C for each size/method pair."""
    with CANDIDATES_PATH.open(encoding="utf-8") as file:
        candidates = json.load(file)
    weights = {}
    for name, game in candidates["games"].items():
        candidate = game["candidates"][0]["C"]
        mode = game["mode"]
        method = game["method"]
        weights[(mode, method)] = (float(candidate), float(candidate))
    return weights


def _load_transformed_game(mode: str, method: str, scenario: str) -> tuple:
    """Load a bimatrix and apply the fitted correction in each player's orientation."""
    matrix_path = (
        f"results/{mode}/accuracy/games/average/"
        f"{method}_{scenario}_Bimatrix_Avg.npy"
    )
    bimatrix = np.load(matrix_path)
    if bimatrix.ndim != 3 or bimatrix.shape[2] != 2:
        raise ValueError(f"Expected a two-payoff bimatrix at {matrix_path}.")

    fit_split = "100-50" if mode in ("full", "half") else "80-30"
    params_path = Path(f"results/final_{fit_split}/{method}_params.json")
    with params_path.open(encoding="utf-8") as file:
        coefficients = json.load(file)["params"]
    params = [coefficients[name] for name in ("x", "y", "alpha", "beta")]
    grid = get_actual_grid(method, mode)
    expected_shape = (len(grid), len(grid))
    if bimatrix.shape[:2] != expected_shape:
        raise ValueError(
            f"Bimatrix shape {bimatrix.shape[:2]} does not match "
            f"{mode}/{method} grid shape {expected_shape}."
        )

    player_one = bimatrix[:, :, 0]
    player_two = bimatrix[:, :, 1].T
    player_one = apply_f(
        player_one, get_dataset_size_descriptor(mode), grid, params
    )
    player_two = apply_f(
        player_two, get_dataset_size_descriptor(mode), grid, params
    )
    player_one = evaluate_game.monotonize_matrix(player_one)
    player_two = evaluate_game.monotonize_matrix(player_two)
    return (
        evaluate_game.to_clipped_gain(player_one),
        evaluate_game.to_clipped_gain(player_two).T,
    )


def _make_table_data(mode: str, transformed: bool, selected_weights) -> dict:
    """Build the original visualizer schema for one dataset size."""
    protection = {
        method: evaluate_game.get_protection(method, mode)
        for method in METHODS
    }
    leakage = {
        method: evaluate_game.get_leakage(method, mode)
        for method in METHODS
    }
    results = {}

    for method in METHODS:
        if transformed:
            real_game = evaluate_game.load_game(mode, method, "Full_FL")
            simulated_games = [
                _load_transformed_game(mode, method, scenario)
                for scenario in SCENARIOS
            ]
        else:
            real_game = evaluate_game.load_game(mode, method, "Full_FL")
            simulated_games = [
                evaluate_game.load_game(mode, method, scenario)
                for scenario in SCENARIOS
            ]

        entries = {}
        c_value = selected_weights[(mode, method)][0]
        weight_pairs = (
            (0.0, 0.0),
            (0.0, c_value),
            (c_value, 0.0),
            (c_value, c_value),
        )
        for weights in weight_pairs:
            key = evaluate_game.weight_pair_key(weights)
            real = evaluate_game.play_game(
                *real_game, protection[method], leakage[method], weights
            )
            simulated = [
                evaluate_game.play_game(
                    *game, protection[method], leakage[method], weights
                )
                for game in simulated_games
            ]
            comparisons = {
                order: {
                    f"sim_{index + 1}": evaluate_game.compare_with_real(
                        real[order], game[order]
                    )
                    for index, game in enumerate(simulated)
                }
                for order in evaluate_game.MOVE_ORDERS
            }
            entries[key] = {
                "weights": {
                    "first_player": weights[0],
                    "second_player": weights[1],
                },
                "games": {
                    "real": real,
                    "sim_1": simulated[0],
                    "sim_2": simulated[1],
                },
                "comparison": comparisons,
            }
        results[method] = entries

    return {
        "metadata": {
            "methods": list(METHODS),
            "utility_formula": (
                "utility = clipped_gain - C * leakage(own choice)"
            ),
            "C_pair": {
                method: {
                    "first_player": selected_weights[(mode, method)][0],
                    "second_player": selected_weights[(mode, method)][1],
                }
                for method in METHODS
            },
            "table_label": f"{mode} (transformed)" if transformed else mode,
        },
        "results": {mode: results},
    }


def main() -> None:
    """Write tables for the configured dataset split."""
    if DATASET_SPLIT not in SPLIT_MODES:
        raise ValueError(f"Unsupported DATASET_SPLIT: {DATASET_SPLIT}")
    selected_weights = _load_selected_weights()
    modes = SPLIT_MODES[DATASET_SPLIT]
    output_dir = f"results/final_{DATASET_SPLIT}"
    for mode in modes:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for transformed in (False, True):
            data = _make_table_data(mode, transformed, selected_weights)
            suffix = "_transformed" if transformed else ""
            output_path = Path(output_dir) / f"nash_table_{mode}{suffix}.png"
            nesh_visualisation.draw_table(data, mode, str(output_path))


if __name__ == "__main__":
    main()
