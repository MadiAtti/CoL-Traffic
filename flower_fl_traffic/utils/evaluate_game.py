"""Search for Nash equilibria in real and self-division privacy games.

The first player selects a matrix row and the second player selects a column.
The payoff tensor stores both players' gains at each cell in ``[row, column]``
orientation. The search clips negative gains, subtracts a weighted leakage penalty, and runs alternating
best-response dynamics from the no-defence cell. Each game is evaluated with
both possible move orders.

The search first evaluates the four measured NT games, then evaluates their
fitted, transformed Self-Division approximations. Each game is tested with
``(C1, C2)`` equal to ``(0, 0)``, ``(0, 0.1)``, ``(0.1, 0)``, and ``(0.1, 0.1)``.
All results are written to ``results/nash/nash_equilibria.json``.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from omegaconf import OmegaConf
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualizer import (
    apply_f,
    get_actual_grid,
    get_dataset_size_descriptor,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the config only once
cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "conf/base.yaml")).config

analyze = "accuracy"   # must match build_matrices.py

OUTPUT_PATH = "results/nash/nash_equilibria.json"
SPLIT_OUTPUTS = {
    "100-50": ("full", "half"),
    "80-30": ("80percent", "30percent"),
}

# (scenario folder name, (first player, second player))
REAL_GAME = ("Full_FL", ("p1", "p2"))
TRANSFORMED_SD_KEY = "sd_transformed"
SELECTED_C_FILES = {
    "100-50": os.path.join(PROJECT_ROOT, "results/nash/c_candidates_100-50.json"),
    "80-30": os.path.join(PROJECT_ROOT, "results/nash/c_candidates_80-30.json"),
}
FIT_SPLITS = {
    "full": "100-50",
    "half": "100-50",
    "80percent": "80-30",
    "30percent": "80-30",
}


def get_weight_pairs(mode, method):
    """Return (0,0), (0,C), (C,0), and (C,C) for one game."""
    split = FIT_SPLITS[mode]
    with open(SELECTED_C_FILES[split], "r", encoding="utf-8") as file:
        candidates = json.load(file)
    selected = next(
        game["candidates"][0]["C"]
        for game in candidates["games"].values()
        if game["mode"] == mode and game["method"] == method
    )
    c = float(selected)
    return ((0.0, 0.0), (0.0, c), (c, 0.0), (c, c))

# who moves first in the dynamics: label -> player number (1 = first player of the pair)
MOVE_ORDERS = {"first_player_starts": 1, "second_player_starts": 2}


# --------------------------------------------------------------------------- #
# Best-response dynamics
# --------------------------------------------------------------------------- #
def best_response_dynamics(U1, U2, start=(0, 0), first=1, max_iter=1000):
    """Run alternating best responses for a two-player matrix game.

    Args:
        U1: Utility matrix for the row player.
        U2: Utility matrix for the column player.
        start: Initial ``(row, column)`` cell.
        first: Player who moves first; either ``1`` or ``2``.
        max_iter: Maximum number of turns before stopping.

    Returns:
        A tuple of the final cell, visited-cell path, and status. Status is
        ``"nash"`` for convergence, ``"cycle"`` for a repeated state, or
        ``"max_iter"`` when the turn limit is reached.
    """
    i, j = start
    turn = first
    path = [(i, j)]
    seen = {(i, j, turn)}   # (state, whose turn) pairs, used to detect cycles
    stable_steps = 0        # number of consecutive turns in which nobody moved

    for _ in range(max_iter):
        if turn == 1:
            # Player 1 best-responds along column j
            new_i = int(np.argmax(U1[:, j]))
            # Tie-breaking: stay if the current choice is already optimal,
            # so the dynamics do not bounce between equal-valued cells.
            if U1[i, j] >= U1[new_i, j]:
                new_i = i
            moved = new_i != i
            i = new_i
        else:
            # Player 2 best-responds along row i
            new_j = int(np.argmax(U2[i, :]))
            if U2[i, j] >= U2[i, new_j]:
                new_j = j
            moved = new_j != j
            j = new_j

        stable_steps = 0 if moved else stable_steps + 1
        if moved:
            path.append((i, j))

        # Two consecutive "no move" turns -> both players are best-responding
        if stable_steps == 2:
            return (i, j), path, "nash"

        # Switch player and check whether this exact situation occurred before
        turn = 2 if turn == 1 else 1
        state = (i, j, turn)
        if state in seen:
            return (i, j), path, "cycle"
        seen.add(state)

    return (i, j), path, "max_iter"


# --------------------------------------------------------------------------- #
# Grid, protection / leakage
# --------------------------------------------------------------------------- #
def get_levels(method, mode="full"):
    """Raw configured defence levels of a method / data size (None = no defence)."""
    if method == "Noise":
        level_map = {
            "full": "full_noise_levels",
            "half": "half_noise_levels",
            "80percent": "percent80_noise_levels",
            "30percent": "percent30_noise_levels",
        }
        if mode not in level_map:
            raise ValueError(f"Unsupported dataset mode for noise: {mode}")
        return list(cfg[level_map[mode]])
    if method == "Suppression":
        return list(cfg["sup_levels"])
    raise ValueError(f"Unsupported privacy method: {method}")


def get_protection(method, mode="full"):
    """
    Normalised protection level (in [0, 1]) of every grid point, in the order
    of the matrix axes: the configured levels plus a final 1.0 (max privacy).

        Noise:        x / (x + 1)      None (no defence) -> 0.0
        Suppression:  (14 - x) / 14    None (no defence) -> 0.0
    """
    levels = get_levels(method, mode)
    if method == "Noise":
        vals = [x / (x + 1) if x is not None else 0.0 for x in levels]
    else:
        vals = [(14 - x) / 14 if x is not None else 0.0 for x in levels]
    return np.array(vals + [1.0], dtype=float)


def get_leakage(method, mode="full"):
    """
    Leakage = 1 - protection: 1.0 for "no defence" (index 0), 0.0 for maximum
    privacy (last index). Derived from get_protection() in one place so the
    first and last grid points cannot get swapped.
    """
    return 1.0 - get_protection(method, mode)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def monotonize_matrix(matrix):
    """
    Make the matrix monotonically NON-INCREASING along every row and column
    using isotonic regression (rows first, then columns).
    """
    result = matrix.copy()
    ir = IsotonicRegression(increasing=False)
    for i in range(matrix.shape[0]):
        result[i, :] = ir.fit_transform(np.arange(matrix.shape[1]), result[i, :])
    for j in range(matrix.shape[1]):
        result[:, j] = ir.fit_transform(np.arange(matrix.shape[0]), result[:, j])
    return result


def to_clipped_gain(gain):
    """Clip an already player-wise monotonized gain matrix from below at zero."""
    if not np.isfinite(gain).all():
        raise ValueError("The gain matrix contains non-finite values (missing max-privacy cells?); "
                         "fill or drop them before running the Nash search.")
    return np.maximum(gain, 0.0)


def apply_leakage_penalty(G1, G2, leakage, weights):
    """
    Subtract the leakage penalty from the (clipped) gain matrices.

        U1[i, j] = G1[i, j] - weights[0] * leakage[i]   (player 1 leaks according to its row choice)
        U2[i, j] = G2[i, j] - weights[1] * leakage[j]   (player 2 leaks according to its column choice)
    """
    assert G1.shape == G2.shape == (len(leakage), len(leakage)), \
        f"shape mismatch: {G1.shape} vs {len(leakage)} grid points"
    return G1 - weights[0] * leakage[:, None], G2 - weights[1] * leakage[None, :]


def load_game(mode, method, scenario):
    """Load and player-wise monotonize a seed-averaged game."""
    path = (f"results/{mode}/{analyze}/games/average/"
            f"{method}_{scenario}_Bimatrix_Avg.npy")
    bm = np.load(path)
    player_one = monotonize_matrix(bm[:, :, 0])
    player_two = monotonize_matrix(bm[:, :, 1].T).T
    return to_clipped_gain(player_one), to_clipped_gain(player_two)


def load_transformed_sd_game(mode, method):
    """Load and correct the averaged Self-Division gain surface.

    The fitted correction is applied to the mean SD surface in own/other
    orientation. The second player's payoff is its transpose, as required for
    the corresponding symmetric two-player game.
    """
    average_dir = f"results/{mode}/{analyze}/games/average"
    pred_path = f"{average_dir}/{method}_Final_Pred.npy"
    fit_split = FIT_SPLITS[mode]
    params_path = f"results/final_{fit_split}/{method}_params.json"
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)["params"]
    coefficients = [params[name] for name in ("x", "y", "alpha", "beta")]
    predicted = np.load(pred_path)
    if predicted.ndim != 2 or predicted.shape[0] != predicted.shape[1]:
        raise ValueError(f"Expected a square averaged SD matrix at {pred_path}.")
    grid = get_actual_grid(method, mode)
    if predicted.shape != (len(grid), len(grid)):
        raise ValueError(
            f"SD matrix shape {predicted.shape} does not match the "
            f"{mode}/{method} privacy grid ({len(grid)}, {len(grid)})."
        )
    transformed = apply_f(
        predicted,
        get_dataset_size_descriptor(mode),
        grid,
        coefficients,
    )
    transformed = monotonize_matrix(transformed)
    return to_clipped_gain(transformed), to_clipped_gain(transformed.T)


def weight_pair_key(weights):
    """Return a stable output key for an ordered pair of leakage coefficients."""
    return f"C1={weights[0]:g}_C2={weights[1]:g}"


# --------------------------------------------------------------------------- #
# Playing a game
# --------------------------------------------------------------------------- #
def describe_equilibrium(cell, status, path, G1, G2, U1, U2, protection, leakage, weights):
    """Collect everything that is known about the cell the dynamics ended in."""
    i, j = cell
    return {
        "cell": [int(i), int(j)],
        "status": status,                          # "nash" | "cycle" | "max_iter"
        "converged": status == "nash",
        "protection": [float(protection[i]), float(protection[j])],
        "leakage": [float(leakage[i]), float(leakage[j])],
        "gain": [float(G1[i, j]), float(G2[i, j])],   # clipped gain, before the penalty
        "leakage_penalty": [float(weights[0] * leakage[i]), float(weights[1] * leakage[j])],
        "utility": [float(U1[i, j]), float(U2[i, j])],
        "path": [[int(a), int(b)] for a, b in path],
        "num_moves": len(path) - 1,
    }


def play_game(G1, G2, protection, leakage, weights):
    """
    Apply the leakage penalty and run the best-response dynamics from (0, 0),
    once with each player moving first.

    Returns {move_order_label: equilibrium description}.
    """
    U1, U2 = apply_leakage_penalty(G1, G2, leakage, weights)

    out = {}
    for label, first in MOVE_ORDERS.items():
        cell, path, status = best_response_dynamics(U1, U2, start=(0, 0), first=first)
        out[label] = describe_equilibrium(cell, status, path, G1, G2, U1, U2,
                                          protection, leakage, weights)
    return out


def compare_with_real(real_eq, sim_eq):
    """Compare one simulated equilibrium with the real one (same move order, same weights)."""
    di = abs(sim_eq["cell"][0] - real_eq["cell"][0])
    dj = abs(sim_eq["cell"][1] - real_eq["cell"][1])
    return {
        "matches_real": sim_eq["cell"] == real_eq["cell"],
        "cell_distance_l1": di + dj,
        "cell_distance_linf": max(di, dj),
        # sim minus real, per player [first, second]
        "protection_diff": [s - r for s, r in zip(sim_eq["protection"], real_eq["protection"])],
        "utility_diff": [s - r for s, r in zip(sim_eq["utility"], real_eq["utility"])],
    }


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
SCHEMA_DOC = {
    "cell": "[row, column] grid index of the equilibrium; row = first player's choice, "
            "column = second player's choice; index 0 = no defence, last = maximum privacy",
    "protection / leakage": "[first player, second player]; leakage = 1 - protection",
    "gain": "clipped (and optionally monotonised) accuracy gain at the cell, before the penalty",
    "leakage_penalty": "weight * leakage per player",
    "utility": "gain - leakage_penalty, per player",
    "status": "nash = pure Nash equilibrium reached, cycle = best responses loop, max_iter = limit hit",
    "path": "cells visited by the dynamics (only turns in which somebody moved), starting at [0, 0]",
    "first_player_starts / second_player_starts": "which player of the pair moves first in the dynamics",
    "comparison": "transformed SD equilibrium vs. measured NT equilibrium; differences are transformed - real",
    "summary": "aggregates over all ordered C pairs and both move orders",
}


def build_output(modes, methods, weight_pairs_by_game, results, grids):
    """Assemble the final, fully nested JSON document."""
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for method in methods:
            entries = results[mode][method]
            n_total = len(entries) * len(MOVE_ORDERS)
            matches = sum(
                entry["comparison"][order][TRANSFORMED_SD_KEY]["matches_real"]
                for entry in entries.values()
                for order in MOVE_ORDERS
            )
            status_counts = {}
            for entry in entries.values():
                for order in MOVE_ORDERS:
                    status = entry["games"][TRANSFORMED_SD_KEY][order]["status"]
                    status_counts[status] = status_counts.get(status, 0) + 1

            summary[mode][method] = {
                "num_C_pairs": len(entries),
                "num_comparisons": n_total,
                "transformed_matches_real": matches,
                "transformed_match_rate": matches / n_total,
                "transformed_status_counts": status_counts,
            }

    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "analyze": analyze,
            "monotonize": "per-player before seed/player averaging; after f for transformed SD",
            "start_cell": [0, 0],
            "utility_formula": "utility = clipped_gain - C * leakage(own choice), leakage = 1 - protection",
            "C_pairs": {
                mode: {
                    method: [
                        {"first_player": w1, "second_player": w2}
                        for w1, w2 in weight_pairs_by_game[(mode, method)]
                    ]
                    for method in methods
                }
                for mode in modes
            },
            "modes": modes,
            "methods": methods,
            "games": {
                "real": {"scenario": REAL_GAME[0], "players": list(REAL_GAME[1])},
                TRANSFORMED_SD_KEY: {
                    "scenario": "Transformed_SD_average",
                    "players": ["P1", "P2"],
                },
            },
            "move_orders": list(MOVE_ORDERS),
        },
        "schema": SCHEMA_DOC,
        # mode -> method -> {raw_levels, protection, leakage}; one entry per grid index
        "grids": grids,
        # mode -> method -> ordered C-pair key -> measured and transformed outcomes
        "results": results,
        "summary": summary,
    }


if __name__ == "__main__":
    modes = ["full", "half", "80percent", "30percent"]
    methods = ["Noise", "Suppression"]
    results, grids = {}, {}

    # Stage one: evaluate all measured NT payoff matrices for every mode.
    for mode in modes:
        results[mode], grids[mode] = {}, {}
        for method in methods:
            protection = get_protection(method, mode)
            leakage = get_leakage(method, mode)
            grids[mode][method] = {
                "raw_levels": get_levels(method, mode) + ["echo"],
                "protection": [float(x) for x in protection],
                "leakage": [float(x) for x in leakage],
            }

            real_name, real_players = REAL_GAME
            G_real = load_game(mode, method, real_name)
            results[mode][method] = {}
            weight_pairs = get_weight_pairs(mode, method)
            for weights in weight_pairs:
                key = weight_pair_key(weights)
                print(
                    f"\nMeasured NT | {method} | {mode} | "
                    f"C=({weights[0]:g}, {weights[1]:g})"
                )
                real = play_game(*G_real, protection, leakage, weights)
                results[mode][method][key] = {
                    "C": {
                        "first_player": weights[0],
                        "second_player": weights[1],
                    },
                    "weights": {
                        "first_player": weights[0],
                        "second_player": weights[1],
                    },
                    "games": {
                        "real": {
                            "scenario": real_name,
                            "players": list(real_players),
                            **real,
                        }
                    },
                    "comparison": {},
                }
                for order in MOVE_ORDERS:
                    result = real[order]
                    print(
                        f"  [{order}] cell={tuple(result['cell'])} "
                        f"protection={tuple(round(p, 3) for p in result['protection'])} "
                        f"[{result['status']}]"
                    )

    # Stage two: apply the fitted correction and evaluate the SD approximations.
    for mode in modes:
        for method in methods:
            protection = get_protection(method, mode)
            leakage = get_leakage(method, mode)
            transformed_sd = load_transformed_sd_game(mode, method)
            weight_pairs = get_weight_pairs(mode, method)
            for weights in weight_pairs:
                key = weight_pair_key(weights)
                entry = results[mode][method][key]
                transformed = play_game(
                    *transformed_sd, protection, leakage, weights
                )
                entry["games"][TRANSFORMED_SD_KEY] = {
                    "scenario": "Transformed_SD_average",
                    "players": ["P1", "P2"],
                    **transformed,
                }
                for order in MOVE_ORDERS:
                    entry["comparison"][order] = {
                        TRANSFORMED_SD_KEY: compare_with_real(
                            entry["games"]["real"][order],
                            transformed[order],
                        )
                    }
                    result = transformed[order]
                    print(
                        f"Transformed SD | {method} | {mode} | "
                        f"C=({weights[0]:g}, {weights[1]:g}) | "
                        f"cell={tuple(result['cell'])} "
                        f"protection={tuple(round(p, 3) for p in result['protection'])} "
                        f"[{result['status']}]"
                    )

    weight_pairs_by_game = {
        (mode, method): get_weight_pairs(mode, method)
        for mode in modes
        for method in methods
    }
    output = build_output(modes, methods, weight_pairs_by_game, results, grids)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    for split, split_modes in SPLIT_OUTPUTS.items():
        split_output = {
            **output,
            "metadata": {
                **output["metadata"],
                "modes": list(split_modes),
            },
            "grids": {mode: grids[mode] for mode in split_modes},
            "results": {mode: results[mode] for mode in split_modes},
            "summary": {
                mode: output["summary"][mode]
                for mode in split_modes
            },
        }
        split_path = f"results/final_{split}/nash_equilibria.json"
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split_output, f, indent=2)
        print(f"Saved: {split_path}")

        all_modes_path = (
            f"results/final_{split}/nash_equilibria_all_sizes.json"
        )
        with open(all_modes_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {all_modes_path}")