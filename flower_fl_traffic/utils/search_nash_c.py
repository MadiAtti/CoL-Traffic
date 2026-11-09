"""Find useful symmetric leakage weights for the four small/large games.

The search uses the measured (real) game only.  It scans C from zero upward,
keeps the first value whose best-response dynamics reaches maximum privacy for
both move orders, and selects intermediate values whose equilibria cover the
available protection grid.  Results are written to
``results/nash/c_candidates.json``.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from utils import evaluate_game


# Select the dataset pair used by this run.  The script searches all four
# games in the selected pair: small/large x DP/SUP.
DATASET_SPLIT = "80-30"  # Change to "100-50" for the 50%/100% pair.
C_MAX = 1.0
C_STEP = 0.01
CANDIDATE_COUNT = 7
OUTPUT_PATH = Path(f"results/nash/c_candidates_{DATASET_SPLIT}.json")
PLOT_PATH = Path(f"results/nash/c_candidates_{DATASET_SPLIT}.png")

DATASET_MODES = {
    "100-50": ("half", "full"),
    "80-30": ("30percent", "80percent"),
}
GAME_SPECS = {
    "small-dp": ("Noise", "small"),
    "small-sup": ("Suppression", "small"),
    "large-dp": ("Noise", "large"),
    "large-sup": ("Suppression", "large"),
}
SCENARIOS = ("Full_FL", "P1_Subnet", "P2_Subnet")


def _signature(result):
    """Return both move-order equilibria as a stable cell signature."""
    return tuple(
        tuple(result[order]["cell"])
        for order in evaluate_game.MOVE_ORDERS
    )


def _is_corner(signature, index):
    return all(cell == (index, index) for cell in signature)


def _is_allowed_corner(signature, max_index):
    """Return whether every equilibrium cell is one of the four corners."""
    allowed = {
        (0, 0),
        (0, max_index),
        (max_index, 0),
        (max_index, max_index),
    }
    return all(cell in allowed for cell in signature)


def _load_monotone_game(mode, method, scenario):
    """Load one measured game in player-wise monotone orientation."""
    raw_game = evaluate_game.load_game(mode, method, scenario)
    return (
        evaluate_game.monotonize_matrix(raw_game[0]),
        evaluate_game.monotonize_matrix(raw_game[1].T).T,
    )


def _scan_game(mode, method, scenario, c_values):
    protection = evaluate_game.get_protection(method, mode)
    leakage = evaluate_game.get_leakage(method, mode)
    game = _load_monotone_game(mode, method, scenario)
    max_index = len(protection) - 1
    values = []

    for c in c_values:
        result = evaluate_game.play_game(
            *game, protection, leakage, (float(c), float(c))
        )
        signature = _signature(result)
        values.append({
            "C": float(c),
            "signature": [list(cell) for cell in signature],
            "protection": [
                result[order]["protection"] for order in evaluate_game.MOVE_ORDERS
            ],
            "statuses": [
                result[order]["status"] for order in evaluate_game.MOVE_ORDERS
            ],
        })
    return values, max_index


def _select_candidates(scans, max_index, count):
    """Select the first common C that reaches max-max in every game."""

    c_values = [item["C"] for item in scans[0]]
    common_max_index = next(
        (
            index for index in range(len(c_values))
            if all(
                _is_corner(
                    tuple(tuple(cell) for cell in scan[index]["signature"]),
                    max_index,
                )
                for scan in scans
            )
        ),
        None,
    )
    if common_max_index is None:
        raise RuntimeError(
            "The three games do not share a C that reaches (max,max)."
        )

    valid_indices = [
        index for index in range(common_max_index + 1)
        if all(
            _is_allowed_corner(
                tuple(tuple(cell) for cell in scan[index]["signature"]),
                max_index,
            )
            for scan in scans
        )
    ]
    if 0 not in valid_indices or common_max_index not in valid_indices:
        raise RuntimeError(
            "The shared C range does not consist of the four corner states."
        )

    return [scans[0][common_max_index]], common_max_index


def _transition_items(scan, end_index):
    """Return scan points where this game's equilibrium signature changes."""
    transitions = []
    previous = None
    for item in scan[: end_index + 1]:
        signature = tuple(tuple(cell) for cell in item["signature"])
        if signature != previous:
            transitions.append(item)
            previous = signature
    return transitions


def find_candidates(c_max=10.0, c_step=0.01, count=7):
    """Find endpoint-preserving and spatially distributed C candidates."""
    output = {
        "metadata": {
            "criterion": (
                "For all three games and both move orders, every selected C "
                "has an equilibrium in {(0,0), (0,max), (max,0), (max,max)}; "
                "the endpoints are (0,0) and (max,max)."
            ),
            "c_max": c_max,
            "c_step": c_step,
            "candidate_count": 1,
            "selection": "first common C where all scenarios and move orders reach max-max",
        },
        "games": {},
    }
    if DATASET_SPLIT not in DATASET_MODES:
        raise ValueError(f"Unsupported DATASET_SPLIT: {DATASET_SPLIT}")
    small_mode, large_mode = DATASET_MODES[DATASET_SPLIT]
    c_values = np.arange(0.0, c_max + c_step / 2, c_step)
    for name, (method, size) in GAME_SPECS.items():
        mode = large_mode if size == "large" else small_mode
        scans = {
            scenario: _scan_game(mode, method, scenario, c_values)
            for scenario in SCENARIOS
        }
        max_index = scans["Full_FL"][1]
        scan_values = [scans[scenario][0] for scenario in SCENARIOS]
        if not all(
            _is_corner(tuple(tuple(cell) for cell in scan[0]["signature"]), 0)
            for scan in scan_values
        ):
            raise RuntimeError(f"{name} does not start at the (0,0) corner in all games.")
        candidates, common_max_index = _select_candidates(
            scan_values, max_index, count
        )
        candidate_values = [item["C"] for item in candidates]
        candidate_details = [
            {
                "C": c_value,
                "games": {
                    scenario: next(
                        item
                        for item in scan
                        if item["C"] == c_value
                    )
                    for scenario, scan in zip(SCENARIOS, scan_values)
                },
            }
            for c_value in candidate_values
        ]

        output["games"][name] = {
            "mode": mode,
            "method": method,
            "scenarios": list(SCENARIOS),
            "max_privacy_index": max_index,
            "c_at_max_privacy": candidate_values[-1],
            "candidates": candidate_details,
            "scans": {
                scenario: scan[: common_max_index + 1]
                for scenario, scan in zip(SCENARIOS, scan_values)
            },
            "transitions": {
                scenario: _transition_items(scan, common_max_index)
                for scenario, scan in zip(SCENARIOS, scan_values)
            },
        }
    return output


def save_plot(result, path):
    """Plot scenarios by row and size-method pairs by column."""
    fig, axes = plt.subplots(
        3, 4, figsize=(13, 8), sharex=False, sharey=True,
        gridspec_kw={"hspace": 0.42, "wspace": 0.2},
    )
    axes = np.asarray(axes).reshape(3, 4)
    scenario_labels = {
        "Full_FL": "Full_FL (p1-p2)",
        "P1_Subnet": "P1 subnet (p11-p12)",
        "P2_Subnet": "P2 subnet (p21-p22)",
    }
    order_labels = {
        "first_player_starts": "P1 starts",
        "second_player_starts": "P2 starts",
    }

    for column, (name, game) in enumerate(result["games"].items()):
        mode = game["mode"]
        method = game["method"]
        protection = evaluate_game.get_protection(method, mode)
        best_c = game["candidates"][0]["C"]
        interesting_max = max(
            item["C"]
            for scenario in SCENARIOS
            for item in game["transitions"][scenario]
        )
        x_max = interesting_max * 1.08 if interesting_max else C_STEP
        x_max = min(x_max, C_MAX)
        x_ticks = sorted({0.0, float(best_c), float(interesting_max)})
        x_labels = [
            "0" if value == 0 else
            f"best C = {value:g}" if np.isclose(value, best_c) else
            f"{value:g}"
            for value in x_ticks
        ]
        for row, scenario in enumerate(SCENARIOS):
            axis = axes[row, column]
            transitions = game["transitions"][scenario]
            for order_index, order in enumerate(evaluate_game.MOVE_ORDERS):
                xs = [item["C"] for item in transitions]
                ys = [
                    protection[item["signature"][order_index][order_index]]
                    for item in transitions
                ]
                if xs[-1] < x_max and ys[-1] == protection[-1]:
                    xs.append(x_max)
                    ys.append(ys[-1])
                axis.step(
                    xs,
                    ys,
                    where="post",
                    color=("#1f4e79" if order_index == 0 else "#4c9f70"),
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    label=order_labels[order],
                )
            axis.axvline(
                best_c,
                color="#d62728",
                alpha=0.95,
                linewidth=2.4,
                linestyle="--",
                zorder=3,
            )
            if row == 0:
                axis.set_title(
                    f"{name}\n{DATASET_SPLIT}",
                    fontsize=10,
                    fontweight="bold",
                    pad=14,
                )
            if column == 0:
                axis.text(
                    -0.28,
                    0.5,
                    scenario_labels[scenario],
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                )
            axis.set_ylim(-0.05, 1.05)
            axis.set_xlim(0, x_max)
            axis.set_xticks(x_ticks)
            axis.set_xticklabels(x_labels, fontsize=8)
            axis.grid(alpha=0.25)
            axis.tick_params(labelsize=8)
            if row == len(SCENARIOS) - 1:
                axis.set_xlabel("C", fontsize=9)
            if row == 0:
                axis.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"C scan ({DATASET_SPLIT}) — red line: shared best C",
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.995,
        bottom=0.08,
        top=0.91,
        hspace=0.42,
        wspace=0.22,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    result = find_candidates(C_MAX, C_STEP, CANDIDATE_COUNT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_plot(result, PLOT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
