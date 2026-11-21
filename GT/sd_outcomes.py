"""Realized outcomes of Self-Division decisions over the (C1, C2) grid.

Implements the evaluation of CoL-SD Section 7 for one configuration
(dataset size x privacy mechanism) on every ``(C1, C2)`` cell of the grid.

Utility (capped game, as in ``evaluate_game``)
    ``u_n = max(Delta_n, 0) - C_n * (1 - p_n)``. A negative accuracy gain is
    capped at 0 because a player falls back on its local model, so the only
    way to lose is the leakage term.

True NE
    Best-response dynamics on the measured Full_FL game, from ``(0, 0)``,
    Player 1 moving first (the default start order).

SD decision of a player
    In its own transformed SD game, a player identifies with whichever half
    carries its own weight. Either half may be "itself", so it solves both
    orientations and keeps the equilibrium in which its own SD-predicted
    utility is larger; a tie is broken at random (seeded, reproducible).

    * Player 1, game ``P1_Subnet``: (A) weights ``(C1, C2)``, own choice is the
      row; (B) weights ``(C2, C1)``, own choice is the column.
    * Player 2, game ``P2_Subnet``: (A) weights ``(C1, C2)``, own choice is the
      column; (B) weights ``(C2, C1)``, own choice is the row.

    Only orientations whose dynamics converge are candidates. If neither
    converges, orientation A's final cell is used and the cell is flagged.

Realized profile
    ``(P1's own SD choice, P2's own SD choice)`` played on the true game.

Player outcome (``OUTCOMES``), one per player and (C1, C2) cell::

    0 hit           realized profile == true NE
    1 miss, ++      realized profile != true NE, realized utility above the
                    player's true-NE utility (hence positive)
    2 miss, +       realized profile != true NE, positive realized utility,
                    but not above the true-NE utility
    3 miss, -       realized profile != true NE, negative realized utility
    4 miss, 0       realized profile != true NE, zero realized utility
    5 no pure NE    the true game has no pure NE here (best-response cycle);
                    not part of the summary table, reported separately

A true NE never has negative utility (each player can deviate to maximum
privacy), so "++" always means a positive utility.

The summary pools the two players: each percentage is over 2 x 2500
player-cells. ``u = 0`` means ``|u| <= ZERO_TOL``.

(C1, C2) grid (``case_c_grid``, ``c_ranges``)
    50 x 50 from 0 to C_max, where C_max is the smallest C at which both
    players reach p = 1 on both axes, widened so that the p = 1 band is the
    last ~10% of each axis. NT and DD: one C_max per mechanism for both sizes;
    RS: one per size and mechanism. Stored in ``GT/results/<UC>/c_ranges.json``.

Summary table (``tab_outcomes_rows.tex``)
    One row per mechanism x size. NT Suppression depends on the start order
    (two mirror equilibria in many cells) and is given for both first movers;
    NT DP, DD and RS are start-order invariant and have no first-mover column.

Outputs in ``usecases.output_dir(split)/sd_outcomes/``: ``*_outcomes.npz``
(``*_p2first_*`` for NT Suppression), ``outcome_cells.csv``,
``outcome_summary.csv/json``, ``tab_outcomes_rows.tex``. The interactive
cherry-pick table is ``notebooks/sd_cherry_pick.ipynb``
(``utils/make_cherry_pick_notebook.py``).

Usage (from the ``GT`` directory)::

    python utils/sd_outcomes.py --split 80-30     # NT
    python utils/sd_outcomes.py --split DD
    python utils/sd_outcomes.py --split RS
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

GT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GT_ROOT))

import evaluate_game  # noqa: E402
import usecases  # noqa: E402

SPLITS = usecases.SPLITS          # NT "100-50" / "80-30", "RS", "DD"
SIZE_NAMES = {name: m.size for name, m in usecases.MODES.items()}
SIZE_LABELS = {"full": "100% data", "half": "50% data",
               "80percent": "80% data", "30percent": "30% data"}
METHODS = ("Suppression", "Noise")
MECH_NAMES = {"Suppression": "Sup", "Noise": "DP"}

C_MAX, C_STEP = 0.49, 0.01
TOL = 1e-12
SEED = 0
ZERO_TOL = 1e-9
OUTCOMES = ("hit", "miss, ++", "miss, +", "miss, -", "miss, 0", "no pure NE")
TABLE_OUTCOMES = OUTCOMES[:5]
NO_PURE_NE = 5


def c_grid(c_max=C_MAX, c_step=C_STEP):
    count = int(round(c_max / c_step)) + 1
    return np.round(np.arange(count) * c_step, 10)


# (C1, C2) grid: 50 x 50 from 0 to C_max. C_max is chosen so that both players
# reach the p = 1 band on both axes: at C1 = C_max the true NE (P1 first) has
# p1 = 1 for every C2 on the grid, and at C2 = C_max p2 = 1 for every C1. The
# threshold is then widened so that the p = 1 band is the last ~10% of each axis.
# NT and DD use one C_max per mechanism for both sizes (the larger threshold, so
# nothing is cut off for either size); RS, whose sizes' gains are ~5x apart, one
# per size and mechanism. Ranges are stored in GT/results/<UC>/c_ranges.json.
GRID_POINTS = 50
BAND_SHARE = 0.10


def _true_game(mode, method):
    return (evaluate_game.load_game(mode, method, "Full_FL"),
            evaluate_game.get_leakage(method, mode),
            len(evaluate_game.get_protection(method, mode)) - 1)


def _round_up(x):
    """Round up to two significant digits."""
    exponent = np.floor(np.log10(x)) - 1
    return float(np.ceil(x / 10 ** exponent) * 10 ** exponent)


def p1_band_threshold(mode, method, points=GRID_POINTS):
    """Smallest C such that, on a grid 0..C, both players are at p = 1 along the far edge
    of their own axis (P1 on C1 = C, P2 on C2 = C) for every value of the other weight."""
    game, leakage, top = _true_game(mode, method)
    gain00 = float(min(game[0][0, 0], game[1][0, 0]))
    step = gain00 / 200
    c = step
    while True:
        others = np.linspace(0.0, c, points)
        p1_ok = all(solve(game, leakage, (c, o), 1)[0][0] == top for o in others)
        p2_ok = p1_ok and all(solve(game, leakage, (o, c), 1)[0][1] == top for o in others)
        if p2_ok:
            return float(c)
        c += step


def c_ranges(split, refresh=False):
    """{"ranges": {"<mode>/<method>": {"C_max", "threshold"}}} of a split (cached as JSON)."""
    path = usecases.output_dir(split) / "c_ranges.json"
    if path.is_file() and not refresh:
        return json.loads(path.read_text())
    large, small = usecases.SPLITS[split]
    per_size = usecases.usecase_of(split) == "RS"
    ranges = {}
    for method in METHODS:
        for modes in ([[small], [large]] if per_size else [[small, large]]):
            thresholds = {m: p1_band_threshold(m, method) for m in modes}
            c_max = _round_up(max(thresholds.values()) / (1.0 - BAND_SHARE))
            for m in modes:
                ranges[f"{m}/{method}"] = {"C_max": c_max, "threshold": thresholds[m]}
    payload = {"rule": "both players reach p = 1 on both axes; band = last 10% of each axis",
               "points": GRID_POINTS, "per_size": per_size, "ranges": ranges}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def case_c_grid(mode, method, points=GRID_POINTS):
    """The (C1, C2) grid of one case: ``points`` values from 0 to its C_max."""
    c_max = c_ranges(usecases.split_of(mode))["ranges"][f"{mode}/{method}"]["C_max"]
    return np.round(np.linspace(0.0, c_max, points), 12)


def load_games(mode, method):
    """True (Full_FL) game and both players' transformed SD games, all capped."""
    return (evaluate_game.load_game(mode, method, "Full_FL"),
            evaluate_game.load_transformed_scenario_game(mode, method, "P1_Subnet"),
            evaluate_game.load_transformed_scenario_game(mode, method, "P2_Subnet"))


def solve(game, leakage, weights, first):
    """Best-response dynamics from (0, 0): ((i, j), status, U1, U2)."""
    U1, U2 = evaluate_game.apply_leakage_penalty(*game, leakage, weights)
    cell, _, status = evaluate_game.best_response_dynamics(U1, U2, start=(0, 0), first=first)
    return cell, status, U1, U2


def sd_decision(game, leakage, c_own, c_other, own_is_row_in_a, first, rng):
    """One player's SD decision in its own transformed game.

    Orientation A puts the player's weight on the row (P1) or column (P2)
    half; orientation B swaps the two weights, so the player is the other
    half. Returns a dict with the chosen own / partner choice, both predicted
    utilities, the chosen orientation, and tie / convergence flags.

    ``first`` is the real player who moves first (1: the C1 player, 2: the C2
    player). In orientation A the subgame's row half carries C1, so the order
    is kept; in orientation B the row half carries C2, so it is swapped.
    """
    options = []
    for label, own_row in (("A", own_is_row_in_a), ("B", not own_is_row_in_a)):
        weights = (c_own, c_other) if own_row else (c_other, c_own)
        sub_first = first if label == "A" else 3 - first
        (i, j), status, U1, U2 = solve(game, leakage, weights, sub_first)
        own, partner = (i, j) if own_row else (j, i)
        u_own, u_partner = (U1[i, j], U2[i, j]) if own_row else (U2[i, j], U1[i, j])
        options.append({"orientation": label, "own": own, "partner": partner,
                        "u_own": float(u_own), "u_partner": float(u_partner),
                        "converged": status == "nash"})
    candidates = [o for o in options if o["converged"]]
    tie = False
    if not candidates:
        chosen = dict(options[0])
    elif len(candidates) == 2 and abs(candidates[0]["u_own"] - candidates[1]["u_own"]) <= TOL:
        tie = candidates[0]["own"] != candidates[1]["own"] or candidates[0]["partner"] != candidates[1]["partner"]
        chosen = dict(candidates[int(rng.integers(2))] if tie else candidates[0])
    else:
        chosen = dict(max(candidates, key=lambda o: o["u_own"]))
    chosen["tie"] = tie
    chosen["any_converged"] = bool(candidates)
    return chosen


def analyse(mode, method, cs, first=1, seed=SEED):
    """Evaluate every (C1, C2) cell; arrays are indexed [k1, k2]."""
    protection = evaluate_game.get_protection(method, mode)
    leakage = evaluate_game.get_leakage(method, mode)
    true_game, sd1_game, sd2_game = load_games(mode, method)
    rng = np.random.default_rng(seed)
    n = len(cs)
    ints = ("i_star", "j_star", "i_real", "j_real",
            "sd1_own", "sd1_partner", "sd2_own", "sd2_partner")
    floats = ("u1_star", "u2_star", "u1_real", "u2_real",
              "sd1_u_own", "sd1_u_partner", "sd2_u_own", "sd2_u_partner")
    bools = ("true_converged", "sd1_tie", "sd2_tie", "sd1_converged", "sd2_converged")
    out = {k: np.zeros((n, n), dtype=int) for k in ints}
    out.update({k: np.zeros((n, n)) for k in floats})
    out.update({k: np.zeros((n, n), dtype=bool) for k in bools})
    out.update({k: np.empty((n, n), dtype="<U1") for k in ("sd1_orientation", "sd2_orientation")})

    for k1, c1 in enumerate(cs):
        for k2, c2 in enumerate(cs):
            (i, j), status, U1, U2 = solve(true_game, leakage, (c1, c2), first)
            d1 = sd_decision(sd1_game, leakage, c1, c2, True, first, rng)
            d2 = sd_decision(sd2_game, leakage, c2, c1, False, first, rng)
            ir, jr = d1["own"], d2["own"]
            cell = (k1, k2)
            out["i_star"][cell], out["j_star"][cell] = i, j
            out["u1_star"][cell], out["u2_star"][cell] = U1[i, j], U2[i, j]
            out["true_converged"][cell] = status == "nash"
            out["i_real"][cell], out["j_real"][cell] = ir, jr
            out["u1_real"][cell], out["u2_real"][cell] = U1[ir, jr], U2[ir, jr]
            for tag, d in (("sd1", d1), ("sd2", d2)):
                out[f"{tag}_own"][cell], out[f"{tag}_partner"][cell] = d["own"], d["partner"]
                out[f"{tag}_u_own"][cell], out[f"{tag}_u_partner"][cell] = d["u_own"], d["u_partner"]
                out[f"{tag}_orientation"][cell] = d["orientation"]
                out[f"{tag}_tie"][cell] = d["tie"]
                out[f"{tag}_converged"][cell] = d["any_converged"]

    hit = (out["i_real"] == out["i_star"]) & (out["j_real"] == out["j_star"])
    out["hit"] = hit & out["true_converged"]
    for n in (1, 2):
        u = out[f"u{n}_real"]
        better = u > out[f"u{n}_star"] + ZERO_TOL
        miss = np.where(better, 1, np.where(u > ZERO_TOL, 2, np.where(u < -ZERO_TOL, 3, 4)))
        outcome = np.where(hit, 0, miss)
        out[f"outcome{n}"] = np.where(out["true_converged"], outcome, NO_PURE_NE)
    out["better_than_ne"] = (out["true_converged"] & ~hit
                             & ((out["u1_real"] > out["u1_star"] + TOL)
                                | (out["u2_real"] > out["u2_star"] + TOL)))
    out.update({"C": cs, "protection": protection, "mode": mode, "method": method,
                "first": first, "seed": seed})
    return out


def summarize(res):
    """Outcome percentages, pooled over both players (sum 100%) and per player."""
    o1, o2 = res["outcome1"], res["outcome2"]
    pooled = np.concatenate([o1.ravel(), o2.ravel()])
    pct = lambda arr: {name: 100.0 * np.count_nonzero(arr == k) / arr.size  # noqa: E731
                       for k, name in enumerate(OUTCOMES)}
    return {
        "mode": res["mode"], "method": res["method"],
        "size": SIZE_NAMES[res["mode"]], "mechanism": MECH_NAMES[res["method"]],
        "cells": int(o1.size), "player_cells": int(pooled.size),
        "pct": pct(pooled),
        "pct_p1": pct(o1), "pct_p2": pct(o2),
        "hit_cells_pct": 100.0 * np.mean(res["hit"]),
        "better_than_ne_pct": 100.0 * np.mean(res["better_than_ne"]),
        "sd_ties": {"p1": int(res["sd1_tie"].sum()), "p2": int(res["sd2_tie"].sum())},
        "sd_nonconverged": {"p1": int((~res["sd1_converged"]).sum()),
                            "p2": int((~res["sd2_converged"]).sum())},
        "sd_orientation_B_pct": {"p1": 100.0 * np.mean(res["sd1_orientation"] == "B"),
                                 "p2": 100.0 * np.mean(res["sd2_orientation"] == "B")},
    }


def save_npz(res, path):
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    arrays.update({"mode": res["mode"], "method": res["method"],
                   "first": res["first"], "seed": res["seed"]})
    np.savez_compressed(path, **arrays)


def write_cells_csv(results, path):
    """Long-form table: one row per configuration x (C1, C2)."""
    fields = ("i_star", "j_star", "u1_star", "u2_star", "i_real", "j_real", "u1_real", "u2_real",
              "sd1_own", "sd1_partner", "sd1_u_own", "sd1_u_partner", "sd1_orientation", "sd1_tie",
              "sd2_own", "sd2_partner", "sd2_u_own", "sd2_u_partner", "sd2_orientation", "sd2_tie")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "method", "C1", "C2", "outcome_p1", "outcome_p2", "hit",
                         "true_converged", "better_than_ne", *fields])
        for res in results:
            for k1, c1 in enumerate(res["C"]):
                for k2, c2 in enumerate(res["C"]):
                    cell = (k1, k2)
                    writer.writerow([res["mode"], res["method"], f"{c1:.6g}", f"{c2:.6g}",
                                     OUTCOMES[res["outcome1"][cell]],
                                     OUTCOMES[res["outcome2"][cell]],
                                     int(res["hit"][cell]),
                                     int(res["true_converged"][cell]),
                                     int(res["better_than_ne"][cell]),
                                     *(f"{res[k][cell]:.10g}" if res[k].dtype.kind == "f"
                                       else res[k][cell] for k in fields)])


# Rows of the summary table: DP is start-order invariant (one row per size);
# Suppression has mirror equilibria, so it is reported for both start orders.
TABLE_ROWS = (("Noise", 1), ("Suppression", 1), ("Suppression", 2))
START_LABEL = {1: "P1", 2: "P2"}


def ordered(summaries):
    """DP (size 1, 2), Sup with P1 first (size 1, 2), Sup with P2 first (size 1, 2)."""
    rank = {key: i for i, key in enumerate(TABLE_ROWS)}
    return sorted(summaries, key=lambda s: (rank[(s["method"], s["first"])], s["size"]))


def latex_rows(summaries, usecase="NT"):
    """Body rows of the summary table: use-case & mechanism & size [& first mover] & outcomes.

    The first-mover column exists only where the start order matters (NT, whose
    Suppression rows are given for both orders); RS / DD tables leave it out.
    """
    with_start = usecase == "NT"
    lines = []
    for s in ordered(summaries):
        cells = " & ".join(f"${s['pct'][name]:.1f}$" for name in TABLE_OUTCOMES)
        start = ""
        if with_start:
            start = ("--" if s["method"] == "Noise" else START_LABEL[s["first"]]) + " & "
        lines.append(f"    {usecase} & {s['mechanism']} & {s['size'].replace('size', 'Size ')} & "
                     f"{start}{cells} \\\\")
    return "\n".join(lines)


def run_split(split, cs):
    out_dir = usecases.output_dir(split) / "sd_outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, summaries = [], []
    # The start order matters only for NT Suppression; RS / DD (and DP everywhere)
    # are reported with P1 moving first.
    rows = TABLE_ROWS if usecases.usecase_of(split) == "NT" else TABLE_ROWS[:2]
    for method, first in rows:
        for mode in sorted(SPLITS[split], key=lambda m: SIZE_NAMES[m]):
            res = analyse(mode, method, case_c_grid(mode, method), first=first)
            suffix = "" if first == 1 else "_p2first"
            save_npz(res, out_dir / f"{mode}_{method}{suffix}_outcomes.npz")
            results.append(res)
            summaries.append({**summarize(res), "first": first})
    summaries = ordered(summaries)
    write_cells_csv([r for r in results if r["first"] == 1], out_dir / "outcome_cells.csv")
    with open(out_dir / "outcome_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mechanism", "size", "first_mover", "player", *OUTCOMES])
        for s in summaries:
            for who, key in (("both", "pct"), ("P1", "pct_p1"), ("P2", "pct_p2")):
                writer.writerow([s["mechanism"], s["size"], START_LABEL[s["first"]], who,
                                 *(f"{s[key][c]:.2f}" for c in OUTCOMES)])
    (out_dir / "outcome_summary.json").write_text(json.dumps({
        "split": split,
        "grid": {f"{r['mode']}/{r['method']}": {"C_min": float(r["C"][0]), "C_max": float(r["C"][-1]),
                                                "C_step": float(r["C"][1] - r["C"][0]),
                                                "cells": int(len(r["C"]) ** 2)}
                 for r in results if r["first"] == 1},
        "start_order": "DP: P1 moves first (order-invariant); Suppression: both orders",
        "tie_seed": SEED, "outcomes": OUTCOMES, "zero_tol": ZERO_TOL,
        "configurations": summaries}, indent=2), encoding="utf-8")
    (out_dir / "tab_outcomes_rows.tex").write_text(
        latex_rows(summaries, usecases.usecase_of(split)) + "\n", encoding="utf-8")

    print(f"\n=== SD outcomes {split}: % of 2 x {len(cs) ** 2} player-cells ===")
    head = f"{'mech':4s} {'size':6s} {'first':5s}" + "".join(f"{c:>13s}" for c in TABLE_OUTCOMES)
    print(head)
    print("-" * len(head))
    for s in summaries:
        print(f"{s['mechanism']:4s} {s['size']:6s} {START_LABEL[s['first']]:5s}"
              + "".join(f"{s['pct'][c]:12.2f}%" for c in TABLE_OUTCOMES))
    print(f"Saved outputs to {out_dir}/")
    return results, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="80-30")
    args = parser.parse_args(argv)
    cs = c_grid()
    for split in (SPLITS if args.split == "all" else (args.split,)):
        run_split(split, cs)


if __name__ == "__main__":
    main()
