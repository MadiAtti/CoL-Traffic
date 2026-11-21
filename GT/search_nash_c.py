"""Protection-level maps and start-order diagnostics over the (C1, C2) grid.

For every configuration (dataset size x privacy mechanism) of a split, and
every ordered pair of leakage weights ``(C1, C2)`` on the configuration's
50 x 50 grid (``sd_outcomes.case_c_grid``: 0 .. C_max, where C_max is chosen
so that both players reach the p = 1 band on both axes; ``GT/results/<UC>/c_ranges.json``), this script maps the chosen protection levels p:

True game NE ``(p1*, p2*)``
    Best-response dynamics on the measured Full_FL game with
    u_n = max(Delta_n, 0) - C_n (1 - p_n).

Realized SD profile ``(p1_SD, p2_SD)``
    Complete information: both players know the true ``(C1, C2)``. Each player
    solves its own fitted, transformed self-division game in both
    orientations (its own weight on either half) and keeps the equilibrium
    with the larger own predicted utility (``sd_outcomes.sd_decision``).
    ``p11`` / ``p22`` are the players' own SD choices, ``p12`` is P1's
    prediction of P2 and ``p21`` is P2's prediction of P1.

All games are solved with ``evaluate_game.best_response_dynamics`` from
``(0, 0)``. The primary pass lets P1 move first; a diagnostic pass lets P2 move
first and reports every cell whose outcome depends on the start order.
Utilities and the outcome table are in ``sd_outcomes.py``.

Splits: ``80-30`` (NT), ``DD``, ``RS`` (and the NT side branch ``100-50``).
Outputs in ``usecases.output_dir(split)/c_grid/`` (``GT/results/<NT|DD|RS>/c_grid/``):

    * ``<mode>_<method>_c_grid.png``    measured-game NE: p1 map | p2 map
    * ``p1_SD/<mode>_<method>_c_grid.png``  P1's transformed SD game: p11 | p12
    * ``p2_SD/<mode>_<method>_c_grid.png``  P2's transformed SD game: p21 | p22
    * ``start_order_diagnostics.png`` (also in ``p1_SD/``, ``p2_SD/``)
      cells where letting P2 move first changes the outcome
    * ``<mode>_<method>_c_grid.npz``    cell-level arrays, both start orders
    * ``cells.csv``                     long-form cell table, all configs
    * ``summary.csv`` / ``summary.json`` per-configuration metrics

Usage (from the ``GT`` directory)::

    python utils/search_nash_c.py --split 80-30
    python utils/search_nash_c.py --split DD
    python utils/search_nash_c.py --split RS
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

GT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GT_ROOT))

import evaluate_game  # noqa: E402
import sd_outcomes  # noqa: E402
import usecases  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SPLITS = usecases.SPLITS          # NT "100-50" / "80-30", "DD", "RS"
METHODS = ("Noise", "Suppression")
SIZE_LABELS = {"full": "100% data", "half": "50% data",
               "80percent": "80% data", "30percent": "30% data",
               "DD_30": "DD 30% data", "DD_80": "DD 80% data",
               "RS_NF1": "RS NF 1x data", "RS_NF5": "RS NF 5x data"}
METHOD_LABELS = {"Noise": "DP (noise)", "Suppression": "SUP (suppression)"}

# LEGACY: the old fixed 0..0.49 C grid; the analysis uses the per-case grid
# (sd_outcomes.case_c_grid). Kept for c_grid() and old command lines.
C_MAX = 0.49
C_STEP = 0.01
TOL = 1e-9            # tolerance for comparing protection levels
DPI = 300
MIN_LABEL_CELLS = 12  # regions smaller than this get no text label
MIN_LABEL_DEPTH = 2   # ... and regions thinner than ~2 * this many cells

ORDERS = {"p1_first": 1, "p2_first": 2}
PRIMARY_ORDER = "p1_first"

# Categorical difference map shared by the misdirection and start-order plots
# (Okabe-Ito colours, colour-blind safe). Index = 2 * (p1 differs) + (p2 differs).
DIFF_CATEGORIES = (
    ("same", "#f3f3f3"),
    ("only $p_2$ differs", "#56b4e9"),
    ("only $p_1$ differs", "#e69f00"),
    ("$p_1$ and $p_2$ differ", "#cc79a7"),
)



# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
def c_grid(c_max=C_MAX, c_step=C_STEP):
    """Return the 1D grid of leakage weights (rounded to avoid float drift)."""
    count = int(round(c_max / c_step)) + 1
    return np.round(np.arange(count) * c_step, 10)


def analyse_configuration(mode, method, cs):
    """Evaluate every (C1, C2) cell for both start orders.

    Arrays are indexed ``[k1, k2]`` with ``C1 = cs[k1]`` and ``C2 = cs[k2]``.
    """
    protection = evaluate_game.get_protection(method, mode)
    out = {"C": cs, "protection": protection}
    for order, first in ORDERS.items():
        r = sd_outcomes.analyse(mode, method, cs, first=first)
        status = lambda ok: np.where(ok, "nash", "cycle").astype(object)  # noqa: E731
        a = {"i_star": r["i_star"], "j_star": r["j_star"],
             "i_sd": r["sd1_own"], "j_sd": r["sd2_own"],              # p11, p22
             "j_sd1": r["sd1_partner"], "i_sd2": r["sd2_partner"]}    # p12, p21
        s = {"status_true": status(r["true_converged"]),
             "status_sd1": status(r["sd1_converged"]),
             "status_sd2": status(r["sd2_converged"])}
        out[order] = {
            **a, **s,
            "p1_star": protection[a["i_star"]],
            "p2_star": protection[a["j_star"]],
            "p1_sd": protection[a["i_sd"]],
            "p2_sd": protection[a["j_sd"]],
            "p12_sd": protection[a["j_sd1"]],
            "p21_sd": protection[a["i_sd2"]],
        }
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def is_corner_only(*p_arrays):
    """Return whether every protection level is 0 or 1 within ``TOL``."""
    values = np.concatenate([np.ravel(p) for p in p_arrays])
    return bool(np.all((np.abs(values) <= TOL) | (np.abs(values - 1) <= TOL)))


def distinct_outcomes(p1, p2):
    """Distinct (p1, p2) pairs, rounded at the tolerance scale, with counts."""
    pairs = np.round(np.stack([p1.ravel(), p2.ravel()], axis=1), 9)
    values, counts = np.unique(pairs, axis=0, return_counts=True)
    return [{"p1": float(v[0]), "p2": float(v[1]), "cells": int(c)}
            for v, c in zip(values, counts)]


def diff_category(i_a, j_a, i_b, j_b):
    """0 same, 1 only p2 differs, 2 only p1 differs, 3 both differ."""
    return (i_a != i_b).astype(int) * 2 + (j_a != j_b).astype(int)


def compute_metrics(mode, method, res):
    """Summary metrics of one configuration (primary order + start-order diagnostics)."""
    r, o = res[PRIMARY_ORDER], res["p2_first"]
    total = r["i_star"].size
    pct = lambda mask: 100.0 * float(np.count_nonzero(mask)) / total  # noqa: E731

    mis = diff_category(r["i_star"], r["j_star"], r["i_sd"], r["j_sd"])
    ord_true = diff_category(r["i_star"], r["j_star"], o["i_star"], o["j_star"])
    ord_sd = diff_category(r["i_sd"], r["j_sd"], o["i_sd"], o["j_sd"])
    case = "B" if is_corner_only(r["p1_star"], r["p2_star"], r["p1_sd"], r["p2_sd"]) else "A"

    metrics = {
        "mode": mode,
        "size": SIZE_LABELS[mode],
        "method": method,
        "case": case,
        "cells": total,
        "match_pct": pct(mis == 0),
        "misdirected_pct": pct(mis > 0),
        "p1_misdirected_pct": pct(mis >= 2),
        "p2_misdirected_pct": pct(mis % 2 == 1),
        "mean_abs_dp1": float(np.mean(np.abs(r["p1_sd"] - r["p1_star"]))),
        "mean_abs_dp2": float(np.mean(np.abs(r["p2_sd"] - r["p2_star"]))),
    }
    for key, cat in (("true", ord_true), ("sd", ord_sd)):
        metrics[f"order_sensitive_{key}_cells"] = int(np.count_nonzero(cat))
        metrics[f"order_sensitive_{key}_pct"] = pct(cat > 0)
        metrics[f"order_sensitive_{key}_breakdown"] = {
            name: int(np.count_nonzero(cat == idx))
            for idx, (name, _) in enumerate(DIFF_CATEGORIES) if idx
        }
    metrics["nonconverged"] = {
        f"{order}_{key}": int(np.count_nonzero(res[order][key] != "nash"))
        for order in ORDERS for key in ("status_true", "status_sd1", "status_sd2")
    }
    metrics["distinct_true_outcomes"] = distinct_outcomes(r["p1_star"], r["p2_star"])
    metrics["distinct_sd_outcomes"] = distinct_outcomes(r["p1_sd"], r["p2_sd"])
    return metrics


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _step(cs):
    return float(cs[1] - cs[0]) if len(cs) > 1 else C_STEP


def _extent(cs):
    half = _step(cs) / 2
    return (cs[0] - half, cs[-1] + half, cs[0] - half, cs[-1] + half)


def _fmt_p(value):
    """Compact protection label: 0, 1, 0.21 (small RS / DD levels: 0.001, 0.005)."""
    if not value % 1:
        return f"{value:.0f}"
    if value < 0.01:
        return f"{value:.3g}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _c_ticks(cs):
    """Axis ticks for a C grid: 0.1 steps on the NT 0..0.49 grid, round values otherwise."""
    if cs[-1] >= 0.3:
        return np.round(np.arange(0, cs[-1] + 1e-9, 0.1), 2)
    from matplotlib.ticker import MaxNLocator
    return [t for t in MaxNLocator(5).tick_values(0, cs[-1]) if 0 <= t <= cs[-1] + 1e-12]


def boundary_segments(labels, cs):
    """Line segments separating neighbouring cells with different labels (x = C1, y = C2)."""
    half = _step(cs) / 2
    segs = []
    n1, n2 = labels.shape
    for k1 in range(n1 - 1):
        for k2 in np.flatnonzero(labels[k1, :] != labels[k1 + 1, :]):
            x = cs[k1] + half
            segs.append([(x, cs[k2] - half), (x, cs[k2] + half)])
    for k2 in range(n2 - 1):
        for k1 in np.flatnonzero(labels[:, k2] != labels[:, k2 + 1]):
            y = cs[k2] + half
            segs.append([(cs[k1] - half, y), (cs[k1] + half, y)])
    return segs


def draw_boundaries(ax, labels, cs, color, lw=1.6, ls="solid", alpha=1.0, zorder=4):
    ax.add_collection(LineCollection(boundary_segments(labels, cs), colors=color,
                                     linewidths=lw, linestyles=ls, alpha=alpha, zorder=zorder))


def label_regions(ax, labels, cs, text_of, skip=()):
    """Write ``text_of(k1, k2)`` at the deepest cell of every large, thick region."""
    for value in np.unique(labels):
        if value in skip:
            continue
        components, count = ndimage.label(labels == value)
        for comp in range(1, count + 1):
            mask = components == comp
            if np.count_nonzero(mask) < MIN_LABEL_CELLS:
                continue
            depth = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
            if depth.max() < MIN_LABEL_DEPTH:
                continue
            k1, k2 = np.unravel_index(np.argmax(depth), mask.shape)
            ax.text(cs[k1], cs[k2], text_of(k1, k2),
                    ha="center", va="center", fontsize=8.5, zorder=6,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                          "edgecolor": "#888888", "linewidth": 0.6, "alpha": 0.92})


def format_axes(ax, cs, title):
    lo, hi, _, _ = _extent(cs)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ticks = _c_ticks(cs)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"$C_1$ (P1 leakage weight)")
    ax.set_ylabel(r"$C_2$ (P2 leakage weight)")
    ax.set_title(title, fontsize=11.5, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def draw_categories(ax, cat, cs):
    ax.imshow(cat.T, origin="lower", extent=_extent(cs),
              cmap=ListedColormap([c for _, c in DIFF_CATEGORIES]),
              vmin=-0.5, vmax=len(DIFF_CATEGORIES) - 0.5, interpolation="nearest", zorder=1)


def level_colours(n_levels):
    """One distinct colour per protection level (index 0 = no defence ... n-1 = max)."""
    return plt.get_cmap("YlGnBu")(np.linspace(0.06, 0.95, n_levels))


def plot_level_maps(res, maps, suptitle, path):
    """Two protection-level maps over the (C1, C2) grid (P1 moves first).

    ``maps`` is a pair of ``(index_array, subscript, description)``.
    """
    cs = res["C"]
    protection = res["protection"]
    n_levels = len(protection)
    cmap = ListedColormap(level_colours(n_levels))
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_levels), n_levels)
    level_names = [_fmt_p(q) for q in protection]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), gridspec_kw={"wspace": 0.12})
    for ax, (idx, sub, desc) in zip(axes, maps):
        image = ax.imshow(idx.T, origin="lower", extent=_extent(cs), cmap=cmap, norm=norm,
                          interpolation="nearest", zorder=1)
        draw_boundaries(ax, idx, cs, "#222222", lw=1.4)

        def text_of(k1, k2, idx=idx, sub=sub):
            return f"$p_{{{sub}}}$ = {level_names[idx[k1, k2]]}"
        label_regions(ax, idx, cs, text_of)
        occurring = sorted(set(np.unique(idx).tolist()))
        format_axes(ax, cs, f"$p_{{{sub}}}$ ({desc})\n"
                            f"levels reached: {', '.join(level_names[k] for k in occurring)}")

    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, ticks=range(n_levels))
    cbar.ax.set_yticklabels(level_names)
    cbar.set_label("protection level $p$  (0 = no defence, 1 = maximum privacy)")
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_configuration(res, metrics, out_dir, stem):
    """Measured game (p1 | p2), P1's SD game (p11 | p12), P2's SD game (p21 | p22)."""
    r = res[PRIMARY_ORDER]
    tag = f"{metrics['size']} · {METHOD_LABELS[metrics['method']]} · P1 moves first"
    plot_level_maps(res, ((r["i_star"], "1", "P1 protection level"),
                          (r["j_star"], "2", "P2 protection level")),
                    f"Measured game NE · {tag}", out_dir / f"{stem}.png")
    plot_level_maps(res, ((r["i_sd"], "11", "P1's own SD choice"),
                          (r["j_sd1"], "12", "P1's prediction of P2")),
                    f"P1's self-division game NE (fitted $f$) · {tag}",
                    out_dir / "p1_SD" / f"{stem}.png")
    plot_level_maps(res, ((r["i_sd2"], "21", "P2's prediction of P1"),
                          (r["j_sd"], "22", "P2's own SD choice")),
                    f"P2's self-division game NE (fitted $f$) · {tag}",
                    out_dir / "p2_SD" / f"{stem}.png")


START_ORDER_VIEWS = {
    # folder: (title, rows of (row index key, column index key, row label), subscripts)
    "": ("Start-order sensitivity",
         (("i_star", "j_star", "True game NE"), ("i_sd", "j_sd", "Realized SD profile")),
         ("1", "2")),
    "p1_SD": ("Start-order sensitivity of P1's self-division game (fitted $f$)",
              (("i_sd", "j_sd1", "P1's SD game NE"),), ("11", "12")),
    "p2_SD": ("Start-order sensitivity of P2's self-division game (fitted $f$)",
              (("i_sd2", "j_sd", "P2's SD game NE"),), ("21", "22")),
}


# Start-order figure colours: a quiet neutral for "same outcome" and the first
# three categorical slots of the reference palette (validated all-pairs, CVD
# dE >= 9) for what changes. Index = diff_category: 0 same, 1 only the column
# player's p, 2 only the row player's p, 3 both.
ORDER_COLOURS = ("#ecebe7", "#eb6834", "#2a78d6", "#1baf7a")
SHORT_SIZE = {"full": "100%", "half": "50%", "80percent": "80%", "30percent": "30%"}
SHORT_METHOD = {"Noise": "DP", "Suppression": "Sup"}


def plot_start_order(split, results, title, rows, subs, path):
    """Where does letting P2 move first change the outcome? One panel per game x configuration.

    Rows: the games in ``rows``; columns: configurations, smaller size first,
    DP before Sup. Grey cells keep the P1-first outcome; coloured cells change,
    by which protection level changes. The share of changed cells is in each
    panel title (counts and breakdowns are in ``summary.json``).
    """
    small = SPLITS[split][1]
    keys = sorted(results, key=lambda k: (k[0] != small, k[1] != "Noise"))
    pa, pb = (f"$p_{{{sub}}}$" for sub in subs)
    nrows, ncols = len(rows), len(keys)
    # NT shares one C grid; RS / DD have one per configuration, so every panel keeps its own scale
    shared = len({float(results[k]["C"][-1]) for k in keys}) == 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1.2, 3.75 * nrows + 1.1),
                             squeeze=False, gridspec_kw={"hspace": 0.42, "wspace": 0.12 if shared else 0.38})
    cmap = ListedColormap(ORDER_COLOURS)
    for col, key in enumerate(keys):
        res = results[key]
        cs = res["C"]
        a, b = res["p1_first"], res["p2_first"]
        for row, (ik, jk, label) in enumerate(rows):
            ax = axes[row, col]
            cat = diff_category(a[ik], a[jk], b[ik], b[jk])
            ax.imshow(cat.T, origin="lower", extent=_extent(cs), cmap=cmap, vmin=-0.5, vmax=3.5,
                      interpolation="nearest")
            share = 100.0 * np.count_nonzero(cat) / cat.size
            verdict = "no change" if share == 0 else f"{share:.1f}% of cells change"
            ax.set_title(f"{usecases.mode(key[0]).label} · {SHORT_METHOD[key[1]]}\n{verdict}",
                         fontsize=12, fontweight="bold" if share >= 5 else "normal", pad=6)
            lo, hi, _, _ = _extent(cs)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ticks = _c_ticks(cs)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=9, length=2, colors="#52514e")
            for spine in ax.spines.values():
                spine.set_color("#c3c2b7")
            if row == nrows - 1:
                ax.set_xlabel("$C_1$", fontsize=11)
            elif shared:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel("$C_2$", fontsize=11)
                ax.annotate(label, xy=(-0.34 if shared else -0.42, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontsize=12.5, fontweight="bold")
            elif shared:
                ax.set_yticklabels([])
    names = ("same outcome", f"only {pb} changes", f"only {pa} changes", f"{pa} and {pb} change")
    handles = [Patch(facecolor=ORDER_COLOURS[k], edgecolor="#9a9a96", label=names[k]) for k in (0, 2, 1, 3)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.01 if nrows > 1 else 0.02))
    fig.suptitle(f"{title} ({split}): does letting P2 move first change the outcome?",
                 fontsize=14, y=1.0 if nrows > 1 else 0.93)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
CELL_FIELDS = ("p1_star", "p2_star", "p1_sd", "p2_sd", "p12_sd", "p21_sd")


def save_npz(res, path):
    """Save every per-cell array for both start orders."""
    arrays = {"C": res["C"], "protection": res["protection"]}
    for order in ORDERS:
        for key, value in res[order].items():
            arrays[f"{order}__{key}"] = value.astype(str) if value.dtype == object else value
    np.savez_compressed(path, **arrays)


def write_cells_csv(results, path):
    """Long-form table: one row per configuration x start order x (C1, C2)."""
    header = ["mode", "method", "order", "C1", "C2", "cell_star", "cell_sd", *CELL_FIELDS,
              "match", "status_true", "status_sd1", "status_sd2"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (mode, method), res in results.items():
            cs = res["C"]
            for order in ORDERS:
                r = res[order]
                for k1, c1 in enumerate(cs):
                    for k2, c2 in enumerate(cs):
                        cell = (k1, k2)
                        writer.writerow([
                            mode, method, order, f"{c1:.6g}", f"{c2:.6g}",
                            f"{r['i_star'][cell]}-{r['j_star'][cell]}",
                            f"{r['i_sd'][cell]}-{r['j_sd'][cell]}",
                            *(f"{r[field][cell]:.10g}" for field in CELL_FIELDS),
                            int(r["i_star"][cell] == r["i_sd"][cell]
                                and r["j_star"][cell] == r["j_sd"][cell]),
                            r["status_true"][cell], r["status_sd1"][cell], r["status_sd2"][cell],
                        ])


SUMMARY_COLUMNS = ("size", "method", "case", "match_pct", "misdirected_pct",
                   "p1_misdirected_pct", "p2_misdirected_pct", "mean_abs_dp1", "mean_abs_dp2",
                   "order_sensitive_true_cells", "order_sensitive_true_pct",
                   "order_sensitive_sd_cells", "order_sensitive_sd_pct")


def write_summary(split, metrics_list, cs, out_dir):
    """Write summary.csv / summary.json and print a readable table."""
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        for m in metrics_list:
            writer.writerow([f"{m[c]:.4g}" if isinstance(m[c], float) else m[c]
                             for c in SUMMARY_COLUMNS])
    payload = {
        "split": split,
        # Each configuration has its own grid (sd_outcomes.case_c_grid); the
        # global ``cs`` (--c-max / --c-step) is not used for the analysis.
        "grid": {"rule": "per configuration: 0 .. C_max, both players reach p = 1 on both axes; "
                         "band = last 10% of each axis (sd_outcomes.c_ranges)",
                 "points": sd_outcomes.GRID_POINTS, "cells": int(sd_outcomes.GRID_POINTS ** 2),
                 "C_max": {f"{m['mode']}/{m['method']}": m["C_max"] for m in metrics_list}},
        "tolerance": TOL,
        "primary_order": PRIMARY_ORDER,
        "definitions": {
            "match": "realized SD cell == true-NE cell (P1 moves first)",
            "p1/p2_misdirected": "p1_SD != p1* / p2_SD != p2*",
            "mean_abs_dp": "mean over cells of |p_SD - p*|",
            "case": "B if every true and SD outcome lies in {0,1}^2, else A",
            "order_sensitive": "outcome differs between P1-first and P2-first passes",
        },
        "configurations": metrics_list,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n = sd_outcomes.GRID_POINTS
    print(f"\n=== Summary {split} ({n}x{n} grid per configuration, P1 moves first) ===")
    head = (f"{'size':10s} {'method':11s} case  match%  P1 off%  P2 off%  "
            f"mean|dp1|  mean|dp2|   start-order variant: true / SD")
    print(head)
    print("-" * len(head))
    for m in metrics_list:
        print(f"{m['size']:10s} {m['method']:11s}  {m['case']}   {m['match_pct']:6.2f}  "
              f"{m['p1_misdirected_pct']:6.2f}   {m['p2_misdirected_pct']:6.2f}   "
              f"{m['mean_abs_dp1']:7.3f}    {m['mean_abs_dp2']:7.3f}     "
              f"{m['order_sensitive_true_cells']:4d} ({m['order_sensitive_true_pct']:5.2f}%) / "
              f"{m['order_sensitive_sd_cells']:4d} ({m['order_sensitive_sd_pct']:5.2f}%)")
    for m in metrics_list:
        bad = {k: v for k, v in m["nonconverged"].items() if v}
        if bad:
            print(f"  non-converged dynamics in {m['mode']}/{m['method']}: {bad}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_split(split, cs):
    """Analyse, plot and export the four configurations of one split."""
    out_dir = usecases.output_dir(split) / "c_grid"
    for sub in ("", "p1_SD", "p2_SD"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for stale in ("move_order_diagnostics.png",):
        (out_dir / stale).unlink(missing_ok=True)
    results, metrics_list = {}, []
    for mode in SPLITS[split]:
        for method in METHODS:
            start = time.perf_counter()
            # 50 x 50 from 0 to the case's C_max (both players reach the p = 1 band on both axes)
            case_cs = sd_outcomes.case_c_grid(mode, method)
            res = analyse_configuration(mode, method, case_cs)
            metrics = compute_metrics(mode, method, res)
            metrics["C_max"], metrics["C_step"] = float(case_cs[-1]), float(case_cs[1] - case_cs[0])
            results[(mode, method)] = res
            metrics_list.append(metrics)
            stem = f"{mode}_{method}_c_grid"
            save_npz(res, out_dir / f"{stem}.npz")
            plot_configuration(res, metrics, out_dir, stem)
            (out_dir / f"{stem}_squares.png").unlink(missing_ok=True)   # superseded
            print(f"[{split}] {mode}/{method}: case {metrics['case']}, "
                  f"{len(metrics['distinct_true_outcomes'])} distinct true NE outcomes, "
                  f"{len(metrics['distinct_sd_outcomes'])} distinct SD outcomes "
                  f"({time.perf_counter() - start:.1f}s)")
    for sub, (title, rows, subs) in START_ORDER_VIEWS.items():
        plot_start_order(split, results, title, rows, subs,
                         out_dir / sub / "start_order_diagnostics.png")
    write_cells_csv(results, out_dir / "cells.csv")
    write_summary(split, metrics_list, cs, out_dir)
    print(f"Saved outputs to {out_dir}/")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="80-30")
    # LEGACY: unused since the per-configuration grid (sd_outcomes.case_c_grid); kept for old calls.
    parser.add_argument("--c-max", type=float, default=C_MAX, help="LEGACY, unused")
    parser.add_argument("--c-step", type=float, default=C_STEP, help="LEGACY, unused")
    args = parser.parse_args(argv)
    cs = c_grid(args.c_max, args.c_step)
    for split in (SPLITS if args.split == "all" else (args.split,)):
        run_split(split, cs)


if __name__ == "__main__":
    main()
