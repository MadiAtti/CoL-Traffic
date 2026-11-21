"""Cross-domain validation of the Transformed Self-Division operator.

Implements the test of CoL-SD Section 6.3 ("Cross-Domain Validation",
Table ``tab:ver``). The operator ``f`` of Equation ``eq:scale``,

    f(Delta_tilde) = (x + y * ``|D_n|``) * (1 + alpha * p_own + beta * p_other) * Delta_tilde,

is fitted per *source* use-case and privacy mechanism. It is then applied to the
SD approximation ``Delta_tilde`` of every other *target* use-case (same
mechanism, the target's own ``|D_n|`` and privacy grid). The RMSE of the result
against the target's true ``Delta`` is reported for both target dataset sizes.
The last table rows give the RMSE of the uncorrected ``Delta_tilde`` for
comparison.

The definitions match ``utils/visualizer.py`` exactly, and so the in-domain fit
behind Table ``tab:boosting``:

    * ``Delta`` and ``Delta_tilde`` are the averaged, monotonized surfaces
      (``<method>_Final_Real.npy`` / ``<method>_Final_Pred.npy`` for NT);
    * ``f(Delta_tilde)`` is compared as it is (not monotonized), as in the
      RS / DD ``Transformed_SD/fit_per_mechanism.py``;
    * the cell where both players echo (last row, last column) is excluded
      from every RMSE.

DATA SLOTS
    Every input is declared in the ``DATA SLOTS`` section below: NT from its
    own pipeline (``average_plots.py`` + ``visualizer.py``), RS and DD from
    their own outputs, copied into the common end-point by
    ``import_usecase.py`` (surfaces and their own fit of f). With
    ``--data placeholder`` (the default) every input is replaced by a
    deterministic synthetic surface, for testing only; use ``--data real``.

Outputs (``GT/results/cross_validation/<data>/``):
    * ``cross_validation.json`` - every number, plus provenance per input
    * ``cross_validation.csv``  - one row per source x target x size
    * ``tab_ver_rows.tex``      - the body rows of Table ``tab:ver``
    * ``tab_boosting_nt.tex``   - the NT rows of Table ``tab:boosting``
    * ``tab_boosting_rows.tex`` - all rows of Table ``tab:boosting`` (RS, DD, NT)

Usage (from the ``GT`` directory)::

    python utils/cross_validation.py                   # all placeholders
    python utils/cross_validation.py --data real        # real where available
    python utils/cross_validation.py --data real --refit   # refit every f (slow)
"""

import argparse
import csv
import json
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GT_ROOT = Path(__file__).resolve().parent
if str(GT_ROOT) not in sys.path:
    sys.path.insert(0, str(GT_ROOT))

import usecases  # noqa: E402  (also puts NT on the path)
from utils import visualizer  # noqa: E402  (NT: apply_f, monotonize_matrix, rmse, grids)

PROJECT_ROOT = usecases.NT_ROOT   # NT slots below are relative to NT/

DOMAINS = ("RS", "DD", "NT")
MECHANISMS = ("Sup", "DP")
SIZES = ("size1", "size2")          # size1 = smaller dataset, size2 = larger
NT_METHOD = {"Sup": "Suppression", "DP": "Noise"}   # paper name -> NT code name
NT_MODE = {"size1": "30percent", "size2": "80percent"}
NT_SPLIT = "80-30"


# =========================================================================== #
# DATA SLOTS
# =========================================================================== #
@dataclass
class SurfaceSlot:
    """One use-case x mechanism x dataset size.

    ``true_path`` / ``sd_path`` point to ``.npy`` matrices in the
    ``[p_own, p_other]`` orientation, with the echo level (p = 1) last on
    both axes. ``grid`` holds the normalized protection levels of those axes.
    ``D_n`` is one player's dataset size, in the domain's unit.
    """
    D_n: float | None
    true_path: str | None = None
    sd_path: str | None = None
    grid: list | None = None           # None -> read the NT grid from conf/base.yaml


@dataclass
class DomainSlot:
    """One use-case: its size unit, its surfaces, and its fitted operator."""
    unit: str
    surfaces: dict = field(default_factory=dict)   # (mechanism, size) -> SurfaceSlot
    # (mechanism) -> (x, y, alpha, beta) or a path to a visualizer params JSON
    params: dict = field(default_factory=dict)
    params_source: str = ""


def _nt_slot(mech, size):
    method, mode = NT_METHOD[mech], NT_MODE[size]
    folder = f"results/{mode}/accuracy/games/average"
    return SurfaceSlot(
        D_n=visualizer.DATASET_SIZE_DESCRIPTORS[mode],     # millions of flows
        true_path=f"{folder}/{method}_Final_Real.npy",
        sd_path=f"{folder}/{method}_Final_Pred.npy",
        grid=None,
    )


def _imported_slot(mode, mech):
    """RS / DD surface as copied by utils/import_usecase.py into the common end-point."""
    folder = usecases.average_dir(mode)
    method = NT_METHOD[mech]
    return SurfaceSlot(D_n=usecases.mode(mode).D_n,
                       true_path=str(folder / f"{method}_Final_Real.npy"),
                       sd_path=str(folder / f"{method}_Final_Pred.npy"),
                       grid=list(usecases.stored_grid(method, mode)) if (folder / f"{method}_grid.npy").is_file() else None)


def _imported_domain(usecase, small, large):
    return DomainSlot(
        unit=usecases.UNITS[usecase],
        surfaces={(m, s): _imported_slot(small if s == "size1" else large, m)
                  for m in MECHANISMS for s in SIZES},
        # their own fit (paper Table tab:boosting), copied by utils/import_usecase.py
        params={m: str(usecases.params_dir(usecase) / f"{NT_METHOD[m]}_params.json") for m in MECHANISMS},
        params_source=f"{usecase} fit_per_mechanism.py (via utils/import_usecase.py)",
    )


DATA = {
    # RS - Recommender System, |D_n| in millions of ratings; DD - Drug Discovery,
    # |D_n| in thousands of compounds. Their outputs reach the common end-point via
    # python utils/import_usecase.py.
    "RS": _imported_domain("RS", "RS_NF1", "RS_NF5"),
    "DD": _imported_domain("DD", "DD_30", "DD_80"),
    # ----------------------------------------------------------------------- #
    # NT - Network Traffic (this repository). |D_n| in millions of flows.
    # The real surfaces and the fitted operator already exist.
    # ----------------------------------------------------------------------- #
    "NT": DomainSlot(
        unit="millions of flows",
        surfaces={(m, s): _nt_slot(m, s) for m in MECHANISMS for s in SIZES},
        params={m: f"results/final_{NT_SPLIT}/{NT_METHOD[m]}_params.json"
                for m in MECHANISMS},
        params_source=f"utils/visualizer.py fit, results/final_{NT_SPLIT}/",
    ),
}

# Used only for a slot that has no real data (or with --data placeholder).
# Rough magnitudes from the paper so the plumbing produces plausible numbers;
# these are NOT results.
PLACEHOLDER = {
    "grid": [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0],
    "gain": {"RS": 0.002, "DD": 0.11, "NT": 0.06},       # Delta at (0, 0)
    "D_n": {"RS": {"size1": 0.5, "size2": 1.0},
            "DD": {"size1": 30.0, "size2": 80.0},
            "NT": {"size1": 0.474519, "size2": 1.265384}},
    "params": (1.0, 0.0, 0.0, 0.0),                      # identity operator
}


# =========================================================================== #
# Loading
# =========================================================================== #
@dataclass
class Surface:
    true: np.ndarray
    sd: np.ndarray
    grid: np.ndarray
    D_n: float
    placeholder: bool
    provenance: str


def _stable_seed(*parts):
    return zlib.crc32("|".join(parts).encode())


def placeholder_surface(domain, mech, size):
    """Deterministic synthetic monotone Delta / Delta_tilde pair."""
    rng = np.random.default_rng(_stable_seed(domain, mech, size))
    grid = np.asarray(PLACEHOLDER["grid"], dtype=float)
    own, other = np.meshgrid(grid, grid, indexing="ij")
    gain = PLACEHOLDER["gain"][domain] * (1.25 if size == "size1" else 1.0)
    curve = 1.3 if mech == "Sup" else 1.0
    true = gain * (1.0 - 1.2 * own ** curve - 0.9 * other ** curve)
    true += rng.normal(0.0, 0.03 * gain, true.shape)
    # SD bias in both directions (paper Sec. 6.1): under-estimate for the smaller
    # dataset, over-estimate for the larger one, privacy-dependent on top.
    bias = 0.85 if size == "size1" else 1.15
    sd = true * bias * (1.0 + 0.15 * own - 0.10 * other)
    sd += rng.normal(0.0, 0.03 * gain, sd.shape)
    return (visualizer.monotonize_matrix(true), visualizer.monotonize_matrix(sd), grid,
            PLACEHOLDER["D_n"][domain][size])


def _nt_grid(mech, size):
    return visualizer.get_actual_grid(NT_METHOD[mech], mode=NT_MODE[size])


def load_surface(domain, mech, size, data_mode):
    """Return the real surface of a slot, or a placeholder if unavailable."""
    slot = DATA[domain].surfaces[(mech, size)]
    paths = [slot.true_path, slot.sd_path]
    real_ok = (data_mode == "real" and slot.D_n is not None
               and all(p and (PROJECT_ROOT / p).is_file() for p in paths))
    if not real_ok:
        true, sd, grid, D_n = placeholder_surface(domain, mech, size)
        reason = "forced (--data placeholder)" if data_mode == "placeholder" else "no real data"
        return Surface(true, sd, grid, D_n, True, f"PLACEHOLDER: {reason}")
    true = np.load(PROJECT_ROOT / slot.true_path)
    sd = np.load(PROJECT_ROOT / slot.sd_path)
    grid = np.asarray(slot.grid if slot.grid is not None else _nt_grid(mech, size), float)
    if true.shape != sd.shape or true.shape != (len(grid), len(grid)):
        raise ValueError(f"{domain}/{mech}/{size}: shapes {true.shape}, {sd.shape} "
                         f"do not match the {len(grid)}-level grid")
    return Surface(true, sd, grid, float(slot.D_n), False, f"{slot.true_path}, {slot.sd_path}")


def load_params(domain, mech, data_mode):
    """Return ((x, y, alpha, beta), provenance, placeholder?) of a source operator."""
    spec = DATA[domain].params.get(mech)
    if isinstance(spec, str):
        path = PROJECT_ROOT / spec
        if data_mode == "real" and path.is_file():
            p = json.loads(path.read_text())["params"]
            return (p["x"], p["y"], p["alpha"], p["beta"]), spec, False
        return PLACEHOLDER["params"], "PLACEHOLDER: identity operator", True
    if spec is None:
        return PLACEHOLDER["params"], "PLACEHOLDER: identity operator", True
    return tuple(spec), DATA[domain].params_source, data_mode == "placeholder"


# =========================================================================== #
# Core
# =========================================================================== #
def echo_mask(shape):
    """All cells except the one where both players echo."""
    mask = np.ones(shape, dtype=bool)
    mask[-1, -1] = False
    return mask


def transfer_rmse(params, target):
    """RMSE of f_source(Delta~_target) against Delta_target (not monotonized, as in RS / DD)."""
    transformed = visualizer.apply_f(target.sd, target.D_n, target.grid, params)
    return float(visualizer.rmse(transformed, target.true, mask=echo_mask(target.true.shape)))


def sd_rmse(target):
    """RMSE of the uncorrected Delta_tilde_target against Delta_target."""
    return float(visualizer.rmse(target.sd, target.true, mask=echo_mask(target.true.shape)))


def refit(domain, mech, surfaces):
    """Refit f for one source with the visualizer's search (same for all domains)."""
    small, big = surfaces[(domain, mech, "size1")], surfaces[(domain, mech, "size2")]
    params, *_ = visualizer.fit_f_joint(big.sd, big.true, big.D_n,
                                        small.sd, small.true, small.D_n,
                                        big.grid, small.grid)
    return tuple(float(v) for v in params)


def run(data_mode, do_refit):
    surfaces = {(d, m, s): load_surface(d, m, s, data_mode)
                for d in DOMAINS for m in MECHANISMS for s in SIZES}
    operators = {}
    for d in DOMAINS:
        for m in MECHANISMS:
            if do_refit:
                print(f"refitting f for {d}/{m} ...", flush=True)
                placeholder = any(surfaces[(d, m, s)].placeholder for s in SIZES)
                operators[(d, m)] = (refit(d, m, surfaces), "refit (utils/visualizer.fit_f_joint)",
                                     placeholder)
            else:
                operators[(d, m)] = load_params(d, m, data_mode)

    rows = []
    for (src, mech), (params, provenance, src_ph) in operators.items():
        x, y = params[0], params[1]
        for tgt in DOMAINS:
            for size in SIZES:
                target = surfaces[(tgt, mech, size)]
                base = sd_rmse(target)
                value = transfer_rmse(params, target)
                rows.append({
                    "source": src, "mechanism": mech, "target": tgt, "size": size,
                    "in_domain": src == tgt,
                    "D_n_target": target.D_n,
                    "g_target": x + y * target.D_n,     # size correction at target
                    "rmse_transfer": value,
                    "rmse_sd": base,
                    "ratio_to_sd": value / base if base > 0 else float("nan"),
                    "params": dict(zip(("x", "y", "alpha", "beta"), params)),
                    "params_provenance": provenance,
                    "placeholder": bool(src_ph or target.placeholder),
                })
    return surfaces, operators, rows


# =========================================================================== #
# Output
# =========================================================================== #
def fmt(value, placeholder=False):
    """Three significant figures, as in the paper tables; dagger on placeholders."""
    text = f"${value:.3g}$"
    return text + r"$^\dagger$" if placeholder else text


def tab_ver_rows(rows):
    """Body rows of Table tab:ver (source blocks, then the SD baseline rows)."""
    by_key = {(r["source"], r["mechanism"], r["target"], r["size"]): r for r in rows}
    lines = []
    for i, src in enumerate(DOMAINS):
        for j, mech in enumerate(MECHANISMS):
            cells = []
            for tgt in DOMAINS:
                for size in SIZES:
                    r = by_key[(src, mech, tgt, size)]
                    cells.append("-" if r["in_domain"] else fmt(r["rmse_transfer"], r["placeholder"]))
            head = (r"\multirow{6}{*}{\rotatebox{90}{\bf{Source}}}" if i == j == 0 else "")
            dom = rf"\multirow{{2}}{{*}}{{\textit{{{src}}}}}" if j == 0 else ""
            if head:
                lines.append(f"        {head}")
            lines.append(f"        & {dom} & {mech} & " + " & ".join(cells) + r" \\")
    lines.append(r"        \hline")
    for j, mech in enumerate(MECHANISMS):
        cells = []
        for tgt in DOMAINS:
            for size in SIZES:
                r = by_key[(tgt, mech, tgt, size)]   # baseline is source-independent
                cells.append(fmt(r["rmse_sd"], r["placeholder"]))
        lead = (r"\multicolumn{2}{c}{\multirow{2}{*}{SD $\widetilde{\Delta}$}}" if j == 0
                else r"\multicolumn{2}{c}{}")
        lines.append(f"        {lead} & {mech} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def tab_boosting_nt(rows):
    """NT rows of Table tab:boosting (in-domain parameters and RMSEs)."""
    by_key = {(r["source"], r["mechanism"], r["target"], r["size"]): r for r in rows}
    lines = []
    for j, mech in enumerate(MECHANISMS):
        r1, r2 = by_key[("NT", mech, "NT", "size1")], by_key[("NT", mech, "NT", "size2")]
        ph = r1["placeholder"] or r2["placeholder"]
        p = r1["params"]
        cells = [fmt(p[k], ph) for k in ("x", "y", "alpha", "beta")]
        cells += [fmt(r1["rmse_sd"], ph), fmt(r1["rmse_transfer"], ph),
                  fmt(r2["rmse_sd"], ph), fmt(r2["rmse_transfer"], ph)]
        lead = r"    \multirow{2}{*}{\textit{NT}}" if j == 0 else "    "
        lines.append(f"{lead} & {mech} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def tab_boosting_all(rows):
    """Rows of Table tab:boosting for RS, DD and NT (in-domain parameters and RMSEs)."""
    by_key = {(r["source"], r["mechanism"], r["target"], r["size"]): r for r in rows}
    lines = []
    for i, uc in enumerate(DOMAINS):
        if i:
            lines.append(r"    \midrule")
        for j, mech in enumerate(MECHANISMS):
            r1, r2 = by_key[(uc, mech, uc, "size1")], by_key[(uc, mech, uc, "size2")]
            ph = r1["placeholder"] or r2["placeholder"]
            p = r1["params"]
            cells = [fmt(p[k], ph) for k in ("x", "y", "alpha", "beta")]
            cells += [fmt(r1["rmse_sd"], ph), fmt(r1["rmse_transfer"], ph),
                      fmt(r2["rmse_sd"], ph), fmt(r2["rmse_transfer"], ph)]
            lead = rf"    \multirow{{2}}{{*}}{{\textit{{{uc}}}}}" if j == 0 else "    "
            lines.append(f"{lead} & {mech} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def summarize(rows):
    """Numbers the Section 6.3 text needs (factor range, negative size corrections)."""
    cross = [r for r in rows if not r["in_domain"]]
    worse = [r for r in cross if r["ratio_to_sd"] > 1]
    ratios = [r["ratio_to_sd"] for r in cross]
    into_nt = [r["ratio_to_sd"] for r in cross if r["target"] == "NT"]
    from_nt = [r["ratio_to_sd"] for r in cross if r["source"] == "NT"]
    return {
        "cross_cells": len(cross),
        "cells_worse_than_sd": len(worse),
        "ratio_to_sd_min": min(ratios), "ratio_to_sd_max": max(ratios),
        "ratio_into_NT": [min(into_nt), max(into_nt)],
        "ratio_from_NT": [min(from_nt), max(from_nt)],
        "negative_g": [f"{r['source']}->{r['target']} {r['mechanism']} {r['size']}: "
                       f"g = {r['g_target']:.3g}" for r in cross if r["g_target"] < 0],
        "any_placeholder": any(r["placeholder"] for r in rows),
    }


def write_outputs(data_mode, surfaces, operators, rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    banner = "% PLACEHOLDER DATA: values marked with a dagger are not results.\n"
    nt_ph = any(r["placeholder"] for r in rows if r["source"] == r["target"] == "NT")
    (out_dir / "tab_ver_rows.tex").write_text(
        (banner if summary["any_placeholder"] else "") + tab_ver_rows(rows) + "\n")
    (out_dir / "tab_boosting_nt.tex").write_text(
        (banner if nt_ph else "") + tab_boosting_nt(rows) + "\n")
    in_domain_ph = any(r["placeholder"] for r in rows if r["in_domain"])
    (out_dir / "tab_boosting_rows.tex").write_text(
        (banner if in_domain_ph else "") + tab_boosting_all(rows) + "\n")

    with open(out_dir / "cross_validation.csv", "w", newline="") as f:
        cols = ["source", "mechanism", "target", "size", "in_domain", "D_n_target",
                "g_target", "rmse_transfer", "rmse_sd", "ratio_to_sd", "placeholder"]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "data_mode": data_mode,
        "definition": "RMSE(mono(f_source(Delta_tilde_target)), Delta_target), "
                      "echo/echo cell excluded; same mechanism for source and target",
        "summary": summary,
        "inputs": {f"{d}/{m}/{s}": {"placeholder": v.placeholder, "D_n": v.D_n,
                                    "grid": v.grid.tolist(), "provenance": v.provenance}
                   for (d, m, s), v in surfaces.items()},
        "operators": {f"{d}/{m}": {"params": list(p), "provenance": src, "placeholder": ph}
                      for (d, m), (p, src, ph) in operators.items()},
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    # provenance paths relative to the repo root, so no local absolute path is written
    text = text.replace(json.dumps(str(usecases.REPO_ROOT) + "/")[1:-1], "")
    (out_dir / "cross_validation.json").write_text(text)
    return summary


def print_table(rows, summary):
    by_key = {(r["source"], r["mechanism"], r["target"], r["size"]): r for r in rows}
    header = f"{'source':10s}" + "".join(f"{t + ' ' + s:>13s}" for t in DOMAINS for s in SIZES)
    print("\nTransferred RMSE  (* = placeholder input)")
    print(header)
    print("-" * len(header))
    for src in DOMAINS:
        for mech in MECHANISMS:
            cells = []
            for tgt in DOMAINS:
                for size in SIZES:
                    r = by_key[(src, mech, tgt, size)]
                    cells.append("-" if r["in_domain"]
                                 else f"{r['rmse_transfer']:.3g}{'*' if r['placeholder'] else ''}")
            print(f"{src + ' ' + mech:10s}" + "".join(f"{c:>13s}" for c in cells))
    for mech in MECHANISMS:
        cells = [f"{by_key[(t, mech, t, s)]['rmse_sd']:.3g}" for t in DOMAINS for s in SIZES]
        print(f"{'SD ' + mech:10s}" + "".join(f"{c:>13s}" for c in cells))
    print(f"\nTransfer / SD ratio over {summary['cross_cells']} cross cells: "
          f"{summary['ratio_to_sd_min']:.3g} .. {summary['ratio_to_sd_max']:.3g}; "
          f"worse than SD in {summary['cells_worse_than_sd']}")
    for line in summary["negative_g"]:
        print(f"  negative size correction: {line}")
    if summary["any_placeholder"]:
        print("\nWARNING: some inputs are placeholders; see cross_validation.json 'inputs'.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", choices=("placeholder", "real"), default="placeholder",
                        help="placeholder: synthetic inputs everywhere (default); "
                             "real: use every slot that has data, placeholder elsewhere")
    parser.add_argument("--refit", action="store_true",
                        help="refit f for every source with visualizer.fit_f_joint (slow)")
    parser.add_argument("--out", default=None, help="output folder")
    args = parser.parse_args(argv)

    surfaces, operators, rows = run(args.data, args.refit)
    out_dir = Path(args.out) if args.out else GT_ROOT / "results" / "cross_validation" / args.data
    summary = write_outputs(args.data, surfaces, operators, rows, out_dir)
    print_table(rows, summary)
    print(f"\nSaved outputs to {out_dir}/")


if __name__ == "__main__":
    main()
