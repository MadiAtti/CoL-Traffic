"""Copy the RS and DD results into the common end-point of ``utils/usecases.py``.

RS and DD keep their own pipelines (seed averaging, isotonic smoothing, fit of
the SD operator f, figures). This script only copies their finished outputs
into the standard matrices that the shared game-theory scripts read; nothing
is recomputed.

Inputs (their outputs)::

    surfaces  per-client isotonic grids, logged [P1_Param, P2_Param] orientation
              DD: ``DD/results/<Small_data_30%|Large_data_80%>/CSVs_Isotonic/``
                  ``<sup|dp>_<real|sim_p1|sim_p2>.csv`` (``Gain_Acc_Smoothed``)
              RS: ``RS/results/<small_data_NF1|large_data_NF5>/isotonic_csv/``
                  ``<sup|dp>_<real|sim_p1|sim_p2>.csv`` (``Isotonic_Gain_RMSE``)
              Client 0 owns P1_Param, client 1 owns P2_Param; in the SD files the
              clients are the halves (sim_p1: p11, p12; sim_p2: p21, p22).
    operator  their fit of f (paper Table ``tab:boosting``)
              DD: ``DD/results/Transformed_SD/data_30%/Difference/<mech>_difference_transformed.csv``
              RS: ``RS_PARAMS`` below, the RS ``fit_per_mechanism.py`` result as
                  recorded in ``DD/results/Transformed_SD/fit_per_mechanism.py``
                  (the RS ``Difference`` CSVs still hold an older 5-parameter fit).

Outputs
    ``usecases.average_dir(mode)`` = ``GT/inputs/<UC>/<mode>/``: bimatrices,
    ``Final_Real/Pred``, grid, manifest;
    ``usecases.params_dir(split)/<Method>_params.json``.

Usage (from ``GT``)::

    python utils/import_usecase.py            # RS and DD
    python utils/import_usecase.py --usecase DD
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GT_ROOT))

import usecases  # noqa: E402  (also puts NT on the path)

MECH_FILE = {"Noise": "dp", "Suppression": "sup"}
SOURCES = {   # mode -> (use-case, isotonic CSV folder, their Transformed_SD folder for a check)
    "DD_30": ("DD", "DD/results/Small_data_30%/CSVs_Isotonic", "DD/results/Transformed_SD/data_30%"),
    "DD_80": ("DD", "DD/results/Large_data_80%/CSVs_Isotonic", "DD/results/Transformed_SD/data_80%"),
    "RS_NF1": ("RS", "RS/results/small_data_NF1/isotonic_csv", "RS/results/Transformed_SD/NF_1"),
    "RS_NF5": ("RS", "RS/results/large_data_NF5/isotonic_csv", "RS/results/Transformed_SD/NF_5"),
}
VALUE = {"DD": "Gain_Acc_Smoothed", "RS": "Isotonic_Gain_RMSE"}
RS_PARAMS = {"Suppression": (0.758332, -0.0442311, 0.608741, 0.066067),
             "Noise": (1.1639, -0.0838379, 0.0754697, 0.0173887)}
DD_PARAMS_FILE = "DD/results/Transformed_SD/data_30%/Difference/{mech}_difference_transformed.csv"


def own_grid(df, client, column, levels):
    """One client's grid, from the logged orientation into its own [p_own, p_other]."""
    g = (df[df.Client == client]
         .pivot_table(index="P1_Param", columns="P2_Param", values=column, aggfunc="mean")
         .reindex(index=levels, columns=levels))
    if g.isna().any().any():
        raise ValueError(f"incomplete grid for client {client} ({column})")
    return g.to_numpy() if client == 0 else g.to_numpy().T


def their_params(usecase, method):
    if usecase == "RS":
        return RS_PARAMS[method], "RS fit_per_mechanism.py (RS_PARAMS in DD fit_per_mechanism.py)"
    path = DD_PARAMS_FILE.format(mech=MECH_FILE[method])
    row = pd.read_csv(usecases.REPO_ROOT / path).iloc[0]
    return tuple(float(row[k]) for k in ("x", "y", "alpha", "beta")), path


def import_mode(name):
    usecase, folder, check_folder = SOURCES[name]
    out = usecases.average_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": name, "usecase": usecase, "source": folder,
                "D_n": usecases.mode(name).D_n, "unit": usecases.UNITS[usecase], "methods": {}}
    for method in usecases.METHODS:
        frames = {s: pd.read_csv(usecases.REPO_ROOT / folder / f"{MECH_FILE[method]}_{s}.csv")
                  for s in ("real", "sim_p1", "sim_p2")}
        levels = sorted(frames["real"].P1_Param.unique())
        if levels[-1] != 1.0:
            raise ValueError(f"{name}/{method}: grid must end with the echo level p = 1")
        g = {(s, c): own_grid(df, c, VALUE[usecase], levels) for s, df in frames.items() for c in (0, 1)}
        p1, p2 = g[("real", 0)], g[("real", 1)]
        p11, p12, p21, p22 = (g[(s, c)] for s in ("sim_p1", "sim_p2") for c in (0, 1))
        real, pred = (p1 + p2) / 2, (p11 + p12 + p21 + p22) / 4
        np.save(out / f"{method}_grid.npy", np.asarray(levels, dtype=float))
        np.save(out / f"{method}_Full_FL_Bimatrix_Avg.npy", np.stack([p1, p2.T], -1))
        np.save(out / f"{method}_P1_Subnet_Bimatrix_Avg.npy", np.stack([p11, p12.T], -1))
        np.save(out / f"{method}_P2_Subnet_Bimatrix_Avg.npy", np.stack([p21, p22.T], -1))
        np.save(out / f"{method}_Final_Real.npy", real)
        np.save(out / f"{method}_Final_Pred.npy", pred)
        deviation = max(
            float(np.abs(arr - pd.read_csv(usecases.REPO_ROOT / check_folder / f"{kind}_{MECH_FILE[method]}.csv")
                         .pivot_table(index="P_own", columns="P_other", values="Gain_Acc").values).max())
            for kind, arr in (("real", real), ("sim", pred)))
        manifest["methods"][method] = {"grid": [float(v) for v in levels],
                                       "max_deviation_from_their_Transformed_SD": deviation}
        print(f"  {name}/{method}: Delta(0,0) = {real[0, 0]:.4g}, "
              f"max |surface - their Transformed_SD| = {deviation:.1e}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_params(usecase):
    split = usecase
    large, small = usecases.SPLITS[split]
    for method in usecases.METHODS:
        params, source = their_params(usecase, method)
        payload = {"method": method, "split": split, "usecase": usecase, "source": source,
                   "params": dict(zip(("x", "y", "alpha", "beta"), params)),
                   "D_n": {"small": usecases.mode(small).D_n, "big": usecases.mode(large).D_n}}
        path = usecases.params_dir(split) / f"{method}_params.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=4))
        print(f"  {usecase}/{method}: f = {params}  ({source})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--usecase", choices=("RS", "DD", "all"), default="all")
    args = parser.parse_args(argv)
    for usecase in ("DD", "RS"):
        if args.usecase in (usecase, "all"):
            for name, (uc, _, _) in SOURCES.items():
                if uc == usecase:
                    print(f"{name}:")
                    import_mode(name)
            write_params(usecase)


if __name__ == "__main__":
    main()
