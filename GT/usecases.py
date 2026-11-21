"""Where each use-case's game inputs live and where the shared outputs go.

Each use-case produces its results with its own pipeline. The shared
game-theory scripts (``evaluate_game``, ``sd_outcomes``) only need, per
dataset size ("mode"):

* the standard matrices in ``average_dir(mode)``:
  ``<Method>_{Full_FL,P1_Subnet,P2_Subnet}_Bimatrix_Avg.npy`` (game
  orientation [row choice, column choice]) and, for RS / DD,
  ``<Method>_grid.npy`` (normalised p, echo = 1 last);
* the fitted SD operator f in ``params_dir(split)/<Method>_params.json``;
* the per-player dataset size ``|D_n|`` in the unit f was fitted with.

NT writes these with its own ``NT/utils/average_plots.py`` + ``NT/utils/visualizer.py``
(in ``NT/results/``); RS and DD are copied from their own outputs into
``GT/inputs/<UC>/`` by ``import_usecase.py``. Every shared output goes to
``output_dir(split)`` = ``GT/results/<NT|DD|RS>/``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GT_ROOT = Path(__file__).resolve().parent            # <repo>/GT
REPO_ROOT = GT_ROOT.parent                             # <repo>: holds NT/, DD/, RS/, GT/
NT_ROOT = REPO_ROOT / "NT"                             # the NT experiment code and results

METHODS = ("Noise", "Suppression")


@dataclass(frozen=True)
class Mode:
    usecase: str      # "NT", "RS" or "DD"
    size: str         # "size1" (smaller) or "size2" (larger)
    label: str        # e.g. "30%"
    D_n: float        # one player's dataset size, in the unit of the use-case's fit


MODES = {
    # NT: millions of flows (one player's parquet)
    "full": Mode("NT", "size2", "100%", 1.581731),
    "half": Mode("NT", "size1", "50%", 0.790865),
    "80percent": Mode("NT", "size2", "80%", 1.265384),
    "30percent": Mode("NT", "size1", "30%", 0.474519),
    # RS: millions of ratings (Samples / 2 / 1e6, as in RS fit_per_mechanism.py)
    "RS_NF1": Mode("RS", "size1", "NF 1x", 1003980 / 2 / 1e6),
    "RS_NF5": Mode("RS", "size2", "NF 5x", 5019837 / 2 / 1e6),
    # DD: thousands of compounds (Samples / 2 / 1e3, as in DD fit_per_mechanism.py)
    "DD_30": Mode("DD", "size1", "30%", 70854 / 2 / 1e3),
    "DD_80": Mode("DD", "size2", "80%", 188945 / 2 / 1e3),
}

SPLITS = {   # split -> (large mode, small mode)
    "100-50": ("full", "half"),
    "80-30": ("80percent", "30percent"),
    "RS": ("RS_NF5", "RS_NF1"),
    "DD": ("DD_80", "DD_30"),
}

UNITS = {"NT": "millions of flows", "RS": "millions of ratings", "DD": "thousands of compounds"}


def mode(name):
    try:
        return MODES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset mode: {name}") from exc


def usecase_of(name):
    """Use-case of a mode or a split."""
    return mode(SPLITS[name][0] if name in SPLITS else name).usecase


def split_of(name):
    return next(split for split, members in SPLITS.items() if name in members)


def average_dir(name):
    """Folder with a mode's standard matrices."""
    if mode(name).usecase == "NT":
        return NT_ROOT / "results" / name / "accuracy" / "games" / "average"
    return GT_ROOT / "inputs" / mode(name).usecase / name


def params_dir(split):
    """Folder with the fitted operator f of a split (``<Method>_params.json``)."""
    if usecase_of(split) == "NT":
        return NT_ROOT / "results" / f"final_{split}"
    return GT_ROOT / "inputs" / usecase_of(split)


def output_dir(split):
    """Folder for the shared game-theory outputs of a split."""
    if usecase_of(split) == "NT":
        return GT_ROOT / "results" / ("NT" if split == "80-30" else f"NT_{split}")
    return GT_ROOT / "results" / usecase_of(split)


def stored_grid(method, name):
    """RS / DD protection grid stored with the matrices; None for NT (conf/base.yaml)."""
    if mode(name).usecase == "NT":
        return None
    return np.load(average_dir(name) / f"{method}_grid.npy")


def grid(method, name):
    """Normalised protection grid (echo = 1 last) of a mode."""
    stored = stored_grid(method, name)
    if stored is not None:
        return np.asarray(stored, dtype=float)
    from utils.visualizer import get_actual_grid   # NT: from NT/conf/base.yaml
    return get_actual_grid(method, name)


def nt_on_path():
    """Make the NT package (``utils``) importable."""
    import sys
    if str(NT_ROOT) not in sys.path:
        sys.path.insert(0, str(NT_ROOT))


nt_on_path()
