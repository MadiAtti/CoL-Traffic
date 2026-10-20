"""
multi_seed_experiment.py
------------------------
Runs the local vs federated training experiment over seeds,
in parallel across seeds (one process per seed), saves per-seed JSON
results, then writes an averaged CSV, a per-seed analysis (std, min/max and
the seeds where they occur) and plots the curves (1x / 5x / 10x).

Output layout
-------------
results/multi-split
  seed_<n>.json
  averaged_results.csv
  per_seed_results.csv
  per_seed_stats.csv
  curves_all.png
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

SEEDS       = [10, 11, 12]        # seeds used in "run" mode
PLOT_SEEDS  = [0, 2, 3, 6, 7, 8, 9, 10, 11, 12]   # seeds loaded in "plot" mode
NUM_WORKERS = 2       # parallel seeds (tune to your CPU)

# --- "rerun" mode: re-train only the invalid (0.0) 10x entries -------------
RERUN_SEEDS   = None   # None = auto-detect every seed in PLOT_SEEDS with invalid entries
RERUN_KEYS    = [10]   # only the 10x dataset
RERUN_WORKERS = 1      # 1 = sequential, safest for RAM / Ray
MAX_RETRIES   = 3      # attempts per config if the federated result is invalid
RESULTS_DIR = Path("results/multi-split")

TARGET_COLUMN = "application_name"
BATCH_SIZE    = 256
LR            = 1e-3
NUM_EPOCHS    = 2
FED_ROUNDS    = 20
RATIOS        = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DATASET_KEYS  = [1, 5, 10]            # 1x / 5x / 10x

FEATURE_COLUMNS = [
    "protocol",
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "src2dst_stddev_ps",
    "src2dst_max_ps",
    "dst2src_min_ps",
    "dst2src_mean_ps",
    "dst2src_stddev_ps",
    "dst2src_max_ps",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",
    "bidirectional_rst_packets",
]

# Metrics analysed per seed in the "plot" mode analysis
ANALYSIS_METRICS = ["local_p1", "local_p2", "fed_p1", "fed_p2", "avg_gain", "norm_impr_p1"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (imported lazily inside workers to avoid top-level import issues)
# ──────────────────────────────────────────────────────────────────────────────

def _worker_imports():
    """Heavy imports done inside the worker process."""
    global fl, L, DataLoader, CustomDataset, TrafficLightningModule
    global get_on_fit_config, create_client_fn, player_specific_metrics
    global OmegaConf, set_seed, silence_log

    from logger_silencer import silence_log as _sl
    silence_log = _sl
    silence_log()

    import flwr as _fl
    import lightning as _L
    from omegaconf import OmegaConf as _OC
    from torch.utils.data import DataLoader as _DL

    from data.custom_dataset import CustomDataset as _CD
    from federated.server import get_on_fit_config as _gof
    from federated.universal_client import create_client_fn as _ccf
    from models.neural_network import TrafficLightningModule as _TLM
    from seed import set_seed as _ss
    from utils.metrics import player_specific_metrics as _psm

    fl                    = _fl
    L                     = _L
    DataLoader            = _DL
    OmegaConf             = _OC
    CustomDataset         = _CD
    TrafficLightningModule = _TLM
    get_on_fit_config     = _gof
    create_client_fn      = _ccf
    player_specific_metrics = _psm
    set_seed              = _ss


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(seed: int, keys=None) -> dict[int, pd.DataFrame]:
    keys = DATASET_KEYS if keys is None else keys
    base = pd.read_parquet("dataset/dataset.parquet")
    out: dict[int, pd.DataFrame] = {}
    if 1 in keys:
        out[1] = base.sample(frac=0.1, random_state=seed)
    if 5 in keys:
        out[5] = base.sample(frac=0.5, random_state=seed)
    if 10 in keys:
        out[10] = base
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────────────────────────

def make_loaders(p1_df, p2_df, le):
    def to_loader(df, shuffle=True):
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = le.transform(df[TARGET_COLUMN].values).astype(np.int64)
        return DataLoader(
            CustomDataset(X=X, y=y), batch_size=BATCH_SIZE, shuffle=shuffle
        )
    return to_loader(p1_df), to_loader(p2_df)


# ──────────────────────────────────────────────────────────────────────────────
# Local training
# ──────────────────────────────────────────────────────────────────────────────

def train_local(p1_df, p2_df, le) -> tuple[float, float]:
    silence_log()
    results = []
    for df in [p1_df, p2_df]:
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = le.transform(df[TARGET_COLUMN].values).astype(np.int64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=0
        )
        tr_ld = DataLoader(
            CustomDataset(X=X_tr, y=y_tr), batch_size=BATCH_SIZE, shuffle=True
        )
        te_ld = DataLoader(
            CustomDataset(X=X_te, y=y_te), batch_size=BATCH_SIZE, shuffle=False
        )
        model = TrafficLightningModule(
            input_dim=len(FEATURE_COLUMNS),
            num_classes=len(le.classes_),
            lr=LR,
        )
        trainer = L.Trainer(
            max_epochs=NUM_EPOCHS * FED_ROUNDS,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, tr_ld)
        m = trainer.validate(model, dataloaders=te_ld, verbose=False)[0]
        results.append(float(m["val_acc"]))

    return results[0], results[1]


# ──────────────────────────────────────────────────────────────────────────────
# Federated training
# ──────────────────────────────────────────────────────────────────────────────

def train_federated(p1_df, p2_df, le) -> tuple[float, float]:
    tr_p1, tr_p2 = make_loaders(p1_df, p2_df, le)
    te_p1, te_p2 = make_loaders(p1_df, p2_df, le)

    config = OmegaConf.create({
        "num_clients": 2,
        "config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "federated_rounds": FED_ROUNDS,
            "max_grad_norm": 1.0,
            "seed": 0,
        },
        "dataset": {
            "input_dim": len(FEATURE_COLUMNS),
            "num_classes": len(le.classes_),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "mode": "full",
        },
    })

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config(client1_noise=0.0, client2_noise=0.0),
        evaluate_metrics_aggregation_fn=player_specific_metrics,
    )

    history = fl.simulation.start_simulation(
        client_fn=create_client_fn([tr_p1, tr_p2], [te_p1, te_p2], config),
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=FED_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={
            "logging_level": logging.ERROR,
            "log_to_driver": False,
            "num_cpus": 2,
            "runtime_env": {"env_vars": {"OMP_NUM_THREADS": "1"}},
        },
    )

    p1_hist = history.metrics_distributed.get("client1_accuracy", [])
    p2_hist = history.metrics_distributed.get("client2_accuracy", [])
    p1_acc = p1_hist[-1][1] if p1_hist else 0.0
    p2_acc = p2_hist[-1][1] if p2_hist else 0.0

    if not (p1_acc > 0.0 and p2_acc > 0.0):
        # Diagnostics: show where the value gets lost
        print("  P1 acc history:", p1_hist, flush=True)
        print("  P2 acc history:", p2_hist, flush=True)
        print("  distributed losses:", history.losses_distributed, flush=True)
        raise RuntimeError(f"Invalid federated accuracy: p1={p1_acc}, p2={p2_acc}")

    return float(p1_acc), float(p2_acc)


# ──────────────────────────────────────────────────────────────────────────────
# Per-seed worker  (runs in a separate process)
# ──────────────────────────────────────────────────────────────────────────────

def is_valid_entry(entry: dict) -> bool:
    """Invalid = any accuracy is missing, NaN or exactly 0.0 (lost client evaluation)."""
    vals = [
        entry["local"]["player1"], entry["local"]["player2"],
        entry["federated"]["player1"], entry["federated"]["player2"],
    ]
    return all(v is not None and not np.isnan(v) and v > 0.0 for v in vals)


def find_bad_configs(seeds: list[int], dataset_keys=None) -> dict[int, list[str]]:
    """Scan seed_<n>.json files. Returns {seed: [invalid config tags]}."""
    bad: dict[int, list[str]] = {}
    for seed in seeds:
        path = RESULTS_DIR / f"seed_{seed}.json"
        if not path.exists():
            continue
        with open(path) as f:
            res = json.load(f)["results"]
        tags = [
            tag for tag, e in res.items()
            if (dataset_keys is None or e["dataset_key"] in dataset_keys)
            and not is_valid_entry(e)
        ]
        if tags:
            bad[seed] = tags
    return bad


def train_federated_checked(p1, p2, le, tag: str, seed: int):
    """Retry federated training if the result is invalid. Returns None if all attempts fail."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return train_federated(p1, p2, le)
        except RuntimeError as exc:
            print(f"[seed={seed}] {tag} attempt {attempt}/{MAX_RETRIES} failed: {exc}", flush=True)
    return None


def run_seed(seed: int, dataset_keys=None, only_tags=None) -> dict:
    """
    Trains dataset-sizes x ratios for one seed and writes results/seed_{seed}.json.

    dataset_keys : which datasets to run (default: all)
    only_tags    : if given, only these configs (e.g. {"10x_r0.125"}) are (re)run;
                   existing valid entries are kept, and a valid existing LOCAL result
                   is reused so only the federated part is retrained.
    The JSON is updated after every config, and invalid results are never saved.
    """
    _worker_imports()
    set_seed(seed)

    dataset_keys = DATASET_KEYS if dataset_keys is None else dataset_keys

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"seed_{seed}.json"

    seed_results: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            seed_results = json.load(f)["results"]

    datasets = load_dataset(seed, dataset_keys)

    for key in dataset_keys:
        dataset = datasets[key]
        le = LabelEncoder()
        le.fit(dataset[TARGET_COLUMN])

        for ratio in RATIOS:
            tag = f"{key}x_r{ratio}"
            if only_tags is not None and tag not in only_tags:
                continue

            split_idx = int(len(dataset) * ratio / (1 + ratio))
            p1 = dataset.iloc[:split_idx]
            p2 = dataset.iloc[split_idx:]
            print(f"[seed={seed}] {tag} | P1:{len(p1)} P2:{len(p2)}", flush=True)

            # reuse an existing valid local result (local training is the slow part)
            old = seed_results.get(tag)
            if old and old["local"]["player1"] > 0 and old["local"]["player2"] > 0:
                p1_loc, p2_loc = old["local"]["player1"], old["local"]["player2"]
            else:
                p1_loc, p2_loc = train_local(p1, p2, le)

            fed = train_federated_checked(p1, p2, le, tag, seed)
            if fed is None:
                print(f"[seed={seed}] {tag} SKIPPED (all {MAX_RETRIES} attempts invalid)", flush=True)
                continue
            p1_fed, p2_fed = fed

            seed_results[tag] = {
                "dataset_key": key,
                "ratio":       ratio,
                "local":       {"player1": p1_loc, "player2": p2_loc},
                "federated":   {"player1": p1_fed, "player2": p2_fed},
            }
            print(
                f"[seed={seed}] {tag} "
                f"local=({p1_loc:.4f},{p2_loc:.4f}) "
                f"fed=({p1_fed:.4f},{p2_fed:.4f})",
                flush=True,
            )

            # save after every config so progress is never lost
            with open(out_path, "w") as f:
                json.dump({"seed": seed, "results": seed_results}, f, indent=2)

    print(f"[seed={seed}] Saved -> {out_path}", flush=True)
    return seed_results


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(all_seed_results: list[dict]) -> pd.DataFrame:
    """
    Average metrics across seeds.
    Returns a DataFrame indexed by (dataset_key, ratio).
    """
    rows = []
    for seed_res in all_seed_results:
        for tag, v in seed_res.items():
            rows.append({
                "dataset_key":        v["dataset_key"],
                "ratio":              v["ratio"],
                "local_p1":           v["local"]["player1"],
                "local_p2":           v["local"]["player2"],
                "federated_p1":       v["federated"]["player1"],
                "federated_p2":       v["federated"]["player2"],
            })

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["dataset_key", "ratio"])
        .agg(
            local_p1_mean=("local_p1", "mean"),
            local_p1_std=("local_p1", "std"),
            local_p2_mean=("local_p2", "mean"),
            local_p2_std=("local_p2", "std"),
            fed_p1_mean=("federated_p1", "mean"),
            fed_p1_std=("federated_p1", "std"),
            fed_p2_mean=("federated_p2", "mean"),
            fed_p2_std=("federated_p2", "std"),
        )
        .reset_index()
    )
    return agg


def load_seed_curves(seeds: list[int]) -> dict[int, dict]:
    """Load seed_<n>.json files. Returns {seed: results_dict}."""
    results_by_seed: dict[int, dict] = {}
    for seed in seeds:
        path = RESULTS_DIR / f"seed_{seed}.json"
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping seed {seed}")
            continue
        with open(path) as f:
            data = json.load(f)
        results_by_seed[seed] = data["results"]
    return results_by_seed


# ──────────────────────────────────────────────────────────────────────────────
# Per-seed analysis (std, min/max and which seeds they come from)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_per_seed(results_by_seed: dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every (dataset_key, ratio) configuration and metric, compute the mean,
    std, min / max across seeds and the seed in which min / max occurs.

    Returns (long_df, stats_df):
      long_df  – one row per (seed, config) with all metrics
      stats_df – one row per (config, metric) with summary statistics
    """
    rows = []
    for seed, res in results_by_seed.items():
        for tag, v in res.items():
            lp1, lp2 = v["local"]["player1"], v["local"]["player2"]
            fp1, fp2 = v["federated"]["player1"], v["federated"]["player2"]
            rows.append({
                "seed": seed,
                "dataset_key": v["dataset_key"],
                "ratio": v["ratio"],
                "local_p1": lp1, "local_p2": lp2,
                "fed_p1": fp1, "fed_p2": fp2,
                "avg_gain": (fp1 + fp2) / 2 - (lp1 + lp2) / 2,
                "norm_impr_p1": (fp1 - lp1) / (1 - lp1) if lp1 < 1 else np.nan,
            })
    long_df = pd.DataFrame(rows)

    stats = []
    for (key, ratio), g in long_df.groupby(["dataset_key", "ratio"]):
        g = g.set_index("seed")
        for m in ANALYSIS_METRICS:
            s = g[m].dropna()
            if s.empty:
                continue
            stats.append({
                "dataset_key": key, "ratio": ratio, "metric": m,
                "n_seeds": len(s),
                "mean": s.mean(), "std": s.std(),
                "min": s.min(), "min_seed": s.idxmin(),
                "max": s.max(), "max_seed": s.idxmax(),
                "range": s.max() - s.min(),
            })
    stats_df = pd.DataFrame(stats)
    return long_df, stats_df


def print_seed_analysis(results_by_seed: dict[int, dict]) -> None:
    long_df, stats_df = analyze_per_seed(results_by_seed)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    for m in ANALYSIS_METRICS:
        sub = stats_df[stats_df["metric"] == m].drop(columns="metric")
        print(f"\n=== {m}: seedek közötti statisztika konfigurációnként ===")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Which seed is how often the min / max?
    print("\n=== Melyik seed hányszor min / max (avg_gain alapján) ===")
    g = stats_df[stats_df["metric"] == "avg_gain"]
    counts = pd.DataFrame({
        "times_min": g["min_seed"].value_counts(),
        "times_max": g["max_seed"].value_counts(),
    }).fillna(0).astype(int).sort_index()
    print(counts.to_string())

    # Seed × configuration table for avg_gain
    print("\n=== avg_gain: seed × (dataset_key, ratio) ===")
    pivot = long_df.pivot_table(
        index="seed", columns=["dataset_key", "ratio"], values="avg_gain"
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_path  = RESULTS_DIR / "per_seed_results.csv"
    stats_path = RESULTS_DIR / "per_seed_stats.csv"
    long_df.to_csv(long_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    print(f"\nMentve → {long_path}, {stats_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (one figure with 2 rows × 3 columns)
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(agg: pd.DataFrame, n_seeds: int):
    """
    One figure with 2 rows × 3 columns:
      Row 0: accuracy curves (local vs federated, P1 & P2, ±std)  for 1x / 5x / 10x
      Row 1: federated gain bar chart                              for 1x / 5x / 10x
    """
    # colour palette: same hue per series, darker shade for fed
    PALETTE = {
        "local_p1":  "#1f77b4",
        "local_p2":  "#ff7f0e",
        "fed_p1":    "#2ca02c",
        "fed_p2":    "#d62728",
    }

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    fig.suptitle(
        f"Local vs Federated accuracy  –  averaged over {n_seeds} seeds",
        fontsize=14, y=1.01,
    )

    for col_idx, key in enumerate(DATASET_KEYS):
        sub    = agg[agg["dataset_key"] == key].sort_values("ratio")
        ratios = sub["ratio"].values

        # ── top row: accuracy curves ──────────────────────────────────────────
        ax = axes[0, col_idx]
        for col_mean, col_std, label, color in [
            ("local_p1_mean", "local_p1_std", "Local P1",     PALETTE["local_p1"]),
            ("local_p2_mean", "local_p2_std", "Local P2",     PALETTE["local_p2"]),
            ("fed_p1_mean",   "fed_p1_std",   "Federated P1", PALETTE["fed_p1"]),
            ("fed_p2_mean",   "fed_p2_std",   "Federated P2", PALETTE["fed_p2"]),
        ]:
            y     = sub[col_mean].values
            y_std = sub[col_std].values
            ax.plot(ratios, y, marker="o", label=label, color=color, linewidth=1.8)
            ax.fill_between(ratios, y - y_std, y + y_std, alpha=0.13, color=color)

        ax.set_xscale("log")
        ax.set_xlabel("Data ratio  (P1 / P2)", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_title(f"Dataset {key}x", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

        # ── bottom row: federated gain bars ───────────────────────────────────
        ax2 = axes[1, col_idx]
        avg_local = (sub["local_p1_mean"].values + sub["local_p2_mean"].values) / 2
        avg_fed   = (sub["fed_p1_mean"].values   + sub["fed_p2_mean"].values)   / 2
        gain      = avg_fed - avg_local

        bar_colors = [PALETTE["fed_p1"] if g >= 0 else PALETTE["fed_p2"] for g in gain]
        ax2.bar(range(len(ratios)), gain, color=bar_colors, alpha=0.82)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(range(len(ratios)))
        ax2.set_xticklabels([str(r) for r in ratios], rotation=30, fontsize=8)
        ax2.set_xlabel("Data ratio", fontsize=9)
        ax2.set_ylabel("Avg gain  (fed − local)", fontsize=9)
        ax2.set_title(f"Federated gain  {key}x", fontsize=10)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out = RESULTS_DIR / "curves_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    mode = "plot"  # "run", "rerun" or "plot"

    results_by_seed: dict[int, dict] = {}

    if mode == "run":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Each seed runs in its own process to avoid Ray / Lightning state clashes.
        print(f"Launching {len(SEEDS)} seeds with {NUM_WORKERS} parallel workers ...")

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = {pool.submit(run_seed, s): s for s in SEEDS}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    results_by_seed[seed] = future.result()
                    print(f"OK Seed {seed} complete")
                except Exception as exc:
                    print(f"FAILED Seed {seed}: {exc}")

    elif mode == "rerun":
        seeds = PLOT_SEEDS if RERUN_SEEDS is None else RERUN_SEEDS
        bad = find_bad_configs(seeds, RERUN_KEYS)
        if not bad:
            print("No invalid entries found - nothing to rerun.")
        else:
            print("Invalid configs found:")
            for s, tags in bad.items():
                print(f"  seed {s}: {len(tags)} configs -> {tags}")
            print(f"\nRerunning {len(bad)} seeds with {RERUN_WORKERS} worker(s) ...")

            with ProcessPoolExecutor(max_workers=RERUN_WORKERS) as pool:
                futures = {
                    pool.submit(run_seed, s, RERUN_KEYS, set(tags)): s
                    for s, tags in bad.items()
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        future.result()
                        print(f"OK Seed {seed} rerun complete")
                    except Exception as exc:
                        print(f"FAILED Seed {seed} rerun: {exc}")

            still_bad = find_bad_configs(seeds, RERUN_KEYS)
            if still_bad:
                print("\nStill invalid after rerun:")
                for s, tags in still_bad.items():
                    print(f"  seed {s}: {tags}")
            else:
                print("\nAll previously invalid configs are now valid.")

        results_by_seed = load_seed_curves(seeds=PLOT_SEEDS)

    elif mode == "plot":
        results_by_seed = load_seed_curves(seeds=PLOT_SEEDS)

    else:
        raise ValueError(f"Unknown mode: {mode!r} (use 'run', 'rerun' or 'plot')")

    if not results_by_seed:
        print("No results collected - aborting.")
        return

    all_seed_results = list(results_by_seed.values())

    # per-seed analysis (std, min/max + seeds)
    print_seed_analysis(results_by_seed)

    # aggregate
    agg = aggregate(all_seed_results)

    csv_path = RESULTS_DIR / "averaged_results.csv"
    agg.to_csv(csv_path, index=False)
    print(f"\nAveraged results saved -> {csv_path}")
    print(agg.to_string(index=False))

    # plot
    plot_curves(agg, n_seeds=len(all_seed_results))

    print("\nDone. All outputs are in the results/ directory.")


if __name__ == "__main__":
    main()