"""
multi_seed_experiment.py
------------------------
Runs the local vs federated training experiment over seeds 0–9,
in parallel across seeds (one process per seed), saves per-seed JSON
results, then writes an averaged CSV and plots 3 curves (1x / 5x / 10x).

Output layout
-------------
results/multi-split
  seed_0.json
  seed_1.json
  ...
  seed_9.json
  averaged_results.csv
  curves_1x.png
  curves_5x.png
  curves_10x.png
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

SEEDS       = list(range(10))          # 0 … 9
NUM_WORKERS = min(4, len(SEEDS))       # parallel seeds (tune to your CPU)
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

def load_dataset(seed: int) -> dict[int, pd.DataFrame]:
    base = pd.read_parquet("dataset/dataset.parquet")
    return {
        1:  base.sample(frac=0.1, random_state=seed),
        5:  base.sample(frac=0.5, random_state=seed),
        10: base,
    }


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

    p1_acc = history.metrics_distributed["client1_accuracy"][-1][1]
    p2_acc = history.metrics_distributed["client2_accuracy"][-1][1]
    return float(p1_acc), float(p2_acc)


# ──────────────────────────────────────────────────────────────────────────────
# Per-seed worker  (runs in a separate process)
# ──────────────────────────────────────────────────────────────────────────────

def run_seed(seed: int) -> dict:
    """
    Trains all dataset-sizes × ratios for one seed.
    Returns a nested dict and also writes results/seed_{seed}.json.
    """
    _worker_imports()
    set_seed(seed)

    datasets = load_dataset(seed)
    seed_results: dict = {}   # key: (dataset_key, ratio)

    for key in DATASET_KEYS:
        dataset = datasets[key]
        le = LabelEncoder()
        le.fit(dataset[TARGET_COLUMN])

        for ratio in RATIOS:
            split_idx = int(len(dataset) * ratio / (1 + ratio))
            p1 = dataset.iloc[:split_idx]
            p2 = dataset.iloc[split_idx:]

            tag = f"{key}x_r{ratio}"
            print(f"[seed={seed}] {tag} | P1:{len(p1)} P2:{len(p2)}", flush=True)

            p1_loc, p2_loc = train_local(p1, p2, le)
            p1_fed, p2_fed = train_federated(p1, p2, le)

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

    # ── save per-seed JSON ────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"seed_{seed}.json"
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "results": seed_results}, f, indent=2)
    print(f"[seed={seed}] Saved → {out_path}", flush=True)

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


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (one figure per dataset_key)
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(agg: pd.DataFrame):
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
        f"Local vs Federated accuracy  –  averaged over {len(SEEDS)} seeds",
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_seed_results: list[dict] = []

    # ── parallel seed execution ───────────────────────────────────────────────
    # Each seed runs in its own process to avoid Ray / Lightning state clashes.
    # Reduce NUM_WORKERS if you run out of RAM.
    print(f"Launching {len(SEEDS)} seeds with {NUM_WORKERS} parallel workers …")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(run_seed, s): s for s in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                all_seed_results.append(result)
                print(f"✓ Seed {seed} complete")
            except Exception as exc:
                print(f"✗ Seed {seed} FAILED: {exc}")

    if not all_seed_results:
        print("No results collected — aborting.")
        return

    # ── aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate(all_seed_results)

    csv_path = RESULTS_DIR / "averaged_results.csv"
    agg.to_csv(csv_path, index=False)
    print(f"\nAveraged results saved → {csv_path}")
    print(agg.to_string(index=False))

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_curves(agg)

    print("\nDone. All outputs are in the results/ directory.")


if __name__ == "__main__":
    main()