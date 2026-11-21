import os
import sys

# Guarantee strictly CPU-bound execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import time
import datetime
import random
import numpy as np
import torch
import flwr as fl
import flwr.common as flc
from flwr.server.strategy import FedAvg
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from client import RecommenderClient, get_parameters, set_parameters
from dataset import get_client_datasets_rs
from models import PureMatrixFactorization

# Canonical 4 base ratios spanning all 7 points via client symmetry
CANONICAL_RATIOS = {
    "Ratio_1_8": (1/8, "1/8", 8/1, "8/1"),
    "Ratio_1_4": (1/4, "1/4", 4/1, "4/1"),
    "Ratio_1_2": (1/2, "1/2", 2/1, "2/1"),
    "Ratio_1_1": (1.0, "1/1", 1.0, "1/1"),
}

RATIO_ORDER = ["1/8", "1/4", "1/2", "1/1", "2/1", "4/1", "8/1"]


def set_deterministic_environment(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


class DummyClientProxy(fl.server.client_proxy.ClientProxy):
    def __init__(self, cid: str):
        try:
            super().__init__(cid)
        except Exception:
            self.cid = cid
    def get_properties(self, ins, timeout, group_id=0): pass
    def get_parameters(self, ins, timeout, group_id=0): pass
    def fit(self, ins, timeout, group_id=0): pass
    def evaluate(self, ins, timeout, group_id=0): pass
    def reconnect(self, ins, timeout, group_id=0): pass


def make_base_conf(seed: int = 42, epochs: int = 2, lr: float = 0.0075, batch_size: int = 256, lambda_reg: float = 0.01):
    return {
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "lambda_reg": lambda_reg,
        "seed": seed,
    }


def train_standalone_model(train_ds, test_ds, client_id: int, seed: int, total_epochs: int,
                           num_users: int, num_items: int, latent_dim: int, max_norm: float,
                           lr: float, batch_size: int, lambda_reg: float, device: torch.device):
    set_deterministic_environment(seed)
    conf = make_base_conf(seed=seed, epochs=total_epochs, lr=lr, batch_size=batch_size, lambda_reg=lambda_reg)

    model = PureMatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=latent_dim,
        max_norm=max_norm
    ).to(device)

    client = RecommenderClient(
        model=model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        conf=conf,
        privacy_mode='none',
        privacy_param=0.0,
        dp_clip=1.0,
        cid=str(client_id)
    )

    client.fit(get_parameters(model), {})
    _, _, alone_metrics = client.evaluate(get_parameters(client.model), {})

    rmse = float(alone_metrics["rmse"])
    del model, client
    gc.collect()
    return rmse


def ratio_worker_task(task: dict):
    seed = task["seed"]
    curve = task["curve"]
    ratio_folder = task["ratio_folder"]
    path0 = task["path0"]
    path1 = task["path1"]
    rounds = task["rounds"]
    local_epochs = task["local_epochs"]
    latent_dim = task["latent_dim"]
    max_norm = task["max_norm"]
    lr = task["lr"]
    batch_size = task["batch_size"]
    lambda_reg = task["lambda_reg"]

    device = torch.device("cpu")
    torch.set_num_threads(max(1, mp.cpu_count() // task["total_workers"]))
    set_deterministic_environment(seed)

    # 1. Load datasets ONCE into memory (Zero redundant I/O)
    t0, te0 = get_client_datasets_rs(path0, privacy_param=0.0, privacy_mode='none', seed=seed)
    t1, te1 = get_client_datasets_rs(path1, privacy_param=0.0, privacy_mode='none', seed=seed)

    # 2. Derive global tensor boundaries from in-memory tensors
    num_users_0 = max(t0.users.max().item(), te0.users.max().item()) + 1
    num_users_1 = max(t1.users.max().item(), te1.users.max().item()) + 1
    num_items = max(
        t0.items.max().item(), te0.items.max().item(),
        t1.items.max().item(), te1.items.max().item()
    ) + 1

    # 3. Compute Standalone Baselines using in-memory datasets
    standalone_epochs = rounds * local_epochs
    alone_rmse_0 = train_standalone_model(
        t0, te0, 0, seed, standalone_epochs, num_users_0, num_items,
        latent_dim, max_norm, lr, batch_size, lambda_reg, device
    )
    alone_rmse_1 = train_standalone_model(
        t1, te1, 1, seed, standalone_epochs, num_users_1, num_items,
        latent_dim, max_norm, lr, batch_size, lambda_reg, device
    )

    # 4. Federated Model Instantiation & Training
    conf0 = make_base_conf(seed=seed, epochs=local_epochs, lr=lr, batch_size=batch_size, lambda_reg=lambda_reg)
    conf1 = make_base_conf(seed=seed, epochs=local_epochs, lr=lr, batch_size=batch_size, lambda_reg=lambda_reg)

    model0 = PureMatrixFactorization(num_users_0, num_items, embedding_dim=latent_dim, max_norm=max_norm).to(device)
    model1 = PureMatrixFactorization(num_users_1, num_items, embedding_dim=latent_dim, max_norm=max_norm).to(device)

    c0 = RecommenderClient(model0, t0, te0, conf0, 'none', 0.0, 1.0, "0")
    c1 = RecommenderClient(model1, t1, te1, conf1, 'none', 0.0, 1.0, "1")

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2
    )

    proxy0, proxy1 = DummyClientProxy("0"), DummyClientProxy("1")
    global_weights = get_parameters(model0)

    for rnd in range(1, rounds + 1):
        c0.set_parameters(global_weights)
        c1.set_parameters(global_weights)

        w0, num0, _ = c0.fit(global_weights, {})
        w1, num1, _ = c1.fit(global_weights, {})

        res0 = flc.FitRes(
            status=flc.Status(code=flc.Code.OK, message=""),
            parameters=flc.ndarrays_to_parameters(w0),
            num_examples=num0,
            metrics={}
        )
        res1 = flc.FitRes(
            status=flc.Status(code=flc.Code.OK, message=""),
            parameters=flc.ndarrays_to_parameters(w1),
            num_examples=num1,
            metrics={}
        )

        agg_params, _ = strategy.aggregate_fit(server_round=rnd, results=[(proxy0, res0), (proxy1, res1)], failures=[])
        if agg_params is not None:
            global_weights = flc.parameters_to_ndarrays(agg_params)

    # 5. Evaluate Joint Error
    _, _, fed_metrics_0 = c0.evaluate(global_weights, {})
    _, _, fed_metrics_1 = c1.evaluate(global_weights, {})

    fed_rmse_0 = float(fed_metrics_0["rmse"])
    fed_rmse_1 = float(fed_metrics_1["rmse"])

    # Relative Error Reduction: Gain = (Theta_alone - Phi_fed) / Theta_alone
    gain_0 = (alone_rmse_0 - fed_rmse_0) / alone_rmse_0 if alone_rmse_0 != 0 else 0.0
    gain_1 = (alone_rmse_1 - fed_rmse_1) / alone_rmse_1 if alone_rmse_1 != 0 else 0.0

    r_val_0, r_str_0, r_val_1, r_str_1 = CANONICAL_RATIOS[ratio_folder]

    records = [
        {
            "Seed": seed,
            "Curve": curve,
            "Client": 0,
            "Ratio_Val": r_val_0,
            "Ratio_Str": r_str_0,
            "RMSE_Alone": alone_rmse_0,
            "RMSE_Fed": fed_rmse_0,
            "Gain_Acc": gain_0
        },
        {
            "Seed": seed,
            "Curve": curve,
            "Client": 1,
            "Ratio_Val": r_val_1,
            "Ratio_Str": r_str_1,
            "RMSE_Alone": alone_rmse_1,
            "RMSE_Fed": fed_rmse_1,
            "Gain_Acc": gain_1
        }
    ]

    del model0, model1, c0, c1, t0, t1, te0, te1
    gc.collect()
    return records


def save_and_render_results(all_records: list, csv_dir: str, plot_dir: str):
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(csv_dir, "rs_ratio_results.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    curves = sorted(df["Curve"].unique())
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for idx, curve in enumerate(curves):
        sub_df = df[df["Curve"] == curve]
        grouped = sub_df.groupby("Ratio_Str")["Gain_Acc"].agg(["mean", "std", "count"]).reset_index()
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
        grouped["Ratio_Str"] = pd.Categorical(grouped["Ratio_Str"], categories=RATIO_ORDER, ordered=True)
        grouped = grouped.sort_values("Ratio_Str").reset_index(drop=True)

        color = colors[idx % len(colors)]
        plt.plot(grouped["Ratio_Str"], grouped["mean"], marker="o", color=color, linewidth=2.0, label=f"Curve {curve}")
        plt.fill_between(
            grouped["Ratio_Str"],
            grouped["mean"] - grouped["sem"],
            grouped["mean"] + grouped["sem"],
            color=color,
            alpha=0.15
        )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.title("Normalized Accuracy Improvement vs. Data Size Ratio (RecSys)")
    plt.xlabel("Datasize Ratio (Client / Partner)")
    plt.ylabel("Normalized Improvement Gain (RER)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")

    plot_path = os.path.join(plot_dir, "rs_ratio_improvement.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optimized CPU FL Ratio Experiment for RecSys")
    parser.add_argument("--src_root", type=str, default="experiments", help="Path to experiments folder")
    parser.add_argument("--out_csv_dir", type=str, default="src/Results/CSVs_Ratio_RS/")
    parser.add_argument("--out_plot_dir", type=str, default="src/Results/Plots_Ratio_RS/")
    parser.add_argument("--curves", nargs="+", default=["Curve_1x", "Curve_5x", "Curve_10x"], help="Curves to run")
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10, help="Flower FedAvg rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local SGD epochs per round")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0075)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--max_norm", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2, help="CPU worker processes")
    args = parser.parse_args()

    start_time = time.time()
    print("===========================================================")
    print(f" 🚀 LAUNCHING RS RATIO SUITE (CPU) | Seeds {args.seed_start}..{args.seed_end}")
    print(f" Directory Target: {args.src_root}")
    print("===========================================================")

    tasks = []
    for seed in range(args.seed_start, args.seed_end + 1):
        for curve in args.curves:
            # Iterate only through the 4 non-redundant ratio folders
            for ratio_folder in CANONICAL_RATIOS.keys():
                p0 = os.path.join(args.src_root, curve, ratio_folder, "Client_0")
                p1 = os.path.join(args.src_root, curve, ratio_folder, "Client_1")

                if not (os.path.exists(p0) and os.path.exists(p1)):
                    print(f"⚠️  Skipping missing directory: {p0} or {p1}")
                    continue

                tasks.append({
                    "seed": seed,
                    "curve": curve.replace("Curve_", ""),
                    "ratio_folder": ratio_folder,
                    "path0": p0,
                    "path1": p1,
                    "rounds": args.rounds,
                    "local_epochs": args.local_epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "lambda_reg": args.lambda_reg,
                    "latent_dim": args.latent_dim,
                    "max_norm": args.max_norm,
                    "total_workers": args.workers,
                })

    if not tasks:
        print(f"❌ Fatal: No valid data found in {args.src_root}. Check paths.")
        sys.exit(1)

    print(f"⚡ Enqueued {len(tasks)} tasks (4 canonical folders per curve/seed) across {args.workers} CPU workers.")

    all_records = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = {executor.submit(ratio_worker_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            try:
                records = future.result()
                all_records.extend(records)
                completed += 1
                print(f"   [{completed}/{len(tasks)}] Done: Curve {records[0]['Curve']} | Seed {records[0]['Seed']} | Evaluated Pair: {records[0]['Ratio_Str']} & {records[1]['Ratio_Str']}")
                save_and_render_results(all_records, args.out_csv_dir, args.out_plot_dir)
            except Exception as e:
                print(f"❌ Worker Failure: {e}")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("===========================================================")
    print(f" 🎉 COMPLETED RS RATIO SUITE in {total_time} | Results saved.")
    print("===========================================================")


if __name__ == "__main__":
    main()