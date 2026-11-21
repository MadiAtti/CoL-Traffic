import os
import sys
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
SPARSECHEM_ROOT = os.path.join(PROJECT_ROOT, "packages", "sparsechem")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SPARSECHEM_ROOT not in sys.path:
    sys.path.insert(0, SPARSECHEM_ROOT)

import sparsechem as sc
from client import DrugDiscoveryClient, get_parameters, set_parameters
from dataset import get_client_datasets

RATIO_MAP = {
    (10, 80): (1/8, "1/8", 8/1, "8/1"),
    (10, 40): (1/4, "1/4", 4/1, "4/1"),
    (10, 20): (1/2, "1/2", 2/1, "2/1"),
    (10, 10): (1.0, "1/1", 1.0, "1/1"),
}

CANONICAL_RATIO_DIRS = {
    "1/8": ["Ratio_1_8", "Ratio_10_80"],
    "1/4": ["Ratio_1_4", "Ratio_10_40"],
    "1/2": ["Ratio_1_2", "Ratio_10_20"],
    "1/1": ["Ratio_1_1", "Ratio_1", "Ratio_10_10", "Ratio_1.0", "Ratio_1-1"],
}

RATIO_ORDER = ["1/8", "1/4", "1/2", "1/1", "2/1", "4/1", "8/1"]
CURVES = ['10x']

def set_deterministic_environment(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def make_base_conf():
    c = sc.ModelConfig(
        input_size=32000,
        hidden_sizes=[40],
        output_size=2808,
        batch_size=64,
        lr=1e-3,
        last_dropout=0.2,
        weight_decay=1e-5,
        non_linearity="relu",
        last_non_linearity="relu",
    )
    c.epochs = 5
    return c

def resolve_ratio_directory(src_root: str, curve: str, ratio_key: str):
    patterns = CANONICAL_RATIO_DIRS[ratio_key]
    for pattern in patterns:
        candidate = os.path.join(src_root, f"Curve_{curve}", pattern)
        p0 = os.path.join(candidate, "Client_0")
        p1 = os.path.join(candidate, "Client_1")
        if os.path.exists(p0) and os.path.exists(p1):
            return p0, p1, pattern
    return None, None, None

def train_standalone_model(data_path: str, client_id: int, seed: int, rounds: int, device: torch.device):
    set_deterministic_environment(seed)
    conf = make_base_conf()
    conf.seed = seed
    conf.epochs = rounds * 5

    train_ds, test_ds = get_client_datasets(data_path, client_id, 0.0, 'none', conf, seed)
    model = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf)).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    client = DrugDiscoveryClient(model, train_ds, test_ds, conf, loss_fn, 'none', 0.0, 1.0, str(client_id), True)
    client.fit(get_parameters(model), {})
    _, _, alone_metrics = client.evaluate(get_parameters(client.model), {})

    acc = float(alone_metrics["accuracy"])
    del model, client, train_ds, test_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc

def ratio_worker_task(task: dict):
    seed = task["seed"]
    pct0, pct1 = task["pct0"], task["pct1"]
    path0, path1 = task["path0"], task["path1"]
    rounds = task["rounds"]
    gpu_id = task.get("gpu_id", None)

    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, mp.cpu_count() // task["total_workers"]))

    set_deterministic_environment(seed)

    # 1. Compute Standalone Baselines in Worker
    alone_acc_0 = train_standalone_model(path0, 0, seed, rounds, device)
    alone_acc_1 = train_standalone_model(path1, 1, seed, rounds, device)

    # 2. Federated Training Setup
    conf = make_base_conf()
    conf.seed = seed
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    t0, te0 = get_client_datasets(path0, 0, 0.0, 'none', conf, seed)
    t1, te1 = get_client_datasets(path1, 1, 0.0, 'none', conf, seed)

    model0 = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf)).to(device)
    model1 = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf)).to(device)

    c0 = DrugDiscoveryClient(model0, t0, te0, conf, loss_fn, 'none', 0.0, 1.0, "0", True)
    c1 = DrugDiscoveryClient(model1, t1, te1, conf, loss_fn, 'none', 0.0, 1.0, "1", True)

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

    # 3. Collaborative Evaluation
    _, _, fed_metrics_0 = c0.evaluate(global_weights, {})
    _, _, fed_metrics_1 = c1.evaluate(global_weights, {})

    fed_acc_0 = float(fed_metrics_0["accuracy"])
    fed_acc_1 = float(fed_metrics_1["accuracy"])

    # Relative Error Reduction (RER)
    gain_0 = (fed_acc_0 - alone_acc_0) / (1.0 - alone_acc_0) if (1.0 - alone_acc_0) != 0 else 0.0
    gain_1 = (fed_acc_1 - alone_acc_1) / (1.0 - alone_acc_1) if (1.0 - alone_acc_1) != 0 else 0.0

    r_val_0, r_str_0, r_val_1, r_str_1 = RATIO_MAP[(pct0, pct1)]

    records = [
        {
            "Seed": seed,
            "Client": 0,
            "Client_Pct": pct0,
            "Partner_Pct": pct1,
            "Ratio_Val": r_val_0,
            "Ratio_Str": r_str_0,
            "Acc_Alone": alone_acc_0,
            "Acc_Fed": fed_acc_0,
            "Gain_Acc": gain_0
        },
        {
            "Seed": seed,
            "Client": 1,
            "Client_Pct": pct1,
            "Partner_Pct": pct0,
            "Ratio_Val": r_val_1,
            "Ratio_Str": r_str_1,
            "Acc_Alone": alone_acc_1,
            "Acc_Fed": fed_acc_1,
            "Gain_Acc": gain_1
        }
    ]

    del model0, model1, c0, c1, t0, t1, te0, te1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records

def save_and_render_results(all_records: list, csv_dir: str, plot_dir: str):
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(csv_dir, "dd_ratio_results.csv")
    df.to_csv(csv_path, index=False)

    grouped = df.groupby("Ratio_Str")["Gain_Acc"].agg(["mean", "std", "count"]).reset_index()
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped["Ratio_Str"] = pd.Categorical(grouped["Ratio_Str"], categories=RATIO_ORDER, ordered=True)
    grouped = grouped.sort_values("Ratio_Str").reset_index(drop=True)

    plt.figure(figsize=(9, 5))
    plt.plot(grouped["Ratio_Str"], grouped["mean"], marker="o", color="#1f77b4", linewidth=2.5, label="Drug Discovery (DD)")
    plt.fill_between(
        grouped["Ratio_Str"],
        grouped["mean"] - grouped["sem"],
        grouped["mean"] + grouped["sem"],
        color="#1f77b4",
        alpha=0.2
    )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.title("Normalized Accuracy Improvement vs. Data Size Ratio (Appendix A)")
    plt.xlabel("Datasize Ratio (Client / Partner)")
    plt.ylabel("Normalized Improvement Gain (RER)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")

    plot_path = os.path.join(plot_dir, "dd_ratio_improvement.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Optimized FL Dataset Ratio Experiment (Appendix A)")
    parser.add_argument("--src_root", type=str, default=os.path.join(PROJECT_ROOT, "src", "Datasets"))
    parser.add_argument("--out_csv_dir", type=str, default=os.path.join(PROJECT_ROOT, "src", "Results", "CSVs_Ratio"))
    parser.add_argument("--out_plot_dir", type=str, default=os.path.join(PROJECT_ROOT, "src", "Results", "Plots_Ratio"))
    parser.add_argument("--seed_start", type=int, default=1, help="Initial seed index")
    parser.add_argument("--seed_end", type=int, default=10, help="Final seed index (inclusive)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2, help="Max parallel process workers")
    args = parser.parse_args()

    start_time = time.time()
    print("===========================================================")
    print(f" 🚀 LAUNCHING DD RATIO SUITE | Seeds {args.seed_start}..{args.seed_end}")
    print("===========================================================")

    tasks = []
    num_gpus = torch.cuda.device_count()

    for seed in range(args.seed_start, args.seed_end + 1):
        for curve in CURVES:
            for (pct0, pct1), (r_val, r_str, _, _) in RATIO_MAP.items():
                p0, p1, resolved_dir = resolve_ratio_directory(args.src_root, curve, r_str)
                if p0 is None:
                    print(f"⚠️  Missing split on disk for ratio {r_str}! Searched patterns: {CANONICAL_RATIO_DIRS[r_str]}")
                    continue

                gpu_id = (len(tasks) % num_gpus) if num_gpus > 0 else None
                tasks.append({
                    "seed": seed,
                    "curve": curve,
                    "pct0": pct0,
                    "pct1": pct1,
                    "path0": p0,
                    "path1": p1,
                    "rounds": args.rounds,
                    "gpu_id": gpu_id,
                    "total_workers": args.workers,
                })

    if not tasks:
        print("❌ Fatal: No valid ratio directories found. Check --src_root.")
        sys.exit(1)

    print(f"⚡ Enqueued {len(tasks)} tasks across {args.workers} workers (Detected GPUs: {num_gpus})")

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
                print(f"   [{completed}/{len(tasks)}] Done: Seed {records[0]['Seed']} | Ratio Pair {records[0]['Ratio_Str']}-{records[1]['Ratio_Str']}")
                save_and_render_results(all_records, args.out_csv_dir, args.out_plot_dir)
            except Exception as e:
                print(f"❌ Worker Failure: {e}")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("===========================================================")
    print(f" 🎉 COMPLETED in {total_time} | CSV & Plots Updated Successfully.")
    print("===========================================================")

if __name__ == "__main__":
    main()