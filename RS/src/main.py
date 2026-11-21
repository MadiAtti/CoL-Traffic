import os
import argparse
import random
import glob
import pandas as pd
import numpy as np
import torch
import time
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.isotonic import IsotonicRegression

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

import flwr as fl
from flwr.server.strategy import FedAvg

from dataset import get_client_datasets_rs
from models import PureMatrixFactorization
from client import RecommenderClient, get_parameters

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================

# Small dataset
PRIVACY_PARAMS_BY_MECH = {
    "dp": [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0],
    "sup": [0.0, 0.05, 0.08, 0.2, 0.4, 0.8, 1.0],
}

# Large dataset
PRIVACY_PARAMS_BY_MECH = {
    "dp": [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0],
    "sup": [0.0, 0.05, 0.08, 0.2, 0.4, 0.8, 1.0],
}

SCENARIO_MAP = {"full": "real", "p1": "sim_p1", "p2": "sim_p2"}
CLIENT_MAP = {"full": {0: "P1", 1: "P2"}, "p1": {0: "P11", 1: "P12"}, "p2": {0: "P21", 1: "P22"}}

class DummyClientProxy(fl.server.client_proxy.ClientProxy):
    def __init__(self, cid):
        try: super().__init__(cid)
        except Exception: self.cid = cid
    def get_properties(self, ins, timeout, group_id=0): pass
    def get_parameters(self, ins, timeout, group_id=0): pass
    def fit(self, ins, timeout, group_id=0): pass
    def evaluate(self, ins, timeout, group_id=0): pass
    def reconnect(self, ins, timeout, group_id=0): pass

def set_deterministic_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_global_vocabulary(data_root):
    parquet_files = glob.glob(os.path.join(data_root, "**/*.parquet"), recursive=True)
    max_u, max_i = 0, 0
    for f in parquet_files:
        df = pd.read_parquet(f)
        if 'user_id' in df.columns: max_u = max(max_u, df['user_id'].max())
        if 'item_id' in df.columns: max_i = max(max_i, df['item_id'].max())
    return int(max_u) + 1, int(max_i) + 1

def get_data_paths_for_scenario(base_dir, scenario):
    if scenario == "full": return os.path.join(base_dir, "client_1"), os.path.join(base_dir, "client_2")
    elif scenario == "p1": return os.path.join(base_dir, "client_1/self_division/sub_client_1"), os.path.join(base_dir, "client_1/self_division/sub_client_2")
    elif scenario == "p2": return os.path.join(base_dir, "client_2/self_division/sub_client_1"), os.path.join(base_dir, "client_2/self_division/sub_client_2")
    raise ValueError("Invalid Scenario")

def train_and_eval_local_baseline_rs(data_path, client_id, seed, vocab_sizes):
    """Establishes the Alone baseline using the absolute 20 iteration budget."""
    set_deterministic_environment(seed)
    num_users, num_items = vocab_sizes
    
    # 20 total iterations for the standalone model
    conf = {'batch_size': 256, 'lr': 0.0075, 'epochs': 20, 'lambda_reg': 0.01, 'seed': seed}
    
    train_ds, test_ds = get_client_datasets_rs(data_path, 0.0, 'none', seed)
    model = PureMatrixFactorization(num_users, num_items, embedding_dim=4, max_norm=0.5)
    c = RecommenderClient(model, train_ds, test_ds, conf, 'none', 0.0, 1.0, client_id)
    
    c.fit([model.item_emb.weight.cpu().detach().numpy()], {})
    _, _, alone_metrics = c.evaluate([c.model.item_emb.weight.cpu().detach().numpy()], {})
    return {"alone_rmse": alone_metrics["rmse"]}

def worker_task(kwargs):
    torch.set_num_threads(1)
    seed, scenario, mech = kwargs["seed"], kwargs["scenario"], kwargs["mech"]
    p1_val, p2_val = kwargs["p1_val"], kwargs["p2_val"]
    vocab_sizes, baselines, rounds = kwargs["vocab_sizes"], kwargs["baselines"], kwargs["rounds"]
    
    set_deterministic_environment(seed)
    num_users, num_items = vocab_sizes
    
    # DYNAMIC BUDGET: If 5 rounds, local epochs = 4. Total iterations = 20.
    local_epochs = max(1, int(20 / rounds))
    conf = {'batch_size': 256, 'lr': 0.0075, 'epochs': local_epochs, 'lambda_reg': 0.01, 'seed': seed}
    dp_clip_rate = 1.0 
    
    path1, path2 = get_data_paths_for_scenario(kwargs["data_root"], scenario)
    t1, te1 = get_client_datasets_rs(path1, p1_val, mech, seed)
    t2, te2 = get_client_datasets_rs(path2, p2_val, mech, seed)
    
    model1 = PureMatrixFactorization(num_users, num_items, embedding_dim=4, max_norm=0.5)
    model2 = PureMatrixFactorization(num_users, num_items, embedding_dim=4, max_norm=0.5)
    
    c1 = RecommenderClient(model1, t1, te1, conf, mech, p1_val, dp_clip_rate, "0")
    c2 = RecommenderClient(model2, t2, te2, conf, mech, p2_val, dp_clip_rate, "1")
    
    global_weights = [model1.item_emb.weight.cpu().detach().numpy()]
    strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=2, min_available_clients=2)
    proxy1, proxy2 = DummyClientProxy("0"), DummyClientProxy("1")
    
    for rnd in range(1, rounds + 1):
        w1, num1, _ = c1.fit(global_weights, {})
        w2, num2, _ = c2.fit(global_weights, {})

        # Prevent FedAvg zero-division crash when both players operate at p >= 1.0 (returning 0 examples)
        if num1 == 0 and num2 == 0:
            num1, num2 = 1, 1

        res1 = fl.common.FitRes(status=fl.common.Status(code=fl.common.Code.OK, message=""), parameters=fl.common.ndarrays_to_parameters(w1), num_examples=num1, metrics={})
        res2 = fl.common.FitRes(status=fl.common.Status(code=fl.common.Code.OK, message=""), parameters=fl.common.ndarrays_to_parameters(w2), num_examples=num2, metrics={})
        agg_params, _ = strategy.aggregate_fit(server_round=rnd, results=[(proxy1, res1), (proxy2, res2)], failures=[])
        if agg_params is not None: global_weights = fl.common.parameters_to_ndarrays(agg_params)

    _, _, fed_metrics_1 = c1.evaluate(global_weights, {})
    _, _, fed_metrics_2 = c2.evaluate(global_weights, {})
    
    records = []
    for c_id, fed_metrics in zip([0, 1], [fed_metrics_1, fed_metrics_2]):
        b = baselines[c_id]
        alone_err = b["alone_rmse"]
        fed_err = fed_metrics["rmse"]
        
        # Relative Error Reduction
        raw_gain_rmse = (alone_err - fed_err) / alone_err if alone_err != 0 else 0.0
        
        records.append({
            "Seed": seed, "Scenario": scenario, "Mechanism": mech, "Client": c_id,
            "P1_Param": p1_val, "P2_Param": p2_val,
            "RMSE_Alone": alone_err, "RMSE_Fed": fed_err, "Raw_Gain_RMSE": raw_gain_rmse
        })
    return records

def apply_2d_isotonic_regression(matrix):
    """Applies Isotonic Regression iteratively to monotonize the 2D grid."""
    smoothed = matrix.copy()
    iso = IsotonicRegression(increasing=False)
    for _ in range(5):
        for i in range(smoothed.shape[0]):
            smoothed[i, :] = iso.fit_transform(np.arange(smoothed.shape[1]), smoothed[i, :])
        for j in range(smoothed.shape[1]):
            smoothed[:, j] = iso.fit_transform(np.arange(smoothed.shape[0]), smoothed[:, j])
    return smoothed

def generate_outputs(df, csv_dir, plot_dir):
    os.makedirs(csv_dir, exist_ok=True)
    raw_plot_dir = os.path.join(plot_dir, "Raw")
    iso_plot_dir = os.path.join(plot_dir, "Isotonic")
    
    for mech in df['Mechanism'].unique():
        os.makedirs(os.path.join(raw_plot_dir, mech), exist_ok=True)
        os.makedirs(os.path.join(iso_plot_dir, mech), exist_ok=True)
        
    rwg_cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", ["#d73027", "#ffffff", "#1a9850"])
    
    # 1. First Pass: Compute Isotonic smoothing and append to the main DataFrame
    df['Isotonic_Gain_RMSE'] = df['Raw_Gain_RMSE']
    for mech in df['Mechanism'].unique():
        for scenario in df['Scenario'].unique():
            for client in df['Client'].unique():
                mask = (df['Mechanism'] == mech) & (df['Scenario'] == scenario) & (df['Client'] == client)
                sub_df = df[mask]
                if sub_df.empty: continue
                
                pivot_mean = sub_df.pivot_table(index='P1_Param', columns='P2_Param', values='Raw_Gain_RMSE', aggfunc='mean')
                smoothed_matrix = apply_2d_isotonic_regression(pivot_mean.values)
                smoothed_df = pd.DataFrame(smoothed_matrix, index=pivot_mean.index, columns=pivot_mean.columns)
                
                for idx, row in sub_df.iterrows():
                    df.loc[idx, 'Isotonic_Gain_RMSE'] = smoothed_df.loc[row['P1_Param'], row['P2_Param']]

    # 2. Second Pass: Export exact CSVs and Plots
    for mech in df['Mechanism'].unique():
        for scenario in df['Scenario'].unique():
            mask_scenario = (df['Mechanism'] == mech) & (df['Scenario'] == scenario)
            sub_df_scenario = df[mask_scenario]
            if sub_df_scenario.empty: continue
            
            clean_scenario = SCENARIO_MAP[scenario]
            
            # Export CSV perfectly formatted: e.g., dp_real.csv
            csv_filename = os.path.join(csv_dir, f"{mech}_{clean_scenario}.csv")
            sub_df_scenario.to_csv(csv_filename, index=False)
            print(f"📁 Saved CSV: {csv_filename}")
            
            for client in sub_df_scenario['Client'].unique():
                client_sub = sub_df_scenario[sub_df_scenario['Client'] == client]
                clean_client = CLIENT_MAP[scenario][client]
                
                # Raw Plot
                raw_mean = client_sub.pivot_table(index='P1_Param', columns='P2_Param', values='Raw_Gain_RMSE', aggfunc='mean')
                raw_std = client_sub.pivot_table(index='P1_Param', columns='P2_Param', values='Raw_Gain_RMSE', aggfunc='std').fillna(0)
                
                v_min, v_max = raw_mean.values.min(), raw_mean.values.max()
                if v_min >= 0: v_min = -1e-2
                if v_max <= 0: v_max = 1e-2
                norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)

                raw_annot = np.empty(raw_mean.shape, dtype=object)
                for i in range(raw_mean.shape[0]):
                    for j in range(raw_mean.shape[1]):
                        raw_annot[i, j] = f"{raw_mean.iloc[i, j]:.3f}\n±{raw_std.iloc[i, j]:.3f}"
                
                plt.figure(figsize=(9, 7))
                sns.heatmap(raw_mean, annot=raw_annot, fmt="", cmap=rwg_cmap, norm=norm, annot_kws={"size": 10}, cbar_kws={'label': 'Raw RMSE Gain'})
                plt.title(f"RAW RMSE Gain - {clean_scenario.upper()} - {clean_client} ({mech.upper()})")
                plt.xlabel(f"P2 Privacy Parameter ({mech.upper()})")
                plt.ylabel(f"P1 Privacy Parameter ({mech.upper()})")
                
                raw_plot_filename = os.path.join(raw_plot_dir, mech, f"{clean_scenario}_{clean_client}_{mech}.png")
                plt.savefig(raw_plot_filename, dpi=300, bbox_inches='tight')
                plt.close()

                # Isotonic Plot
                iso_mean = client_sub.pivot_table(index='P1_Param', columns='P2_Param', values='Isotonic_Gain_RMSE', aggfunc='mean')
                
                # Retrieve the raw standard deviation again to overlay on the smoothed mean
                raw_std = client_sub.pivot_table(index='P1_Param', columns='P2_Param', values='Raw_Gain_RMSE', aggfunc='std').fillna(0)
                
                iso_annot = np.empty(iso_mean.shape, dtype=object)
                for i in range(iso_mean.shape[0]):
                    for j in range(iso_mean.shape[1]):
                        # Format includes both the Isotonic mean and the Raw standard deviation
                        iso_annot[i, j] = f"{iso_mean.iloc[i, j]:.3f}\n±{raw_std.iloc[i, j]:.3f}"
                
                plt.figure(figsize=(9, 7))
                sns.heatmap(iso_mean, annot=iso_annot, fmt="", cmap=rwg_cmap, norm=norm, annot_kws={"size": 10}, cbar_kws={'label': 'Isotonic Regressed RMSE Gain'})
                plt.title(f"ISOTONIC RMSE Gain - {clean_scenario.upper()} - {clean_client} ({mech.upper()})")
                plt.xlabel(f"P2 Privacy Parameter ({mech.upper()})")
                plt.ylabel(f"P1 Privacy Parameter ({mech.upper()})")
                
                iso_plot_filename = os.path.join(iso_plot_dir, mech, f"{clean_scenario}_{clean_client}_{mech}.png")
                plt.savefig(iso_plot_filename, dpi=300, bbox_inches='tight')
                plt.close()

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_csv_dir", type=str, required=True)
    parser.add_argument("--out_plot_dir", type=str, required=True)
    parser.add_argument("--seed_range", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()

    os.makedirs(args.out_csv_dir, exist_ok=True)
    vocab_sizes = extract_global_vocabulary(args.data_root)

    tasks = []
    for seed in range(1, args.seed_range + 1):
        for scenario in ["full", "p1", "p2"]:
            print(f"\n--- Computing Baselines [Seed {seed}/{args.seed_range}] [{scenario}] ---")
            path1, path2 = get_data_paths_for_scenario(args.data_root, scenario)
            baselines = {
                0: train_and_eval_local_baseline_rs(path1, 0, seed, vocab_sizes),
                1: train_and_eval_local_baseline_rs(path2, 1, seed, vocab_sizes)
            }
            
            for mech, privacy_params in PRIVACY_PARAMS_BY_MECH.items():
                for p1_val in privacy_params:
                    for p2_val in privacy_params:
                        tasks.append({
                            "seed": seed, "scenario": scenario, "mech": mech,
                            "p1_val": p1_val, "p2_val": p2_val,
                            "data_root": args.data_root, "rounds": args.rounds,
                            "vocab_sizes": vocab_sizes, "baselines": baselines
                        })

    all_records = []
    print(f"\n⚡ Dispatching {len(tasks)} tasks to {args.workers} workers...")
    
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            try:
                all_records.extend(future.result())
                completed += 1
                if completed % 50 == 0: print(f"   ... Progress: {completed}/{len(tasks)} tasks.")
            except Exception as e: print(f"Task failed: {e}")

    df = pd.DataFrame(all_records)
    
    print("\n🎨 Generating Raw and Isotonic Output Heatmaps and CSV grids...")
    generate_outputs(df, args.out_csv_dir, args.out_plot_dir)

    total_runtime = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=int(total_runtime)))
    print(f"Done. Total experiment runtime: {formatted_time}")

if __name__ == "__main__":
    main()