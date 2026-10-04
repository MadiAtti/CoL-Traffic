"""
evaluate_capacity.py
--------------------
Evaluates isolated, local model training across varying dataset sizes (100% down to 5%)
AND varying model capacities (40, 80, 120, 160, 200 neurons).
Includes deep Data Distribution tracking, Multi-Task Stratified Subsampling, 
and automated logging to a dedicated directory.
"""

import os
import sys
import csv
import argparse
import warnings
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")

TRAIN_CONTEXT = {}

# --- Path Injection ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPARSECHEM_ROOT = os.path.join(PROJECT_ROOT, "packages", "sparsechem")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SPARSECHEM_ROOT not in sys.path:
    sys.path.insert(0, SPARSECHEM_ROOT)

import sparsechem as sc
import packages.utils.data_utils as du

# --- Configuration Constants ---
INPUT_SIZE = 32000
OUTPUT_SIZE = 2808
DEFAULT_DATA_DIR = ""

# ==========================================
# Automated Logging Mechanism
# ==========================================
class Logger(object):
    """Duplicates stdout to both the console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_base_conf(n_train_total, epochs, hidden_size):
    """Constructs the base SparseChem configuration with dynamic hidden sizes."""
    batch_size = max(64, int(n_train_total * 0.02))
    c = sc.ModelConfig(
        input_size=INPUT_SIZE,
        hidden_sizes=[hidden_size], # <--- Dynamically updated here
        output_size=OUTPUT_SIZE,
        batch_size=batch_size,
        lr=1e-3,
        last_dropout=0.2,
        weight_decay=1e-5,
        non_linearity="relu",
        last_non_linearity="relu",
    )
    c.epochs = epochs
    return c

# ==========================================
# Data Distribution & Stratification Logic
# ==========================================
def get_stratified_nested_permutation(Y_train, seed=42):
    rng = np.random.RandomState(seed)
    num_samples, num_tasks = Y_train.shape
    
    Y_csc = Y_train.tocsc()
    Y_bin = Y_csc.copy()
    Y_bin.data = (Y_bin.data > 0.5).astype(int)
    
    essential_indices = set()
    
    for t in range(num_tasks):
        active_rows = Y_csc.indices[Y_csc.indptr[t]:Y_csc.indptr[t+1]]
        active_vals = Y_bin.data[Y_csc.indptr[t]:Y_csc.indptr[t+1]]
        actual_active_rows = active_rows[active_vals == 1]
        
        if len(actual_active_rows) > 0:
            chosen = rng.choice(actual_active_rows, size=min(2, len(actual_active_rows)), replace=False)
            essential_indices.update(chosen)
            
    essential_list = list(essential_indices)
    rng.shuffle(essential_list)
    
    all_indices = set(range(num_samples))
    remaining_indices = list(all_indices - essential_indices)
    rng.shuffle(remaining_indices)
    
    permutation = np.array(essential_list + remaining_indices)
    return permutation

def analyze_data_distribution(Y_sparse, percent, keep_num):
    total_possible = Y_sparse.shape[0] * Y_sparse.shape[1]
    observed = Y_sparse.nnz
    density = observed / total_possible if total_possible > 0 else 0
    
    actives = (Y_sparse.data > 0.5).sum()
    inactives = observed - actives
    imbalance_ratio = actives / max(1, observed)
    
    Y_csc = Y_sparse.tocsc()
    Y_bin = Y_csc.copy()
    Y_bin.data = (Y_bin.data > 0.5).astype(int)
    actives_per_task = Y_bin.sum(axis=0).A1
    
    dead_tasks = (actives_per_task == 0).sum()
    rare_tasks = (actives_per_task < 5).sum()
    
    print(f"📊 DATA DISTRIBUTION: {percent}% Split ({keep_num} samples)")
    print(f"   ├─ Density:   {density:.4%} ({observed:,} observed out of {total_possible:,} matrix slots)")
    print(f"   ├─ Imbalance: {actives:,} Actives vs {inactives:,} Inactives ({imbalance_ratio:.2%} are hits)")
    print(f"   └─ Task Health: {dead_tasks} Dead Tasks (0 hits) | {rare_tasks} Rare Tasks (<5 hits)")

# ==========================================
# Training and Evaluation Core
# ==========================================
def evaluate_model(model, test_loader, conf, device):
    model.eval()
    preds_batches, targets_batches = [], []
    observed_preds, observed_targets = [], []
    output_size = int(conf.output_size)

    with torch.no_grad():
        for batch in test_loader:
            b_x = torch.sparse_coo_tensor(
                batch["x_ind"], batch["x_data"], 
                size=[batch["batch_size"], conf.input_size]
            ).to(device)

            logits = model(b_x)
            logits_sig = torch.sigmoid(logits).cpu().numpy()

            batch_size = int(batch["batch_size"])
            dense_targets = np.full((batch_size, output_size), np.nan, dtype=float)

            y_ind = batch["y_ind"].cpu().numpy()
            y_data = batch["y_data"].cpu().numpy()

            if y_ind.size > 0:
                rows, cols = y_ind[0].astype(int), y_ind[1].astype(int)
                dense_targets[rows, cols] = y_data
                observed_preds.extend(logits_sig[rows, cols].reshape(-1).tolist())
                observed_targets.extend(y_data.reshape(-1).tolist())

            preds_batches.append(logits_sig)
            targets_batches.append(dense_targets)

    preds_all = np.vstack(preds_batches) if preds_batches else np.zeros((0, output_size))
    targets_all = np.vstack(targets_batches) if targets_batches else np.zeros((0, output_size))

    if len(observed_preds) > 0:
        obs_preds_bin = (np.asarray(observed_preds) > 0.5).astype(int)
        obs_targets_bin = (np.asarray(observed_targets) > 0.5).astype(int)
        acc = accuracy_score(obs_targets_bin, obs_preds_bin)
        prec = precision_score(obs_targets_bin, obs_preds_bin, average="binary", zero_division=0)
        rec = recall_score(obs_targets_bin, obs_preds_bin, average="binary", zero_division=0)
        f1 = f1_score(obs_targets_bin, obs_preds_bin, average="binary", zero_division=0)
    else:
        acc = prec = rec = f1 = float('nan')

    from sklearn.metrics import roc_auc_score
    task_aucs, task_f1s = [], []
    
    for t in range(output_size):
        mask = ~np.isnan(targets_all[:, t])
        if mask.sum() == 0: continue
        y_true, y_score = targets_all[mask, t], preds_all[mask, t]

        if np.unique(y_true).size > 1:
            try: task_aucs.append(roc_auc_score((y_true > 0.5).astype(int), y_score))
            except Exception: pass
        try: task_f1s.append(f1_score((y_true > 0.5).astype(int), (y_score > 0.5).astype(int), zero_division=0))
        except Exception: pass

    macro_task_auc = float(np.mean(task_aucs)) if len(task_aucs) > 0 else float('nan')
    macro_task_f1 = float(np.mean(task_f1s)) if len(task_f1s) > 0 else float('nan')

    if preds_all.size == 0: global_acc = float('nan')
    else:
        preds_bin_all = (preds_all > 0.5).astype(int)
        targets_all_zero = np.nan_to_num(targets_all, nan=0.0).astype(int)
        try: global_acc = float(accuracy_score(targets_all_zero.flatten(), preds_bin_all.flatten()))
        except Exception: global_acc = float('nan')

    # Explicit memory cleanup for evaluation variables
    del preds_batches, targets_batches, preds_all, targets_all
    
    return {
        "observed_acc": float(acc), "observed_prec": float(prec),
        "observed_rec": float(rec), "observed_f1": float(f1),
        "macro_task_auc": macro_task_auc, "macro_task_f1": macro_task_f1,
        "global_acc": global_acc,
    }

def train_and_evaluate(percent, hidden_size, X_train, Y_train, test_dataset, device, args, permutation=None):
    num_samples = X_train.shape[0]
    keep_num = max(1, int((percent / 100.0) * num_samples))
    indices = np.sort(permutation[:keep_num])

    X_train_sub = X_train[indices]
    Y_train_sub = Y_train[indices]
    
    # Only print distribution analysis on the first capacity pass to avoid log spam
    if hidden_size == 40:
        analyze_data_distribution(Y_train_sub, percent, keep_num)
    
    train_dataset = sc.SparseDataset(X_train_sub, Y_train_sub)
    conf = make_base_conf(keep_num, args.epochs, hidden_size)
    conf.output_size = Y_train_sub.shape[1]
    
    model = sc.TrunkAndHead(conf=conf, trunk=sc.Trunk(conf)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=sc.sparse_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=sc.sparse_collate)

    print(f"⚙️  Training model on {percent}% Data with {hidden_size} Neurons...")
    model.train()
    for ep in range(args.epochs):
        for batch in train_loader:
            b_x = torch.sparse_coo_tensor(batch["x_ind"], batch["x_data"], size=[batch["batch_size"], conf.input_size]).to(device)
            y_ind, y_data = batch["y_ind"].to(device), batch["y_data"].to(device)
            
            optimizer.zero_grad()
            logits = model(b_x)
            logits_subset = logits[y_ind[0], y_ind[1]]
            
            if logits_subset.numel() > 0:
                loss = loss_fn(logits_subset, y_data).mean()
                loss.backward()
                optimizer.step()
                del loss # Force destruction of the computation graph

            # Aggressive cleanup of batch variables to prevent memory leakage
            del b_x, y_ind, y_data, logits, logits_subset
                
    metrics = evaluate_model(model, test_loader, conf, device)

    # VERY IMPORTANT: Destroy the model and empty the CUDA cache before the next iteration
    del model
    del optimizer
    del train_loader
    del test_loader
    del train_dataset
    
    gc.collect() # Force Python Garbage Collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Purge the PyTorch GPU cache

    return metrics

def run_capacity_sweep(hidden_size):
    """Run the full percentage sweep for one hidden-size setting."""
    X_train_full = TRAIN_CONTEXT["X_train_full"]
    Y_train_full = TRAIN_CONTEXT["Y_train_full"]
    test_dataset = TRAIN_CONTEXT["test_dataset"]
    device = TRAIN_CONTEXT["device"]
    args = TRAIN_CONTEXT["args"]
    permutation = TRAIN_CONTEXT["permutation"]

    print(f"\n\n{'='*60}")
    print(f"🚀 STARTING CAPACITY RUN: {hidden_size} NEURONS")
    print(f"{'='*60}")

    results = []

    for percent in range(100, 0, -5):
        print("\n" + "-"*40)
        metrics = train_and_evaluate(
            percent, hidden_size, X_train_full, Y_train_full, test_dataset, device, args, permutation=permutation
        )

        print(f"✅ Results for {percent}% Data ({hidden_size} neurons):")
        print(f"   ├─ Accuracy: {metrics['global_acc']:.4f}")
        print(f"   ├─ ObsF1:    {metrics['observed_f1']:.4f}")
        print(f"   ├─ MacroAUC: {metrics['macro_task_auc']:.4f}")
        print(f"   └─ MacroF1:  {metrics['macro_task_f1']:.4f}")

        results.append({
            "hidden_neurons": hidden_size,
            "percentage": percent,
            "accuracy": metrics["global_acc"],
            "observed_accuracy": metrics["observed_acc"],
            "observed_precision": metrics["observed_prec"],
            "observed_recall": metrics["observed_rec"],
            "observed_f1": metrics["observed_f1"],
            "macro_task_auc": metrics["macro_task_auc"],
            "macro_task_f1": metrics["macro_task_f1"],
        })

    return hidden_size, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate local model training across varying dataset sizes and model capacities.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to the 'data_2_split' directory.")
    parser.add_argument("--out_dir", type=str, default="results_capacity", help="Directory to save logs and CSVs.")
    parser.add_argument("--client_idx", type=int, default=0, help="Which client's data partition to use (0 or 1).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel CPU workers to use for capacity runs.")
    args = parser.parse_args()

    # --- Setup Output Directory and Logging ---
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, "experiment_log.txt")
    sys.stdout = Logger(log_file)

    print(f"==================================================")
    print(f" INITIALIZING CAPACITY EXPERIMENT")
    print(f" Results will be saved to: ./{args.out_dir}/")
    print(f"==================================================")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    print(f"Loading data from {args.data_dir} for client {args.client_idx}...")
    X_train_full, Y_train_full = du.load_ratio_split_data(args.data_dir, args.client_idx, train=True)
    X_test, Y_test = du.load_ratio_split_data(args.data_dir, args.client_idx, train=False)
    
    test_dataset = sc.SparseDataset(X_test, Y_test)

    permutation = get_stratified_nested_permutation(Y_train_full, seed=args.seed)
    TRAIN_CONTEXT.update({
        "X_train_full": X_train_full,
        "Y_train_full": Y_train_full,
        "test_dataset": test_dataset,
        "device": device,
        "args": args,
        "permutation": permutation,
    })

    if args.workers > 1 and torch.cuda.is_available():
        print("Parallel workers with CUDA can oversubscribe one GPU and slow things down.")
        print("Falling back to 1 worker; use --workers only for CPU runs.")
        args.workers = 1

    # The capacities requested
    capacities = [40]  # For testing purposes, you can adjust this list

    def write_results(hidden_size, results):
        csv_file = os.path.join(args.out_dir, f"metrics_H{hidden_size}.csv")
        fieldnames = ["hidden_neurons", "percentage", "accuracy", "observed_accuracy", "observed_precision",
                      "observed_recall", "observed_f1", "macro_task_auc", "macro_task_f1"]

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n📁 Saved results for {hidden_size} neurons to: {csv_file}")

    if args.workers == 1:
        for hidden_size in capacities:
            _, results = run_capacity_sweep(hidden_size)
            write_results(hidden_size, results)
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            futures = {executor.submit(run_capacity_sweep, hidden_size): hidden_size for hidden_size in capacities}

            collected = {}
            for future in as_completed(futures):
                hidden_size, results = future.result()
                collected[hidden_size] = results

            for hidden_size in capacities:
                write_results(hidden_size, collected[hidden_size])

    print(f"\n🎉 Entire capacity sweep complete. Master log saved to {log_file}.")

if __name__ == "__main__":
    main()