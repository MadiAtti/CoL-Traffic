"""
prep_subsets.py
---------------
Generates highly robust 30% and 80% subsets of the original drug discovery dataset.
Utilizes Multi-Task Stratification to prevent rare tasks from being destroyed.
Reconstructs the base data from the valid data_100% splits to bypass corrupted source files.
"""
import sys
import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import packages.utils.data_utils as du

SEED = 42
# We use the known-working 100% data directory to reconstruct the full dataset
DEFAULT_DATA_DIR = "/home/student/Matan/Federated_Learning/drug_discovery/new/src/Datasets/data_100%/full/data_2_split/"

def get_stratified_nested_permutation(Y_train, seed=42):
    """Protects rare positive hits by forcing them to the front of the permutation."""
    print("Generating Multi-Task Stratified Permutation...")
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

def create_and_save_subset(ecfp_tr, ic50_tr, permutation, percent, root_dir):
    num_samples = ecfp_tr.shape[0]
    keep_num = int((percent / 100.0) * num_samples)
    
    # Slice the nested permutation and sort to maintain matrix integrity
    idx = np.sort(permutation[:keep_num])
    
    ecfp_sub = ecfp_tr[idx]
    ic50_sub = ic50_tr[idx]
    
    os.makedirs(root_dir, exist_ok=True)
    sp.save_npz(os.path.join(root_dir, "x_tr.npz"), ecfp_sub)
    sp.save_npz(os.path.join(root_dir, "y_tr.npz"), ic50_sub)
    print(f"✅ Saved {percent}% dataset ({keep_num} samples) to {root_dir}")

def main():
    print("Reconstructing 100% Base Data from valid splits...")
    
    # Bypass the corrupted source directory by loading the valid 100% partitions
    X_0_tr, Y_0_tr = du.load_ratio_split_data(DEFAULT_DATA_DIR, 0, train=True)
    X_0_te, Y_0_te = du.load_ratio_split_data(DEFAULT_DATA_DIR, 0, train=False)
    X_1_tr, Y_1_tr = du.load_ratio_split_data(DEFAULT_DATA_DIR, 1, train=True)
    X_1_te, Y_1_te = du.load_ratio_split_data(DEFAULT_DATA_DIR, 1, train=False)
    
    # Reconstruct the massive 100% dataset pool
    ecfp_tr = sp.vstack([X_0_tr, X_0_te, X_1_tr, X_1_te]).tocsr()
    ic50_tr = sp.vstack([Y_0_tr, Y_0_te, Y_1_tr, Y_1_te]).tocsr()
    
    print(f"-> Reconstructed Matrix Size: {ecfp_tr.shape[0]} samples, {ic50_tr.shape[1]} tasks.")

    # Generate the master stratified layout
    permutation = get_stratified_nested_permutation(ic50_tr, seed=SEED)
    
    # Save the files directly inside your `src/Datasets` folder
    src_root = Path(__file__).resolve().parent
    out_80 = str(src_root / "Datasets/data_80") + os.path.sep
    out_30 = str(src_root / "Datasets/data_30") + os.path.sep
    
    
    create_and_save_subset(ecfp_tr, ic50_tr, permutation, 80, out_80)
    create_and_save_subset(ecfp_tr, ic50_tr, permutation, 30, out_30)

if __name__ == "__main__":
    main()