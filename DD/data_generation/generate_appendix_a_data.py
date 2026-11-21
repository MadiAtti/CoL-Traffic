"""
generate_all_experiments_data.py
--------------------------------
Generates the exact stratified splits for the 1x, 5x, and 10x curves
across all 7 asymmetry ratios.
"""
import sys
import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import packages.utils.data_utils as du

SEED = 42
DEFAULT_DATA_DIR = "/home/student/Matan/Federated_Learning/drug_discovery/new/src/Datasets/data_100%/full/data_2_split/"

# Define the matrix of experiments
CURVES = {'1x': 0.1, '5x': 0.5, '10x': 1.0}
RATIOS = {
    '1_8': 1/8, '1_4': 1/4, '1_2': 1/2, '1_1': 1.0, 
    '2_1': 2.0, '4_1': 4.0, '8_1': 8.0
}

def get_stratified_nested_permutation(Y_train, seed=42):
    # Stratification logic retained from original script to protect rare hits
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
    remaining_indices = list(set(range(num_samples)) - essential_indices)
    rng.shuffle(remaining_indices)
    
    return np.array(essential_list + remaining_indices)

def main():
    print("Reconstructing 100% Base Data from valid splits...")
    X_0_tr, Y_0_tr = du.load_ratio_split_data(DEFAULT_DATA_DIR, 0, train=True)
    X_0_te, Y_0_te = du.load_ratio_split_data(DEFAULT_DATA_DIR, 0, train=False)
    X_1_tr, Y_1_tr = du.load_ratio_split_data(DEFAULT_DATA_DIR, 1, train=True)
    X_1_te, Y_1_te = du.load_ratio_split_data(DEFAULT_DATA_DIR, 1, train=False)
    
    ecfp_tr = sp.vstack([X_0_tr, X_0_te, X_1_tr, X_1_te]).tocsr()
    ic50_tr = sp.vstack([Y_0_tr, Y_0_te, Y_1_tr, Y_1_te]).tocsr()
    
    N_total = ecfp_tr.shape[0]
    permutation = get_stratified_nested_permutation(ic50_tr, seed=SEED)
    src_root = Path(__file__).resolve().parent / "Datasets"
    
    for curve_name, vol_pct in CURVES.items():
        D_total = int(N_total * vol_pct)
        # Extract the master slice for this volume curve
        idx_curve = np.sort(permutation[:D_total])
        ecfp_curve = ecfp_tr[idx_curve]
        ic50_curve = ic50_tr[idx_curve]
        
        for ratio_name, R in RATIOS.items():
            p1_size = int(D_total * (R / (1 + R)))
            
            # Sub-split into Player 1 and Player 2
            ecfp_p1, ic50_p1 = ecfp_curve[:p1_size], ic50_curve[:p1_size]
            ecfp_p2, ic50_p2 = ecfp_curve[p1_size:], ic50_curve[p1_size:]
            
            # Save structures
            out_dir_p1 = src_root / f"Curve_{curve_name}" / f"Ratio_{ratio_name}" / "Client_0"
            out_dir_p2 = src_root / f"Curve_{curve_name}" / f"Ratio_{ratio_name}" / "Client_1"
            
            os.makedirs(out_dir_p1, exist_ok=True)
            os.makedirs(out_dir_p2, exist_ok=True)
            
            sp.save_npz(os.path.join(out_dir_p1, "x_tr.npz"), ecfp_p1)
            sp.save_npz(os.path.join(out_dir_p1, "y_tr.npz"), ic50_p1)
            sp.save_npz(os.path.join(out_dir_p2, "x_tr.npz"), ecfp_p2)
            sp.save_npz(os.path.join(out_dir_p2, "y_tr.npz"), ic50_p2)
            
    print("✅ All experimental dataset proportions successfully generated.")

if __name__ == "__main__":
    main()