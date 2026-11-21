"""
run_subsets.py
--------------
Automates the P1/P2/Train/Test splitting process across the 30% and 80% datasets.
"""
import sys
from pathlib import Path
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_data import split_with_overlap

def process_subset(folder_name):
    src_root = Path(__file__).resolve().parent
    dd = src_root / "Datasets" / folder_name
    
    print(f"\n========================================")
    print(f" Processing Subset: {folder_name}")
    print(f"========================================")

    if not dd.exists():
        print(f"Error: {dd} does not exist. Run prep_subsets.py first.")
        return

    # Load the truncated array
    ecfp_tr = sp.load_npz(str(dd / "x_tr.npz"))
    ic50_tr = sp.load_npz(str(dd / "y_tr.npz"))

    # Execute the stratified splitting logic
    split_with_overlap(
        ratio=1, 
        ecfp_tr=ecfp_tr, 
        ic50_tr=ic50_tr, 
        root_dir=str(dd) + "/", 
        overlap=2808
    )

if __name__ == "__main__":
    process_subset("data_30")
    process_subset("data_80")