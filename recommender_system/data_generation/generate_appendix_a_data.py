"""
generate_all_experiments_data_rs.py
-----------------------------------
Generates nested stratified experimental splits for the 1x, 5x, and 10x curves
across all 7 asymmetry ratios for Recommender Systems (Netflix dataset).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DEFAULT_DATA_PATH = (
    "/home/student/Matan/Federated_Learning/"
    "recomender_systems/data/processed/NF_100.parquet"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "data" / "experiments"
)

# Volume curves representing 10%, 50%, and 100% of the base user population
CURVES = {'1x': 0.1, '5x': 0.5, '10x': 1.0}

# Asymmetry ratios R = |U_Client0| / |U_Client1|
RATIOS = {
    '1_8': 1 / 8,
    '1_4': 1 / 4,
    '1_2': 1 / 2,
    '1_1': 1.0,
    '2_1': 2.0,
    '4_1': 4.0,
    '8_1': 8.0,
}


def get_nested_user_permutation(unique_users: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Generates a deterministic permutation of unique user IDs to ensure that
    Curve_1x is a strict subset of Curve_5x, which is a strict subset of Curve_10x.
    """
    rng = np.random.RandomState(seed)
    permuted_users = unique_users.copy()
    rng.shuffle(permuted_users)
    return permuted_users


def partition_client_data(
    client_df: pd.DataFrame,
    client_dir: Path,
    seed: int = 42,
    test_size: float = 0.2
) -> tuple[int, int]:
    """
    Splits local client ratings into 80% train and 20% verification,
    enforces cold-start removal, and saves PyArrow Parquet files.
    """
    if len(client_df) < 2:
        return 0, 0

    train_df, eval_df = train_test_split(
        client_df, test_size=test_size, random_state=seed
    )

    # Eliminate evaluation ratings for unobserved users or items (cold-start mitigation)
    known_users = train_df['user_id'].unique()
    known_items = train_df['item_id'].unique()
    eval_filtered = eval_df[
        eval_df['user_id'].isin(known_users) & eval_df['item_id'].isin(known_items)
    ]

    if len(eval_filtered) == 0 and len(eval_df) > 0:
        # Prevent zero-length evaluation tensors if splits are excessively small
        eval_df = eval_df
    else:
        eval_df = eval_filtered

    os.makedirs(client_dir, exist_ok=True)
    train_path = client_dir / "train.parquet"
    eval_path = client_dir / "eval.parquet"

    train_df.to_parquet(train_path, engine='pyarrow', index=False)
    eval_df.to_parquet(eval_path, engine='pyarrow', index=False)

    return len(train_df), len(eval_df)


def main():
    data_file = Path(DEFAULT_DATA_PATH)
    if not data_file.exists():
        raise FileNotFoundError(f"Input base parquet file does not exist: {data_file}")

    print(f"Loading 100% RS Base Dataset from: {data_file}")
    df = pd.read_parquet(data_file)

    required_cols = {'user_id', 'item_id', 'rating'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    unique_users = df['user_id'].unique()
    total_users = len(unique_users)
    total_ratings = len(df)
    print(f"Loaded {total_ratings:,} ratings across {total_users:,} unique users.")

    permuted_users = get_nested_user_permutation(unique_users, seed=SEED)
    output_root = Path(DEFAULT_OUTPUT_DIR)

    for curve_name, vol_pct in CURVES.items():
        curve_user_count = int(total_users * vol_pct)
        # Slicing the prefix guarantees nested subsets across curves
        curve_users = permuted_users[:curve_user_count]
        df_curve = df[df['user_id'].isin(curve_users)].copy()

        print(f"\n--- Generating Curve {curve_name} ({vol_pct * 100:.0f}% users: {curve_user_count:,}) ---")

        for ratio_name, R in RATIOS.items():
            # Calculate client split boundaries based on user proportions
            p1_user_count = int(curve_user_count * (R / (1.0 + R)))
            p1_users = curve_users[:p1_user_count]
            p2_users = curve_users[p1_user_count:]

            df_c0 = df_curve[df_curve['user_id'].isin(p1_users)].copy()
            df_c1 = df_curve[df_curve['user_id'].isin(p2_users)].copy()

            # Align shared item space across clients
            common_items = np.intersect1d(df_c0['item_id'].unique(), df_c1['item_id'].unique())
            total_active_ratings = len(df_c0) + len(df_c1)

            if len(common_items) > 0 and total_active_ratings >= 5000:
                df_c0 = df_c0[df_c0['item_id'].isin(common_items)].copy()
                df_c1 = df_c1[df_c1['item_id'].isin(common_items)].copy()

            out_dir_c0 = output_root / f"Curve_{curve_name}" / f"Ratio_{ratio_name}" / "Client_0"
            out_dir_c1 = output_root / f"Curve_{curve_name}" / f"Ratio_{ratio_name}" / "Client_1"

            tr_0, ev_0 = partition_client_data(df_c0, out_dir_c0, seed=SEED)
            tr_1, ev_1 = partition_client_data(df_c1, out_dir_c1, seed=SEED)

            actual_ratio = (tr_0 + ev_0) / (tr_1 + ev_1) if (tr_1 + ev_1) > 0 else 0
            print(
                f"  Ratio {ratio_name:<4} | Client_0: {tr_0 + ev_0:>7,} ratings "
                f"| Client_1: {tr_1 + ev_1:>7,} ratings | Realized Ratio: {actual_ratio:.3f}"
            )

    print("\n✅ All experimental RS curves and asymmetry ratios successfully generated.")


if __name__ == "__main__":
    main()