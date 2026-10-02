import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def execute_self_division_for_client(client_dir: str, dataset_name: str, client_name: str):
    """
    Implements the Self-Division heuristic locally for a specific client.
    """
    train_path = os.path.join(client_dir, 'train.parquet')
    eval_path = os.path.join(client_dir, 'eval.parquet')
    
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        print(f"  ⚠ Skipping {client_name} in {dataset_name}: Missing baseline parquets.")
        return

    train_df = pd.read_parquet(train_path)
    eval_df = pd.read_parquet(eval_path)
    client_df = pd.concat([train_df, eval_df], ignore_index=True)
    
    unique_users = client_df['user_id'].unique()
    
    if len(unique_users) < 2:
        print(f"    ⚠ Insufficient users for Self-Division in {client_name}. Falling back to random split.")
        df_shuffled = client_df.sample(frac=1, random_state=42).reset_index(drop=True)
        mid_point = len(df_shuffled) // 2
        sub_1_df = df_shuffled.iloc[:mid_point].copy()
        sub_2_df = df_shuffled.iloc[mid_point:].copy()
    else:
        np.random.seed(42)
        np.random.shuffle(unique_users)
        
        mid_point = len(unique_users) // 2
        sub_1_users = unique_users[:mid_point]
        sub_2_users = unique_users[mid_point:]
        
        sub_1_df = client_df[client_df['user_id'].isin(sub_1_users)].copy()
        sub_2_df = client_df[client_df['user_id'].isin(sub_2_users)].copy()

    common_items = np.intersect1d(sub_1_df['item_id'].unique(), sub_2_df['item_id'].unique())
    total_before = len(sub_1_df) + len(sub_2_df)
    
    if len(common_items) > 0:
        s1_filtered = sub_1_df[sub_1_df['item_id'].isin(common_items)]
        s2_filtered = sub_2_df[sub_2_df['item_id'].isin(common_items)]
        
        if total_before < 5000:
            print(f"    ⚠ Micro-dataset detected. Bypassing strict item intersection for Self-Division.")
        else:
            sub_1_df = s1_filtered
            sub_2_df = s2_filtered
    else:
        print(f"    ⚠ Empty item intersection. Bypassing strict item-space reduction.")
    
    subsets = {
        'sub_client_1': sub_1_df,
        'sub_client_2': sub_2_df
    }
    
    for sub_name, sub_df in subsets.items():
        if len(sub_df) < 2:
            print(f"    ⚠ Skipping {sub_name}: Insufficient data.")
            continue
            
        sub_train, sub_eval = train_test_split(sub_df, test_size=0.2, random_state=42)
        
        known_users = sub_train['user_id'].unique()
        known_items = sub_train['item_id'].unique()
        eval_filtered = sub_eval[sub_eval['user_id'].isin(known_users) & sub_eval['item_id'].isin(known_items)]
        
        if len(eval_filtered) == 0 and len(sub_eval) > 0:
            print(f"    ⚠ Cold-start filtering removed all eval data for {sub_name}. Retaining raw eval set.")
        else:
            sub_eval = eval_filtered
        
        output_dir = os.path.join(client_dir, 'self_division', sub_name)
        os.makedirs(output_dir, exist_ok=True)
        
        sub_train.to_parquet(os.path.join(output_dir, 'train.parquet'), engine='pyarrow', index=False)
        sub_eval.to_parquet(os.path.join(output_dir, 'eval.parquet'), engine='pyarrow', index=False)
        print(f"    ✓ {sub_name}: Exported {len(sub_train)} train and {len(sub_eval)} eval records.")

if __name__ == "__main__":
    base_datasets_dir = "./processed_datasets"
    
    if not os.path.exists(base_datasets_dir):
        raise FileNotFoundError(f"Base data directory not found: {base_datasets_dir}. Please run split_data.py first.")
        
    dataset_folders = ["NF_1_IID", "NF_5_IID"]
    
    print("Starting localized Self-Division partitioning loop...\n")
    for dataset_name in dataset_folders:
        dataset_path = os.path.join(base_datasets_dir, dataset_name)
        if not os.path.exists(dataset_path):
            continue
            
        print(f"Entering Dataset Benchmark: {dataset_name}")
        for client_name in ['client_1', 'client_2']:
            client_dir = os.path.join(dataset_path, client_name)
            execute_self_division_for_client(client_dir, dataset_name, client_name)