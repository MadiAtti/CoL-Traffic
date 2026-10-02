import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def partition_for_flower(df: pd.DataFrame, dataset_name: str, base_output_dir: str):
    """
    Partitions a base dataset into 2 clients with disjoint users.
    """
    print(f"Partitioning {dataset_name} for Flower clients...")
    
    unique_users = df['user_id'].unique()
    
    if len(unique_users) < 2:
        print("  ⚠ Insufficient users for disjoint partitioning. Falling back to random IID ratings split.")
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        mid_point = len(df_shuffled) // 2
        client_1_df = df_shuffled.iloc[:mid_point].copy()
        client_2_df = df_shuffled.iloc[mid_point:].copy()
    else:
        np.random.seed(42)  
        np.random.shuffle(unique_users)
        
        mid_point = len(unique_users) // 2
        client_1_users = unique_users[:mid_point]
        client_2_users = unique_users[mid_point:]
        
        client_1_df = df[df['user_id'].isin(client_1_users)].copy()
        client_2_df = df[df['user_id'].isin(client_2_users)].copy()
    
    common_items = np.intersect1d(client_1_df['item_id'].unique(), client_2_df['item_id'].unique())
    total_before = len(client_1_df) + len(client_2_df)
    
    if len(common_items) > 0:
        c1_filtered = client_1_df[client_1_df['item_id'].isin(common_items)]
        c2_filtered = client_2_df[client_2_df['item_id'].isin(common_items)]
        
        if total_before < 5000:
            print(f"  ⚠ Micro-dataset detected (size: {total_before}). Bypassing strict item intersection to prevent data decimation and size inversions.")
        else:
            client_1_df = c1_filtered
            client_2_df = c2_filtered
    else:
        print("  ⚠ Empty item intersection. Bypassing strict item-space reduction to prevent data loss.")
    
    clients_data = {
        'client_1': client_1_df,
        'client_2': client_2_df
    }
    
    for client_name, client_df in clients_data.items():
        if len(client_df) < 2:
            print(f"  ⚠ Skipping {client_name}: Insufficient data.")
            continue
            
        train_df, eval_df = train_test_split(
            client_df, test_size=0.2, random_state=42
        )
        
        known_users = train_df['user_id'].unique()
        known_items = train_df['item_id'].unique()
        eval_filtered = eval_df[eval_df['user_id'].isin(known_users) & eval_df['item_id'].isin(known_items)]
        
        if len(eval_filtered) == 0 and len(eval_df) > 0:
            print(f"  ⚠ Cold-start filtering removed all eval data for {client_name}. Retaining raw eval set to prevent zero-length arrays.")
        else:
            eval_df = eval_filtered
        
        client_dir = os.path.join(base_output_dir, dataset_name, client_name)
        os.makedirs(client_dir, exist_ok=True)
        
        train_path = os.path.join(client_dir, 'train.parquet')
        eval_path = os.path.join(client_dir, 'eval.parquet')
        
        train_df.to_parquet(train_path, engine='pyarrow', index=False)
        eval_df.to_parquet(eval_path, engine='pyarrow', index=False)
        
        print(f"  -> {client_name}: Saved {len(train_df)} train and {len(eval_df)} eval ratings.")

if __name__ == "__main__":
    processed_dir = "./processed"
    base_output_dir = "./processed_datasets"
    
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed repository directory does not exist: {processed_dir}")
        
    target_datasets = ["NF_1_IID", "NF_5_IID"]
    
    for dataset_name in target_datasets:
        input_path = os.path.join(processed_dir, f"{dataset_name}.parquet")
        
        if not os.path.exists(input_path):
            print(f"Skipping {dataset_name}: File not found at {input_path}")
            continue
            
        df = pd.read_parquet(input_path)
        partition_for_flower(df, dataset_name, base_output_dir)