import pandas as pd
import numpy as np
import os
from typing import List

# Configuration
RAW_DATA_DIR = './raw'
OUTPUT_DIR = './processed'

def parse_netflix_data(file_paths: List[str]) -> pd.DataFrame:
    """Parses the irregular Netflix txt files into a structured DataFrame."""
    print("Parsing raw text files...")
    data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        with open(file_path, 'r') as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    user_id, rating, _ = line.split(',')
                    data.append([int(user_id), movie_id, float(rating)])
                    
    return pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])

def apply_density_filter(df: pd.DataFrame, min_ratings: int = 10) -> pd.DataFrame:
    """Iteratively removes users and items with less than `min_ratings`."""
    print(f"Applying density filter (min {min_ratings} ratings)...")
    if len(df) < 5000:
        min_ratings = 2 
    elif len(df) < 25000:
        min_ratings = 3
    elif len(df) < 100000:
        min_ratings = 5
    else:
        min_ratings = 10

    while True:
        start_len = len(df)
        
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        df = df[df['user_id'].isin(valid_users)]
        
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_ratings].index
        df = df[df['item_id'].isin(valid_items)]
        
        if len(df) == start_len:
            break
            
    return df

def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Item and User average discounting, then clamps to [-2, 2]."""
    print("Applying normalization and clamping...")
    
    item_avgs = df.groupby('item_id')['rating'].transform('mean')
    df['rating'] = df['rating'] - item_avgs
    
    user_avgs = df.groupby('user_id')['rating'].transform('mean')
    df['rating'] = df['rating'] - user_avgs
    
    df['rating'] = np.clip(df['rating'], -2.0, 2.0)
    
    return df

def generate_subsets(df: pd.DataFrame, target_sizes: dict, output_dir: str):
    """Generates nested user-wise subsets and saves them as Parquet files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    user_rating_counts = df['user_id'].value_counts().to_dict()
    
    # FIX: Sort users by rating frequency (ascending). 
    # This ensures we accumulate smaller user profiles first, allowing us to 
    # precisely hit microscopic target thresholds without catastrophic overshoot.
    sorted_users = sorted(user_rating_counts.keys(), key=lambda u: user_rating_counts[u])
    
    for name, target_size in target_sizes.items():
        print(f"Generating {name} (Target size: ~{target_size} ratings)...")
        
        current_size = 0
        selected_users = []
        
        for user in sorted_users:
            selected_users.append(user)
            current_size += user_rating_counts[user]
            
            if current_size >= target_size and len(selected_users) >= 2:
                break
                
        subset_df = df[df['user_id'].isin(selected_users)].copy()
        
        subset_df['user_id'] = subset_df['user_id'].astype('category').cat.codes
        subset_df['item_id'] = subset_df['item_id'].astype('category').cat.codes
        
        output_path = os.path.join(output_dir, f"{name}.parquet")
        subset_df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"  ✓ Saved {name} with {len(subset_df)} ratings to {output_path}")

if __name__ == "__main__":
    files = [os.path.join(RAW_DATA_DIR, f'combined_data_{i}.txt') for i in range(1, 5)]
    
    df_raw = parse_netflix_data(files)
    df_filtered = apply_density_filter(df_raw, min_ratings=10)
    df_normalized = apply_normalization(df_filtered)
    
    total_ratings = len(df_normalized)
    print(f"\nTotal filtered and normalized ratings available: {total_ratings}")
    
    dynamic_target_sizes = {
        'NF_1_IID': int(total_ratings * 0.01),
        'NF_5_IID': int(total_ratings * 0.05),     
    }
        
    generate_subsets(df_normalized, dynamic_target_sizes, OUTPUT_DIR)
    print("\nData provisioning complete.")