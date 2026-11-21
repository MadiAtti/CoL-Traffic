import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(dataframe['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def apply_suppression(df, p, seed=42):
    """
    Suppression (Sup) mechanism: Removes a subset of the dataset.
    The reduction size is determined by the privacy parameter p.
    """
    if p <= 0.0:
        return df
    if p >= 1.0:
        return df.iloc[0:0] # Return empty dataframe for full privacy
        
    rng = np.random.RandomState(seed)
    keep_n = int((1.0 - p) * len(df))
    kept_indices = rng.choice(df.index, size=keep_n, replace=False)
    return df.loc[kept_indices].reset_index(drop=True)

def get_client_datasets_rs(data_path, privacy_param, privacy_mode, seed=42):
    """Loads parquet data and applies the requested privacy mechanisms."""
    df_train = pd.read_parquet(f"{data_path}/train.parquet")
    df_eval = pd.read_parquet(f"{data_path}/eval.parquet")
    
    if privacy_mode == 'sup' and privacy_param > 0.0:
        df_train = apply_suppression(df_train, privacy_param, seed=seed)
        
    train_ds = RatingsDataset(df_train)
    test_ds = RatingsDataset(df_eval)
    return train_ds, test_ds