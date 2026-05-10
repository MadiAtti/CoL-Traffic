import os
from unittest import case

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def dataset_splitter(cfg):
    '''
    This function performs the hierarchical splitting of the dataset into P1/P2 and then into P11/P12 and P21/P22.
    It saves the splits as Parquet files for consistent loading in future runs.
    '''
    ds_cfg = cfg.dataset
    ds_mode = ds_cfg.mode
    

    p1_path = "dataset/{ds_mode}{p1}".format(ds_mode=ds_mode, p1=ds_cfg.paths.p1)
    p2_path = "dataset/{ds_mode}{p2}".format(ds_mode=ds_mode, p2=ds_cfg.paths.p2)

    print(f"Checking for existing dataset splits at:\n - {p1_path}\n - {p2_path}")

    if os.path.exists(p1_path) and os.path.exists(p2_path):
        print("Dataset splits already exist. Skipping splitting.")
        return
        
    print("Dataset splits not found. Performing splitting...")

    dataset = pd.read_parquet(ds_cfg.input_path, columns=ds_cfg.feature_columns + [ds_cfg.target_column]).dropna()
    
    X = StandardScaler().fit_transform(dataset[ds_cfg.feature_columns].values.astype(np.float32))
    y = LabelEncoder().fit_transform(dataset[ds_cfg.target_column].values).astype(np.int32)

    # Decrease the dataset size for faster experimentation (optional, can be removed for full dataset)
    if ds_mode == "full":
        pass  # Use the full dataset without sampling
    elif ds_mode == "half": # Reduce to half the dataset
        X, _, y, _ = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    elif ds_mode == "quarter": # Reduce to quarter of the dataset
        X, _, y, _ = train_test_split(X, y, test_size=0.75, stratify=y, random_state=42)

    # Base -> P1, P2
    X1, X2, y1, y2 = train_test_split(X, y, test_size=ds_cfg.initial_split_ratio, stratify=y, random_state=42)

    #Save the base splits (P1 and P2) to Parquet files for future runs (ensures consistent splits across runs)
    pd.DataFrame(X1, columns=ds_cfg.feature_columns).assign(**{ds_cfg.target_column: y1}).to_parquet(p1_path, index=False)
    pd.DataFrame(X2, columns=ds_cfg.feature_columns).assign(**{ds_cfg.target_column: y2}).to_parquet(p2_path, index=False)


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("conf/base.yaml")
    dataset_splitter(cfg)