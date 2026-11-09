"""Build cached P1/P2 partitions from the source traffic Parquet file.

The source is cleaned, standardized, and label-encoded before optional
stratified downsampling and a stratified participant split. Existing complete
P1/P2 splits are reused rather than overwritten.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def dataset_splitter(cfg):
    """Prepare and persist the configured P1/P2 dataset splits.

    Args:
        cfg: Hydra configuration specifying source data, columns, mode, and
            participant split ratio.

    Raises:
        ValueError: If the selected dataset mode is unsupported.
        FileNotFoundError: If exactly one cached participant split exists.
    """
    ds_cfg = cfg.dataset
    ds_mode = ds_cfg.mode
    supported_modes = {"full", "half", "80percent", "30percent"}
    if ds_mode not in supported_modes:
        raise ValueError(f"Unsupported dataset mode: {ds_mode}")

    p1_path = "dataset/{ds_mode}{p1}".format(ds_mode=ds_mode, p1=ds_cfg.paths.p1)
    p2_path = "dataset/{ds_mode}{p2}".format(ds_mode=ds_mode, p2=ds_cfg.paths.p2)

    print(f"Checking for existing dataset splits at:\n - {p1_path}\n - {p2_path}")

    p1_exists = os.path.exists(p1_path)
    p2_exists = os.path.exists(p2_path)
    if p1_exists and p2_exists:
        print("Dataset splits already exist. Skipping splitting.")
        return
    if p1_exists or p2_exists:
        raise FileNotFoundError(
            "Only one of the cached P1/P2 splits exists. Refusing to overwrite "
            "the existing split; restore its matching file before continuing."
        )
        
    print("Dataset splits not found. Performing splitting...")

    dataset = pd.read_parquet(ds_cfg.input_path, columns=ds_cfg.feature_columns + [ds_cfg.target_column]).dropna()
    
    X = StandardScaler().fit_transform(dataset[ds_cfg.feature_columns].values.astype(np.float32))
    y = LabelEncoder().fit_transform(dataset[ds_cfg.target_column].values).astype(np.int32)

    # Decrease the dataset size for faster experimentation (optional, can be removed for full dataset)
    if ds_mode == "full":
        pass  # Use the full dataset without sampling
    elif ds_mode == "half": # Reduce to half the dataset
        X, _, y, _ = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    elif ds_mode == "80percent": # Reduce to 80% of the dataset
        X, _, y, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    elif ds_mode == "30percent": # Reduce to 30% of the dataset
        X, _, y, _ = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)
    else:
        raise ValueError(f"Unsupported dataset mode: {ds_mode}")

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