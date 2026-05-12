import os, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset

def setup_directories(cfg):
    '''
    Folder structure initialization based on the provided configuration.
    '''

    mode_dirs = [
        'full',
        'half',
        'quarter'
    ]
    
    base_results_dirs = [
        '1_local_baseline', 
        '2_suppression', 
        '3_noise'
    ]

    os.makedirs('results', exist_ok=True)

    for base in mode_dirs:
        os.makedirs(os.path.join('results', base), exist_ok=True)
        os.makedirs(os.path.join('dataset', base), exist_ok=True)

        for method in base_results_dirs:
            os.makedirs(os.path.join('results', base, method), exist_ok=True)
            os.makedirs(os.path.join('results', base, method, 'P1'), exist_ok=True)
            os.makedirs(os.path.join('results', base, method, 'P2'), exist_ok=True)
            os.makedirs(os.path.join('results', base, "plots"), exist_ok=True)
            os.makedirs(os.path.join('results', base, "games"), exist_ok=True)
            os.makedirs(os.path.join('results', base, "games/average"), exist_ok=True)
            
    if hasattr(cfg.dataset, 'paths'):
        for key in cfg.dataset.paths.keys():
            path = cfg.dataset.paths[key]
            if path:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)

def prepare_data_and_loaders(cfg):
    '''
    1. Load raw data, scale features, and encode target.
    2. Perform hierarchical splits (Base -> P1/P2, then P1 -> P11/P12, P2 -> P21/P22) and save them if not already done.
    3. Create DataLoader objects for each split and return them in a structured dict format'''
    ds_cfg = cfg.dataset
    seed = cfg.config.seed
    mode = ds_cfg.mode

    all_data = {}

    base_path = f"dataset/{mode}/"
    
    # Load pre-split data from Parquet files if they exist (ensures consistent splits across runs)
    p1_data = pd.read_parquet(f"{base_path}{ds_cfg.paths.p1}")
    p2_data = pd.read_parquet(f"{base_path}{ds_cfg.paths.p2}")
    X1, y1 = p1_data[ds_cfg.feature_columns].values.astype(np.float32), p1_data[ds_cfg.target_column].values.astype(np.int32)
    X2, y2 = p2_data[ds_cfg.feature_columns].values.astype(np.float32), p2_data[ds_cfg.target_column].values.astype(np.int32)
    
    all_data['p1'] = (X1, y1)
    all_data['p2'] = (X2, y2)

    # P1 -> P11, P12
    X11, X12, y11, y12 = train_test_split(X1, y1, test_size=ds_cfg.initial_split_ratio, stratify=y1, random_state=seed)
    # P2 -> P21, P22
    X21, X22, y21, y22 = train_test_split(X2, y2, test_size=ds_cfg.initial_split_ratio, stratify=y2, random_state=seed)
    
    # Overwrite the sub-splits to Parquet files (P11, P12, P21, P22) and store them in the all_data dict
    data_to_save = {'p11':(X11,y11), 'p12':(X12,y12), 'p21':(X21,y21), 'p22':(X22,y22)}
    for key, (sX, sy) in data_to_save.items():
        all_data[key] = (sX, sy)
        pd.DataFrame(sX, columns=ds_cfg.feature_columns).assign(**{ds_cfg.target_column: sy}).to_parquet(f"{base_path}{ds_cfg.paths[key]}", index=False)

    # Create DataLoader objects for each split, returning them in a structured dict format
    loaders = {}
    for key in ds_cfg.paths.keys():
        X_p, y_p = all_data[key]        
        X_tr, X_te, y_tr, y_te = train_test_split(X_p, y_p, test_size=ds_cfg.test_split_ratio, stratify=y_p, random_state=seed)
        
        loaders[key] = {
            "train": DataLoader(CustomDataset(X_tr, y_tr), batch_size=cfg.config.batch_size, shuffle=True),
            "test":  DataLoader(CustomDataset(X_te, y_te), batch_size=cfg.config.batch_size, shuffle=False)
        }
    
    return loaders

if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("conf/base.yaml")
    setup_directories(cfg)
