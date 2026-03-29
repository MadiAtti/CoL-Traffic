import os, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data.custom_dataset import CustomDataset

def setup_directories(cfg):
    base_results_dirs = [
        '1_local_baseline', 
        '2_suppression', 
        '3_noise'
    ]
    
    for base in base_results_dirs:
        os.makedirs(base, exist_ok=True)
        for sub in ['P1', 'P2']:
            os.makedirs(os.path.join(base, sub), exist_ok=True)
            
    if hasattr(cfg.dataset, 'paths'):
        for key in cfg.dataset.paths.keys():
            path = cfg.dataset.paths[key]
            if path:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)

    print("Mappastruktúra sikeresen inicializálva.")

def prepare_data_and_loaders(cfg):
    ds_cfg = cfg.dataset
    seed = cfg.config.seed
    
    # 1. Alapadat betöltés és skálázás
    df = pd.read_parquet(ds_cfg.input_path, columns=ds_cfg.feature_columns + [ds_cfg.target_column]).dropna()
    X = StandardScaler().fit_transform(df[ds_cfg.feature_columns].values.astype(np.float32))
    y = LabelEncoder().fit_transform(df[ds_cfg.target_column].values).astype(np.int32)

    # 2. Hierarchikus split és mentés (ha a p1 hiányzik, újrageneráljuk mindet)
    if not os.path.exists(ds_cfg.paths.p1):
        # Base -> P1, P2
        X1, X2, y1, y2 = train_test_split(X, y, test_size=ds_cfg.initial_split_ratio, stratify=y, random_state=seed)
        # P1 -> P11, P12
        X11, X12, y11, y12 = train_test_split(X1, y1, test_size=ds_cfg.initial_split_ratio, stratify=y1, random_state=seed)
        # P2 -> P21, P22
        X21, X22, y21, y22 = train_test_split(X2, y2, test_size=ds_cfg.initial_split_ratio, stratify=y2, random_state=seed)
        
        data_to_save = {'p1':(X1,y1), 'p2':(X2,y2), 'p11':(X11,y11), 'p12':(X12,y12), 'p21':(X21,y21), 'p22':(X22,y22)}
        for key, (sX, sy) in data_to_save.items():
            pd.DataFrame(sX, columns=ds_cfg.feature_columns).assign(**{ds_cfg.target_column: sy}).to_parquet(ds_cfg.paths[key], index=False)

    # 3. Loaderek létrehozása a kért dict formátumban
    loaders = {}
    for key in ds_cfg.paths.keys():
        df_p = pd.read_parquet(ds_cfg.paths[key])
        X_p, y_p = df_p[ds_cfg.feature_columns].values.astype(np.float32), df_p[ds_cfg.target_column].values.astype(np.int32)
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_p, y_p, test_size=ds_cfg.test_split_ratio, stratify=y_p, random_state=seed)
        
        loaders[key] = {
            "train": DataLoader(CustomDataset(X_tr, y_tr), batch_size=cfg.config.batch_size, shuffle=True),
            "test":  DataLoader(CustomDataset(X_te, y_te), batch_size=cfg.config.batch_size, shuffle=False)
        }
    
    return loaders