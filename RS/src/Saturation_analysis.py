import os
import time
import datetime
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONF = {
    'batch_size': 256, 
    'lr_base': 0.0075, 
    'epochs': 20,      
    'lambda_reg': 0.01,
    'features': 4,     
    'max_norm': 0.5,   
    'seeds': [18169, 22596, 30593, 42212, 57582, 74694, 77654, 82958, 87048, 90839] # Explicit evaluation seeds
}

def set_deterministic_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# DATASET & PURE MATRIX FACTORIZATION
# ==========================================
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=4, max_norm=0.5):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim, max_norm=max_norm)
        self.item_emb = nn.Embedding(num_items, embedding_dim, max_norm=max_norm)
        
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        u = self.user_emb(user_indices)
        i = self.item_emb(item_indices)
        
        prediction = (u * i).sum(dim=1)
        l2_penalty = (u.norm(2, dim=1)**2 + i.norm(2, dim=1)**2)
        return prediction, l2_penalty

# ==========================================
# PARALLEL WORKER SUB-ROUTINE
# ==========================================
def worker_task(kwargs):
    torch.set_num_threads(1) 
    device = torch.device("cpu")
    
    p = kwargs["p"]
    seed = kwargs["seed"]
    df_subset = kwargs["df_subset"].copy()
    
    set_deterministic_environment(seed)
    
    # Dynamically remap IDs to contiguous local integers
    df_subset['user_id'] = df_subset['user_id'].astype('category').cat.codes
    df_subset['item_id'] = df_subset['item_id'].astype('category').cat.codes
    
    num_users = df_subset['user_id'].max() + 1
    num_items = df_subset['item_id'].max() + 1
    
    train_df, test_df = train_test_split(df_subset, test_size=0.2, random_state=seed)
    
    # Eliminate Cold-Start Noise
    known_items = train_df['item_id'].unique()
    test_df = test_df[test_df['item_id'].isin(known_items)]
    
    train_loader = DataLoader(RatingsDataset(train_df), batch_size=CONF['batch_size'], shuffle=True)
    test_loader = DataLoader(RatingsDataset(test_df), batch_size=CONF['batch_size'], shuffle=False)
    
    model = MatrixFactorization(num_users, num_items, embedding_dim=CONF['features'], max_norm=CONF['max_norm']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=CONF['lr_base'])
    loss_fn = nn.MSELoss(reduction='sum')
    
    model.train()
    for epoch in range(CONF['epochs']):
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            
            preds, l2_reg = model(users, items)
            preds = torch.clamp(preds, min=-2.0, max=2.0)
            
            mse_loss = loss_fn(preds, ratings)
            loss = mse_loss + CONF['lambda_reg'] * l2_reg.sum()
            
            loss.backward()
            optimizer.step()
            
    model.eval()
    total_squared_error, samples = 0.0, 0
    with torch.no_grad():
        for users, items, ratings in test_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            preds, _ = model(users, items)
            preds = torch.clamp(preds, min=-2.0, max=2.0)
            
            total_squared_error += loss_fn(preds, ratings).item()
            samples += len(ratings)
            
    rmse = np.sqrt(total_squared_error / max(1, samples))
    return {
        "Data_Percentage": p,
        "Seed": seed,
        "Total_Records": len(df_subset),
        "Alone_RMSE": rmse
    }

# ==========================================
# COORDINATOR MAIN ENGINE 
# ==========================================
def apply_density_filter(df: pd.DataFrame, min_ratings: int = 10) -> pd.DataFrame:
    """Dynamically degrades the density threshold for micro-datasets."""
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
        df = df[df['user_id'].isin(user_counts[user_counts >= min_ratings].index)]
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= min_ratings].index)]
        if len(df) == start_len:
            break
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/student/Matan/Federated_Learning/CoL-Traffic/recomender_systems/data/processed")
    parser.add_argument("--workers", type=int, default=18) 
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "Saturation_Analysis_12")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "dataset_saturation_metrics_multi_seed.csv")

    if os.path.exists(csv_path):
        results = pd.read_csv(csv_path).to_dict("records")
        completed_keys = {
            (row["Data_Percentage"], row["Seed"])
            for row in results
        }
        print(f"📄 Resuming from {len(results)} saved evaluations in {csv_path}")
    else:
        results = []
        completed_keys = set()

    print(f"🚀 Initializing Data Saturation Engine with {len(CONF['seeds'])}-Seed Evaluation")
    start_time = time.time()
    
    p50_path = os.path.join(args.data_dir, "NF_100.parquet")
    print("📥 Loading extraction baseline file NF_100.parquet...")
    df_50 = pd.read_parquet(p50_path)
    
    user_counts = df_50['user_id'].value_counts().to_dict()
    # 1% and then 5% with an increment of 5%
    percentages = [100] 
    total_estimated_ratings = len(df_50) 
    
    print(f"🧬 Generating density-filtered subsets across {len(CONF['seeds'])} defined seeds...")
    print(f"⚡ Dispatching one percentage batch at a time ({len(percentages)} percentages × {len(CONF['seeds'])} seeds) to {args.workers} workers...")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        for p in percentages:
            tasks = []

            for seed in CONF['seeds']:
                if (p, seed) in completed_keys:
                    continue

                unique_users = df_50['user_id'].unique()
                np.random.seed(seed)
                np.random.shuffle(unique_users)

                # Calculate target count based on the micro-percentage scale
                target_count = int(total_estimated_ratings * (p / 100.0))
                current_size = 0
                selected_users = []

                for user in unique_users:
                    selected_users.append(user)
                    current_size += user_counts[user]
                    if current_size >= target_count:
                        break

                sub_df = df_50[df_50['user_id'].isin(selected_users)].copy()
                sub_df = apply_density_filter(sub_df, min_ratings=10)

                tasks.append({
                    "p": p,
                    "seed": seed,
                    "df_subset": sub_df
                })

            if not tasks:
                print(f"   ... Percentage {p}% already complete; skipping.")
                continue

            print(f"   ... Starting percentage {p}% ({len(tasks)} seed evaluations).")
            futures = {executor.submit(worker_task, task): task for task in tasks}
            completed = 0
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                        pd.DataFrame([res]).to_csv(
                            csv_path,
                            mode="a",
                            header=not os.path.exists(csv_path),
                            index=False,
                        )
                    completed += 1
                    print(f"   ... Percentage {p}% progress: {completed}/{len(tasks)} seeds complete.")
                except Exception as e:
                    print(f"❌ Sub-task collapsed for percentage {p}%: {e}")

            print(f"   ... Finished percentage {p}%.")

    del df_50

    results_df = pd.DataFrame(results).sort_values(by=["Data_Percentage", "Seed"])
    results_df.to_csv(csv_path, index=False)
    
    # Compute average RMSE per percentage point
    avg_df = results_df.groupby("Data_Percentage", as_index=False)["Alone_RMSE"].mean()
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 7))
    
    # Plot individual seed points with low density (opacity)
    sns.scatterplot(
        data=results_df, 
        x="Data_Percentage", 
        y="Alone_RMSE", 
        color="#d73027", 
        alpha=0.35, 
        s=45,
        edgecolor="none",
        label="Individual Seed Runs"
    )
    
    # Plot bold mean line
    sns.lineplot(
        data=avg_df, 
        x="Data_Percentage", 
        y="Alone_RMSE", 
        marker="o", 
        color="#d73027", 
        linewidth=3.0,
        markersize=8,
        label="Average RMSE (Mean)"
    )
    
    plt.title(f"Baseline Model Saturation Analysis\nStandalone Core RMSE vs. Dataset Percentage Scale ({len(CONF['seeds'])}-Seed Multi-Run)", fontsize=14)
    plt.xlabel("Dataset Percentage Scale (%)", fontsize=12)
    plt.ylabel("Standalone Core RMSE (Lower is Better)", fontsize=12)
    # plt.gca().invert_yaxis()  # Removed so lower RMSE stays at the bottom
    plt.xscale('log') 
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, "saturation_curve_multi_seed.png")
    plt.savefig(plot_path, dpi=300)
    
    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"📊 Monotonic error curve with seed dispersion exported: {plot_path}")
    print(f"⏱️  Total Grid Compute Runtime: {elapsed}")

if __name__ == "__main__":
    main()