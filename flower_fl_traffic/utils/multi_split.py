import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_silencer import silence_log
silence_log()  # ← MINDEN más import előtt

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import flwr as fl
import lightning as L
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data.custom_dataset import CustomDataset
from utils.metrics import player_specific_metrics
from seed import set_seed
from models.neural_network import TrafficLightningModule
from federated.server import get_on_fit_config
from federated.universal_client import create_client_fn

TARGET_COLUMN = "application_name"
BATCH_SIZE    = 256
LR            = 1e-3
NUM_EPOCHS    = 2
FED_ROUNDS    = 20

FEATURE_COLUMNS = [
    "protocol",
    "bidirectional_min_ps",
    "bidirectional_mean_ps",
    "bidirectional_stddev_ps",
    "bidirectional_max_ps",
    "src2dst_stddev_ps",
    "src2dst_max_ps",
    "dst2src_min_ps",
    "dst2src_mean_ps",
    "dst2src_stddev_ps",
    "dst2src_max_ps",
    "bidirectional_stddev_piat_ms",
    "bidirectional_max_piat_ms",
    "bidirectional_rst_packets",
]


def load_dataset(seed: int):
    dataset = pd.read_parquet("dataset/dataset.parquet")
    return {
        1:  dataset.sample(frac=0.1, random_state=seed),
        5:  dataset.sample(frac=0.5, random_state=seed),
        10: dataset,
    }


def make_loaders(player1_data, player2_data, le=None):
    if le is None:
        le = LabelEncoder()
        le.fit(pd.concat([player1_data, player2_data])[TARGET_COLUMN])

    def df_to_loader(df, shuffle=True):
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = le.transform(df[TARGET_COLUMN].values).astype(np.int64)
        return DataLoader(CustomDataset(X=X, y=y), batch_size=BATCH_SIZE, shuffle=shuffle)

    return df_to_loader(player1_data), df_to_loader(player2_data), le


def train_local(player1_data, player2_data, le):
    silence_log()
    """Két független NN modell tanítása, mindegyik a saját adatán."""
    results = []

    for df in [player1_data, player2_data]:
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = le.transform(df[TARGET_COLUMN].values).astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=0
        )

        train_loader = DataLoader(CustomDataset(X=X_train, y=y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(CustomDataset(X=X_test,  y=y_test),  batch_size=BATCH_SIZE, shuffle=False)

        model = TrafficLightningModule(
            input_dim=len(FEATURE_COLUMNS),
            num_classes=len(le.classes_),
            lr=LR,
        )

        trainer = L.Trainer(
            max_epochs=NUM_EPOCHS * FED_ROUNDS,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, train_loader)
        metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)[0]
        results.append(float(metrics["val_acc"]))

    print(f"  [Local Training] Player1 Accuracy: {results[0]:.4f}, Player2 Accuracy: {results[1]:.4f}")
    return results[0], results[1]


def train_federated(player1_data, player2_data, le):
    """Federated learning a két player között."""
    train_p1, train_p2, _ = make_loaders(player1_data, player2_data, le)
    test_p1,  test_p2,  _ = make_loaders(player1_data, player2_data, le)

    config = OmegaConf.create({
        "num_clients": 2,
        "config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "federated_rounds": FED_ROUNDS,
            "max_grad_norm": 1.0,
            "seed": 0,
        },
        "dataset": {
            "input_dim": len(FEATURE_COLUMNS),
            "num_classes": len(le.classes_),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "mode": "full",
        }
    })

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config(client1_noise=0.0, client2_noise=0.0),
        evaluate_metrics_aggregation_fn=player_specific_metrics,
    )

    history = fl.simulation.start_simulation(
        client_fn=create_client_fn([train_p1, train_p2], [test_p1, test_p2], config),
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=FED_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={
            "logging_level": logging.ERROR,
            "log_to_driver": False,  # ← ez tiltja a raylet és worker outputot
            "num_cpus": 2,
            "runtime_env": {"env_vars": {"OMP_NUM_THREADS": "1"}}
        }
    )
    

    p1_acc = history.metrics_distributed["client1_accuracy"][-1][1]
    p2_acc = history.metrics_distributed["client2_accuracy"][-1][1]

    print(f"  [Federated Training] Player1 Accuracy: {p1_acc:.4f}, Player2 Accuracy: {p2_acc:.4f}")
    return p1_acc, p2_acc



def print_results(results):
    for (key, ratio), acc in results.items():
        p1_imp = acc['federated']['player1'] - acc['local']['player1']
        p2_imp = acc['federated']['player2'] - acc['local']['player2']
        print(f"\nDataset: {key}x | Ratio: {ratio}")
        print(f"  Local:     P1={acc['local']['player1']:.4f}, P2={acc['local']['player2']:.4f}")
        print(f"  Federated: P1={acc['federated']['player1']:.4f}, P2={acc['federated']['player2']:.4f}")
        print(f"  Javulás:   P1={p1_imp:+.4f}, P2={p2_imp:+.4f}")


if __name__ == "__main__":
    seed   = 0
    ratios = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    results = {}

    set_seed(seed)
    datasets = load_dataset(seed)

    for key, dataset in datasets.items():
        # LabelEncoder egyszer az egész datasetre — konzisztens label mapping
        le = LabelEncoder()
        le.fit(dataset[TARGET_COLUMN])

        for ratio in ratios:
            split_index  = int(len(dataset) * ratio / (1 + ratio))
            player1_data = dataset.iloc[:split_index]
            player2_data = dataset.iloc[split_index:]

            print(f"\n{'='*60}")
            print(f"Dataset: {key}x | Ratio: {ratio:.3f} | P1: {player1_data.shape} | P2: {player2_data.shape}")

            p1_loc, p2_loc = train_local(player1_data, player2_data, le)
            p1_fed, p2_fed = train_federated(player1_data, player2_data, le)

            results[(key, ratio)] = {
                "local":     {"player1": p1_loc, "player2": p2_loc},
                "federated": {"player1": p1_fed, "player2": p2_fed},
            }

    print_results(results)