import argparse
import random
import flwr as fl
import numpy as np
import torch
import json
import os
from flower_fl_traffic.old_config import local_params
from flower_fl_traffic.data.old_preprocessing import prepare_data, create_data_loaders
from flower_fl_traffic.federated.old_client import FlowerClient
from flower_fl_traffic.federated.old_strategy import get_strategy

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    # 1. Argumentumok és Seed beállítása
    parser = argparse.ArgumentParser(description="Flower Federated Learning Simulation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--exp", type=str, default="fl", choices=["fl", "dp"], help="Experiment type")
    args = parser.parse_args()

    set_seed(args.seed)

    # 2. Adatok előkészítése
    # A prepare_all_data visszaadja az összes Player (P1, P2, P11 stb.) adatait
    data_dict = prepare_data(args.seed)

    # 3. Kliensadatok térképe (Client Data Map)
    # A Flower CID (Client ID) alapján fogja lekérni az adatokat.
    # Példa: CID "0" -> Player 1, CID "1" -> Player 2
    client_data_map = {
        "0": create_data_loaders(data_dict['X_p1_train'], data_dict['X_p1_test'], 
                                data_dict['y_p1_train'], data_dict['y_p1_test'], 
                                local_params['batch_size']),
        "1": create_data_loaders(data_dict['X_p2_train'], data_dict['X_p2_test'], 
                                data_dict['y_p2_train'], data_dict['y_p2_test'], 
                                local_params['batch_size']),
    }

    # 4. Kliens generáló függvény
    def client_fn(cid: str) -> FlowerClient:
        train_loader, test_loader = client_data_map[cid]
        return FlowerClient(cid, train_loader, test_loader)

    # 5. Stratégia kiválasztása (Zajszint beállítása, ha DP fut)
    current_noise = local_params['noise_levels'][0] if args.exp == "dp" else None
    strategy = get_strategy(current_noise=current_noise)

    # 6. SZIMULÁCIÓ INDÍTÁSA
    print(f"Indul a Flower szimuláció! Típus: {args.exp.upper()}, Zaj: {current_noise}")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_data_map),
        config=fl.server.ServerConfig(num_rounds=local_params['federated_rounds']),
        strategy=strategy,
        ray_init_args={"num_cpus": 2} # Erőforrás korlátozás
    )

    # 7. Eredmények mentése
    save_path = f"results/flower_{args.exp}/{args.seed}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        # A history objektum tartalmazza a körök pontosságát és veszteségét
        json.dump(history.metrics_centralized, f, indent=4)

    print(f"Szimuláció kész! Eredmények mentve: {save_path}")

if __name__ == "__main__":
    main()