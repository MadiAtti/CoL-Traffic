import flwr as fl
import ray
from itertools import product
from federated.universal_client import create_client_fn
from federated.server import get_on_fit_config
from utils.save import save_federated_history, setup_file
from utils.metrics import player_specific_metrics

def run_experiment(config, train_loaders, test_loaders, subdir, mode):
    """
    Univerzális futtató DP (zaj) és Suppression (feature elnyomás) kísérletekhez.
    mode: "dp" vagy "sup"
    """
    # 1. Beállítások szétválasztása mód alapján
    if mode == "dp":
        base_dir = "3_noise"
        metric_name = "noise"
        levels = config.config.noise_levels
        param_keys = ["p1_noise", "p2_noise"]
    else:  # sup mód
        base_dir = "2_suppression"
        metric_name = "features"
        levels = config.config.sup_levels
        param_keys = ["p1_features", "p2_features"]

    # 2. Inicializálás (fájl ürítése)
    setup_file(config, subdir, base_dir=base_dir)
    
    # 3. Mátrix generálása (7x7)
    scenarios = list(product(levels, levels))

    for val1, val2 in scenarios:
        print(f"\n" + "="*60)
        print(f"📡 SZIMULÁCIÓ ({mode.upper()}) | P1: {val1} | P2: {val2}")
        print("="*60)

        # 4. Dinamikus config összeállítása (p1_noise=val1... VAGY p1_features=val1...)
        fit_params = {param_keys[0]: val1, param_keys[1]: val2}
        
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
            on_fit_config_fn=get_on_fit_config(**fit_params),
            evaluate_metrics_aggregation_fn=player_specific_metrics,
        )

        # 5. Szimuláció
        history = fl.simulation.start_simulation(
            client_fn=create_client_fn(train_loaders, test_loaders, config),
            num_clients=config.num_clients,
            config=fl.server.ServerConfig(num_rounds=config.config.federated_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )

        # 6. Mentés
        save_federated_history(
            history, config, val1, val2, subdir, 
            base_dir=base_dir, metric_name=metric_name
        )

        # Eredmény kiírása a konzolra
        res_p1 = history.metrics_distributed["client1_accuracy"][-1][1]
        res_p2 = history.metrics_distributed["client2_accuracy"][-1][1]
        print(f"📊 VÉGEREDMÉNY: P1: {res_p1:.2%}, P2: {res_p2:.2%}")

        ray.shutdown()