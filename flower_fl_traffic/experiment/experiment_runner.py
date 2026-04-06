import logging

from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from utils.logger_silencer import silence_log
import flwr as fl
import ray
import multiprocessing as mp
from itertools import product
from federated.universal_client import create_client_fn
from federated.server import get_on_fit_config
from utils.save import save_federated_history, setup_file
from utils.metrics import player_specific_metrics

def _run_single_scenario(args):
    """
    Helper function to run a single simulation scenario in a separate process.
    """

    silence_log()

    (val1, val2, config, raw_train_loaders, test_loaders, subdir, 
     mode, base_dir, metric_name, param_keys) = args

    active_loaders = []

    if mode == "sup":
        total_f = config.dataset.input_dim  # Ez a 14
        active_loaders = []
        
        for i, limit in enumerate([val1, val2]):
            limit = int(limit)
            
            # 1. KINYERÉS ÉS MÁSOLÁS (Hogy ne legyen shared memory ütközés)
            # A régi kódod alapján: X_p1_train, y_p1_train kinyerése
            X_orig = raw_train_loaders[i].dataset.X.cpu().numpy().copy()
            y_orig = raw_train_loaders[i].dataset.y.cpu().numpy().copy()
            
            # 2. FIZIKAI VÁGÁS (Ahogy a régi kódod create_suppressed_dataset-je csinálta)
            # Csak az első 'limit' számú oszlopot tartjuk meg
            feature_indices = list(range(limit))
            X_cut = X_orig[:, feature_indices]
            
            # 3. CUSTOM DATASET INICIALIZÁLÁS
            # Átadjuk a vágott X-et (X_cut), és megmondjuk, hova pakolja a 14-es vázban
            new_ds = CustomDataset(
                X=X_cut, 
                y=y_orig, 
                feature_indices=feature_indices, 
                total_features=total_f
            )
            
            active_loaders.append(DataLoader(
                new_ds, 
                batch_size=config.config.batch_size, 
                shuffle=True
            ))
    else:
        # Ha nem suppression, akkor az eredeti loadereket használjuk
        active_loaders = raw_train_loaders

    print(f"\n🚀 Starting Parallel Simulation ({mode.upper()}) | Client1: {val1} | Client2: {val2}")
    
    # Create strategy with the appropriate on_fit_config for the current scenario
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

    # Start the Flower simulation
    # Note: client_resources is set to 1 CPU per client to allow effective parallelism
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn(active_loaders, test_loaders, config),
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.config.federated_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={
            "logging_level": logging.ERROR,
            "log_to_driver": True,
            "num_cpus": 2,
            "_temp_dir": "/dev/shm/ray_tmp",
        }
    )

    # Save the history and results
    save_federated_history(
        history, config, val1, val2, subdir, 
        base_dir=base_dir, metric_name=metric_name
    )

    # Extract final results for console logging
    res_1 = history.metrics_distributed["client1_accuracy"][-1][1]
    res_2 = history.metrics_distributed["client2_accuracy"][-1][1]
    
    print(f"✅ Scenario Finished | C1: {val1}, C2: {val2} | Acc: {res_1:.2%}, {res_2:.2%}")
    
    # We return the results just in case, though saving happens inside
    return (val1, val2, res_1, res_2)


def run_experiment(config, train_loaders, test_loaders, subdir, mode):
    """
    Universal experiment runner that parallelizes scenarios using multiprocessing.
    """
    # Setup directories and parameters based on mode
    if mode == "dp": # Differentially Private mode
        base_dir = "3_noise"
        metric_name = "noise"
        levels = config.config.noise_levels
        param_keys = ["client1_noise", "client2_noise"]
    else:  # Suppression mode
        base_dir = "2_suppression"
        metric_name = "features"
        levels = config.config.sup_levels
        param_keys = ["client1_features", "client2_features"]

    # Initalize file for saving results 
    setup_file(config, subdir, base_dir=base_dir)
    
    # Generate all combinations of noise/suppression levels for Client1 and Client2
    scenarios = list(product(levels, levels))

    # Prepare tasks for the multiprocessing pool
    tasks = []
    for val1, val2 in scenarios:
        tasks.append((
            val1, val2, config, train_loaders, test_loaders, subdir, 
            mode, base_dir, metric_name, param_keys
        ))

    # Determine the number of parallel processes
    # Since each scenario uses 2 clients (2 CPUs), 4 processes use 8 CPUs.
    # Adjust this number based on your available RAM.
    num_parallel_scenarios = 4

    print(f"\n{'#'*60}")
    print(f"🔥 Starting Parallel Runner | Mode: {mode.upper()} | Processes: {num_parallel_scenarios}")
    print(f"{'#'*60}\n")

    # Use Multiprocessing Pool to run scenarios in parallel
    try:
        with mp.Pool(processes=num_parallel_scenarios) as pool:
            pool.map(_run_single_scenario, tasks)
    except Exception as e:
        print(f"❌ Error during parallel execution: {e}")
    finally:
        # Final cleanup of Ray after all scenarios in this mode are done
        ray.shutdown()

    print(f"\n✨ All {mode.upper()} scenarios for {subdir} completed.")