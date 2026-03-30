import flwr as fl
import ray
from itertools import product
from federated.universal_client import create_client_fn
from federated.server import get_on_fit_config
from utils.save import save_federated_history, setup_file
from utils.metrics import player_specific_metrics

def run_experiment(config, train_loaders, test_loaders, subdir, mode):
    """
    Universal experiment runner for both DP and Suppression modes.
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

    # Run each scenario
    for val1, val2 in scenarios:
        print(f"\n" + "="*60)
        print(f"📡 Simulation ({mode.upper()}) | Client1: {val1} | Client2: {val2}")
        print("="*60)

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

        # Start the simulation with the created strategy and client function
        history = fl.simulation.start_simulation(
            client_fn=create_client_fn(train_loaders, test_loaders, config),
            num_clients=config.num_clients,
            config=fl.server.ServerConfig(num_rounds=config.config.federated_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )

        # Save the history and results in the specified format
        save_federated_history(
            history, config, val1, val2, subdir, 
            base_dir=base_dir, metric_name=metric_name
        )

        # Print the final results to the console
        res_1 = history.metrics_distributed["client1_accuracy"][-1][1]
        res_2 = history.metrics_distributed["client2_accuracy"][-1][1]
        print(f"📊 Final Results: Client1: {res_1:.2%}, Client2: {res_2:.2%}")

        ray.shutdown()