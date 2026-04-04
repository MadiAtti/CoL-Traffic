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

    (val1, val2, config, train_loaders, test_loaders, subdir, 
     mode, base_dir, metric_name, param_keys) = args

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
        client_fn=create_client_fn(train_loaders, test_loaders, config),
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.config.federated_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
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
    num_parallel_scenarios = 7

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