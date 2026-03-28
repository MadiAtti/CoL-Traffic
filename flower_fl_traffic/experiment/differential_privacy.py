from itertools import product
import flwr as fl

from federated.universal_client import create_client_fn
from federated.server import get_on_fit_config
from utils.save import save_federated_history, setup_file
from utils.metrics import player_specific_metrics

def run_dp_experiment(config, train_loaders, test_loaders, subdir):
    
    setup_file(config, subdir)

    noise_levels = config.config.noise_levels
    noise_scenarios = list(product(noise_levels, noise_levels))
    
    for client1_noise, client2_noise in noise_scenarios:
        print(f"\n" + "="*60)
        print(f"📡 SZIMULÁCIÓ | Client 1 Zaj: {client1_noise} | Client 2 Zaj: {client2_noise}")
        print("="*60)
        
        ## 3. Define strategy 
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  
            fraction_evaluate=1.0,  
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
            on_fit_config_fn=get_on_fit_config(client1_noise, client2_noise),
            evaluate_metrics_aggregation_fn=player_specific_metrics,
        )
        
        ## 5. Start simulation
        history = fl.simulation.start_simulation(
            client_fn=create_client_fn(train_loaders, test_loaders, config),
            num_clients=config.num_clients,
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
        )

        save_path = save_federated_history(history, config, client1_noise, client2_noise, subdir)

        final_client1 = history.metrics_distributed["client1_accuracy"][-1][1]
        final_client2 = history.metrics_distributed["client2_accuracy"][-1][1]

        print(f"\n📊 VÉGEREDMÉNY:")
        print(f"   👤 Client 1 (Noise: {client1_noise}): {final_client1:.2%}")
        print(f"   👤 Client 2 (Noise: {client2_noise}): {final_client2:.2%}")
        print("-" * 60)
