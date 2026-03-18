import flwr as fl
from federated.client import FlowerClient
from federated.strategy import get_strategy

def run_flower_simulation(client_data_map, local_params, noise=None):
    
    def client_fn(cid: str):
        train_loader, test_loader = client_data_map[cid]
        return FlowerClient(cid, train_loader, test_loader, local_params)

    strategy = get_strategy(local_params, current_noise=noise)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_data_map),
        config=fl.server.ServerConfig(num_rounds=local_params['federated_rounds']),
        strategy=strategy,
    )
    
    return history