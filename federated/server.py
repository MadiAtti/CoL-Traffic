import flwr as fl
from federated.client import FlowerClient
from federated.strategy import get_strategy
from config import local_params

def run_flower_simulation(client_data_map, noise=None):
    
    def client_fn(cid: str):
        train_loader, test_loader = client_data_map[cid]
        return FlowerClient(cid, train_loader, test_loader)

    strategy = get_strategy( current_noise=noise)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_data_map),
        config=fl.server.ServerConfig(num_rounds=local_params['federated_rounds']),
        strategy=strategy,
    )
    
    return history