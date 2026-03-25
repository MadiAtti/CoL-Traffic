import flwr as fl

def get_strategy(current_noise=None):
    
    def on_fit_config_fn(server_round: int):
        return {
            "server_round": server_round,
            "noise_multiplier": current_noise,
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             
        min_fit_clients=2,           
        min_available_clients=2,    
        on_fit_config_fn=on_fit_config_fn,
    )
    return strategy