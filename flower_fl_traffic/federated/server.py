def get_on_fit_config(**kwargs):
    """
    Factory function to create an on_fit_config_fn for Flower that includes the specified parameters for DP and Suppression experiments.
    This function generates a configuration function that will be called during the fit phase of each federated round, 
    allowing us to pass the appropriate noise levels or feature limits to the clients based on the current round and the experiment settings.
    """
    def fit_config_fn(server_round: int):
        # Base configuration that includes the current round number, which can be used by 
        # clients to determine when to apply feature suppression or noise addition
        config = {"current_round": server_round}

        # Update the configuration with the specific parameters for the current scenario (DP or Suppression) that were passed as keyword arguments when creating the on_fit_config_fn. 
        # This allows us to dynamically set the noise levels or feature limits for each client based on the scenario being tested in the experiment.
        config.update(kwargs)
        
        return config

    return fit_config_fn