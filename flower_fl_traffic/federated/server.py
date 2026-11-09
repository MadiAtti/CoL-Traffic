"""Factories for Flower server-side per-round configuration callbacks."""


def get_on_fit_config(**kwargs):
    """Create a callback that sends scenario parameters to every client.

    Args:
        **kwargs: Scenario-specific key/value pairs such as client noise
            multipliers or retained-feature counts.

    Returns:
        A Flower ``on_fit_config_fn`` that includes ``current_round`` and the
        supplied scenario parameters.
    """
    def fit_config_fn(server_round: int):
        """Return the shared fit configuration for ``server_round``."""
        # Base configuration that includes the current round number, which can be used by 
        # clients to determine when to apply feature suppression or noise addition
        config = {"current_round": server_round}

        # Update the configuration with the specific parameters for the current scenario (DP or Suppression) that were passed as keyword arguments when creating the on_fit_config_fn. 
        # This allows us to dynamically set the noise levels or feature limits for each client based on the scenario being tested in the experiment.
        config.update(kwargs)
        
        return config

    return fit_config_fn