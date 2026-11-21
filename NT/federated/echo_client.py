"""Flower client implementing the maximum-privacy ``echo`` boundary.

During fitting, the client returns the received global parameters unchanged
and performs no local optimization. It still evaluates the global model on
its local test loader so each game's participant-specific accuracy is
available to the result logger.
"""

from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import torch

from models.neural_network import TrafficNN
from utils.evaluation import evaluate_model


class EchoClient(fl.client.NumPyClient):
    """Passive client that does not derive updates from its private data."""

    def __init__(self, cid, model, testloader, cfg, n_samples):
        """Initialize the echo client and its FedAvg example count.

        Args:
            cid: Flower client identifier (``"0"`` or ``"1"``).
            model: Traffic classifier receiving server parameters.
            testloader: Local test data used only for evaluation.
            cfg: Experiment configuration, including the DP delta setting.
            n_samples: Example count reported during fitting. The scenario
                runner sets this to the active peer's training-set size.
        """
        super().__init__()
        self.cid        = cid
        self.model      = model
        self.testloader = testloader
        self.cfg        = cfg
        self.n_samples  = n_samples  # Match the active peer's sample count for equal FedAvg weighting.
        self.device     = torch.device("cpu")
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Parameter handling shared with the regular traffic client.
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        """Return the model state as NumPy arrays for Flower."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load a server-provided parameter sequence into the model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # Fitting returns the received global parameters without local training.
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        """Return the server parameters unchanged without local training."""
        self.set_parameters(parameters)
        return self.get_parameters(config={}), self.n_samples, {}

    # ------------------------------------------------------------------
    # Evaluation remains enabled to measure the global model on local test data.
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        """Measure loss and accuracy on the echo client's local test data."""
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()

        loss, accuracy = evaluate_model(self.model, self.testloader, criterion)

        return (
            float(loss),
            len(self.testloader.dataset),
            {
                "client_id": self.cid,
                "accuracy": float(accuracy),
                "loss": float(loss),
                "epsilon": 0.0,
            },
        )


# ------------------------------------------------------------------
# Factory with the same Flower client-factory interface as the active client.
# ------------------------------------------------------------------

def create_echo_client_fn(trainloader, testloader, cfg):
    """Create a Flower client factory for echo clients.

    Args:
        trainloader: Training loaders used to determine each client's sample
            count.
        testloader: Test loaders supplied to each echo client.
        cfg: Model and experiment configuration.

    Returns:
        A ``client_fn`` that creates an ``EchoClient`` from a Flower client ID.
    """
    def client_fn(cid: str) -> fl.client.Client:
        """Construct the echo client corresponding to ``cid``."""
        idx = int(cid)
        return EchoClient(
            cid        = cid,
            model      = TrafficNN(
                input_dim   = cfg.dataset.input_dim,
                num_classes = cfg.dataset.num_classes,
            ),
            testloader = testloader[idx],
            cfg        = cfg,
            n_samples  = len(trainloader[idx].dataset),
        ).to_client()

    return client_fn