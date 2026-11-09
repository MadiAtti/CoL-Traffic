"""Flower client for standard and differentially private local training.

The server supplies per-client DP noise settings. Each client trains its
received global model locally and reports evaluation metrics tagged with its
stable client ID so aggregation does not depend on result ordering.
"""

from collections import OrderedDict
import os
import flwr as fl
import torch
from opacus import PrivacyEngine
from models.neural_network import TrafficNN
from utils.evaluation import evaluate_model
from utils.training import train_dp, train_standard

class UniversalTrafficClient(fl.client.NumPyClient):
    '''
    Universal client class for federated learning experiments on traffic data.
    This client can handle both Differential Privacy (DP) and Feature Suppression scenarios
    based on the configuration passed during initialization.
    '''

    # Initialization with client ID, model, data loaders, and configuration
    def __init__(self, cid, model, trainloader, testloader, cfg):
        """Initialize a client with its local loaders and experiment settings.

        Args:
            cid: Flower client identifier (``"0"`` or ``"1"``).
            model: Traffic classifier instance.
            trainloader: Local training DataLoader, possibly feature-suppressed.
            testloader: Local evaluation DataLoader.
            cfg: Hydra experiment configuration.
        """
        super().__init__()
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.cfg = cfg 
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.privacy_engine = None

    def get_parameters(self, config):
        """Return model state tensors as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load a server-provided parameter sequence into the model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # Fit method that handles both DP and Suppression based on the config parameters
    def fit(self, parameters, config):
        """Apply one round of local training and return updated parameters.

        Positive noise multipliers use Opacus DP-SGD; zero noise uses ordinary
        stochastic gradient training.
        """
        self.set_parameters(parameters)
        
        # DP handling: Determine the noise level for this client based on the config parameter
        noise = config.get("client1_noise") if self.cid == "0" else config.get("client2_noise")
        
        # Ensure noise is a float and handle the case where it might be None or "None"
        if noise is None or str(noise) == "None": 
            noise = 0.0
        else:
            noise = float(noise)
        
        epochs = self.cfg.config.num_epochs
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.config.lr)
        # Train model based on the specified noise level: if noise > 0, use DP training; otherwise, use standard training
        if noise > 0:
            if self.privacy_engine is None:
                self.privacy_engine = PrivacyEngine()
            self.model = train_dp(
                self.model, self.trainloader, optimizer, epochs, 
                noise, self.cfg.config.max_grad_norm, self.device,
                privacy_engine=self.privacy_engine,
            )
        else:
            self.model = train_standard(
                self.model, self.trainloader, optimizer, epochs, self.device
            )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    # Evaluation method that evaluates the model on the test set and returns the loss and accuracy
    def evaluate(self, parameters, config):
        """Evaluate the received global model and report client-tagged metrics."""
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Use the original evaluation function to compute loss and accuracy on the test set
        loss, accuracy = evaluate_model(self.model, self.testloader, criterion)
        epsilon = (
            self.privacy_engine.get_epsilon(delta=self.cfg.config.dp_delta)
            if self.privacy_engine is not None
            and self.privacy_engine.accountant.history
            else 0.0
        )

        return float(loss), len(self.testloader.dataset), {
            "client_id": self.cid,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "epsilon": float(epsilon),
        }

# Factory function to create a client_fn for Flower that initializes 
# UniversalTrafficClient instances with the appropriate data loaders and configuration
def create_client_fn(trainloader, testloader, cfg):
    """Create a Flower client factory for regular traffic-classification runs.

    Args:
        trainloader: Sequence of local training DataLoaders indexed by client ID.
        testloader: Sequence of local evaluation DataLoaders indexed by ID.
        cfg: Hydra configuration used to construct the model and clients.

    Returns:
        A Flower ``client_fn`` creating ``UniversalTrafficClient`` instances.
    """

    # The client_fn that will be passed to the Flower simulation, which creates a UniversalTrafficClient 
    # instance for each client ID (cid) with the corresponding train and test loaders and configuration
    def client_fn(cid: str):
        """Construct the client associated with Flower's string client ID."""

        return UniversalTrafficClient(cid = cid,
                                    model = TrafficNN(input_dim=cfg.dataset.input_dim, num_classes=cfg.dataset.num_classes),
                                    trainloader = trainloader[int(cid)], 
                                    testloader = testloader[int(cid)], 
                                    cfg = cfg).to_client()
    return client_fn