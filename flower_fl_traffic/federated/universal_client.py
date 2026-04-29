from collections import OrderedDict
import os
import flwr as fl
import torch
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
        super().__init__()
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.cfg = cfg 
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # Fit method that handles both DP and Suppression based on the config parameters
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # active features count for debugging: Determine the number of non-zero columns in the first batch of the training data
        test_batch_x, _ = next(iter(self.trainloader))
        num_active = torch.any(test_batch_x != 0, dim=0).sum().item()
        
        # Expected active features based on the configuration for this client: if client ID is "0", use client1_features; otherwise, use client2_features
        exp = config.get("client1_features") if self.cid == "0" else config.get("client2_features")
        
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
            self.model = train_dp(
                self.model, self.trainloader, optimizer, epochs, 
                noise, self.cfg.config.max_grad_norm, self.device
            )
        else:
            self.model = train_standard(
                self.model, self.trainloader, optimizer, epochs, self.device
            )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    # Evaluation method that evaluates the model on the test set and returns the loss and accuracy
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Use the original evaluation function to compute loss and accuracy on the test set
        loss, accuracy = evaluate_model(self.model, self.testloader, criterion)
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# Factory function to create a client_fn for Flower that initializes 
# UniversalTrafficClient instances with the appropriate data loaders and configuration
def create_client_fn(trainloader, testloader, cfg):

    # The client_fn that will be passed to the Flower simulation, which creates a UniversalTrafficClient 
    # instance for each client ID (cid) with the corresponding train and test loaders and configuration
    def client_fn(cid: str):

        return UniversalTrafficClient(cid = cid,
                                    model = TrafficNN(input_dim=cfg.dataset.input_dim, num_classes=cfg.dataset.num_classes),
                                    trainloader = trainloader[int(cid)], 
                                    testloader = testloader[int(cid)], 
                                    cfg = cfg).to_client()
    return client_fn