import flwr as fl
import torch
from collections import OrderedDict
from utils.training import train_local_model, train_local_dp_model
from utils.evaluation import evaluate_model
from models.neural_network import NeuralNetwork # Feltételezve az új nevet

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, test_loader, local_params):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.params = local_params
        # Minden kliens saját modellel indul
        self.model = NeuralNetwork(local_params['input_dim'], local_params['num_classes'])
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        server_round = config.get("server_round", 1)
        noise = config.get("noise_multiplier", None)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

        if noise is not None:
            self.model, _ = train_local_dp_model(
                self.model, self.train_loader, self.criterion, optimizer, 
                round_num=server_round, noise_multiplier=noise
            )
        else:
            self.model, _ = train_local_model(
                self.model, self.train_loader, self.criterion, optimizer, round_num=server_round
            )

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate_model(self.model, self.test_loader, self.criterion)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}