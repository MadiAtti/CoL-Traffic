from collections import OrderedDict
import os
import flwr as fl
import torch
from models.neural_network import TrafficNN
from utils.evaluation import evaluate_model
from utils.training import train_dp, train_standard

class UniversalTrafficClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader, cfg):
        super().__init__()
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.cfg = cfg 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        if self.cid == "0":
            noise = config.get("p1_noise")
        else:
            noise = config.get("p2_noise")
        
        if noise is None or noise == "None": 
            noise = 0.0
        else:
            noise = float(noise)
        
        epochs = self.cfg.config.num_epochs
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.config.lr)

        client_label = f"Client {self.cid}"
        pid = os.getpid()

        if noise > 0:
            print(f"--- [PID:{pid}] {client_label} training with DP (noise: {noise}) ---")
            self.model = train_dp(
                self.model, self.trainloader, optimizer, epochs, 
                noise, self.cfg.config.max_grad_norm, self.device
            )
        else:
            print(f"--- [PID:{pid}] {client_label} training standard ---")
            self.model = train_standard(
                self.model, self.trainloader, optimizer, epochs, self.device
            )

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        
        loss, accuracy = evaluate_model(self.model, self.testloader, criterion)
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}
    
def create_client_fn(trainloader, testloader, cfg):

    def client_fn(cid: str):

        return UniversalTrafficClient(cid = cid,
                                    model = TrafficNN(input_dim=cfg.dataset.input_dim, num_classes=cfg.dataset.num_classes),
                                    trainloader = trainloader[int(cid)], 
                                    testloader = testloader[int(cid)], 
                                    cfg = cfg).to_client()

    return client_fn