from collections import OrderedDict

import flwr as fl
import torch
from utils.evaluation import evaluate_model

class SimpleClient(fl.client.NumPyClient):
    def __init__(self,
                model, 
                trainloader, 
                testloader) -> None:
        super().__init__()

        ## initalize client with model, trainloader, testloader
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model

        ## set device to GPU if available, else CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.criterion = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters())

    # def get_parameters(self, config):
    #     return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)

        print("KLIENS: Tanítás indul...")
        # Meghívjuk az eredeti tanító függvényedet 1 körre
        # train_local_model(self.model, self.trainloader, self.criterion, self.optimizer, round_num=1)
        # return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        loss, accuracy = evaluate_model(self.model, self.testloader, criterion)
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# Ide jön az adatbetöltésed, majd:
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SimpleClient(model, tl, vtl))