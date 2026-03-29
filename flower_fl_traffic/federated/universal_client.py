from collections import OrderedDict
import os
import flwr as fl
import torch
from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
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
        
        # --- 1. OPTIMALIZÁLT SUPPRESSION KEZELÉSE ---
        f_limit = config.get("p1_features") if self.cid == "0" else config.get("p2_features")
        server_round = config.get("current_round", 1)

        # Csak az ELSŐ KÖRBEN cserélünk Loadert, vagy ha még nem történt meg
        if f_limit is not None and server_round == 1:
            f_limit = int(f_limit)
            total_features = self.cfg.dataset.input_dim
            
            # Kinyerjük az eredeti adatokat (csak egyszer a kísérlet elején)
            raw_x = self.trainloader.dataset.X.numpy()
            raw_y = self.trainloader.dataset.y.numpy()
            
            # Szeletelés és visszahizlalás nullákkal
            x_sub = raw_x[:, :f_limit]
            new_ds = CustomDataset(
                X=x_sub, 
                y=raw_y, 
                feature_indices=range(f_limit), 
                total_features=total_features
            )

            # --- ELLENŐRZÉS: Első 5 sor kiírása ---
            # print(f"\n🔍 [DEBUG - Client {self.cid}] Adatellenőrzés (Limit: {f_limit}):")
            # # Megnézzük az első 5 sort a hálónak átadott formátumban
            # for i in range(min(5, len(new_ds))):
            #     sample_x, _ = new_ds[i]
            #     # Kerekítve írjuk ki, hogy átlátható legyen
            #     formatted_row = [round(float(val), 4) for val in sample_x]
            #     print(f"  Sor {i}: {formatted_row}")
            # print(f"✅ Loader sikeresen frissítve.\n")
            # --------------------------------------
            
            # Loader csere (ez kitart a 20. kör végéig)
            self.trainloader = DataLoader(
                new_ds, 
                batch_size=self.cfg.config.batch_size, 
                shuffle=True
            )
            print(f"✅ [Client {self.cid}] Adatok előkészítve a kísérlethez: {f_limit}/{total_features} feature")

        noise = config.get("p1_noise") if self.cid == "0" else config.get("p2_noise")
        
        # Tisztítjuk a noise értéket
        if noise is None or str(noise) == "None": 
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