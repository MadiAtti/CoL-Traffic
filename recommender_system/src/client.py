import flwr as fl
import torch
import numpy as np
from collections import OrderedDict

def get_parameters(model):
    """As user-sets are disjoint, players only share the item feature matrix Q."""
    return [model.item_emb.weight.cpu().detach().numpy()]

def set_parameters(model, parameters):
    """Updates only the shared item matrix from the FedAvg server."""
    model.item_emb.weight.data = torch.tensor(parameters[0]).to(model.item_emb.weight.device)

class RecommenderClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, conf, privacy_mode, privacy_param, dp_clip, cid):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.conf = conf
        
        # reduction='sum' mirrors the theoretical SGD math bounds
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.privacy_mode = privacy_mode
        self.privacy_param = privacy_param
        self.dp_clip = dp_clip
        self.cid = cid
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        # Synchronize local model parameters with the server's broadcasted weights
        self.set_parameters(parameters)

        # Unified Echoing Protocol for ALL privacy mechanisms (SUP and DP)
        # At maximum privacy (p >= 1.0), bypass training, echo the server's weights,
        # and return 0 examples so this update does not dilute the active partner.
        if self.privacy_param >= 1.0:
            return self.get_parameters(config={}), 0, {}

        original_item_weights = parameters[0].copy()
        self.set_parameters(parameters)

        g = torch.Generator()
        if 'seed' in self.conf:
            g.manual_seed(self.conf['seed'])

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.conf['batch_size'], shuffle=True, generator=g
        )
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.conf['lr'])
        
        self.model.train()
        for _ in range(self.conf['epochs']):
            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                optimizer.zero_grad()
                
                preds, l2_reg = self.model(users, items)
                preds = torch.clamp(preds, min=-2.0, max=2.0)
                
                mse_loss = self.loss_fn(preds, ratings)
                loss = mse_loss + self.conf['lambda_reg'] * l2_reg.sum()
                
                loss.backward()
                optimizer.step()

        # Targeted DP Logic: Add noise strictly to the shared item matrix updates
        if self.privacy_mode == 'dp' and self.privacy_param > 0.0:
            new_item_weights = self.model.item_emb.weight.cpu().detach().numpy()
            delta_q = new_item_weights - original_item_weights
            
            # Clip the update
            l2_norm = np.linalg.norm(delta_q)
            if l2_norm > self.dp_clip:
                delta_q = delta_q * (self.dp_clip / l2_norm)
                
            # Add scaled Gaussian noise
            sigma = self.privacy_param / (1.0 - self.privacy_param)
            noise_std = self.dp_clip * sigma
            noise = np.random.normal(0.0, noise_std, delta_q.shape)
            
            final_item_weights = (original_item_weights + delta_q + noise).astype(np.float32)
            self.model.item_emb.weight.data = torch.tensor(final_item_weights).to(self.device)

        return self.get_parameters(config={}), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.conf['batch_size'], shuffle=False)
        self.model.eval()
        
        total_squared_error, samples = 0.0, 0
        with torch.no_grad():
            for users, items, ratings in test_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                
                preds, _ = self.model(users, items)
                preds = torch.clamp(preds, min=-2.0, max=2.0)
                
                total_squared_error += self.loss_fn(preds, ratings).item()
                samples += len(ratings)
                
        rmse = np.sqrt(total_squared_error / max(1, samples))
        return float(rmse), samples, {"rmse": float(rmse)}