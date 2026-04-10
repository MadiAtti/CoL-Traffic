import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

class TrafficLightningModule(L.LightningModule):
    """
    LightningModule wrapper for the TrafficNN architecture.
    Handles training logic, validation, and optimization.
    """
    def __init__(self, input_dim, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Calculate hidden layer sizes based on original heuristic
        hidden_layer1_neurons = int((2 / 3) * input_dim + num_classes)
        hidden_layer2_neurons = input_dim
        
        # Core network architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer1_neurons, hidden_layer2_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer2_neurons, num_classes)
        )
        
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """Standard forward pass through the sequential model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Encapsulates the training loop logic for a single batch."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Logging for monitoring progress
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Encapsulates the validation loop logic for a single batch."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Logs are monitored by the EarlyStopping callback
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Initializes the optimizer using the learning rate from config."""
        return optim.Adam(self.parameters(), lr=self.lr)

# class TrafficNN(nn.Module):
#     '''
#     Simple feedforward neural network for traffic classification.
#     The architecture consists of two hidden layers with ReLU activations.
#     The number of neurons in the hidden layers is determined by a common 
#     heuristic based on the input dimension and the number of classes.
#     '''
#     def __init__(self, input_dim, num_classes):
#         super(TrafficNN, self).__init__()
        
#         hidden_layer1_neurons = int((2 / 3) * input_dim + num_classes)
#         hidden_layer2_neurons = input_dim
        
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_layer1_neurons),
#             nn.ReLU(),
#             nn.Linear(hidden_layer1_neurons, hidden_layer2_neurons),
#             nn.ReLU(),
#             nn.Linear(hidden_layer2_neurons, num_classes)
#         )
        
#     def forward(self, x):
#         return self.model(x)