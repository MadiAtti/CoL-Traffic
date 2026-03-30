import torch
from opacus import PrivacyEngine

def train_standard(model, loader, optimizer, epochs, device):
    """Standard training loop for federated learning without differential privacy."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model

def train_dp(model, loader, optimizer, epochs, noise, max_grad_norm, device):
    """Differential privacy training loop for federated learning."""
    # Initialize the PrivacyEngine to make the model and optimizer private with the specified noise and max_grad_norm
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=noise, max_grad_norm=max_grad_norm,
        poisson_sampling=False
    )
    
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model