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

    # Only wrap the model and optimizer with PrivacyEngine if they haven't been wrapped already 
    # (e.g., in case of multiple calls to this function)
    if not hasattr(model, "grad_sample_module"):
        privacy_engine = PrivacyEngine()
        # poisson_sampling=False, mert Flower-ben fix batch-ekkel dolgozunk
        model, optimizer, loader = privacy_engine.make_private(
            module=model, 
            optimizer=optimizer, 
            data_loader=loader,
            noise_multiplier=noise, 
            max_grad_norm=max_grad_norm,
            poisson_sampling=False
        )
    
    # Standard training loop with the private model and optimizer
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # After training, we return the original model (not the wrapped version) to ensure compatibility with Flower's expectations
    if hasattr(model, "original_module"):
        return model.original_module
    
    return model