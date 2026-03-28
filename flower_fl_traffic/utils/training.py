import torch
from opacus import PrivacyEngine

def train_standard(model, loader, optimizer, epochs, device):
    """Sima tanítás (Baseline, FL, Suppression - mindenhez jó, ami nem zajos)"""
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
    """Zajos tanítás (DP-FL)"""
    privacy_engine = PrivacyEngine()
    # Az Opacus 'felokosítja' a modellt és a loadert
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