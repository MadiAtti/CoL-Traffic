"""Training loops shared by federated clients."""

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def train_standard(model, loader, optimizer, epochs, device):
    """Train a model for the requested epochs without differential privacy.

    Args:
        model: PyTorch model to update.
        loader: Iterable of feature and label batches.
        optimizer: Optimizer attached to ``model``.
        epochs: Number of complete passes over ``loader``.
        device: PyTorch device for model inputs and labels.

    Returns:
        The trained model.
    """
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

def train_dp(
    model,
    loader,
    optimizer,
    epochs,
    noise,
    max_grad_norm,
    device,
    privacy_engine=None,
):
    """Train with Opacus per-sample clipping and Gaussian noise addition.

    Args:
        model: PyTorch model to update.
        loader: Iterable of feature and label batches.
        optimizer: Optimizer attached to ``model``.
        epochs: Number of complete passes over the private loader.
        noise: Opacus noise multiplier.
        max_grad_norm: Per-sample gradient clipping threshold.
        device: PyTorch device for model inputs and labels.
        privacy_engine: Optional client-level accountant to reuse across
            federated rounds.

    Returns:
        The underlying trained model after removing per-sample gradient hooks.
    """

    # 1. DP COMPATIBILITY CHECK
    # Opacus cannot handle standard BatchNorm as it leaks privacy between samples.
    # ModuleValidator.fix() replaces BatchNorm with GroupNorm automatically.
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    model.to(device)

    # Keep one accountant per client across FL rounds while creating a fresh
    # optimizer and private wrapper for each round.
    if privacy_engine is None:
        privacy_engine = PrivacyEngine()
    
    # poisson_sampling=False: Essential for Federated Learning as Flower 
    # provides fixed-size batches/iterators.
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,    
        data_loader=loader,
        noise_multiplier=noise,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
    )
    
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # 3. TRAINING LOOP
    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass: Opacus hooks calculate per-sample gradients here
            loss.backward()
            
            # Optimizer step: Gradients are clipped and noise is added here
            optimizer.step()

    # Remove per-sample-gradient hooks before the next FL round re-wraps
    # the underlying model with a fresh optimizer.
    if hasattr(model, "remove_hooks"):
        model.remove_hooks()

    return model._module if hasattr(model, "_module") else model