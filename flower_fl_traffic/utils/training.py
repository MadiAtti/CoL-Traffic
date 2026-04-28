import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

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
    """
    Differential Privacy (DP) training loop for Federated Learning clients.
    
    This function wraps a standard PyTorch model with Opacus to ensure 
    DP guarantees (per-sample gradient clipping and noise addition).
    """

    # 1. DP COMPATIBILITY CHECK
    # Opacus cannot handle standard BatchNorm as it leaks privacy between samples.
    # ModuleValidator.fix() replaces BatchNorm with GroupNorm automatically.
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    
    model.to(device)

    # 2. PRIVACY ENGINE INITIALIZATION
    # We initialize the engine only if the model isn't already "private".
    # In FL, we usually re-wrap every round because the optimizer is re-initialized.
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

    # 4. UNWRAPPING
    # Opacus wraps the model in a 'GradSampleModule'. 
    # We return the underlying '_module' so the weights can be extracted 
    # by the Flower server without serialization/metadata errors.
    return model._module if hasattr(model, "_module") else model