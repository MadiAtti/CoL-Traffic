"""Shared model-evaluation routine for local and federated clients."""

import torch

@torch.no_grad()
def evaluate_model(model, test_loader, criterion):
    """Compute sample-weighted mean loss and accuracy on a data loader.

    Args:
        model: Model whose predictions are evaluated.
        test_loader: Iterable of ``(inputs, labels)`` batches.
        criterion: Loss function evaluated on each batch.

    Returns:
        A pair ``(mean_loss, accuracy)`` as Python floats.

    Raises:
        ValueError: If the loader contains no examples.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate a model with an empty test dataset.")
    return total_loss / total, correct / total
