import torch

@torch.no_grad()
def evaluate_model(model, test_loader, criterion):
    '''
    Evaluate the given model on the test set using the provided criterion (loss function).
    Returns the average loss and accuracy over the test set.
    '''

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(test_loader), correct / total
