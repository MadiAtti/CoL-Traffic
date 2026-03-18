import numpy as np
from opacus import PrivacyEngine
import torch

from config import local_params


def train_local_model(model, train_loader, criterion, optimizer, round_num):
    """Train a model locally for FL for specified number of epochs"""
    model.train()
    epochs_data = []

    for epoch in range(local_params['epochs']):
        print('\t\t\t\tEpoch ', epoch)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / (total_samples / train_loader.batch_size)
        accuracy = correct_predictions / total_samples

        epoch_data = {
            'round': round_num,
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy
        }
        epochs_data.append(epoch_data)

    return model, epochs_data

def train_model(model, train_loader, criterion, optimizer):
    """Train a local baseline model"""
    model.train()
    epochs_data = []
    
    for epoch in range(local_params['epochs'] * local_params['federated_rounds']):
        print('\t\tRound ', epoch)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / (total_samples / train_loader.batch_size)
        accuracy = correct_predictions / total_samples
        
        epoch_data = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy
        }
        epochs_data.append(epoch_data)

    return model, epochs_data

def train_local_dp_model(model, train_loader, criterion, optimizer, round_num, noise_multiplier):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "avg_grad_norm": [],
        "snr": [],
        "noise_norms": []
    }
    if noise_multiplier is not None:
        model.train()
        history["epsilons"] = []
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=local_params["max_grad_norm"],
            clipping="flat",
            poisson_sampling=True
        )
    for epoch in range(local_params['epochs']):
        print('\t\t\t\t\tEpoch ', epoch)
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        grad_norms, noise_norms = [], []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            if noise_multiplier is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=local_params["max_grad_norm"]
                )
                grad_norms.append(grad_norm.item())
                noise_std = noise_multiplier * local_params["max_grad_norm"]
                noise_norm = noise_std * np.sqrt(sum(p.numel() for p in model.parameters()))
                noise_norms.append(noise_norm)
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        avg_noise_norm = np.mean(noise_norms) if noise_norms else 0.0
        snr = avg_grad_norm / avg_noise_norm if avg_noise_norm > 0 else float('inf')
        epoch_loss /= total
        epoch_acc = correct / total
        if noise_multiplier is not None:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            history["epsilons"].append(epsilon)
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        history["avg_grad_norm"].append(avg_grad_norm)
        history["snr"].append(snr)
    return model, history
