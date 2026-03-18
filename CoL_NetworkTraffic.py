import argparse
import random
import json
import os
import warnings
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from copy import deepcopy
from itertools import product
from opacus import PrivacyEngine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Global configuration parameters
local_params = {
    'data_path': "dataset/dataset.parquet",
    'p1_path':   "dataset/p1.parquet",
    'p2_path':   "dataset/p2.parquet",
    'p11_path':  "dataset/p11.parquet",
    'p12_path':  "dataset/p12.parquet",
    'p21_path':  "dataset/p21.parquet",
    'p22_path':  "dataset/p22.parquet",
    'feature_columns': [
        'protocol',
        'bidirectional_min_ps',
        'bidirectional_mean_ps',
        'bidirectional_stddev_ps',
        'bidirectional_max_ps',
        'src2dst_stddev_ps',
        'src2dst_max_ps',
        'dst2src_min_ps',
        'dst2src_mean_ps',
        'dst2src_stddev_ps',
        'dst2src_max_ps',
        'bidirectional_stddev_piat_ms',
        'bidirectional_max_piat_ms',
        'bidirectional_rst_packets'
    ],
    'target_column': 'application_name',
    'initial_split_ratio': 0.5,
    'test_split_ratio': 0.2,
    'epochs': 2,
    'federated_rounds': 20,
    'batch_size': 256,
    'learning_rate': 0.001,
    'input_dim': None,
    'num_classes': None,
    "noise_levels": [None, 0.33, 0.67, 1.00, 1.33, 1.67, 2.00],  # CHANGE HERE privacy-NOISE parameters
    "max_grad_norm": 1.0,
}

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Secure RNG turned off.*")
warnings.filterwarnings("ignore", message="Using a non-full backward hook.*")


class CustomDataset(Dataset):
    def __init__(self, X, y, feature_indices=None, total_features=None):
        if total_features is None:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            # Create a zero-filled tensor with the full feature dimension
            X_selected = np.zeros((len(X), total_features), dtype=np.float32)
            # Fill in the selected features
            for new_idx, orig_idx in enumerate(feature_indices):
                X_selected[:, orig_idx] = X[:, new_idx]

            self.X = torch.tensor(X_selected, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNetwork, self).__init__()
        
        hidden_layer1_neurons = int((2 / 3) * input_dim + num_classes)
        hidden_layer2_neurons = input_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer1_neurons, hidden_layer2_neurons),
            nn.ReLU(),
            nn.Linear(hidden_layer2_neurons, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)


def average_models(model_p1, model_p2):
    global_model = NeuralNetwork(input_dim=len(local_params["feature_columns"]), num_classes=local_params['num_classes'])
    for p_global, p1, p2 in zip(global_model.parameters(), model_p1.parameters(), model_p2.parameters()):
        p_global.data = (p1.data + p2.data) / 2.0
    return global_model


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


@torch.no_grad()
def evaluate_model(model, test_loader, criterion):
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


def create_data_loaders(X_train, X_test, y_train, y_test):
    """Create train and test data loaders"""
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_params['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=local_params['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, test_loader


def generate_feature_combinations():
    """Generate all possible feature combinations for privacy experiments"""
    num_features = len(local_params['feature_columns'])
    combinations = []
    # CHANGE HERE privacy-SUP parameters
    for num_features_to_keep in [14, 12, 10, 8, 6, 4, 2]:
        selected_features = local_params['feature_columns'][:num_features_to_keep]
        combinations.append(selected_features)

    return combinations


def create_data_loaders_sup(X_train, X_test, y_train, y_test, feature_indices, total_features):
    """Create train and test data loaders with feature selection"""
    train_dataset = CustomDataset(X_train, y_train, feature_indices, total_features)
    test_dataset = CustomDataset(X_test, y_test, feature_indices, total_features)

    train_loader = DataLoader(
        train_dataset,
        batch_size=local_params['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=local_params['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader


def create_suppressed_dataset(X, feature_indices):
    """Create a dataset with specified features"""
    return X[:, feature_indices]


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_data(seed):
    input_path = local_params['data_path']
    p1_path = local_params['p1_path']
    p2_path = local_params['p2_path']
    p11_path = local_params['p11_path']
    p12_path = local_params['p12_path']
    p21_path = local_params['p21_path']
    p22_path = local_params['p22_path']

    # Create results directory if it doesn't exist
    os.makedirs('1_local_baseline', exist_ok=True)
    os.makedirs('2_fl_baseline', exist_ok=True)
    os.makedirs('3_suppression', exist_ok=True)
    os.makedirs('4_noise', exist_ok=True)
    os.makedirs('1_local_baseline/P1', exist_ok=True)
    os.makedirs('2_fl_baseline/P1', exist_ok=True)
    os.makedirs('3_suppression/P1', exist_ok=True)
    os.makedirs('4_noise/P1', exist_ok=True)
    os.makedirs('1_local_baseline/P2', exist_ok=True)
    os.makedirs('2_fl_baseline/P2', exist_ok=True)
    os.makedirs('3_suppression/P2', exist_ok=True)
    os.makedirs('4_noise/P2', exist_ok=True)

    # Check if dataset is present
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    # Load columns of interest and drop NA rows
    columns_to_load = local_params['feature_columns'] + [local_params['target_column']]
    df = pd.read_parquet(input_path, columns=columns_to_load).dropna()

    # Separate features and target
    X = df[local_params['feature_columns']].values.astype(np.float32)
    y_raw = df[local_params['target_column']].values

    # Encode the target
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Update local_params with dynamic shape info
    local_params['input_dim'] = X.shape[1]
    local_params['num_classes'] = len(np.unique(y))

    # Split into P1 & P2, and further into P11, P12 & P21 & P22 if not already done
    if not (os.path.exists(p1_path) and os.path.exists(p2_path)):
        X_p1, X_p2, y_p1, y_p2 = train_test_split(
            X, y,
            test_size=local_params['initial_split_ratio'],
            stratify=y,
            random_state=seed
        )

        X_p11, X_p12, y_p11, y_p12 = train_test_split(
            X_p1, y_p1,
            test_size=local_params['initial_split_ratio'],
            stratify=y_p1,
            random_state=seed
        )

        X_p21, X_p22, y_p21, y_p22 = train_test_split(
            X_p2, y_p2,
            test_size=local_params['initial_split_ratio'],
            stratify=y_p2,
            random_state=seed
        )

        # Save P1
        df_p1 = pd.DataFrame(X_p1, columns=local_params['feature_columns'])
        df_p1[local_params['target_column']] = y_p1
        df_p1.to_parquet(p1_path, index=False)

        # Save P2
        df_p2 = pd.DataFrame(X_p2, columns=local_params['feature_columns'])
        df_p2[local_params['target_column']] = y_p2
        df_p2.to_parquet(p2_path, index=False)

        # Save P11
        df_p11 = pd.DataFrame(X_p11, columns=local_params['feature_columns'])
        df_p11[local_params['target_column']] = y_p11
        df_p11.to_parquet(p11_path, index=False)

        # Save P12
        df_p12 = pd.DataFrame(X_p12, columns=local_params['feature_columns'])
        df_p12[local_params['target_column']] = y_p12
        df_p12.to_parquet(p12_path, index=False)

        # Save P21
        df_p21 = pd.DataFrame(X_p21, columns=local_params['feature_columns'])
        df_p21[local_params['target_column']] = y_p21
        df_p21.to_parquet(p21_path, index=False)

        # Save P22
        df_p22 = pd.DataFrame(X_p22, columns=local_params['feature_columns'])
        df_p22[local_params['target_column']] = y_p22
        df_p22.to_parquet(p22_path, index=False)

    else:
        # Load each subset from parquet
        df_p1 = pd.read_parquet(p1_path)
        df_p2 = pd.read_parquet(p2_path)
        df_p11 = pd.read_parquet(p11_path)
        df_p12 = pd.read_parquet(p12_path)
        df_p21 = pd.read_parquet(p21_path)
        df_p22 = pd.read_parquet(p22_path)

        # Convert them back to numpy
        X_p1 = df_p1[local_params['feature_columns']].values.astype(np.float32)
        y_p1 = df_p1[local_params['target_column']].values.astype(np.int32)
        X_p2 = df_p2[local_params['feature_columns']].values.astype(np.float32)
        y_p2 = df_p2[local_params['target_column']].values.astype(np.int32)

        X_p11 = df_p11[local_params['feature_columns']].values.astype(np.float32)
        y_p11 = df_p11[local_params['target_column']].values.astype(np.int32)
        X_p12 = df_p12[local_params['feature_columns']].values.astype(np.float32)
        y_p12 = df_p12[local_params['target_column']].values.astype(np.int32)
        X_p21 = df_p21[local_params['feature_columns']].values.astype(np.float32)
        y_p21 = df_p21[local_params['target_column']].values.astype(np.int32)
        X_p22 = df_p22[local_params['feature_columns']].values.astype(np.float32)
        y_p22 = df_p22[local_params['target_column']].values.astype(np.int32)

    # Split each subset into train & test
    X_p1_train, X_p1_test, y_p1_train, y_p1_test = train_test_split(
        X_p1, y_p1,
        test_size=local_params['test_split_ratio'],
        stratify=y_p1,
        random_state=seed
    )
    X_p2_train, X_p2_test, y_p2_train, y_p2_test = train_test_split(
        X_p2, y_p2,
        test_size=local_params['test_split_ratio'],
        stratify=y_p2,
        random_state=seed
    )

    X_p11_train, X_p11_test, y_p11_train, y_p11_test = train_test_split(
        X_p11, y_p11,
        test_size=local_params['test_split_ratio'],
        stratify=y_p11,
        random_state=seed
    )
    X_p12_train, X_p12_test, y_p12_train, y_p12_test = train_test_split(
        X_p12, y_p12,
        test_size=local_params['test_split_ratio'],
        stratify=y_p12,
        random_state=seed
    )
    X_p21_train, X_p21_test, y_p21_train, y_p21_test = train_test_split(
        X_p21, y_p21,
        test_size=local_params['test_split_ratio'],
        stratify=y_p21,
        random_state=seed
    )
    X_p22_train, X_p22_test, y_p22_train, y_p22_test = train_test_split(
        X_p22, y_p22,
        test_size=local_params['test_split_ratio'],
        stratify=y_p22,
        random_state=seed
    )

    return (X_p1_train, X_p1_test, y_p1_train, y_p1_test,
            X_p2_train, X_p2_test, y_p2_train, y_p2_test,
            X_p11_train, X_p11_test, y_p11_train, y_p11_test,
            X_p12_train, X_p12_test, y_p12_train, y_p12_test,
            X_p21_train, X_p21_test, y_p21_train, y_p21_test,
            X_p22_train, X_p22_test, y_p22_train, y_p22_test)


def run_local(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test):

    results = {
        'parameters': local_params,
        'models': {
            'M1': {'training': [], 'evaluation': {}},
            'M2': {'training': [], 'evaluation': {}}
        }
    }

    columns_to_load = local_params['feature_columns'] + [local_params['target_column']]
    df = pd.read_parquet(local_params['data_path'], columns=columns_to_load)
    df = df.dropna()

    X = df[local_params['feature_columns']].values.astype(np.float32)
    y_raw = df[local_params['target_column']].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Update runtime parameters
    local_params['input_dim'] = X.shape[1]
    local_params['num_classes'] = len(np.unique(y))

    # Create dataloaders
    train_loader_p1, test_loader_p1 = create_data_loaders(X_p1_train, X_p1_test, y_p1_train, y_p1_test)
    train_loader_p2, test_loader_p2 = create_data_loaders(X_p2_train, X_p2_test, y_p2_train, y_p2_test)

    print("\tPlayer 1")
    # Train and evaluate Model M1
    model_M1 = NeuralNetwork(local_params['input_dim'], local_params['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer_M1 = optim.Adam(model_M1.parameters(), lr=local_params['learning_rate'])

    model_M1, epochs_data_M1 = train_model(model_M1, train_loader_p1, criterion, optimizer_M1)
    
    results['models']['M1']['training'] = epochs_data_M1

    loss_M1_p1, acc_M1_p1 = evaluate_model(model_M1, test_loader_p1, criterion)
    loss_M1_p2, acc_M1_p2 = evaluate_model(model_M1, test_loader_p2, criterion)
    
    results['models']['M1']['evaluation'] = {
        'P1': {'loss': loss_M1_p1, 'accuracy': acc_M1_p1},
        'P2': {'loss': loss_M1_p2, 'accuracy': acc_M1_p2}
    }

    print("\tPlayer 2")
    # Train and evaluate Model M2
    model_M2 = NeuralNetwork(local_params['input_dim'], local_params['num_classes'])
    optimizer_M2 = optim.Adam(model_M2.parameters(), lr=local_params['learning_rate'])

    model_M2, epochs_data_M2 = train_model(model_M2, train_loader_p2, criterion, optimizer_M2)
    
    results['models']['M2']['training'] = epochs_data_M2

    loss_M2_p2, acc_M2_p2 = evaluate_model(model_M2, test_loader_p2, criterion)
    loss_M2_p1, acc_M2_p1 = evaluate_model(model_M2, test_loader_p1, criterion)
    
    results['models']['M2']['evaluation'] = {
        'P2': {'loss': loss_M2_p2, 'accuracy': acc_M2_p2},
        'P1': {'loss': loss_M2_p1, 'accuracy': acc_M2_p1}
    }

    return results


def run_fl(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test):

    results = {
        'parameters': local_params,
        'federated_training': {
            'rounds': [],
            'final_evaluation': {}
        }
    }

    # Load and preprocess data
    columns_to_load = local_params['feature_columns'] + [local_params['target_column']]
    df = pd.read_parquet(local_params['data_path'], columns=columns_to_load)
    df = df.dropna()

    X = df[local_params['feature_columns']].values.astype(np.float32)
    y_raw = df[local_params['target_column']].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Update runtime parameters
    local_params['input_dim'] = X.shape[1]
    local_params['num_classes'] = len(np.unique(y))

    # Create dataloaders
    train_loader_p1, test_loader_p1 = create_data_loaders(X_p1_train, X_p1_test, y_p1_train, y_p1_test)
    train_loader_p2, test_loader_p2 = create_data_loaders(X_p2_train, X_p2_test, y_p2_train, y_p2_test)

    # Initialize global model
    global_model = NeuralNetwork(local_params['input_dim'], local_params['num_classes'])
    criterion = nn.CrossEntropyLoss()

    # Federated Learning Rounds
    for round_num in range(1, local_params['federated_rounds'] + 1):
        print('\t\tRound ', round_num)

        # Initialize local models with global model parameters
        model_p1 = deepcopy(global_model)
        model_p2 = deepcopy(global_model)

        # Train local models
        optimizer_p1 = optim.Adam(model_p1.parameters(), lr=local_params['learning_rate'])
        optimizer_p2 = optim.Adam(model_p2.parameters(), lr=local_params['learning_rate'])

        print('\t\t\tPlayer 1')
        model_p1, epochs_data_p1 = train_local_model(model_p1, train_loader_p1, criterion, optimizer_p1, round_num)
        print('\t\t\tPlayer 2')
        model_p2, epochs_data_p2 = train_local_model(model_p2, train_loader_p2, criterion, optimizer_p2, round_num)

        # Aggregate models
        global_model = average_models(model_p1, model_p2)

        # Evaluate global model
        loss_p1, acc_p1 = evaluate_model(global_model, test_loader_p1, criterion)
        loss_p2, acc_p2 = evaluate_model(global_model, test_loader_p2, criterion)

        # Store round results
        round_results = {
            'round': round_num,
            'client_training': {
                'P1': epochs_data_p1,
                'P2': epochs_data_p2
            },
            'global_evaluation': {
                'P1': {'loss': loss_p1, 'accuracy': acc_p1},
                'P2': {'loss': loss_p2, 'accuracy': acc_p2}
            }
        }
        results['federated_training']['rounds'].append(round_results)

    # Final evaluation
    final_loss_p1, final_acc_p1 = evaluate_model(global_model, test_loader_p1, criterion)
    final_loss_p2, final_acc_p2 = evaluate_model(global_model, test_loader_p2, criterion)

    results['federated_training']['final_evaluation'] = {
        'P1': {'loss': final_loss_p1, 'accuracy': final_acc_p1},
        'P2': {'loss': final_loss_p2, 'accuracy': final_acc_p2}
    }
    return results


def run_dp_exp(X_p1_train, X_p1_test, X_p2_train, X_p2_test, y_p1_train, y_p1_test, y_p2_train, y_p2_test, p1_noise, p2_noise, experiment_id):
    """Run a single privacy experiment with specific noise level for each client"""

    # Create dataloaders
    train_loader_p1, test_loader_p1 = create_data_loaders(X_p1_train, X_p1_test, y_p1_train, y_p1_test)
    train_loader_p2, test_loader_p2 = create_data_loaders(X_p2_train, X_p2_test, y_p2_train, y_p2_test)

    # Initialize global model with full feature dimension
    input_dim = len(local_params['feature_columns'])  # Use full feature dimension for the model
    global_model = NeuralNetwork(input_dim, local_params['num_classes'])
    criterion = nn.CrossEntropyLoss()

    experiment_results = {
        'experiment_id': experiment_id,
        'p1_noise': p1_noise,
        'p2_noise': p2_noise,
        'rounds': [],
        'final_evaluation': {}
    }

    # Federated Learning Rounds
    for round_num in range(1, local_params['federated_rounds'] + 1):
        print('\t\tRound ', round_num)

        # Initialize local models
        model_p1 = deepcopy(global_model)
        model_p2 = deepcopy(global_model)

        # Train local models
        optimizer_p1 = optim.Adam(model_p1.parameters(), lr=local_params['learning_rate'])
        optimizer_p2 = optim.Adam(model_p2.parameters(), lr=local_params['learning_rate'])

        print('\t\t\tPlayer 1')
        model_p1, epochs_data_p1 = train_local_dp_model(model_p1, train_loader_p1, criterion, optimizer_p1, round_num, p1_noise)
        print('\t\t\tPlayer 2')
        model_p2, epochs_data_p2 = train_local_dp_model(model_p2, train_loader_p2, criterion, optimizer_p2, round_num, p2_noise)

        # Aggregate models
        global_model = average_models(model_p1, model_p2)

        # Evaluate global model
        loss_p1, acc_p1 = evaluate_model(global_model, test_loader_p1, criterion)
        loss_p2, acc_p2 = evaluate_model(global_model, test_loader_p2, criterion)

        # Store round results
        round_results = {
            'round': round_num,
            'client_training': {
                'P1': epochs_data_p1,
                'P2': epochs_data_p2
            },
            'global_evaluation': {
                'P1': {'loss': loss_p1, 'accuracy': acc_p1},
                'P2': {'loss': loss_p2, 'accuracy': acc_p2}
            }
        }
        experiment_results['rounds'].append(round_results)

    # Final evaluation
    final_loss_p1, final_acc_p1 = evaluate_model(global_model, test_loader_p1, criterion)
    final_loss_p2, final_acc_p2 = evaluate_model(global_model, test_loader_p2, criterion)

    experiment_results['final_evaluation'] = {
        'P1': {'loss': final_loss_p1, 'accuracy': final_acc_p1},
        'P2': {'loss': final_loss_p2, 'accuracy': final_acc_p2}
    }

    return experiment_results


def run_dp(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test):

    results = {
        'parameters': local_params,
        'experiments': []
    }

    # Load and preprocess data
    columns_to_load = local_params['feature_columns'] + [local_params['target_column']]
    df = pd.read_parquet(local_params['data_path'], columns=columns_to_load)
    df = df.dropna()

    X = df[local_params['feature_columns']].values.astype(np.float32)
    y_raw = df[local_params['target_column']].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Update runtime parameters
    local_params['num_classes'] = len(np.unique(y))

    # Run experiments for all combinations of feature sets
    current_experiment = 0

    for p1_noise, p2_noise in product(local_params['noise_levels'], local_params['noise_levels']):
        current_experiment += 1
        print("\tExp ", current_experiment, ' / ', (len(local_params['noise_levels']) * len(local_params['noise_levels'])))
        experiment_id = f"exp_{current_experiment}"

        # Run experiment
        experiment_results = run_dp_exp(
            X_p1_train, X_p1_test, X_p2_train, X_p2_test,
            y_p1_train, y_p1_test, y_p2_train, y_p2_test,
            p1_noise, p2_noise, experiment_id
        )

        results['experiments'].append(experiment_results)

    return results


def run_sup_exp(X_p1_train, X_p1_test, X_p2_train, X_p2_test, y_p1_train, y_p1_test, y_p2_train, y_p2_test, p1_features, p2_features, experiment_id):
    """Run a single privacy experiment with specific feature sets for each client"""

    # Get feature indices
    p1_feature_indices = [local_params['feature_columns'].index(f) for f in p1_features]
    p2_feature_indices = [local_params['feature_columns'].index(f) for f in p2_features]

    # Create suppressed datasets
    X_p1_train_sup = create_suppressed_dataset(X_p1_train, p1_feature_indices)
    X_p1_test_sup = create_suppressed_dataset(X_p1_test, p1_feature_indices)
    X_p2_train_sup = create_suppressed_dataset(X_p2_train, p2_feature_indices)
    X_p2_test_sup = create_suppressed_dataset(X_p2_test, p2_feature_indices)

    total_features = len(local_params['feature_columns'])

    # Create dataloaders with proper feature mapping
    train_loader_p1, test_loader_p1 = create_data_loaders_sup(
        X_p1_train_sup, X_p1_test_sup, y_p1_train, y_p1_test,
        p1_feature_indices, total_features
    )
    train_loader_p2, test_loader_p2 = create_data_loaders_sup(
        X_p2_train_sup, X_p2_test_sup, y_p2_train, y_p2_test,
        p2_feature_indices, total_features
    )

    # Initialize global model with full feature dimension
    input_dim = total_features  # Use full feature dimension for the model
    global_model = NeuralNetwork(input_dim, local_params['num_classes'])
    criterion = nn.CrossEntropyLoss()

    experiment_results = {
        'experiment_id': experiment_id,
        'p1_features': p1_features,
        'p2_features': p2_features,
        'rounds': [],
        'final_evaluation': {}
    }

    # Federated Learning Rounds
    for round_num in range(1, local_params['federated_rounds'] + 1):
        print('\t\tRound ', round_num)

        # Initialize local models
        model_p1 = deepcopy(global_model)
        model_p2 = deepcopy(global_model)

        # Train local models
        optimizer_p1 = optim.Adam(model_p1.parameters(), lr=local_params['learning_rate'])
        optimizer_p2 = optim.Adam(model_p2.parameters(), lr=local_params['learning_rate'])

        print('\t\t\tPlayer 1')
        model_p1, epochs_data_p1 = train_local_model(model_p1, train_loader_p1, criterion, optimizer_p1, round_num)
        print('\t\t\tPlayer 2')
        model_p2, epochs_data_p2 = train_local_model(model_p2, train_loader_p2, criterion, optimizer_p2, round_num)

        # Aggregate models
        global_model = average_models(model_p1, model_p2)

        # Evaluate global model
        loss_p1, acc_p1 = evaluate_model(global_model, test_loader_p1, criterion)
        loss_p2, acc_p2 = evaluate_model(global_model, test_loader_p2, criterion)

        # Store round results
        round_results = {
            'round': round_num,
            'client_training': {
                'P1': epochs_data_p1,
                'P2': epochs_data_p2
            },
            'global_evaluation': {
                'P1': {'loss': loss_p1, 'accuracy': acc_p1},
                'P2': {'loss': loss_p2, 'accuracy': acc_p2}
            }
        }
        experiment_results['rounds'].append(round_results)

    # Final evaluation
    final_loss_p1, final_acc_p1 = evaluate_model(global_model, test_loader_p1, criterion)
    final_loss_p2, final_acc_p2 = evaluate_model(global_model, test_loader_p2, criterion)

    experiment_results['final_evaluation'] = {
        'P1': {'loss': final_loss_p1, 'accuracy': final_acc_p1},
        'P2': {'loss': final_loss_p2, 'accuracy': final_acc_p2}
    }

    return experiment_results


def run_sup(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test):

    results = {
        'parameters': local_params,
        'experiments': []
    }

    # Load and preprocess data
    columns_to_load = local_params['feature_columns'] + [local_params['target_column']]
    df = pd.read_parquet(local_params['data_path'], columns=columns_to_load)
    df = df.dropna()

    X = df[local_params['feature_columns']].values.astype(np.float32)
    y_raw = df[local_params['target_column']].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Update runtime parameters
    local_params['num_classes'] = len(np.unique(y))

    # Generate feature combinations for privacy experiments
    feature_combinations = generate_feature_combinations()

    # Run experiments for all combinations of feature sets
    current_experiment = 0

    for p1_features, p2_features in product(feature_combinations, feature_combinations):
        current_experiment += 1
        print("\tExp ", current_experiment, ' / ', (len(feature_combinations) * len(feature_combinations)))
        experiment_id = f"exp_{current_experiment}"

        # Run experiment
        experiment_results = run_sup_exp(
            X_p1_train, X_p1_test, X_p2_train, X_p2_test,
            y_p1_train, y_p1_test, y_p2_train, y_p2_test,
            p1_features, p2_features, experiment_id
        )

        results['experiments'].append(experiment_results)

    return results


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the training script with a specified seed.")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for reproducibility (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    (X_p1_train, X_p1_test, y_p1_train, y_p1_test,
     X_p2_train, X_p2_test, y_p2_train, y_p2_test,
     X_p11_train, X_p11_test, y_p11_train, y_p11_test,
     X_p12_train, X_p12_test, y_p12_train, y_p12_test,
     X_p21_train, X_p21_test, y_p21_train, y_p21_test,
     X_p22_train, X_p22_test, y_p22_train, y_p22_test) = prepare_data(args.seed)
     
    #print("Baseline ...")
    #set_seed(args.seed)
    #loc_res = run_local(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    #with open('1_local_baseline/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(loc_res, f, indent=4)

    #print("Federated ...")
    #set_seed(args.seed)
    #fl_res = run_fl(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    #with open('2_fl_baseline/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(fl_res, f, indent=4)

    #print("Suppression ...")
    #set_seed(args.seed)
    #sup_res = run_sup(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    #with open('3_suppression/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(sup_res, f, indent=4)

    print("Noise ...")
    set_seed(args.seed)
    dp_res = run_dp(X_p1_train, X_p1_test, y_p1_train, y_p1_test, X_p2_train, X_p2_test, y_p2_train, y_p2_test)
    with open('4_noise/' + str(args.seed) + '.json', 'w') as f:
        json.dump(dp_res, f, indent=4)

    #print("P1 Simulated Local ...")
    #set_seed(args.seed)
    #loc_res = run_local(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    #with open('1_local_baseline/P1/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(loc_res, f, indent=4)

    #print("P1 Simulated Federated ...")
    #set_seed(args.seed)
    #fl_res = run_fl(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    #with open('2_fl_baseline/P1/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(fl_res, f, indent=4)

    #print("P1 simulated Suppression ...")
    #set_seed(args.seed)
    #sup_res = run_sup(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    #with open('3_suppression/P1/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(sup_res, f, indent=4)

    print("P1 Simulated Noise ...")
    set_seed(args.seed)
    dp_res = run_dp(X_p11_train, X_p11_test, y_p11_train, y_p11_test, X_p12_train, X_p12_test, y_p12_train, y_p12_test)
    with open('4_noise/P1/' + str(args.seed) + '.json', 'w') as f:
        json.dump(dp_res, f, indent=4)

    #print("P2 Simulated Local ...")
    #set_seed(args.seed)
    #loc_res = run_local(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    #with open('1_local_baseline/P2/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(loc_res, f, indent=4)

    #print("P2 Simulated Federated ...")
    #set_seed(args.seed)
    #fl_res = run_fl(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    #with open('2_fl_baseline/P2/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(fl_res, f, indent=4)

    #print("P2 Simulated Suppression ...")
    #set_seed(args.seed)
    #sup_res = run_sup(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    #with open('3_suppression/P2/' + str(args.seed) + '.json', 'w') as f:
    #    json.dump(sup_res, f, indent=4)

    print("P2 simulated Noise ...")
    set_seed(args.seed)
    dp_res = run_dp(X_p21_train, X_p21_test, y_p21_train, y_p21_test, X_p22_train, X_p22_test, y_p22_train, y_p22_test)
    with open('4_noise/P2/' + str(args.seed) + '.json', 'w') as f:
        json.dump(dp_res, f, indent=4)


if __name__ == "__main__":
    main()
