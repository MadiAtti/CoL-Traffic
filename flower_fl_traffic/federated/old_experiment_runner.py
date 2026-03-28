from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import torch.nn as nn
import torch.optim as optim
from itertools import product


from flower_fl_traffic.old_config import local_params
from flower_fl_traffic.data.old_preprocessing import average_models, create_data_loaders, create_data_loaders_sup, create_suppressed_dataset, generate_feature_combinations
from models import neural_network
from utils.evaluation import evaluate_model
from utils.training import train_local_dp_model, train_local_model, train_model


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
    model_M1 = neural_network(local_params['input_dim'], local_params['num_classes'])
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
    model_M2 = neural_network(local_params['input_dim'], local_params['num_classes'])
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
    global_model = neural_network(local_params['input_dim'], local_params['num_classes'])
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
    global_model = neural_network(input_dim, local_params['num_classes'])
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
    global_model = neural_network(input_dim, local_params['num_classes'])
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
