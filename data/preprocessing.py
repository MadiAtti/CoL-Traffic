import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from config import local_params
from data.custom_dataset import CustomDataset as custom_dataset
from torch.utils.data import DataLoader

from models import neural_network


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

def create_data_loaders(X_train, X_test, y_train, y_test):
    """Create train and test data loaders"""
    train_dataset = custom_dataset(X_train, y_train)
    test_dataset = custom_dataset(X_test, y_test)
    
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


def create_data_loaders_sup(X_train, X_test, y_train, y_test, feature_indices, total_features):
    """Create train and test data loaders with feature selection"""
    train_dataset = custom_dataset(X_train, y_train, feature_indices, total_features)
    test_dataset = custom_dataset(X_test, y_test, feature_indices, total_features)

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

def generate_feature_combinations():
    """Generate all possible feature combinations for privacy experiments"""
    num_features = len(local_params['feature_columns'])
    combinations = []
    # CHANGE HERE privacy-SUP parameters
    for num_features_to_keep in [14, 12, 10, 8, 6, 4, 2]:
        selected_features = local_params['feature_columns'][:num_features_to_keep]
        combinations.append(selected_features)

    return combinations

def average_models(model_p1, model_p2):
    global_model = neural_network(input_dim=len(local_params["feature_columns"]), num_classes=local_params['num_classes'])
    for p_global, p1, p2 in zip(global_model.parameters(), model_p1.parameters(), model_p2.parameters()):
        p_global.data = (p1.data + p2.data) / 2.0
    return global_model

