"""
dataset.py

Manages raw data loading, splitting routines, and suppression mechanism implementation.
Suppression acts as an attribute reduction mechanism, offering absolute privacy 
to hidden subsets of the data.
"""
import random
import packages.utils.data_utils as du
import sparsechem as sc

def hide_h_percent(X_train, Y_train, h: float):
    """
    Applies Data Suppression.
    Randomly removes h% of the training samples, protecting them from exposure 
    during the collaborative training process.
    """
    if h == 0.0:
        return X_train, Y_train
    X_train_num = X_train.shape[0]
    remaining_data_num = int((1 - h) * X_train_num)
    shared_data_indices = random.sample(range(X_train_num), remaining_data_num)
    return X_train[shared_data_indices], Y_train[shared_data_indices]

def get_client_datasets(data_path: str, k: int, privacy_param: float, privacy_mode: str, conf):
    """
    Retrieves and prepares the dataset for a specific client partition.
    
    Args:
        data_path: Directory containing the split data shards.
        k: Client ID index.
        privacy_param: The value dictating privacy strength (noise scale for DP, ratio for suppression).
        privacy_mode: 'suppression' or 'dp'.
        conf: Model configuration to update with output sizes.
    """
    X_train, Y_train = du.load_ratio_split_data(data_path, k, train=True)
    
    # Suppression is applied strictly at the dataset level prior to training
    if privacy_mode == 'suppression' and privacy_param > 0.0:
        X_train, Y_train = hide_h_percent(X_train, Y_train, privacy_param)
        
    train_dataset = sc.SparseDataset(X_train, Y_train)
    
    # Dynamically update the output size for the client's personalized Head
    conf.output_size = Y_train.shape[1]

    X_test, Y_test = du.load_ratio_split_data(data_path, k, train=False)
    test_dataset = sc.SparseDataset(X_test, Y_test)

    return train_dataset, test_dataset