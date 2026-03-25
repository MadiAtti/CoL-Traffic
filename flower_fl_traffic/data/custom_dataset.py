import torch
import numpy as np
from torch.utils.data import Dataset

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