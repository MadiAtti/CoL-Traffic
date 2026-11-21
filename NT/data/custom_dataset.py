"""PyTorch dataset wrapper for traffic features and integer class labels.

The dataset can optionally restore a selected feature subset to the original
input width by zero-filling features that were suppressed.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Store tabular features and labels as tensors for PyTorch loaders."""

    def __init__(self, X, y, feature_indices=None, total_features=None):
        """Create a dataset, optionally placing selected columns in full-width input.

        Args:
            X: Two-dimensional feature array.
            y: One-dimensional class-label array.
            feature_indices: Original feature positions represented by columns
                in ``X`` when ``total_features`` is supplied.
            total_features: Full model input width. If omitted, ``X`` is kept
                at its original width.
        """
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
        """Return the number of examples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return the feature vector and label at ``idx``."""
        return self.X[idx], self.y[idx]