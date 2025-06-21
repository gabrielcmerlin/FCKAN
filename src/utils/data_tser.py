from aeon.datasets import load_regression
from kan.utils import create_dataset_from_data
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class DatasetManager:
    def __init__(self, name, device, batch_size=32):
        self.dtype = torch.get_default_dtype()
        self.dataset_name = name
        self.batch_size = batch_size
        self.device = device
        self.load_data()

    def transform(self, X, y):
        # Convert to tensors with appropriate types for regression
        # X can be [N, L] or [N, C, L]
        X = torch.as_tensor(X, dtype=self.dtype)
        y = torch.as_tensor(np.array(y, dtype=float), dtype=self.dtype).unsqueeze(1)  # shape [N,1]
        return X, y

    def process_data(self):
        # Load regression splits directly
        X_train, y_train = load_regression(name=self.dataset_name, split='train')
        X_test, y_test = load_regression(name=self.dataset_name, split='test')
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        X_train = (X_train - X_train.mean(axis=2, keepdims=True)) / (X_train.std(axis=2, keepdims=True) + 1e-8)
        X_test = (X_test - X_test.mean(axis=2, keepdims=True)) / (X_test.std(axis=2, keepdims=True) + 1e-8)

        # X_train/test shapes may vary: e.g. [N,1,L] or [N,C,L] or [N,L]
        # Remove redundant axis only if singleton at dim=1
        if X_train.ndim == 3 and X_train.shape[1] == 1:
            X_train = X_train[:, 0, :]
        if X_test.ndim == 3 and X_test.shape[1] == 1:
            X_test = X_test[:, 0, :]

        # Transform into torch tensors
        X_train, y_train = self.transform(X_train, y_train)
        X_test, y_test = self.transform(X_test, y_test)

        return X_train, y_train, X_test, y_test

    def load_data(self):
        # Load and keep raw tensors
        self.X_train, self.y_train, self.X_test, self.y_test = self.process_data()
        # Optionally create a dataset for other uses
        self.dataset = create_dataset_from_data(self.X_train, self.y_train, device=self.device)

    def split_data(self, val_size=0.2):
        # Split training into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=val_size, random_state=42
        )
        return X_train, X_val, y_train, y_val

    def load_dataloader_for_training(self):
        # Add channel dim for convolutional FCN expecting [N, C, L]
        X_tr = self.X_train.unsqueeze(1) if self.X_train.ndim == 2 else self.X_train
        X_te = self.X_test.unsqueeze(1) if self.X_test.ndim == 2 else self.X_test

        train_ds = TensorDataset(X_tr, self.y_train)
        test_ds = TensorDataset(X_te, self.y_test)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=16)
        return train_loader, test_loader