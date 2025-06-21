from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from kan.utils import create_dataset_from_data
import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

class DatasetManager:
    def __init__(self, name, device, batch_size=32):
        self.dtype = torch.get_default_dtype()
        self.dataset_name = name
        self.batch_size = batch_size
        self.device = device
        self.label_encoder = LabelEncoder()
        self.load_data()
    
    def get_classes_number(self):   
        return len(np.unique(self.y_train))

    def transform(self, X, y):
        X = torch.as_tensor(X).type(self.dtype)
        y = torch.as_tensor(np.array(y, dtype=int))
        return X, y
    
    def pad_sequences(self, sequences, dtype=np.float32):
        # sequences: list of 2D arrays or 1D arrays
        maxlen = max(seq.shape[-1] for seq in sequences)
        batch_size = len(sequences)
        # Supondo que as séries são 1D
        padded = np.zeros((batch_size, maxlen), dtype=dtype)
        for i, seq in enumerate(sequences):
            length = seq.shape[-1]
            padded[i, :length] = seq
        return padded
    

    def zscore_per_sample(self, X):

        X = np.array(X)

        if X.ndim == 2:
            return np.array([zscore(sample, axis=0) for sample in X])
        elif X.ndim == 3:
            return np.array([zscore(sample, axis=1) for sample in X])
        else:
            print('Erro no Z-score')

    def process_data(self):
        X_train, y_train = load_classification(name=self.dataset_name, split='train')
        X_test, y_test = load_classification(name=self.dataset_name, split='test')
        
        y_train = self.label_encoder.fit_transform(y_train) 
        y_test = self.label_encoder.transform(y_test)

        if isinstance(X_train, (list, np.ndarray)) and isinstance(X_train[0], np.ndarray):
            X_train = self.pad_sequences(X_train)
            X_test = self.pad_sequences(X_test)

        X_train = self.zscore_per_sample(X_train)
        X_test = self.zscore_per_sample(X_test)

        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_train, y_train = self.transform(X_train, y_train)
        X_test, y_test = self.transform(X_test, y_test)
        return X_train, y_train, X_test, y_test

    def load_data(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.process_data()
        self.dataset = create_dataset_from_data(self.X_train, self.y_train, device=self.device)

    def split_data(self, val_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=42)
        return X_train, X_val, y_train, y_val

    def load_dataloader_for_training(self):
        self.X_train, self.X_test = self.X_train.unsqueeze(1), self.X_test.unsqueeze(1)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=16)

        test_dataset = TensorDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=16)

        return train_loader, test_loader