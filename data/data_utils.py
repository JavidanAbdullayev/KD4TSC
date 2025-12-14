""" Data loading and preprocessing utilities."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class TSDataset(Dataset):
    """Time Series Dataset for PyTorch."""
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: Time series data (n_samples, seq_length, n_features)
            y: Labels (n_samples, ) or one-hot (n_samples, n_classes)
            transform: Optional transform to apply
        """
        self.X = torch.FloatTensor(X)
        
        # Handle labels
        if len(y.shape) > 1: # ONe-hot encoded
            self.y = torch.floatTensor(y)
        else:
            self.y = torch.LongTensor(y)

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform

        return x, y
        

def znormalize(x, axis=1, epsilon=1e-8):
    """Z-normalize time series data."""
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    std[std==0] = 1.0 # Avoid division by zero

    return (x - mean) / std


def load_ucr_dataset(root_dir, dataset_name):
    """
    Load a single UCR dataset.

    Args:
        root_dir: Root directory containing UCR datasets
        dataset_name: Name of the dataset

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    dataset_path = os.path.join(root_dir, dataset_name)

    # Load train and test files
    train_file = os.path.join(dataset_path, f'{dataset_name}_TRAIN.tsv')
    test_file = os.path.join(dataset_path, f'{dataset_name}_TEST.tsv')

    df_train = pd.read_csv(train_file, sep='\t', header=None)
    df_test = pd.read_csv(test_file, sep='\t', header=None)

    # Extract labels (first column)
    y_train = df_train.iloc[:, 0].values
    y_test = df_test.iloc[:, 0].values

    # Extract features (remaining columns)
    x_train = df_train.iloc[:, 1:].values
    x_test = df_test.iloc[:, 1:]

    # Z-normalization
    x_train = znormalize(x_train, axis=1)
    x_test = znormalize(x_test, axis=1)

    return x_train, y_train, x_test, y_test


def prepare_data(x_train, y_train, x_test, y_test, one_hot=True):
    """
    Prepare data for training.

    Args:
        x_train, y_train, x_test, y_test: Raw data
        one_hot: Whether to one-hot encode labels

    Returns:
        Prepared data with proper shapes and encoding
    """

    # Ensure correct shape for univariate data
    if len(x_train.shape) == 2:
        x_train = x_train[:, :, np.newaxis]
        x_test = x_test[:, :, np.newaxis]

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.fit_transform(y_test)

    nb_classes = len(np.unique(y_train_encoded))

    # One-hot encode if requested
    if one_hot:
        y_train_onehot = np.zeros((len(y_train_encoded), nb_classes))
        y_train_onehot[np.arange(len(y_train_encoded)), y_train_encoded] = 1

        y_test_onehot = np.zeros((len(y_test_encoded), nb_classes))
        y_test_onehot[np.arange(len(y_test_encoded)), y_test_encoded] = 1

        y_train_encoded = y_train_onehot
        y_test_encoded = y_test_onehot

    return x_train, y_train_encoded, x_test, y_test_encoded, nb_classes


def create_data_loaders(x_train, y_train, x_test, y_test, batch_size=64,
                        num_workers=0, pin_memory=True):
    
    """
    Create PyTorch DataLoaders.

    Args:
        x_train, y_train, x_test, y_test: Prepared data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to use pinned memory

    Returns:
        train_loader, test_loader
    """

    train_dataset = TSDataset(x_train, y_train)
    test_dataset = TSDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


