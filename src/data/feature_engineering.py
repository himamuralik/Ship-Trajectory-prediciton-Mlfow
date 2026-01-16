# src/data/feature_engineering.py
import numpy as np
import torch


def create_sequences(data: np.ndarray, seq_length: int, target_cols_idx: list | None = None):
    """
    Create sliding window sequences for time-series prediction.

    Args:
        data: numpy array of shape (timesteps, features)
        seq_length: number of past timesteps (history)
        target_cols_idx: indices of columns to predict (default: all)

    Returns:
        X: (samples, seq_length, features)
        y: (samples, future_steps, target_features)
    """
    if target_cols_idx is None:
        target_cols_idx = list(range(data.shape[1]))

    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # For now assuming predicting next step(s) – adjust horizon later
        future = data[i + seq_length:i + seq_length + 1, target_cols_idx]  # 1 step ahead
        y.append(future)

    X = np.array(X)   # (n_samples, seq_length, n_features)
    y = np.array(y)   # (n_samples, future_steps, n_targets)

    print(f"Created {X.shape[0]} sequences")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
