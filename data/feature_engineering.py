python
import numpy as np

def create_sequences(data, seq_len, target_cols):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-len(target_cols)])
        y.append(data[i+seq_len, -len(target_cols):])
    return np.array(X), np.array(y)

