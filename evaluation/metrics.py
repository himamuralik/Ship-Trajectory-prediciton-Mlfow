python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }
