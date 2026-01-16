import argparse
import os
import mlflow
import mlflow.keras
import tensorflow as tf
import numpy as np
import h5py
import joblib
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import RobustScaler

from models import build_model

ARCHITECTURES = ["lstm", "bilstm", "gru", "bilstm_attention"]


def load_train_valid(data_path="./data/new_york_2015.h5"):
    """Load training and validation data from HDF5 (adjust keys to match your preprocess.py)"""
    with h5py.File(data_path, 'r') as f:
        # Example structure – CHANGE THESE KEYS to match what your preprocess.py saved
        x_train = np.array(f.get('train/X', []))
        y_train = np.array(f.get('train/y', []))
        x_valid = np.array(f.get('valid/X', []))
        y_valid = np.array(f.get('valid/y', []))

        if len(x_train) == 0:
            raise ValueError("No training data found in HDF5. Check preprocess.py output.")
    return x_train, y_train, x_valid, y_valid


def train_model(arch):
    data_path = "./data/new_york_2015.h5"
    log_dir = f"./logs/tuning_{arch}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Loading data for {arch}...")
    x_train, y_train, x_valid, y_valid = load_train_valid(data_path)

    # Scaling
    hist_scaler = RobustScaler().fit(x_train.reshape(-1, x_train.shape[-1]))
    targ_scaler = RobustScaler().fit(y_train.reshape(-1, y_train.shape[-1]))

    x_train_s = hist_scaler.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    y_train_s = targ_scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    x_valid_s = hist_scaler.transform(x_valid.reshape(-1, x_valid.shape[-1])).reshape(x_valid.shape)
    y_valid_s = targ_scaler.transform(y_valid.reshape(-1, y_valid.shape[-1])).reshape(y_valid.shape)

    joblib.dump({"history_scaler": hist_scaler, "target_scaler": targ_scaler},
                os.path.join(log_dir, "scalers.joblib"))

    # Hyperparameter tuning
    tuner = RandomSearch(
        lambda hp: build_model(
            arch,
            input_shape=(x_train.shape[1], x_train.shape[2]),
            output_shape=(6, 2),
            hidden_units=hp.Int('units', min_value=32, max_value=512, step=32)
        ),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory=log_dir,
        project_name='ship_trajectory'
    )

    tuner.search(
        x_train_s, y_train_s,
        epochs=50,
        validation_data=(x_valid_s, y_valid_s),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8)]
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(os.path.join(log_dir, "best_model.h5"))

    # MLflow logging
    with mlflow.start_run(run_name=f"train-{arch}"):
        mlflow.log_param("architecture", arch)
        mlflow.log_artifact(os.path.join(log_dir, "best_model.h5"), "model")
        mlflow.log_artifact(os.path.join(log_dir, "scalers.joblib"), "scalers")
        mlflow.keras.log_model(best_model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="all",
                        help="Architecture or 'all' to train all")
    args = parser.parse_args()

    if args.arch == "all":
        to_train = ARCHITECTURES
    else:
        if args.arch not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {args.arch}")
        to_train = [args.arch]

    for arch in to_train:
        print(f"\n{'='*60}\nTraining {arch.upper()}\n{'='*60}")
        train_model(arch)
