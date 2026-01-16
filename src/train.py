# src/train.py
import argparse
import os
import mlflow
import mlflow.keras
import tensorflow as tf
import joblib
import numpy as np
import h5py
from kerastuner.tuners import Hyperband
from kerastuner import HyperParameters, Objective

from models import build_model   # ← your model factory

ARCHITECTURES = ["lstm", "bilstm", "gru", "bilstm_attention"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train ship trajectory prediction model(s)")
    parser.add_argument("--arch", type=str, default="all",
                        help="Architecture to train or 'all' to train every model")
    return parser.parse_args()

def load_data(split="train"):
    # This is placeholder — replace with your actual loading logic
    data_path = "./data/new_york_2015.h5"
    with h5py.File(data_path, "r") as f:
        g = f["splits"][split]
        # Example — adapt to your real structure
        x = np.array([g[sid]["features_x"][()] for sid in g.keys() if "features_x" in g[sid]])
        y = np.array([g[sid]["features_y"][()] for sid in g.keys() if "features_y" in g[sid]])
    return x, y

def main():
    args = parse_args()

    if args.arch.lower() == "all":
        to_train = ARCHITECTURES
    else:
        if args.arch not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {args.arch}. Available: {ARCHITECTURES}")
        to_train = [args.arch]

    print(f"Training architectures: {to_train}")

    # Load data once (assuming same for all models)
    print("Loading training & validation data...")
    x_train, y_train = load_data("train")
    x_val,   y_val   = load_data("valid")

    # You might want to scale here or inside the loop — depends on your logic

    for arch in to_train:
        print(f"\n{'═' * 60}\nStarting training for {arch.upper()}\n{'═' * 60}")

        run_name = f"train-{arch}"
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            mlflow.log_param("architecture", arch)
            mlflow.log_param("input_shape", str(x_train.shape[1:]))

            # Example scalers — adapt to your real preprocessing
            from sklearn.preprocessing import StandardScaler
            hist_scaler = StandardScaler().fit(x_train.reshape(-1, x_train.shape[-1]))
            targ_scaler = StandardScaler().fit(y_train.reshape(-1, y_train.shape[-1]))

            x_train_s = hist_scaler.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
            y_train_s = targ_scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
            x_val_s   = hist_scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
            y_val_s   = targ_scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)

            # Save scalers
            scaler_path = f"artifacts/{arch}_scalers.joblib"
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump({"history_scaler": hist_scaler, "target_scaler": targ_scaler}, scaler_path)
            mlflow.log_artifact(scaler_path, "scalers")

            # Hyperparameter tuning
            tuner = Hyperband(
                lambda hp: build_model(hp, arch, input_shape=x_train.shape[1:]),
                objective=Objective("val_loss", direction="min"),
                max_epochs=50,
                factor=3,
                directory=f"logs/tuning_{arch}",
                project_name="ship_trajectory",
                overwrite=True
            )

            tuner.search(
                x_train_s, y_train_s,
                epochs=60,
                validation_data=(x_val_s, y_val_s),
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
            )

            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            mlflow.log_params(best_hp.values)

            # Build and train final model
            model = tuner.hypermodel.build(best_hp)
            history = model.fit(
                x_train_s, y_train_s,
                epochs=100,
                validation_data=(x_val_s, y_val_s),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(f"artifacts/best_{arch}.h5", save_best_only=True)
                ]
            )

            # Log final metrics
            best_val_loss = min(history.history["val_loss"])
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("epochs_trained", len(history.history["loss"]))

            # Save and log model
            model.save(f"artifacts/final_model_{arch}.h5")
            mlflow.keras.log_model(model, "model")
            mlflow.log_artifact(f"artifacts/final_model_{arch}.h5")

            print(f"→ Training completed for {arch} — run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
