# src/main.py
import argparse
import yaml
import torch
import mlflow
import os

from utils.seed import set_seed
from data.load_data import load_data
from data.preprocess import preprocess_data
from data.feature_engineering import create_sequences
from models.model import get_model_class     # ← new factory function (see below)
from models.train import train_model
from evaluation.latency import measure_latency
from evaluation.metrics import compute_metrics
from utils.logging import log_experiment

ARCHITECTURES = ["lstm", "bilstm", "gru", "bilstm_attention"]


def parse_args():
    parser = argparse.ArgumentParser(description="Ship Trajectory Prediction Training")
    parser.add_argument("--arch", type=str, default="all",
                        help="Model architecture or 'all' to train multiple")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to config YAML file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    # Load & prepare data (shared across models)
    df = load_data(cfg["data_path"])
    df, scaler = preprocess_data(df, cfg["feature_cols"])

    X, y = create_sequences(
        df.values,
        cfg["sequence_length"],
        cfg["target_cols"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which architectures to train
    if args.arch.lower() == "all":
        to_train = ARCHITECTURES
    else:
        if args.arch not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {args.arch}")
        to_train = [args.arch]

    print(f"Training architectures: {to_train}")

    for arch_name in to_train:
        print(f"\n{'='*60}\nTraining {arch_name.upper()}\n{'='*60}")

        with mlflow.start_run(run_name=f"train-{arch_name}", nested=True):
            mlflow.log_param("architecture", arch_name)
            mlflow.log_params(cfg)

            # Get model class / instance for this architecture
            ModelClass = get_model_class(arch_name)
            model = ModelClass(
                input_size=X.shape[-1],
                hidden_size=cfg["hidden_size"],
                output_size=y.shape[-1],
                # extra params for attention/gru/etc if needed
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

            # Train
            train_model(model, cfg["dataloader"], optimizer, cfg["epochs"], device)

            # Inference example (first sample)
            sample_input = torch.tensor(X[:1]).float().to(device)
            latency = measure_latency(model, sample_input)

            # Full evaluation
            with torch.no_grad():
                preds = model(torch.tensor(X).float().to(device)).cpu().numpy()
            metrics = compute_metrics(y, preds)

            # Log everything
            log_experiment(cfg, {**metrics, **latency}, model, arch=arch_name)

            # Optional: save model artifact per architecture
            torch.save(model.state_dict(), f"artifacts/model_{arch_name}.pth")
            mlflow.log_artifact(f"artifacts/model_{arch_name}.pth")

            print(f"Completed {arch_name} — run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    main()
