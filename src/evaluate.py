# src/evaluate.py
import argparse
import os
import time
import yaml
import numpy as np
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from pathlib import Path

# CHANGED: Import class directly
from src.data.feature_engineering import create_sequences
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.model import TrajectoryModel
from src.evaluation.metrics import compute_regression_metrics
from src.utils.seed import set_seed

# Configuration / constants
ARTIFACTS_ROOT = Path("artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHITECTURES = ["lstm", "bilstm", "gru", "bilstm_attention"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="all")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    return parser.parse_args()

def load_test_data(config):
    df = load_data(config["data_path"])
    df, _ = preprocess_data(df, config["feature_cols"])
    
    # Simple split for eval (last 20%)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    
    X_test, y_test = create_sequences(
        test_df.values,
        config["sequence_length"],
        config["target_cols"]
    )
    return torch.tensor(X_test).float(), torch.tensor(y_test).float()

def evaluate_model(model, X, y_true):
    model.eval()
    model.to(DEVICE)
    
    start = time.perf_counter()
    with torch.no_grad():
        y_pred = model(X.to(DEVICE)).cpu().numpy()
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000 / len(X)
    
    # Calculate metrics
    metrics = compute_regression_metrics(y_true.numpy(), y_pred)
    metrics["latency_ms_per_sample"] = latency_ms
    
    return metrics, y_pred

def main():
    args = parse_args()
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
        
    set_seed(config["seed"])
    X_test, y_test = load_test_data(config)
    
    to_eval = ARCHITECTURES if args.arch == "all" else [args.arch]
    
    print(f"Evaluating: {to_eval}")
    
    for arch in to_eval:
        model_path = ARTIFACTS_ROOT / f"model_{arch}.pth"
        if not model_path.exists():
            print(f"⚠️ Skipping {arch}: {model_path} not found.")
            continue
            
        print(f"Evaluating {arch}...")
        
        # CHANGED: Instantiate class directly
        model = TrajectoryModel(
            input_size=X_test.shape[-1],
            hidden_size=config["hidden_size"],
            output_size=y_test.shape[-1],
            architecture=arch,
            num_layers=config.get("num_layers", 1),
            dropout=0.0
        )
        
        if DEVICE.type == 'cpu':
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
            
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        print(f"  --> MSE: {metrics['mse']:.4f} | Latency: {metrics['latency_ms_per_sample']:.4f} ms")
        
        with mlflow.start_run(run_name=f"eval-{arch}", nested=True):
            mlflow.log_params({"arch": arch, "mode": "evaluation"})
            mlflow.log_metrics(metrics)

if __name__ == "__main__":
    main()
