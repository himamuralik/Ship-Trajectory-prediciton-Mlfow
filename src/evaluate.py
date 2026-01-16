# src/evaluate.py
"""
Evaluation script for Ship Trajectory Prediction models.
Supports evaluating one architecture or all trained ones.
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from pathlib import Path

from data.feature_engineering import create_sequences  # adjust import as needed
from data.load_data import load_data                  # adjust import
from data.preprocess import preprocess_data
from models.model import get_model_class
from evaluation.metrics import compute_metrics        # if you have this
from evaluation.latency import measure_latency        # if you have this

# ────────────────────────────────────────────────
# Configuration / constants
# ────────────────────────────────────────────────

ARTIFACTS_ROOT = Path("artifacts")
MLRUNS_DIR = Path("mlruns")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCHITECTURES = ["lstm", "bilstm", "gru", "bilstm_attention"]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in km (for lat/lon columns)"""
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ship trajectory models")
    parser.add_argument("--arch", type=str, default="all",
                        help="Architecture to evaluate or 'all' (default)")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to config file")
    return parser.parse_args()


def load_test_data(config):
    """Load and prepare test data (adapt to your real loading logic)"""
    df = load_data(config["data_path"])
    df, _ = preprocess_data(df, config["feature_cols"])  # scaler not needed for eval if normalized already

    # Assuming last part is test — or use proper split
    test_df = df.iloc[int(len(df)*0.8):]  # simplistic example

    X_test, y_test = create_sequences(
        test_df.values,
        config["sequence_length"],
        config["target_cols"]
    )
    return torch.tensor(X_test).float(), torch.tensor(y_test).float()


def evaluate_single_model(model, X, y_true, batch_size=32):
    """Run inference, compute metrics + latencies"""
    model.eval()
    model.to(DEVICE)

    y_pred = []
    latencies_batch = []
    latencies_single = []

    # Batch inference (amortized)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(DEVICE)
            start = time.perf_counter()
            pred = model(batch)
            torch.cuda.synchronize() if DEVICE.type == "cuda" else None
            latencies_batch.append((time.perf_counter() - start) * 1000 / batch.shape[0])

            y_pred.append(pred.cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()

    # Single-sample latency (online / real-time)
    single_sample = X[:1].to(DEVICE)
    for _ in range(20):  # warmup
        _ = model(single_sample)
    single_times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = model(single_sample)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        single_times.append((time.perf_counter() - start) * 1000)

    # Metrics
    metrics = compute_metrics(y_true.numpy(), y_pred)  # if you have this function
    # or compute manually:
    ade = np.mean(np.sqrt(np.mean((y_true.numpy() - y_pred)**2, axis=1)))
    fde = np.mean(np.sqrt(np.mean((y_true.numpy()[:, -1] - y_pred[:, -1])**2, axis=1)))
    rmse = np.sqrt(np.mean((y_true.numpy() - y_pred)**2))
    r2 = r2_score(y_true.numpy().reshape(-1, y_true.shape[-1]), y_pred.reshape(-1, y_pred.shape[-1]))

    metrics.update({
        "ADE": float(ade),
        "FDE": float(fde),
        "RMSE": float(rmse),
        "R2": float(r2),
        "batch_latency_ms_mean": float(np.mean(latencies_batch)),
        "batch_latency_ms_std": float(np.std(latencies_batch)),
        "single_latency_ms_mean": float(np.mean(single_times)),
        "single_latency_ms_p95": float(np.percentile(single_times, 95)),
    })

    return metrics, y_pred, y_true.numpy()


def plot_diagnostics(arch_name, y_true, y_pred, save_path):
    """Create and save diagnostic plots"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Trajectory sample
    axs[0,0].plot(y_true[0, :, 0], y_true[0, :, 1], 'b-', label='True')
    axs[0,0].plot(y_pred[0, :, 0], y_pred[0, :, 1], 'r--', label='Pred')
    axs[0,0].set_title(f"{arch_name} - Sample Trajectory")
    axs[0,0].legend()

    # Error histogram
    errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=-1)).flatten()
    axs[0,1].hist(errors, bins=50, color='skyblue', edgecolor='black')
    axs[0,1].axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.2f}')
    axs[0,1].set_title("Error Distribution")
    axs[0,1].legend()

    # Per-step ADE
    per_step_err = np.sqrt(np.mean((y_true - y_pred)**2, axis=(0,2)))
    axs[1,0].plot(per_step_err, marker='o')
    axs[1,0].set_title("Mean Error per Future Step")
    axs[1,0].set_xlabel("Step")
    axs[1,0].set_ylabel("RMSE")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.arch.lower() == "all":
        to_eval = ARCHITECTURES
    else:
        if args.arch not in ARCHITECTURES:
            raise ValueError(f"Unknown arch: {args.arch}")
        to_eval = [args.arch]

    print(f"Evaluating: {to_eval}")

    X_test, y_test = load_test_data(config)

    results = {}

    with mlflow.start_run(run_name="evaluation-all", description="Multi-model evaluation"):
        table_rows = []

        for arch in to_eval:
            print(f"\nEvaluating {arch}...")

            # Find latest/best run for this arch (simplest: assume artifacts saved)
            model_path = ARTIFACTS_ROOT / f"model_{arch}.pth"
            if not model_path.exists():
                print(f"  → Model not found: {model_path}")
                continue

            ModelClass = get_model_class(arch)
            model = ModelClass(
                input_size=X_test.shape[-1],
                hidden_size=config["hidden_size"],
                output_size=y_test.shape[-1]
            )
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            metrics, y_pred, y_true = evaluate_single_model(model, X_test, y_test)

            # Plot & log
            plot_path = f"plots/diagnostic_{arch}.png"
            os.makedirs("plots", exist_ok=True)
            plot_diagnostics(arch, y_true, y_pred, plot_path)

            # Log to MLflow
            mlflow.log_metrics({f"{arch}_{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(plot_path)

            # For table
            table_rows.append({
                "Model": arch,
                "ADE": f"{metrics['ADE']:.3f}",
                "FDE": f"{metrics['FDE']:.3f}",
                "RMSE": f"{metrics['RMSE']:.3f}",
                "R²": f"{metrics['R2']:.4f}",
                "Batch ms (mean±std)": f"{metrics['batch_latency_ms_mean']:.3f} ± {metrics['batch_latency_ms_std']:.3f}",
                "Single ms (mean/p95)": f"{metrics['single_latency_ms_mean']:.2f} / {metrics['single_latency_ms_p95']:.2f}"
            })

            results[arch] = metrics

        # Print nice table
        if table_rows:
            df_table = pd.DataFrame(table_rows)
            print("\n" + "="*80)
            print("Evaluation Results Comparison")
            print(df_table.to_string(index=False))
            print("="*80 + "\n")

            # Save table as artifact
            table_md = df_table.to_markdown(index=False)
            with open("evaluation_table.md", "w") as f:
                f.write(table_md)
            mlflow.log_artifact("evaluation_table.md")


if __name__ == "__main__":
    main()
