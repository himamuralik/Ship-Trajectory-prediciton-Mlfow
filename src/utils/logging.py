# src/utils/logging.py
"""
MLflow logging helpers for ship trajectory prediction project
"""

import mlflow
import mlflow.pytorch
import os
from typing import Dict, Any


def log_experiment(
    config: Dict[str, Any],
    metrics: Dict[str, float],
    model: Any,
    arch_name: str = "model",
    artifact_dir: str = "artifacts"
):
    """
    Log configuration, metrics, model, and artifacts to MLflow.
    
    Args:
        config: Dictionary of hyperparameters / settings
        metrics: Dictionary of computed metrics (loss, ADE, latency, etc.)
        model: The PyTorch model instance
        arch_name: Name of the architecture (lstm, bilstm, etc.)
        artifact_dir: Directory where local artifacts are saved
    """
    # Log all config parameters
    for key, value in config.items():
        mlflow.log_param(key, value)

    # Log metrics
    mlflow.log_metrics(metrics)

    # Log architecture-specific tags
    mlflow.set_tag("architecture", arch_name)
    mlflow.set_tag("framework", "PyTorch")

    # Log model
    mlflow.pytorch.log_model(
        model,
        artifact_path=f"model_{arch_name}",
        registered_model_name=f"TrajectoryModel-{arch_name}" if mlflow.active_run() else None
    )

    # Log any saved artifacts (e.g. model weights, plots, scalers)
    if os.path.exists(artifact_dir):
        for root, _, files in os.walk(artifact_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, artifact_dir)
                mlflow.log_artifact(file_path, artifact_path=os.path.join("artifacts", relative_path))

    print(f"Logged experiment for {arch_name} to MLflow run: {mlflow.active_run().info.run_id}")


def log_artifact_file(file_path: str, artifact_path: str = None):
    """
    Convenience function to log a single file as artifact
    """
    if not os.path.exists(file_path):
        print(f"Warning: Artifact file not found: {file_path}")
        return
    
    artifact_path = artifact_path or os.path.basename(file_path)
    mlflow.log_artifact(file_path, artifact_path=artifact_path)
    print(f"Logged artifact: {file_path} → {artifact_path}")


def log_plot(fig, filename: str = "plot.png"):
    """
    Save matplotlib figure and log it as artifact
    """
    import matplotlib.pyplot as plt
    
    os.makedirs("artifacts/plots", exist_ok=True)
    save_path = f"artifacts/plots/{filename}"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    
    log_artifact_file(save_path, f"plots/{filename}")
