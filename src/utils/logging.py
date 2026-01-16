import mlflow


def log_experiment(config, metrics, model, arch_name="model"):
    mlflow.log_params(config)
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, f"model_{arch_name}")
