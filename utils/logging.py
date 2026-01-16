python
import mlflow
import mlflow.pytorch

def log_experiment(params, metrics, model):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, "model")

