python
import mlflow
import mlflow.pytorch

def log_experiment(params, metrics, model):
    for k, v in params.items():
        if isinstance(v, (int, float, str)):
            mlflow.log_param(k, v)

    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))

    mlflow.pytorch.log_model(model, artifact_path="model")

