python
import argparse
import yaml
import torch
import mlflow

from utils.seed import set_seed
from data.load_data import load_data
from data.preprocess import preprocess_data
from data.feature_engineering import create_sequences
from models.model import TrajectoryModel
from models.train import train_model
from evaluation.latency import measure_latency
from evaluation.metrics import compute_metrics
from utils.logging import log_experiment

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    df = load_data(cfg["data_path"])
    df, scaler = preprocess_data(df, cfg["feature_cols"])

    X, y = create_sequences(
        df.values,
        cfg["sequence_length"],
        cfg["target_cols"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrajectoryModel(
        input_size=X.shape[-1],
        hidden_size=cfg["hidden_size"],
        output_size=y.shape[-1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    with mlflow.start_run():
        train_model(model, cfg["dataloader"], optimizer, cfg["epochs"], device)

        latency = measure_latency(
            model,
            torch.tensor(X[:1]).float().to(device)
        )

        metrics = compute_metrics(y, model(torch.tensor(X).float().to(device)).cpu().detach().numpy())

        log_experiment(cfg, {**metrics, **latency}, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    main(args.config_path)
