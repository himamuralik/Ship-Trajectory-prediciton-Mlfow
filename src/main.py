import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import set_seed
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.data.feature_engineering import create_sequences
from src.models.model import TrajectoryModel
from src.utils.logging import log_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="all",
                        help="lstm, bilstm, gru, bilstm_attention or 'all'")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    # Load & prepare data (once – shared across architectures)
    df = load_data(cfg["data_path"])
    df, scaler = preprocess_data(df, cfg["feature_cols"])
    X, y = create_sequences(df.values, cfg["sequence_length"], cfg["target_cols"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch_list = ["lstm", "bilstm", "gru", "bilstm_attention"] if args.arch == "all" else [args.arch]

    for arch_name in arch_list:
        print(f"\nTraining {arch_name.upper()}")

        with mlflow.start_run(run_name=f"train-{arch_name}"):
            mlflow.log_param("architecture", arch_name)
            mlflow.log_params(cfg)

            model = TrajectoryModel(
                input_size=X.shape[-1],
                hidden_size=cfg["hidden_size"],
                output_size=y.shape[-1],
                architecture=arch_name,
                num_layers=cfg.get("num_layers", 1),
                dropout=cfg.get("dropout", 0.0)
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

            # Simple training loop (expand with your real train_model function)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

            for epoch in range(cfg["epochs"]):
                model.train()
                total_loss = 0
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    loss = torch.nn.functional.mse_loss(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {total_loss/len(loader):.4f}")

            # Save & log
            torch.save(model.state_dict(), f"artifacts/model_{arch_name}.pth")
            mlflow.log_artifact(f"artifacts/model_{arch_name}.pth")
            log_experiment(cfg, {"final_loss": total_loss/len(loader)}, model, arch_name)


if __name__ == "__main__":
    main()
