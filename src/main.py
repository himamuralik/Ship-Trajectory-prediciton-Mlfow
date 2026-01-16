python
from evaluation.metrics import compute_regression_metrics
from evaluation.latency import measure_latency_and_throughput


python
with mlflow.start_run():

    train_model(
        model,
        dataloader,
        optimizer,
        cfg["epochs"],
        device
    )

    # Predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(
            torch.tensor(X).float().to(device)
        ).cpu().numpy()

    # Accuracy metrics
    accuracy_metrics = compute_regression_metrics(y, y_pred)

    # Latency & throughput
    perf_metrics = measure_latency_and_throughput(
        model,
        torch.tensor(X[:1]).float().to(device),
        runs=cfg.get("latency_runs", 100)
    )

    # Log EVERYTHING
    log_experiment(
        params=cfg,
        metrics={**accuracy_metrics, **perf_metrics},
        model=model
    )
