python
import time
import torch

def measure_latency(model, sample_input, runs=100):
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(sample_input)
            times.append(time.time() - start)

    return {
        "mean_latency_ms": 1000 * sum(times) / len(times),
        "p95_latency_ms": 1000 * sorted(times)[int(0.95 * len(times))]
    }
