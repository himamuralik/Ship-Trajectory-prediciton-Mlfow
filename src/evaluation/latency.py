python
import time
import torch
import numpy as np

def measure_latency_and_throughput(
    model,
    sample_input,
    runs: int = 100
):
    model.eval()
    latencies = []

    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(sample_input)
            end = time.perf_counter()
            latencies.append(end - start)

    latencies = np.array(latencies)

    mean_latency_ms = latencies.mean() * 1000
    p95_latency_ms = np.percentile(latencies, 95) * 1000

    throughput = 1.0 / latencies.mean()  # samples/sec

    return {
        "mean_latency_ms": mean_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "throughput_samples_per_sec": throughput
    }
