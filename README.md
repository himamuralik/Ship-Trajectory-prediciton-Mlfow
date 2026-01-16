# 🚢 Ship Trajectory Prediction with MLflow

![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **A production-ready MLOps pipeline for maritime vessel trajectory forecasting.**
> 
> This repository implements a deep learning (LSTM) training pipeline integrated with **MLflow** for experiment tracking, model versioning, and rigorous **latency benchmarking** (P95/Throughput) suitable for edge deployment analysis.

---

## ⚡ Key Features

* **MLflow Integration:** Automatically logs hyperparameters, training loss, and final evaluation metrics.
* **Latency Benchmarking:** Custom evaluation module (`evaluation/latency.py`) to measure **Mean Latency**, **P95 Latency**, and **Throughput** (samples/sec).
* **Reproducibility:** Seed setting and config-driven experiments (`configs/config.yaml`) for consistent runs.
* **Modular Design:** Clean separation of data loading, preprocessing, model definition, and training loops.

---

## 🛠️ Repository Structure

```text
Ship-Trajectory-prediction-Mlflow
│
├── MLproject                   # MLflow project definition
├── conda.yaml                  # Environment dependencies
├── README.md                   # Project documentation
│
├── src/                        # Source code
│   ├── main.py                 # Entry point for training & evaluation
│   │
│   ├── data/                   # Data pipeline
│   │   ├── load_data.py        # CSV ingestion
│   │   ├── preprocess.py       # StandardScaler & cleaning
│   │   ├── feature_engineering.py # Sliding window sequence generation
│   │
│   ├── models/                 # PyTorch model definitions
│   │   ├── model.py            # LSTM Architecture
│   │   ├── train.py            # Training loop
│   │
│   ├── evaluation/             # Benchmarking modules
│   │   ├── latency.py          # Inference speed measurement
│   │   ├── metrics.py          # RMSE, MAE, R2 calculation
│   │
│   └── utils/                  # Utilities
│       ├── logging.py          # MLflow logging wrappers
│       ├── seed.py             # Reproducibility helpers
```
⚙️ Model Architecture
The core model uses a Long Short-Term Memory (LSTM) network designed for sequential time-series forecasting.

Input: Sequence of vessel states (Lat, Lon, SOG, COG, etc.)

Hidden Layers: Configurable hidden dimension LSTM layers.

Output: Linear head predicting the next step(s) in the trajectory.

Optimization: Adam Optimizer with MSE Loss.
🚀 How to Run
1. Prerequisites
Ensure you have conda installed. Create the environment:

Bash

conda env create -f conda.yaml
conda activate ship-trajectory-env
2. Run with MLflow (Recommended)
This project is set up as an MLflow Project. You can run it directly:

Bash

# Run with default config
mlflow run .

# Run with a specific config file
mlflow run . -P config_path=configs/custom_config.yaml
3. Run Manually
You can also execute the script directly using Python:

Bash

python src/main.py --config_path configs/config.yaml
## 📊 Evaluation & Metrics
The pipeline automatically logs the following metrics to the MLflow server:

Accuracy Metrics
RMSE (Root Mean Squared Error): Overall trajectory deviation.

MAE (Mean Absolute Error): Average distance error.

R² Score: Goodness of fit.

Performance Metrics (Latency)
Crucial for edge deployment (e.g., on buoys or USVs):

Mean Latency (ms): Average inference time per sample.

P95 Latency (ms): The 95th percentile latency (worst-case handling).

Throughput: Number of samples processed per second.

## 👤 Author
Hima Murali

Focus: Maritime Autonomy, MLOps, Time-Series Modeling

## 📄 License
This project is licensed under the MIT License.
