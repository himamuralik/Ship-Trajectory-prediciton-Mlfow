# Ship Trajectory Prediction – MLflow Project

This repository implements a **ship trajectory prediction system** with a clean, reproducible **MLflow Project** setup. It supports training, evaluation, and inference while tracking experiments, parameters, metrics, and models using MLflow.

The project is designed to be **research-grade and interview-ready**, following MLflow best practices.

---

## Project Structure

```
Ship-Trajectory-Prediction-MLflow/
│
├── MLproject              # MLflow project definition
├── conda.yaml             # Reproducible environment
├── config.yaml            # Experiment configuration
├── README.md
│
├── src/
│   ├── main.py             # Training entry point
│   ├── evaluate.py         # Evaluation script
│   ├── inference/
│   │   └── app.py          # Inference / demo app
│   ├── models/             # Model architectures
│   ├── utils/              # Logging, metrics, helpers
│   └── data/               # Data loading & preprocessing
│
└── mlruns/                 # MLflow tracking directory (auto-created)
```

---

## Requirements

* Conda (Miniconda / Anaconda)
* Python 3.10
* CPU-only execution (default)

All dependencies are managed via **MLflow + conda.yaml**.

---

## Environment Setup

MLflow automatically creates the environment defined in `conda.yaml`.

```yaml
channels:
  - pytorch
  - conda-forge

dependencies:
  - python=3.10
  - pytorch
  - cpuonly
  - pip:
      - mlflow>=2.10
      - numpy
      - pandas
      - scikit-learn
```

---

## MLflow Configuration

The project explicitly sets:

* **Tracking URI**: local filesystem (`./mlruns`)
* **Experiment name**: `Ship-Trajectory-Prediction`

This ensures:

* Consistent experiment grouping
* Full reproducibility
* Clean run comparison

---

## Running the Project with MLflow

### 1. Training

Run the default training pipeline:

```bash
mlflow run .
```

Run training for a specific architecture:

```bash
mlflow run . -P arch=lstm
```

Available parameters:

* `config_path` (default: `config.yaml`)
* `arch` (default: `all`)

Each run logs:

* Hyperparameters
* Training & validation metrics
* Model artifacts
* Configuration file

---

### 2. Evaluation

```bash
mlflow run . -e evaluate
```

Evaluation runs are tracked as separate MLflow runs with their own metrics.

---

### 3. Inference / Deployment

```bash
mlflow run . -e deploy
```

This launches a **local inference application** (custom app, not MLflow model serving).

---

## Experiment Tracking

For each MLflow run, the following are logged:

* **Parameters**: architecture, hyperparameters, dataset settings
* **Metrics**: prediction error, trajectory deviation, latency (if applicable)
* **Artifacts**:

  * Trained model
  * Configuration file
  * Plots / logs (if enabled)

Runs are tagged with:

* `architecture`
* `task = trajectory_prediction`

---

## Viewing Results

Launch the MLflow UI:

```bash
mlflow ui
```

Then open:

```
http://localhost:5000
```

---

## Key Design Principles

* **Reproducibility first** – MLflow Projects + conda
* **Edge-aware modeling** – CPU-friendly architectures
* **Robustness to noisy data** – real-world trajectory conditions
* **Clear separation** – training, evaluation, inference

---

## Notes

* This project uses **local MLflow tracking** by default.
* It can be easily extended to:

  * MLflow Model Registry
  * Remote tracking servers
  * Cloud or edge deployment pipelines

---

## License

MIT License

---

## Author

Hima Murali Kattur
AI Systems & Trajectory Prediction

