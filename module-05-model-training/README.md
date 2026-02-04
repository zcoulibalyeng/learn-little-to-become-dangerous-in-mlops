# Module 5: Model Training Pipelines

## Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│  │   DATA   │──►│ FEATURE  │──►│  TRAIN   │──►│ EVALUATE │              │
│  │  INGEST  │   │   ENG    │   │          │   │          │              │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘              │
│       │              │              │              │                    │
│       ▼              ▼              ▼              ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     ARTIFACT STORE                               │   │
│  │  Data Version │ Features │ Model │ Metrics │ Logs                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Modular Training Code

### Project Structure

```
training/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── configs/
│   ├── config.yaml
│   └── hyperparams.yaml
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── train.py
└── requirements.txt
```

### Configuration Management

```yaml
# configs/config.yaml
data:
  train_path: "s3://bucket/data/train.parquet"
  test_path: "s3://bucket/data/test.parquet"
  target_column: "label"
  
features:
  numerical: ["age", "amount", "frequency"]
  categorical: ["category", "region"]
  
model:
  type: "xgboost"
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    
training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  
mlflow:
  tracking_uri: "http://mlflow:5000"
  experiment_name: "fraud_detection"
```

```python
# src/utils/config.py
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class Config:
    data: dict
    features: dict
    model: dict
    training: dict
    mlflow: dict

def load_config(path: str) -> Config:
    cfg = OmegaConf.load(path)
    return OmegaConf.structured(Config(**cfg))
```

### Training Script

```python
# train.py
import argparse
import mlflow
from src.data.load import load_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.config import load_config

def main(config_path: str):
    # Load config
    config = load_config(config_path)
    
    # Setup MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params(config.model.params)
        
        # Load data
        train_df, test_df = load_data(config.data)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        
        # Build features
        X_train, y_train = build_features(train_df, config.features)
        X_test, y_test = build_features(test_df, config.features)
        
        # Train model
        model = train_model(X_train, y_train, config.model)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Save model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training complete. Metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
```

---

## Hyperparameter Tuning

### Optuna

```python
import optuna
import mlflow

def objective(trial):
    # Suggest hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        model = XGBClassifier(**params)
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        mean_score = scores.mean()
        
        mlflow.log_metric("cv_f1_mean", mean_score)
        mlflow.log_metric("cv_f1_std", scores.std())
        
    return mean_score

# Run optimization
with mlflow.start_run(run_name="hyperparameter_tuning"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    # Log best params
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_f1", study.best_value)
    
    print(f"Best params: {study.best_params}")
    print(f"Best F1: {study.best_value}")
```

### Ray Tune

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
    )
    
    for epoch in range(10):
        model.fit(X_train, y_train)
        accuracy = model.score(X_val, y_val)
        tune.report(accuracy=accuracy)

# Define search space
search_space = {
    "n_estimators": tune.choice([50, 100, 200]),
    "max_depth": tune.randint(3, 15),
    "learning_rate": tune.loguniform(1e-3, 1e-1),
}

# Run tuning
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=50,
    scheduler=ASHAScheduler(),
    resources_per_trial={"cpu": 2, "gpu": 0.5}
)

print(f"Best config: {analysis.best_config}")
```

---

## Distributed Training

### PyTorch Distributed

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Create model and wrap with DDP
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            outputs = ddp_model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

---

## Docker for Training

```dockerfile
# Dockerfile.training
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY train.py .

# Set environment
ENV PYTHONPATH=/app

# Run training
ENTRYPOINT ["python", "train.py"]
CMD ["--config", "configs/config.yaml"]
```

```bash
# Build and run
docker build -f Dockerfile.training -t training:latest .

docker run -v $(pwd)/data:/app/data \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  training:latest --config configs/config.yaml
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| Config | Externalize parameters |
| Modular code | Reusable, testable components |
| Experiment tracking | Log all runs |
| Hyperparameter tuning | Find optimal params |
| Containerization | Reproducible environment |

---

👉 **[Continue to Module 6: Model Registry & Versioning](../module-06-model-registry/README.md)**
