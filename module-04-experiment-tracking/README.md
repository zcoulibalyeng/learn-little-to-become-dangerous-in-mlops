# Module 4: Experiment Tracking

## Why Track Experiments?

```
WITHOUT TRACKING:
├── "Which hyperparameters gave best results?"
├── "What data version was used?"
├── "Can't reproduce last week's model"
├── Experiments scattered in notebooks
└── No comparison between runs

WITH TRACKING:
├── All experiments logged automatically
├── Compare runs side by side
├── Reproduce any experiment
├── Share results with team
└── Audit trail for compliance
```

---

## MLflow

### Installation & Setup

```bash
pip install mlflow

# Start UI server
mlflow ui --port 5000
# Open http://localhost:5000
```

### Basic Tracking

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Set experiment
mlflow.set_experiment("fraud_detection")

# Start run
with mlflow.start_run(run_name="rf_baseline"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("data_version", "v1.2")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("confusion_matrix.png")
```

### Autologging

```python
import mlflow

# Enable autolog for framework
mlflow.sklearn.autolog()
# Also: mlflow.tensorflow.autolog(), mlflow.pytorch.autolog(), etc.

# Train - everything logged automatically
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

### Comparing Experiments

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("fraud_detection")

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.f1_score > 0.8",
    order_by=["metrics.f1_score DESC"],
    max_results=10
)

# Compare runs
for run in runs:
    print(f"Run: {run.info.run_id}")
    print(f"  Params: {run.data.params}")
    print(f"  Metrics: {run.data.metrics}")
```

---

## Weights & Biases

### Setup

```bash
pip install wandb
wandb login
```

### Basic Tracking

```python
import wandb

# Initialize run
wandb.init(
    project="fraud-detection",
    name="rf-baseline",
    config={
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.01
    }
)

# Training loop with logging
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": scheduler.get_lr()
    })

# Log final metrics
wandb.summary["best_accuracy"] = best_accuracy
wandb.summary["best_f1"] = best_f1

# Log artifacts
wandb.save("model.pkl")

# Finish
wandb.finish()
```

### Hyperparameter Sweeps

```python
import wandb

# Define sweep config
sweep_config = {
    "method": "bayes",  # or grid, random
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "min": 0.0001,
            "max": 0.1,
            "distribution": "log_uniform_values"
        },
        "n_estimators": {
            "values": [50, 100, 200, 500]
        },
        "max_depth": {
            "min": 3,
            "max": 20
        }
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="fraud-detection")

# Define training function
def train():
    wandb.init()
    config = wandb.config
    
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth
    )
    model.fit(X_train, y_train)
    
    val_loss = evaluate(model, X_val, y_val)
    wandb.log({"val_loss": val_loss})

# Run sweep
wandb.agent(sweep_id, train, count=50)
```

### Visualizations

```python
# Log plots
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(history['loss'])
wandb.log({"loss_curve": wandb.Image(fig)})

# Log confusion matrix
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    y_true=y_test,
    preds=y_pred,
    class_names=["not_fraud", "fraud"]
)})

# Log PR curve
wandb.log({"pr_curve": wandb.plot.pr_curve(
    y_true=y_test,
    y_probas=y_proba,
    labels=["not_fraud", "fraud"]
)})

# Log table
table = wandb.Table(columns=["id", "prediction", "actual"])
for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
    table.add_data(i, pred, actual)
wandb.log({"predictions": table})
```

---

## Experiment Organization

```
BEST PRACTICES:

1. Naming Conventions
   ├── experiment: project_name/model_type
   ├── run: date_description_version
   └── Example: fraud_detection/rf_baseline_v2

2. Tag Experiments
   ├── stage: dev, staging, prod
   ├── team: data-science, ml-eng
   └── dataset: v1.2, augmented

3. Organize by Project
   fraud_detection/
   ├── baseline/
   ├── feature_engineering/
   ├── hyperparameter_tuning/
   └── final_model/

4. Document Everything
   ├── Hypothesis being tested
   ├── Expected outcome
   └── Conclusions
```

---

## Comparison Table

| Feature | MLflow | W&B | Neptune |
|---------|--------|-----|---------|
| Free tier | Unlimited | 100GB | 10GB |
| Self-hosted | ✅ | ❌ | ❌ |
| Model registry | ✅ | ✅ | ✅ |
| Hyperparameter sweep | Manual | Built-in | Built-in |
| Collaboration | Basic | Advanced | Advanced |
| Visualizations | Basic | Advanced | Advanced |

---

## Summary

Key practices:
- ✅ Log all hyperparameters
- ✅ Log all metrics (train, val, test)
- ✅ Version data with each run
- ✅ Log model artifacts
- ✅ Use meaningful run names
- ✅ Tag experiments for filtering

---

👉 **[Continue to Module 5: Model Training Pipelines](../module-05-model-training/README.md)**
