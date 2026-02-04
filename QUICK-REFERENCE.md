# 📋 MLOps Quick Reference

## Complete ML Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps LIFECYCLE                                  │
│                                                                         │
│  DATA          FEATURE        TRAINING       DEPLOYMENT    MONITORING   │
│  ────          ───────        ────────       ──────────    ──────────   │
│  │ Ingest      │ Engineer     │ Train        │ Package     │ Metrics    │
│  │ Validate    │ Store        │ Evaluate     │ Deploy      │ Drift      │
│  │ Version     │ Serve        │ Register     │ Serve       │ Retrain    │
│  ▼             ▼              ▼              ▼             ▼            │
│  DVC           Feast          MLflow         Docker/K8s    Evidently    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## DVC (Data Version Control)

```bash
# Initialize
dvc init
dvc remote add -d storage s3://bucket/dvc

# Track data
dvc add data/dataset.csv
git add data/dataset.csv.dvc .gitignore
git commit -m "Add dataset"

# Push/Pull
dvc push
dvc pull

# Pipeline
dvc repro                    # Run pipeline
dvc dag                      # Show DAG
dvc metrics show             # Show metrics
dvc params diff              # Compare params
```

```yaml
# dvc.yaml
stages:
  train:
    cmd: python train.py
    deps:
      - data/train.csv
      - src/train.py
    params:
      - train.n_estimators
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false
```

---

## MLflow

```python
import mlflow

# Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")

# Track experiment
with mlflow.start_run(run_name="experiment_1"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("plots/confusion.png")
    mlflow.sklearn.log_model(model, "model")

# Autolog
mlflow.sklearn.autolog()
model.fit(X, y)

# Load model
model = mlflow.pyfunc.load_model("runs:/abc123/model")
model = mlflow.pyfunc.load_model("models:/my_model/Production")

# Model Registry
mlflow.register_model("runs:/abc123/model", "my_model")

from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage("my_model", 1, "Production")
```

---

## Feast (Feature Store)

```python
from feast import Entity, Feature, FeatureView, FileSource, FeatureStore

# Define entity
customer = Entity(name="customer_id", value_type=ValueType.INT64)

# Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    schema=[
        Feature(name="total_purchases", dtype=Float32),
        Feature(name="avg_order_value", dtype=Float32),
    ],
    source=FileSource(path="data/customers.parquet"),
)

# Get features
store = FeatureStore(repo_path=".")

# Historical (training)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_features:total_purchases"]
).to_df()

# Online (serving)
features = store.get_online_features(
    features=["customer_features:total_purchases"],
    entity_rows=[{"customer_id": 1}]
).to_dict()
```

---

## Model Serving

### FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/my_model/Production")

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(request: PredictRequest):
    prediction = model.predict([request.features])
    return {"prediction": int(prediction[0])}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Drift Detection (Evidently)

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

# Report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html("drift_report.html")

# Test
tests = TestSuite(tests=[DataDriftTestPreset()])
tests.run(reference_data=train_df, current_data=prod_df)

if tests.as_dict()["summary"]["all_passed"]:
    print("No drift")
else:
    print("Drift detected!")
```

### Statistical Tests

```python
from scipy import stats

# KS Test (numerical)
stat, p_value = stats.ks_2samp(reference, current)
drift = p_value < 0.05

# PSI (Population Stability Index)
def calculate_psi(ref, cur, buckets=10):
    breakpoints = np.percentile(ref, np.linspace(0, 100, buckets+1))
    ref_pct = np.histogram(ref, breakpoints)[0] / len(ref)
    cur_pct = np.histogram(cur, breakpoints)[0] / len(cur)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi  # >0.25 = significant drift
```

---

## Monitoring (Prometheus)

```python
from prometheus_client import Counter, Histogram, start_http_server

PREDICTIONS = Counter('predictions_total', 'Total predictions')
LATENCY = Histogram('prediction_latency_seconds', 'Latency')

@LATENCY.time()
def predict(features):
    PREDICTIONS.inc()
    return model.predict(features)

start_http_server(8001)  # Metrics endpoint
```

---

## CI/CD (GitHub Actions)

```yaml
name: ML Pipeline
on: [push]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install deps
        run: pip install -r requirements.txt
      
      - name: Train
        run: python train.py
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      
      - name: Test
        run: pytest tests/
      
      - name: Deploy
        if: github.ref == 'refs/heads/main'
        run: python deploy.py
```

---

## Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG('ml_pipeline', schedule_interval='@daily',
         start_date=datetime(2024, 1, 1)) as dag:
    
    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_data
    )
    
    train = PythonOperator(
        task_id='train',
        python_callable=train_model
    )
    
    deploy = PythonOperator(
        task_id='deploy',
        python_callable=deploy_model
    )
    
    extract >> train >> deploy
```

---

## Key Commands

```bash
# DVC
dvc init && dvc add data/ && dvc push

# MLflow
mlflow ui --port 5000
mlflow models serve -m "models:/model/Production" -p 8000

# Feast  
feast init && feast apply && feast materialize

# Docker
docker build -t model:v1 .
docker run -p 8000:8000 model:v1

# Kubernetes
kubectl apply -f deployment.yaml
kubectl get pods -l app=model
kubectl logs -l app=model
```

---

## Retraining Triggers

```python
def should_retrain():
    return any([
        # Performance degraded
        get_metric("f1") < 0.80,
        
        # Data drift detected
        get_psi("amount") > 0.25,
        
        # Model too old
        model_age_days() > 30,
        
        # Enough new data
        new_data_count() > 10000
    ])
```

---

## Best Practices Checklist

```
DATA:
☐ Version data with DVC
☐ Validate with Great Expectations
☐ Track lineage

TRAINING:
☐ Track experiments (MLflow/W&B)
☐ Version models
☐ Use reproducible pipelines

DEPLOYMENT:
☐ Containerize (Docker)
☐ Health checks
☐ Autoscaling
☐ A/B testing capability

MONITORING:
☐ Log predictions
☐ Track drift
☐ Alert on anomalies
☐ Collect ground truth

OPERATIONS:
☐ CI/CD pipelines
☐ Automated retraining
☐ Runbooks documented
☐ Model cards
```

---

## Tools by Category

| Category | Tools |
|----------|-------|
| Data Versioning | DVC, LakeFS, Delta Lake |
| Experiment Tracking | MLflow, W&B, Neptune |
| Feature Store | Feast, Tecton, Hopsworks |
| Orchestration | Airflow, Kubeflow, Prefect |
| Serving | TF Serving, TorchServe, Seldon |
| Monitoring | Evidently, Prometheus, Grafana |
| CI/CD | GitHub Actions, GitLab CI |
