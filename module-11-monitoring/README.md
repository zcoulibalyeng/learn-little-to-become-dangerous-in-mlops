# Module 11: Monitoring & Observability

## What to Monitor

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ML MONITORING LAYERS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    MODEL PERFORMANCE                              │  │
│  │  • Accuracy, F1, AUC  • Prediction distribution  • Model decay    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    DATA QUALITY                                   │  │
│  │  • Input distribution  • Missing values  • Schema violations      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    OPERATIONAL                                    │  │
│  │  • Latency  • Throughput  • Error rates  • Resource usage         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    INFRASTRUCTURE                                 │  │
│  │  • CPU/Memory  • GPU utilization  • Network  • Disk               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prometheus & Grafana Setup

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  grafana-data:
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'model-api'
    static_configs:
      - targets: ['model-api:8000']
```

### Instrumenting Your API

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
import time

# Define metrics
PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model_name', 'model_version']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency',
    ['model_name'],
    buckets=[.005, .01, .025, .05, .1, .25, .5, 1]
)

PREDICTION_VALUE = Histogram(
    'model_prediction_value',
    'Distribution of prediction values',
    ['model_name'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODEL_LOAD_TIME = Gauge(
    'model_load_timestamp',
    'When model was loaded',
    ['model_name', 'model_version']
)

def track_prediction(model_name: str, model_version: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            # Track metrics
            PREDICTION_COUNT.labels(
                model_name=model_name,
                model_version=model_version
            ).inc()
            
            PREDICTION_LATENCY.labels(
                model_name=model_name
            ).observe(time.time() - start_time)
            
            PREDICTION_VALUE.labels(
                model_name=model_name
            ).observe(result['probability'])
            
            return result
        return wrapper
    return decorator

# Usage
@track_prediction("fraud_detector", "1.0.0")
def predict(features):
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0][1]
    return {"prediction": int(prediction[0]), "probability": probability}
```

---

## Logging for ML

```python
import logging
import json
from datetime import datetime

# Structured logging
class MLLogger:
    def __init__(self, model_name: str, model_version: str):
        self.logger = logging.getLogger(model_name)
        self.model_name = model_name
        self.model_version = model_version
    
    def log_prediction(
        self,
        request_id: str,
        features: dict,
        prediction: int,
        probability: float,
        latency_ms: float
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "features": features,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": latency_ms
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, request_id: str, error: str, features: dict = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "model_name": self.model_name,
            "error": error,
            "features": features
        }
        self.logger.error(json.dumps(log_entry))

# Usage
ml_logger = MLLogger("fraud_detector", "1.0.0")

@app.post("/predict")
async def predict(request: PredictionRequest):
    request_id = str(uuid4())
    start_time = time.time()
    
    try:
        result = model.predict(request.features)
        
        ml_logger.log_prediction(
            request_id=request_id,
            features=request.features,
            prediction=result["prediction"],
            probability=result["probability"],
            latency_ms=(time.time() - start_time) * 1000
        )
        
        return result
    except Exception as e:
        ml_logger.log_error(request_id, str(e), request.features)
        raise
```

---

## Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: ml-alerts
    rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, model_prediction_latency_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency (p95 > 500ms)"
      
      - alert: PredictionErrorRate
        expr: rate(model_prediction_errors_total[5m]) / rate(model_predictions_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Prediction error rate > 1%"
      
      - alert: ModelNotServing
        expr: up{job="model-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model API is down"
      
      - alert: PredictionDistributionShift
        expr: abs(avg_over_time(model_prediction_value_sum[1h]) - avg_over_time(model_prediction_value_sum[24h] offset 7d)) > 0.1
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Prediction distribution shift detected"
```

---

## Grafana Dashboards

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Predictions per Second",
        "type": "graph",
        "targets": [{
          "expr": "rate(model_predictions_total[1m])"
        }]
      },
      {
        "title": "Prediction Latency (p95)",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Prediction Distribution",
        "type": "heatmap",
        "targets": [{
          "expr": "rate(model_prediction_value_bucket[5m])"
        }]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [{
          "expr": "rate(model_prediction_errors_total[5m]) / rate(model_predictions_total[5m]) * 100"
        }]
      }
    ]
  }
}
```

---

## Ground Truth Collection

```python
# Collect ground truth for model evaluation
class GroundTruthCollector:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def log_prediction(
        self,
        prediction_id: str,
        features: dict,
        prediction: int,
        probability: float
    ):
        self.db.insert("predictions", {
            "prediction_id": prediction_id,
            "features": json.dumps(features),
            "prediction": prediction,
            "probability": probability,
            "created_at": datetime.utcnow()
        })
    
    def update_ground_truth(self, prediction_id: str, actual_label: int):
        self.db.update(
            "predictions",
            {"prediction_id": prediction_id},
            {"actual_label": actual_label, "labeled_at": datetime.utcnow()}
        )
    
    def compute_metrics(self, start_date: datetime, end_date: datetime):
        predictions = self.db.query("""
            SELECT prediction, actual_label
            FROM predictions
            WHERE labeled_at BETWEEN %s AND %s
              AND actual_label IS NOT NULL
        """, [start_date, end_date])
        
        y_pred = [p["prediction"] for p in predictions]
        y_true = [p["actual_label"] for p in predictions]
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred)
        }
```

---

## Summary

| Metric Type | Examples | Tools |
|-------------|----------|-------|
| Model | Accuracy, F1, AUC | Custom, Evidently |
| Data | Distribution, nulls | Great Expectations |
| Operational | Latency, throughput | Prometheus |
| Infrastructure | CPU, memory, GPU | Prometheus, CloudWatch |

---

👉 **[Continue to Module 12: Data & Model Drift Detection](../module-12-drift-detection/README.md)**
