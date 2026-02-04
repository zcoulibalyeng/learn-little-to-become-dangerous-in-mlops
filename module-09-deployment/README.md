# Module 9: Model Deployment

## Deployment Patterns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT PATTERNS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  EMBEDDED                   REST API                  STREAMING         │
│  ┌─────────────┐           ┌─────────────┐          ┌─────────────┐     │
│  │ Application │           │   Client    │          │   Stream    │     │
│  │ ┌─────────┐ │           │             │          │   Source    │     │
│  │ │  Model  │ │           └──────┬──────┘          └──────┬──────┘     │
│  │ └─────────┘ │                  │                        │            │
│  └─────────────┘                  ▼                        ▼            │
│                            ┌─────────────┐          ┌─────────────┐     │
│                            │ Model API   │          │   Kafka/    │     │
│                            │  Service    │          │   Flink     │     │
│                            └─────────────┘          └─────────────┘     │
│                                                                         │
│  Use: Mobile apps,         Use: Web apps,           Use: Real-time      │
│       Edge devices              Microservices            scoring        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Flask API Deployment

```python
# app.py
from flask import Flask, request, jsonify
import mlflow
import pandas as pd

app = Flask(__name__)

# Load model at startup
model = mlflow.pyfunc.load_model("models:/fraud_detector/Production")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data["features"]])
        
        prediction = model.predict(df)
        probability = model.predict_proba(df)
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.json
    df = pd.DataFrame(data["instances"])
    
    predictions = model.predict(df)
    
    return jsonify({
        "predictions": predictions.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

---

## FastAPI Deployment

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Request/Response models
class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

class BatchRequest(BaseModel):
    instances: list[list[float]]

# Load model
model = mlflow.pyfunc.load_model("models:/fraud_detector/Production")
model_version = "1.0.0"

@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": model_version}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=float(probability[0][1]),
            model_version=model_version
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(request: BatchRequest):
    features = np.array(request.instances)
    predictions = model.predict(features)
    
    return {"predictions": predictions.tolist()}
```

---

## Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .
COPY models/ ./models/

# Non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=fraud_detector
      - MODEL_STAGE=Production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detector
  template:
    metadata:
      labels:
        app: fraud-detector
    spec:
      containers:
      - name: model
        image: fraud-detector:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MODEL_NAME
          value: "fraud_detector"
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: tracking-uri
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detector-service
spec:
  selector:
    app: fraud-detector
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detector
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Serverless Deployment

### AWS Lambda

```python
# lambda_handler.py
import json
import boto3
import pickle

# Load model from S3
s3 = boto3.client('s3')
model_obj = s3.get_object(Bucket='models', Key='fraud_detector.pkl')
model = pickle.loads(model_obj['Body'].read())

def handler(event, context):
    try:
        body = json.loads(event['body'])
        features = body['features']
        
        prediction = model.predict([features])
        probability = model.predict_proba([features])
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': int(prediction[0]),
                'probability': float(probability[0][1])
            })
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Functions

```python
# main.py
import functions_framework
from google.cloud import storage
import pickle

# Load model
storage_client = storage.Client()
bucket = storage_client.bucket('models')
blob = bucket.blob('fraud_detector.pkl')
model = pickle.loads(blob.download_as_bytes())

@functions_framework.http
def predict(request):
    data = request.get_json()
    features = data['features']
    
    prediction = model.predict([features])
    
    return {'prediction': int(prediction[0])}
```

---

## Summary

| Method | Use Case | Latency | Scale |
|--------|----------|---------|-------|
| REST API | General purpose | Medium | Manual/HPA |
| Serverless | Sporadic traffic | Cold start | Auto |
| Kubernetes | High availability | Low | Auto |
| Embedded | Edge/Mobile | Lowest | N/A |

---

👉 **[Continue to Module 10: Model Serving](../module-10-serving/README.md)**
