# Module 10: Model Serving

## Serving Patterns

```
BATCH INFERENCE:
├── Scheduled predictions on large datasets
├── Results stored for later use
├── Latency: Minutes to hours
└── Use: Reports, recommendations, scoring

REAL-TIME INFERENCE:
├── On-demand predictions
├── Synchronous API calls
├── Latency: Milliseconds
└── Use: Fraud detection, pricing

STREAMING INFERENCE:
├── Continuous data streams
├── Predictions in near real-time
├── Latency: Sub-second
└── Use: IoT, real-time analytics
```

---

## TensorFlow Serving

```bash
# Pull TF Serving image
docker pull tensorflow/serving

# Save model in SavedModel format
model.save("saved_model/fraud_detector/1")

# Run serving
docker run -p 8501:8501 \
  -v "$(pwd)/saved_model:/models/fraud_detector" \
  -e MODEL_NAME=fraud_detector \
  tensorflow/serving
```

### REST API

```python
import requests
import json

# Predict
data = {"instances": [[1.0, 2.0, 3.0, 4.0]]}
response = requests.post(
    "http://localhost:8501/v1/models/fraud_detector:predict",
    json=data
)
predictions = response.json()["predictions"]
```

### gRPC (Faster)

```python
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "fraud_detector"
request.inputs["input"].CopyFrom(
    tf.make_tensor_proto([[1.0, 2.0, 3.0, 4.0]])
)

response = stub.Predict(request)
```

---

## TorchServe

```bash
# Install
pip install torchserve torch-model-archiver

# Archive model
torch-model-archiver \
  --model-name fraud_detector \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pt \
  --handler handler.py \
  --export-path model_store

# Start server
torchserve --start \
  --model-store model_store \
  --models fraud_detector=fraud_detector.mar
```

### Custom Handler

```python
# handler.py
from ts.torch_handler.base_handler import BaseHandler
import torch
import json

class FraudHandler(BaseHandler):
    def preprocess(self, data):
        inputs = []
        for row in data:
            inputs.append(json.loads(row["body"])["features"])
        return torch.tensor(inputs, dtype=torch.float32)
    
    def inference(self, data):
        with torch.no_grad():
            outputs = self.model(data)
        return outputs
    
    def postprocess(self, data):
        predictions = torch.sigmoid(data).numpy()
        return [{"prediction": float(p), "fraud": p > 0.5} for p in predictions]
```

---

## MLflow Serving

```bash
# Serve directly
mlflow models serve -m "models:/fraud_detector/Production" -p 5001

# Or build Docker
mlflow models build-docker -m "models:/fraud_detector/1" -n fraud-detector

# Run container
docker run -p 5001:8080 fraud-detector
```

```python
# Predict
import requests

response = requests.post(
    "http://localhost:5001/invocations",
    json={"instances": [[1.0, 2.0, 3.0]]}
)
```

---

## Seldon Core (Kubernetes)

```yaml
# seldon-deployment.yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: fraud-detector
spec:
  predictors:
  - name: default
    replicas: 3
    graph:
      name: classifier
      implementation: SKLEARN_SERVER
      modelUri: s3://models/fraud_detector
      children: []
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          resources:
            requests:
              memory: 1Gi
```

### A/B Testing with Seldon

```yaml
spec:
  predictors:
  - name: model-a
    replicas: 2
    traffic: 80
    graph:
      name: model-a
      modelUri: s3://models/fraud_detector_v1
  - name: model-b
    replicas: 1
    traffic: 20
    graph:
      name: model-b
      modelUri: s3://models/fraud_detector_v2
```

---

## Batch Inference

```python
# batch_inference.py
import pandas as pd
from datetime import datetime
import mlflow

def run_batch_inference(
    input_path: str,
    output_path: str,
    model_name: str = "fraud_detector"
):
    # Load data
    df = pd.read_parquet(input_path)
    
    # Load model
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    
    # Feature columns
    feature_cols = ["amount", "category_encoded", "hour", "day_of_week"]
    X = df[feature_cols]
    
    # Predict
    df["prediction"] = model.predict(X)
    df["probability"] = model.predict_proba(X)[:, 1]
    df["scored_at"] = datetime.utcnow()
    
    # Save results
    df.to_parquet(output_path)
    
    print(f"Scored {len(df)} records")
    return df

# Schedule with Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG("batch_scoring", schedule_interval="0 2 * * *") as dag:
    score_task = PythonOperator(
        task_id="score",
        python_callable=run_batch_inference,
        op_kwargs={
            "input_path": "s3://data/daily/{{ ds }}.parquet",
            "output_path": "s3://scores/{{ ds }}.parquet"
        }
    )
```

---

## Model Caching & Optimization

```python
# ONNX conversion for faster inference
import onnx
from skl2onnx import convert_sklearn

# Convert to ONNX
onnx_model = convert_sklearn(
    sklearn_model,
    initial_types=[("input", FloatTensorType([None, 10]))]
)
onnx.save(onnx_model, "model.onnx")

# Run with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: features})
```

---

## Summary

| Tool | Use Case |
|------|----------|
| TF Serving | TensorFlow models |
| TorchServe | PyTorch models |
| MLflow | Framework agnostic |
| Seldon | Kubernetes native |
| ONNX Runtime | Optimized inference |

---

👉 **[Continue to Module 11: Monitoring & Observability](../module-11-monitoring/README.md)**
