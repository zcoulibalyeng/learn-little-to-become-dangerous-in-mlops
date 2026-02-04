# Module 6: Model Registry & Versioning

## Why Model Registry?

```
PROBLEMS WITHOUT REGISTRY:
├── "Which model is in production?"
├── "Where is the model artifact?"
├── "What version trained with what data?"
├── Can't rollback to previous version
└── No approval workflow

MODEL REGISTRY PROVIDES:
├── Central model repository
├── Version control for models
├── Stage transitions (dev→staging→prod)
├── Model lineage and metadata
└── Access control and approval
```

---

## MLflow Model Registry

### Register Model

```python
import mlflow
from mlflow.tracking import MlflowClient

# Register during training
with mlflow.start_run():
    model = train_model()
    
    # Log and register
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="fraud_detector"
    )

# Or register existing run
result = mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="fraud_detector"
)
```

### Manage Versions

```python
client = MlflowClient()

# Get model versions
versions = client.search_model_versions("name='fraud_detector'")
for v in versions:
    print(f"Version: {v.version}, Stage: {v.current_stage}")

# Get latest version
latest = client.get_latest_versions("fraud_detector", stages=["Production"])

# Transition stage
client.transition_model_version_stage(
    name="fraud_detector",
    version=3,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="fraud_detector",
    version=2,
    stage="Archived"
)
```

### Model Stages

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL LIFECYCLE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐              │
│  │  None   │───►│ Staging │───►│Production│───►│Archived │              │
│  │         │    │         │    │          │    │         │              │
│  │ Initial │    │ Testing │    │  Live    │    │ Retired │              │
│  └─────────┘    └─────────┘    └──────────┘    └─────────┘              │
│                      │              ▲                                   │
│                      │              │                                   │
│                      └───Rollback───┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Load Registered Model

```python
import mlflow

# Load by stage
model = mlflow.pyfunc.load_model("models:/fraud_detector/Production")

# Load by version
model = mlflow.pyfunc.load_model("models:/fraud_detector/3")

# Load latest
model = mlflow.pyfunc.load_model("models:/fraud_detector/latest")

# Make predictions
predictions = model.predict(data)
```

---

## Model Metadata

### Add Descriptions and Tags

```python
client = MlflowClient()

# Update model description
client.update_registered_model(
    name="fraud_detector",
    description="XGBoost model for fraud detection. Trained on transaction data."
)

# Update version description
client.update_model_version(
    name="fraud_detector",
    version=3,
    description="Improved feature engineering. F1: 0.92"
)

# Add tags
client.set_model_version_tag(
    name="fraud_detector",
    version=3,
    key="validation_status",
    value="passed"
)

client.set_model_version_tag(
    name="fraud_detector",
    version=3,
    key="data_version",
    value="v2.1"
)
```

### Query by Tags

```python
# Find models with specific tags
versions = client.search_model_versions(
    "name='fraud_detector' and tags.validation_status='passed'"
)
```

---

## Model Signature

```python
from mlflow.models.signature import infer_signature

# Infer from data
signature = infer_signature(X_train, model.predict(X_train))

# Or define manually
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
    ColSpec("double", "age"),
    ColSpec("double", "amount"),
    ColSpec("string", "category"),
])
output_schema = Schema([ColSpec("long", "prediction")])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log with signature
mlflow.sklearn.log_model(model, "model", signature=signature)
```

---

## Model Artifacts Structure

```
model/
├── MLmodel           # Model metadata
├── model.pkl         # Serialized model
├── conda.yaml        # Environment
├── requirements.txt  # Dependencies
├── python_env.yaml   # Python version
└── input_example.json # Sample input
```

### MLmodel File

```yaml
# MLmodel
artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: 3.10.0
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.3.0
model_uuid: abc123
run_id: xyz789
signature:
  inputs: '[{"name": "age", "type": "double"}, ...]'
  outputs: '[{"name": "prediction", "type": "long"}]'
```

---

## CI/CD Integration

```yaml
# .github/workflows/model-promotion.yml
name: Model Promotion

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model name'
        required: true
      version:
        description: 'Model version'
        required: true
      target_stage:
        description: 'Target stage'
        required: true
        default: 'Staging'

jobs:
  promote:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install MLflow
        run: pip install mlflow
      
      - name: Promote Model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python -c "
          from mlflow.tracking import MlflowClient
          client = MlflowClient()
          client.transition_model_version_stage(
              name='${{ github.event.inputs.model_name }}',
              version=${{ github.event.inputs.version }},
              stage='${{ github.event.inputs.target_stage }}'
          )
          "
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| Model Registry | Central model store |
| Versions | Track model iterations |
| Stages | Lifecycle management |
| Signatures | Input/output schema |
| Tags | Searchable metadata |

Best practices:
- ✅ Register all production models
- ✅ Use meaningful descriptions
- ✅ Tag with data versions
- ✅ Validate before promotion
- ✅ Automate stage transitions

---

👉 **[Continue to Module 7: Testing ML Systems](../module-07-testing/README.md)**
