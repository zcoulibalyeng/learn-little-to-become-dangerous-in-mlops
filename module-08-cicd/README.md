# Module 8: CI/CD for Machine Learning

## ML CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML CI/CD PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CODE CHANGE                          DATA/MODEL CHANGE                 │
│      │                                       │                          │
│      ▼                                       ▼                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    CONTINUOUS INTEGRATION                          │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │ │
│  │  │  Lint   │─►│  Test   │─►│  Build  │─►│Validate │                │ │
│  │  │  Code   │  │  Unit   │  │ Docker  │  │  Model  │                │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    CONTINUOUS TRAINING                             │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │ │
│  │  │  Data   │─►│ Feature │─►│  Train  │─►│Register │                │ │
│  │  │Validate │  │   Eng   │  │  Model  │  │  Model  │                │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    CONTINUOUS DEPLOYMENT                           │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │ │
│  │  │ Staging │─►│  Test   │─►│ Canary  │─►│  Prod   │                │ │
│  │  │ Deploy  │  │ Integr. │  │ Deploy  │  │ Deploy  │                │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## GitHub Actions for ML

### Complete Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily retraining

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Lint
        run: |
          flake8 src/
          black --check src/
      
      - name: Unit tests
        run: pytest tests/unit -v
      
      - name: Data tests
        run: pytest tests/data -v

  train:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      model_version: ${{ steps.train.outputs.model_version }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Pull data
        run: dvc pull
      
      - name: Train model
        id: train
        run: |
          python train.py --config configs/prod.yaml
          echo "model_version=$(cat model_version.txt)" >> $GITHUB_OUTPUT
      
      - name: Model validation
        run: pytest tests/model -v

  deploy-staging:
    needs: train
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          python scripts/deploy.py \
            --model-version ${{ needs.train.outputs.model_version }} \
            --environment staging
      
      - name: Integration tests
        run: pytest tests/integration -v --endpoint $STAGING_ENDPOINT

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Promote to production
        run: |
          python scripts/promote_model.py \
            --model-version ${{ needs.train.outputs.model_version }} \
            --from-stage staging \
            --to-stage production
      
      - name: Deploy to production
        run: |
          python scripts/deploy.py \
            --model-version ${{ needs.train.outputs.model_version }} \
            --environment production \
            --canary-percentage 10
```

### Continuous Training Trigger

```yaml
# .github/workflows/continuous-training.yml
name: Continuous Training

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:
    inputs:
      force_retrain:
        description: 'Force retraining'
        type: boolean
        default: false

jobs:
  check-trigger:
    runs-on: ubuntu-latest
    outputs:
      should_train: ${{ steps.check.outputs.should_train }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Check if retraining needed
        id: check
        run: |
          # Check data drift, model decay, or new data
          python scripts/check_retraining_trigger.py
          echo "should_train=$(cat should_train.txt)" >> $GITHUB_OUTPUT

  train:
    needs: check-trigger
    if: needs.check-trigger.outputs.should_train == 'true' || github.event.inputs.force_retrain == 'true'
    uses: ./.github/workflows/ml-pipeline.yml
    secrets: inherit
```

---

## Model Validation Gates

```python
# scripts/validate_model.py
import mlflow
from mlflow.tracking import MlflowClient

def validate_model(model_version: str, thresholds: dict) -> bool:
    """Validate model meets production requirements."""
    client = MlflowClient()
    
    # Get model metrics
    run = client.get_run(model_version)
    metrics = run.data.metrics
    
    # Check thresholds
    validations = []
    
    # Accuracy
    if metrics.get("accuracy", 0) < thresholds["min_accuracy"]:
        print(f"❌ Accuracy {metrics['accuracy']} < {thresholds['min_accuracy']}")
        validations.append(False)
    else:
        print(f"✅ Accuracy: {metrics['accuracy']}")
        validations.append(True)
    
    # F1 Score
    if metrics.get("f1_score", 0) < thresholds["min_f1"]:
        print(f"❌ F1 {metrics['f1_score']} < {thresholds['min_f1']}")
        validations.append(False)
    else:
        print(f"✅ F1 Score: {metrics['f1_score']}")
        validations.append(True)
    
    # No regression from baseline
    baseline = get_production_model_metrics()
    if metrics.get("f1_score", 0) < baseline["f1_score"] * 0.95:
        print("❌ Model regression detected")
        validations.append(False)
    else:
        print("✅ No regression")
        validations.append(True)
    
    return all(validations)

if __name__ == "__main__":
    thresholds = {
        "min_accuracy": 0.85,
        "min_f1": 0.70,
    }
    
    is_valid = validate_model(sys.argv[1], thresholds)
    sys.exit(0 if is_valid else 1)
```

---

## Deployment Strategies

```yaml
# Canary Deployment
deploy-canary:
  steps:
    - name: Deploy canary (10%)
      run: |
        kubectl apply -f k8s/canary-deployment.yaml
        kubectl set image deployment/model-canary model=$IMAGE
    
    - name: Monitor canary
      run: |
        python scripts/monitor_canary.py --duration 30m
    
    - name: Promote or rollback
      run: |
        if [ "$CANARY_SUCCESS" == "true" ]; then
          kubectl apply -f k8s/production-deployment.yaml
        else
          kubectl rollout undo deployment/model-canary
        fi
```

```yaml
# Blue-Green Deployment
deploy-blue-green:
  steps:
    - name: Deploy green environment
      run: |
        kubectl apply -f k8s/green-deployment.yaml
    
    - name: Test green
      run: pytest tests/integration --endpoint $GREEN_ENDPOINT
    
    - name: Switch traffic
      run: |
        kubectl patch service model-service \
          -p '{"spec":{"selector":{"version":"green"}}}'
    
    - name: Cleanup blue
      run: kubectl delete deployment model-blue
```

---

## Summary

| Stage | Actions |
|-------|---------|
| CI | Lint, test, build, validate |
| CT | Data validation, training, registration |
| CD | Staging deploy, integration test, production |

Triggers:
- Code changes → CI pipeline
- Data changes → CT pipeline
- Schedule → Periodic retraining
- Drift detected → Automated retraining

---

👉 **[Continue to Module 9: Model Deployment](../module-09-deployment/README.md)**
