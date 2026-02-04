# Module 7: Testing ML Systems

## Testing Pyramid for ML

```
                    ┌─────────────────┐
                    │   E2E Tests     │  Few, slow, expensive
                    │   (Pipeline)    │
                    ├─────────────────┤
                    │ Integration     │  
                    │ Tests           │
              ┌─────┴─────────────────┴─────┐
              │      Model Tests            │
              │  (Performance, Fairness)    │
        ┌─────┴─────────────────────────────┴─────┐
        │           Data Tests                    │
        │  (Schema, Quality, Distribution)        │
  ┌─────┴─────────────────────────────────────────┴─────┐
  │                 Unit Tests                          │
  │  (Functions, Feature Engineering, Preprocessing)    │
  └─────────────────────────────────────────────────────┘
          Many, fast, cheap
```

---

## Data Tests

### Schema Validation

```python
import pytest
import pandas as pd
from great_expectations.core import ExpectationSuite

def test_schema():
    """Test that data matches expected schema."""
    df = pd.read_parquet("data/train.parquet")
    
    # Required columns exist
    required_columns = ["user_id", "amount", "category", "timestamp", "label"]
    assert all(col in df.columns for col in required_columns)
    
    # Data types
    assert df["user_id"].dtype == "int64"
    assert df["amount"].dtype == "float64"
    assert df["category"].dtype == "object"
    assert df["label"].dtype == "int64"

def test_no_nulls_in_critical_columns():
    """Test critical columns have no null values."""
    df = pd.read_parquet("data/train.parquet")
    
    critical_columns = ["user_id", "amount", "label"]
    for col in critical_columns:
        assert df[col].isnull().sum() == 0, f"Null values in {col}"

def test_value_ranges():
    """Test values are within expected ranges."""
    df = pd.read_parquet("data/train.parquet")
    
    assert df["amount"].min() >= 0, "Negative amounts found"
    assert df["amount"].max() <= 1_000_000, "Unrealistic amount"
    assert df["label"].isin([0, 1]).all(), "Invalid label values"
```

### Data Quality

```python
def test_no_duplicates():
    """Test no duplicate transactions."""
    df = pd.read_parquet("data/train.parquet")
    assert df["transaction_id"].is_unique, "Duplicate transactions found"

def test_class_balance():
    """Test label distribution is acceptable."""
    df = pd.read_parquet("data/train.parquet")
    
    fraud_ratio = df["label"].mean()
    assert 0.001 < fraud_ratio < 0.1, f"Unusual fraud ratio: {fraud_ratio}"

def test_data_freshness():
    """Test data is recent enough."""
    df = pd.read_parquet("data/train.parquet")
    
    max_date = pd.to_datetime(df["timestamp"]).max()
    days_old = (pd.Timestamp.now() - max_date).days
    assert days_old < 7, f"Data is {days_old} days old"
```

### Distribution Tests

```python
from scipy import stats

def test_distribution_stability():
    """Test feature distributions haven't shifted."""
    train_df = pd.read_parquet("data/train.parquet")
    new_df = pd.read_parquet("data/new_data.parquet")
    
    # Kolmogorov-Smirnov test
    for col in ["amount", "frequency"]:
        statistic, p_value = stats.ks_2samp(train_df[col], new_df[col])
        assert p_value > 0.05, f"Distribution shift in {col}"
```

---

## Model Tests

### Performance Tests

```python
def test_model_accuracy():
    """Test model meets minimum accuracy."""
    model = load_model("models/model.pkl")
    X_test, y_test = load_test_data()
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    assert accuracy >= 0.85, f"Accuracy {accuracy} below threshold"

def test_model_f1_score():
    """Test model F1 score for imbalanced data."""
    model = load_model("models/model.pkl")
    X_test, y_test = load_test_data()
    
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    
    assert f1 >= 0.70, f"F1 score {f1} below threshold"

def test_no_regression():
    """Test new model doesn't regress from baseline."""
    new_model = load_model("models/new_model.pkl")
    baseline_model = load_model("models/baseline.pkl")
    X_test, y_test = load_test_data()
    
    new_f1 = f1_score(y_test, new_model.predict(X_test))
    baseline_f1 = f1_score(y_test, baseline_model.predict(X_test))
    
    assert new_f1 >= baseline_f1 * 0.95, "Model regression detected"
```

### Fairness Tests

```python
def test_demographic_parity():
    """Test model is fair across groups."""
    model = load_model("models/model.pkl")
    X_test, y_test = load_test_data()
    
    predictions = model.predict(X_test)
    
    for group in ["male", "female"]:
        group_mask = X_test["gender"] == group
        group_positive_rate = predictions[group_mask].mean()
        overall_positive_rate = predictions.mean()
        
        ratio = group_positive_rate / overall_positive_rate
        assert 0.8 < ratio < 1.2, f"Bias detected for {group}"

def test_equal_opportunity():
    """Test equal true positive rates across groups."""
    model = load_model("models/model.pkl")
    X_test, y_test = load_test_data()
    
    predictions = model.predict(X_test)
    
    tpr_by_group = {}
    for group in X_test["gender"].unique():
        mask = (X_test["gender"] == group) & (y_test == 1)
        tpr = (predictions[mask] == 1).mean()
        tpr_by_group[group] = tpr
    
    tpr_values = list(tpr_by_group.values())
    assert max(tpr_values) - min(tpr_values) < 0.1, "TPR disparity detected"
```

### Robustness Tests

```python
def test_model_on_edge_cases():
    """Test model handles edge cases."""
    model = load_model("models/model.pkl")
    
    edge_cases = pd.DataFrame({
        "amount": [0.0, 999999.99, -1.0],
        "category": ["unknown", "", None],
    })
    
    # Should not crash
    predictions = model.predict(edge_cases)
    assert len(predictions) == len(edge_cases)

def test_model_determinism():
    """Test model produces consistent predictions."""
    model = load_model("models/model.pkl")
    X_test = load_test_data()[0]
    
    pred1 = model.predict(X_test)
    pred2 = model.predict(X_test)
    
    assert (pred1 == pred2).all(), "Non-deterministic predictions"
```

---

## Infrastructure Tests

### Training Pipeline Tests

```python
def test_training_completes():
    """Test training pipeline runs to completion."""
    from src.train import train_pipeline
    
    result = train_pipeline(config="configs/test_config.yaml")
    
    assert result["status"] == "success"
    assert "model_path" in result
    assert os.path.exists(result["model_path"])

def test_model_can_be_serialized():
    """Test model can be saved and loaded."""
    model = train_model(X_train, y_train)
    
    # Save
    joblib.dump(model, "test_model.pkl")
    
    # Load
    loaded_model = joblib.load("test_model.pkl")
    
    # Predictions match
    assert (model.predict(X_test) == loaded_model.predict(X_test)).all()
```

### Serving Tests

```python
def test_prediction_latency():
    """Test prediction latency is acceptable."""
    model = load_model("models/model.pkl")
    X_single = X_test.iloc[[0]]
    
    import time
    start = time.time()
    for _ in range(100):
        model.predict(X_single)
    elapsed = (time.time() - start) / 100
    
    assert elapsed < 0.1, f"Latency {elapsed}s exceeds threshold"

def test_serving_endpoint():
    """Test model serving endpoint."""
    import requests
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": [1.0, 2.0, 3.0]}
    )
    
    assert response.status_code == 200
    assert "prediction" in response.json()
```

---

## ML Test Score

```
SCORING (from Google's paper):

+0.5 points: Manual test with documented results
+1.0 points: Automated test running regularly

CATEGORIES:
├── Data Tests (0-5 points)
├── Model Tests (0-5 points)
├── Infrastructure Tests (0-5 points)
└── Monitoring Tests (0-5 points)

INTERPRETATION:
0    : Research project, not production
0-1  : Serious reliability gaps
1-2  : Basic productionization
2-3  : Reasonably tested
3-5  : Strong automation
>5   : Exceptional
```

---

## Summary

| Test Type | What to Test |
|-----------|--------------|
| Data | Schema, quality, distribution |
| Model | Accuracy, fairness, robustness |
| Infrastructure | Training, serving, latency |
| Integration | End-to-end pipeline |

---

👉 **[Continue to Module 8: CI/CD for Machine Learning](../module-08-cicd/README.md)**
