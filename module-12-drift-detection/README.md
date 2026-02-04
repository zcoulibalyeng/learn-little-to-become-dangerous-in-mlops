# Module 12: Data & Model Drift Detection

## Types of Drift

```
DATA DRIFT (Covariate Shift):
├── Input feature distribution changes
├── Example: Customer demographics shift
├── Detection: Compare feature distributions
└── Action: Monitor, may need retraining

CONCEPT DRIFT:
├── Relationship between X and Y changes
├── Example: Fraud patterns evolve
├── Detection: Monitor model performance
└── Action: Retrain with new data

LABEL DRIFT (Prior Probability Shift):
├── Target distribution changes
├── Example: Fraud rate increases
├── Detection: Monitor prediction distribution
└── Action: Adjust thresholds or retrain

PREDICTION DRIFT:
├── Model output distribution changes
├── May indicate data or concept drift
├── Detection: Monitor prediction statistics
└── Action: Investigate root cause
```

---

## Evidently AI

### Installation

```bash
pip install evidently
```

### Data Drift Report

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Define column mapping
column_mapping = ColumnMapping(
    target='label',
    prediction='prediction',
    numerical_features=['amount', 'age', 'frequency'],
    categorical_features=['category', 'region']
)

# Create report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

# Run on reference and current data
report.run(
    reference_data=train_df,
    current_data=production_df,
    column_mapping=column_mapping
)

# Save HTML report
report.save_html("drift_report.html")

# Get as dictionary
result = report.as_dict()
print(f"Dataset drift detected: {result['metrics'][0]['result']['dataset_drift']}")
```

### Data Drift Test

```python
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataStabilityTestPreset

# Create test suite
tests = TestSuite(tests=[
    DataDriftTestPreset(),
    DataStabilityTestPreset()
])

# Run tests
tests.run(
    reference_data=train_df,
    current_data=production_df,
    column_mapping=column_mapping
)

# Check if tests passed
if tests.as_dict()["summary"]["all_passed"]:
    print("✅ No significant drift detected")
else:
    print("❌ Drift detected!")
    # Get failed tests
    for test in tests.as_dict()["tests"]:
        if test["status"] == "FAIL":
            print(f"  - {test['name']}: {test['description']}")
```

### Model Performance Monitoring

```python
from evidently.metric_preset import ClassificationPreset

# Performance report
performance_report = Report(metrics=[
    ClassificationPreset()
])

performance_report.run(
    reference_data=train_df,  # With predictions
    current_data=production_df,
    column_mapping=column_mapping
)

# Check metrics
result = performance_report.as_dict()
metrics = result['metrics'][0]['result']['current']
print(f"Accuracy: {metrics['accuracy']}")
print(f"F1: {metrics['f1']}")
```

---

## Statistical Drift Tests

### Population Stability Index (PSI)

```python
import numpy as np

def calculate_psi(reference, current, buckets=10):
    """Calculate Population Stability Index."""
    # Create buckets from reference
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Calculate proportions
    ref_counts = np.histogram(reference, breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, breakpoints)[0] / len(current)
    
    # Avoid division by zero
    ref_counts = np.clip(ref_counts, 0.001, None)
    cur_counts = np.clip(cur_counts, 0.001, None)
    
    # Calculate PSI
    psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
    
    return psi

# Interpretation
# PSI < 0.1: No significant change
# PSI 0.1-0.25: Moderate change, investigate
# PSI > 0.25: Significant change, action required

psi = calculate_psi(train_df['amount'], prod_df['amount'])
print(f"PSI for amount: {psi:.4f}")
```

### Kolmogorov-Smirnov Test

```python
from scipy import stats

def detect_drift_ks(reference, current, threshold=0.05):
    """Detect drift using KS test."""
    statistic, p_value = stats.ks_2samp(reference, current)
    
    drift_detected = p_value < threshold
    
    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": drift_detected
    }

# Check each feature
for col in numerical_features:
    result = detect_drift_ks(train_df[col], prod_df[col])
    status = "❌ DRIFT" if result["drift_detected"] else "✅ OK"
    print(f"{col}: {status} (p={result['p_value']:.4f})")
```

### Chi-Square Test (Categorical)

```python
from scipy.stats import chi2_contingency

def detect_drift_categorical(reference, current, threshold=0.05):
    """Detect drift in categorical features."""
    # Get value counts
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    
    # Align categories
    all_categories = set(ref_counts.index) | set(cur_counts.index)
    ref_aligned = [ref_counts.get(c, 0) for c in all_categories]
    cur_aligned = [cur_counts.get(c, 0) for c in all_categories]
    
    # Chi-square test
    contingency = np.array([ref_aligned, cur_aligned])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        "chi2": chi2,
        "p_value": p_value,
        "drift_detected": p_value < threshold
    }
```

---

## Continuous Drift Monitoring

```python
# drift_monitor.py
from datetime import datetime, timedelta
import schedule
import time

class DriftMonitor:
    def __init__(self, reference_data, model_name):
        self.reference = reference_data
        self.model_name = model_name
        self.alerts = []
    
    def check_drift(self):
        # Get recent production data
        current = get_recent_predictions(hours=24)
        
        # Run drift tests
        tests = TestSuite(tests=[DataDriftTestPreset()])
        tests.run(
            reference_data=self.reference,
            current_data=current
        )
        
        result = tests.as_dict()
        
        if not result["summary"]["all_passed"]:
            self.send_alert(result)
            self.log_drift_event(result)
        
        # Log metrics
        self.log_metrics(result)
    
    def send_alert(self, result):
        # Send to Slack, PagerDuty, email, etc.
        message = f"🚨 Drift detected in {self.model_name}"
        # slack.send(message)
        print(message)
    
    def log_metrics(self, result):
        # Log to Prometheus, CloudWatch, etc.
        for metric in result["metrics"]:
            log_metric(
                name=f"drift_{metric['name']}",
                value=metric['value'],
                model=self.model_name
            )

# Schedule monitoring
monitor = DriftMonitor(train_df, "fraud_detector")
schedule.every(1).hours.do(monitor.check_drift)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Drift Response Workflow

```
DRIFT DETECTED
      │
      ▼
┌─────────────────┐
│ Severity Check  │
│ (PSI, p-value)  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Minor      Major
    │         │
    ▼         ▼
 Log &      Alert
Monitor    Team
    │         │
    │    ┌────┴────┐
    │    ▼         ▼
    │  Quick     Root
    │  Fix      Cause
    │    │      Analysis
    │    │         │
    │    └────┬────┘
    │         │
    │         ▼
    │   ┌───────────┐
    │   │ Retrain   │
    │   │  Model    │
    │   └─────┬─────┘
    │         │
    │         ▼
    └────►┌───────────┐
          │  Deploy   │
          │   New     │
          │  Model    │
          └───────────┘
```

---

## Summary

| Drift Type | Detection Method | Action |
|------------|-----------------|--------|
| Data drift | PSI, KS test | Monitor/Retrain |
| Concept drift | Performance decay | Retrain |
| Label drift | Target distribution | Adjust threshold |

Tools:
- Evidently AI
- Alibi Detect
- WhyLabs
- Custom statistical tests

---

👉 **[Continue to Module 13: Continuous Training & Retraining](../module-13-retraining/README.md)**
