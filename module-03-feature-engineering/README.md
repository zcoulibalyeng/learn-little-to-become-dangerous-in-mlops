# Module 3: Feature Engineering & Feature Stores

## Feature Engineering Challenges

```
PROBLEMS:
├── Training-serving skew (different feature code)
├── Feature recomputation (duplicate effort)
├── No feature discovery (reinventing features)
├── Inconsistent features across models
└── No feature versioning

SOLUTION: Feature Store
├── Central feature repository
├── Consistent feature computation
├── Feature sharing across teams
├── Online and offline serving
└── Feature versioning and lineage
```

---

## Feature Store Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FEATURE STORE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     FEATURE REGISTRY                             │   │
│  │  • Feature definitions   • Metadata   • Lineage   • Discovery    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────┐      ┌─────────────────────────────────┐   │
│  │    OFFLINE STORE        │      │       ONLINE STORE              │   │
│  │                         │      │                                 │   │
│  │  • Historical features  │      │  • Low latency serving          │   │
│  │  • Batch processing     │ ───► │  • Real-time predictions        │   │
│  │  • Training data        │      │  • Key-value store (Redis)      │   │
│  │  • Data warehouse       │      │                                 │   │
│  └─────────────────────────┘      └─────────────────────────────────┘   │
│           │                                    │                        │
│           ▼                                    ▼                        │
│  ┌─────────────────┐                ┌─────────────────┐                 │
│  │ Model Training  │                │ Model Serving   │                 │
│  └─────────────────┘                └─────────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Feast Feature Store

### Installation & Setup

```bash
pip install feast

# Initialize project
feast init my_feature_repo
cd my_feature_repo
```

### Define Features

```python
# feature_repo/features.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define entity
customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Customer ID"
)

# Define data source
customer_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="total_purchases", dtype=Float32),
        Feature(name="avg_order_value", dtype=Float32),
        Feature(name="days_since_last_order", dtype=Int64),
        Feature(name="customer_segment", dtype=String),
    ],
    source=customer_source,
)
```

### Apply & Materialize

```bash
# Apply feature definitions
feast apply

# Materialize to online store
feast materialize 2024-01-01T00:00:00 2024-01-15T00:00:00

# Materialize incrementally
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

### Get Features

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# Get historical features (for training)
entity_df = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "event_timestamp": pd.to_datetime(["2024-01-15"] * 3)
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_order_value",
        "customer_features:days_since_last_order",
    ]
).to_df()

# Get online features (for serving)
feature_vector = store.get_online_features(
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_order_value",
    ],
    entity_rows=[{"customer_id": 1}]
).to_dict()
```

### Feature Service

```python
# Define feature service for model
from feast import FeatureService

fraud_detection_fs = FeatureService(
    name="fraud_detection",
    features=[
        customer_features[["total_purchases", "avg_order_value"]],
        transaction_features[["amount", "merchant_category"]],
    ]
)

# Use in training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=fraud_detection_fs
).to_df()

# Use in serving
features = store.get_online_features(
    features=fraud_detection_fs,
    entity_rows=[{"customer_id": 1, "transaction_id": "tx123"}]
)
```

---

## Feature Engineering Patterns

### Aggregation Features

```python
import pandas as pd

def create_aggregation_features(df, entity_col, time_col, value_col, windows):
    """Create rolling aggregation features."""
    features = pd.DataFrame()
    features[entity_col] = df[entity_col]
    
    for window in windows:
        # Group and aggregate
        agg = df.groupby(entity_col).rolling(
            window=f"{window}D",
            on=time_col
        )[value_col].agg(['sum', 'mean', 'count', 'std'])
        
        # Rename columns
        agg.columns = [f"{value_col}_{w}_{window}d" 
                      for w in ['sum', 'mean', 'count', 'std']]
        features = features.merge(agg, on=entity_col)
    
    return features

# Example: 7, 14, 30 day windows
features = create_aggregation_features(
    df=transactions,
    entity_col="customer_id",
    time_col="timestamp",
    value_col="amount",
    windows=[7, 14, 30]
)
```

### Time-Based Features

```python
def create_time_features(df, time_col):
    """Extract time-based features."""
    df = df.copy()
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['day_of_month'] = df[time_col].dt.day
    df['month'] = df[time_col].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hour'] = df['hour'].between(9, 17).astype(int)
    return df
```

### Encoding Features

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

# Target encoding
encoder = ce.TargetEncoder(cols=['category'])
df['category_encoded'] = encoder.fit_transform(df['category'], df['target'])

# Frequency encoding
freq = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(freq)
```

---

## Online vs Offline Features

```
OFFLINE FEATURES (Batch):
├── Used for: Training, batch predictions
├── Latency: Minutes to hours
├── Storage: Data warehouse (BigQuery, Snowflake)
├── Freshness: Updated periodically
└── Example: 30-day purchase history

ONLINE FEATURES (Real-time):
├── Used for: Real-time predictions
├── Latency: Milliseconds
├── Storage: Key-value store (Redis, DynamoDB)
├── Freshness: Updated continuously
└── Example: Current session behavior

STREAMING FEATURES:
├── Computed from event streams
├── Updated in near real-time
├── Example: Rolling 5-minute average
└── Tools: Kafka, Flink, Spark Streaming
```

---

## Feature Store Configuration

```yaml
# feature_store.yaml
project: my_project
registry: data/registry.pb
provider: local  # or gcp, aws

online_store:
  type: redis
  connection_string: localhost:6379

offline_store:
  type: bigquery  # or file, redshift, snowflake

entity_key_serialization_version: 2
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| Feature Store | Central feature repository |
| Offline Store | Historical features for training |
| Online Store | Low-latency features for serving |
| Feature View | Feature definition and schema |
| Feature Service | Bundle features for a model |

Best practices:
- ✅ Share features across models
- ✅ Use consistent feature computation
- ✅ Document feature semantics
- ✅ Version feature definitions

---

👉 **[Continue to Module 4: Experiment Tracking](../module-04-experiment-tracking/README.md)**
