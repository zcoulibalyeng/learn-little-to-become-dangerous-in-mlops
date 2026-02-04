# Module 2: Data Management & Versioning

## Why Version Data?

```
PROBLEMS WITHOUT DATA VERSIONING:
├── Can't reproduce experiments
├── Can't track which data trained which model
├── Can't rollback to previous data state
├── Data changes break models silently
└── No audit trail for compliance

SOLUTION: Treat data like code
├── Version control
├── Branching and merging
├── History and lineage
└── Collaboration
```

---

## DVC (Data Version Control)

### Installation & Setup

```bash
# Install DVC
pip install dvc

# Initialize in Git repo
git init
dvc init

# Configure remote storage
dvc remote add -d myremote s3://mybucket/dvc
# Or: gs://mybucket/dvc, azure://container/path, /local/path
```

### Basic Commands

```bash
# Track data file
dvc add data/dataset.csv
# Creates: data/dataset.csv.dvc (pointer file)
# Creates: data/.gitignore (ignores actual data)

# Commit to git
git add data/dataset.csv.dvc data/.gitignore
git commit -m "Add training dataset"

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Checkout specific version
git checkout v1.0
dvc checkout
```

### DVC Pipelines

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
      - data/raw
    outs:
      - data/prepared

  featurize:
    cmd: python src/featurize.py
    deps:
      - src/featurize.py
      - data/prepared
    params:
      - featurize.max_features
    outs:
      - data/features

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features
    params:
      - train.n_estimators
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/test
    metrics:
      - eval_metrics.json:
          cache: false
```

```yaml
# params.yaml
featurize:
  max_features: 200

train:
  n_estimators: 100
  learning_rate: 0.1
```

```bash
# Run pipeline
dvc repro

# Run specific stage
dvc repro train

# Show pipeline DAG
dvc dag

# Compare experiments
dvc metrics diff

# Show parameters
dvc params diff
```

---

## Data Validation

### Great Expectations

```python
# Install
# pip install great_expectations

import great_expectations as gx

# Create context
context = gx.get_context()

# Create expectation suite
suite = context.add_expectation_suite("my_suite")

# Add expectations
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="my_suite"
)

# Column exists
validator.expect_column_to_exist("user_id")

# Not null
validator.expect_column_values_to_not_be_null("user_id")

# Value ranges
validator.expect_column_values_to_be_between(
    "age", min_value=0, max_value=120
)

# Unique values
validator.expect_column_values_to_be_unique("email")

# Categorical values
validator.expect_column_values_to_be_in_set(
    "status", ["active", "inactive", "pending"]
)

# Save suite
validator.save_expectation_suite()

# Run validation
results = context.run_checkpoint(checkpoint_name="my_checkpoint")
```

### Schema Validation

```python
# Using Pandera
import pandera as pa
from pandera import Column, Check, DataFrameSchema

schema = DataFrameSchema({
    "user_id": Column(int, Check.greater_than(0)),
    "age": Column(int, Check.in_range(0, 120)),
    "email": Column(str, Check.str_matches(r"^[\w\.-]+@[\w\.-]+\.\w+$")),
    "balance": Column(float, Check.greater_than_or_equal_to(0)),
    "status": Column(str, Check.isin(["active", "inactive"])),
    "created_at": Column(pa.DateTime),
})

# Validate
validated_df = schema.validate(df)
```

---

## Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│  │  SOURCE  │──►│  INGEST  │──►│ VALIDATE │──►│TRANSFORM │              │
│  │          │   │          │   │          │   │          │              │
│  │ • APIs   │   │ • Batch  │   │ • Schema │   │ • Clean  │              │
│  │ • DBs    │   │ • Stream │   │ • Quality│   │ • Enrich │              │
│  │ • Files  │   │ • CDC    │   │ • Checks │   │ • Agg    │              │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘              │
│                                                       │                 │
│                                                       ▼                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐            │
│  │  SERVE   │◄──│  CATALOG │◄──│  VERSION │◄──│   STORE    │            │
│  │          │   │          │   │          │   │            │            │
│  │ • APIs   │   │ • Schema │   │ • DVC    │   │ • Lake     │            │
│  │ • Batch  │   │ • Lineage│   │ • Delta  │   │ • Warehouse│            │
│  │ • Stream │   │ • Docs   │   │ • LakeFS │   │ • Feature  │            │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Storage Patterns

### Delta Lake

```python
# pip install delta-spark

from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

# Write Delta table
df.write.format("delta").save("/path/to/delta-table")

# Read
df = spark.read.format("delta").load("/path/to/delta-table")

# Time travel
df = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("/path/to/delta-table")

# Or by timestamp
df = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-01") \
    .load("/path/to/delta-table")

# ACID transactions
deltaTable = DeltaTable.forPath(spark, "/path/to/delta-table")

# Update
deltaTable.update(
    condition="id = 1",
    set={"value": "new_value"}
)

# Merge (upsert)
deltaTable.alias("target").merge(
    updates.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={"value": "source.value"}) \
 .whenNotMatchedInsert(values={"id": "source.id", "value": "source.value"}) \
 .execute()
```

---

## Data Lineage

```
DATA LINEAGE = Tracking data origin, transformations, and usage

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │────►│ Transformed │────►│   Feature   │
│  Source A   │     │   Dataset   │     │    Store    │
└─────────────┘     └─────────────┘     └───────┬─────┘
                           ▲                    │
┌─────────────┐            │                    │
│  Raw Data   │────────────┘                    │
│  Source B   │                                 │
└─────────────┘                                 ▼
                                         ┌─────────────┐
                                         │    Model    │
                                         │   Training  │
                                         └──────┬──────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │ Predictions │
                                         └─────────────┘
```

### Tools for Lineage

```python
# OpenLineage
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job

client = OpenLineageClient(url="http://localhost:5000")

# Emit lineage event
event = RunEvent(
    eventType=RunState.COMPLETE,
    run=Run(runId=str(uuid4())),
    job=Job(namespace="my-namespace", name="my-job"),
    inputs=[...],
    outputs=[...]
)
client.emit(event)
```

---

## Summary

| Tool | Purpose |
|------|---------|
| DVC | Data version control |
| Great Expectations | Data validation |
| Delta Lake | ACID data lake |
| OpenLineage | Data lineage |
| LakeFS | Git for data |

Key practices:
- ✅ Version all data artifacts
- ✅ Validate data at every step
- ✅ Track data lineage
- ✅ Use immutable data storage

---

👉 **[Continue to Module 3: Feature Engineering & Feature Stores](../module-03-feature-engineering/README.md)**
