# Module 14: ML Orchestration

## Orchestration Tools

```
APACHE AIRFLOW:
├── General-purpose workflow orchestration
├── Python-based DAGs
├── Rich UI and monitoring
└── Best for: Mixed workloads, existing Airflow users

KUBEFLOW PIPELINES:
├── Kubernetes-native ML pipelines
├── Component-based architecture
├── Integrates with K8s ecosystem
└── Best for: K8s environments, ML-focused teams

PREFECT:
├── Modern Python workflow
├── Dynamic workflows
├── Cloud and self-hosted
└── Best for: Python teams, dynamic pipelines

DAGSTER:
├── Data-aware orchestration
├── Software-defined assets
├── Strong typing
└── Best for: Data engineering focus
```

---

## Apache Airflow

### ML Pipeline DAG

```python
# dags/ml_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['mlops@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def extract_data(**context):
    """Extract data from source."""
    data = fetch_from_database(
        start_date=context['data_interval_start'],
        end_date=context['data_interval_end']
    )
    context['task_instance'].xcom_push(key='data_path', value=data.path)

def validate_data(**context):
    """Validate data quality."""
    data_path = context['task_instance'].xcom_pull(key='data_path')
    validation_result = run_great_expectations(data_path)
    
    if not validation_result.success:
        raise ValueError("Data validation failed")

def transform_features(**context):
    """Feature engineering."""
    data_path = context['task_instance'].xcom_pull(key='data_path')
    features = create_features(data_path)
    save_to_feature_store(features)

def train_model(**context):
    """Train ML model."""
    import mlflow
    
    with mlflow.start_run():
        features = load_from_feature_store()
        model = train(features)
        
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics(evaluate(model))
        
        run_id = mlflow.active_run().info.run_id
        context['task_instance'].xcom_push(key='run_id', value=run_id)

def validate_model(**context):
    """Validate model performance."""
    run_id = context['task_instance'].xcom_pull(key='run_id')
    metrics = get_run_metrics(run_id)
    
    if metrics['f1'] < 0.8:
        raise ValueError(f"Model F1 {metrics['f1']} below threshold")

def deploy_model(**context):
    """Deploy model to production."""
    run_id = context['task_instance'].xcom_pull(key='run_id')
    register_and_deploy(run_id)

with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training'],
) as dag:
    
    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )
    
    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )
    
    transform = PythonOperator(
        task_id='transform_features',
        python_callable=transform_features,
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
    )
    
    deploy = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
    )
    
    extract >> validate_data_task >> transform >> train >> validate_model_task >> deploy
```

---

## Kubeflow Pipelines

### Pipeline Definition

```python
# pipeline.py
from kfp import dsl
from kfp.components import create_component_from_func

@create_component_from_func
def load_data(data_path: str) -> str:
    import pandas as pd
    df = pd.read_parquet(data_path)
    output_path = "/tmp/data.parquet"
    df.to_parquet(output_path)
    return output_path

@create_component_from_func
def preprocess(data_path: str) -> str:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_parquet(data_path)
    scaler = StandardScaler()
    df[['amount', 'age']] = scaler.fit_transform(df[['amount', 'age']])
    
    output_path = "/tmp/processed.parquet"
    df.to_parquet(output_path)
    return output_path

@create_component_from_func
def train_model(data_path: str, n_estimators: int = 100) -> str:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    df = pd.read_parquet(data_path)
    X = df.drop('label', axis=1)
    y = df['label']
    
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    
    model_path = "/tmp/model.pkl"
    joblib.dump(model, model_path)
    return model_path

@create_component_from_func
def evaluate_model(model_path: str, test_data_path: str) -> float:
    import pandas as pd
    import joblib
    from sklearn.metrics import f1_score
    
    model = joblib.load(model_path)
    df = pd.read_parquet(test_data_path)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    predictions = model.predict(X)
    return f1_score(y, predictions)

@dsl.pipeline(
    name='ML Training Pipeline',
    description='End-to-end ML training'
)
def ml_pipeline(
    data_path: str = 's3://bucket/data.parquet',
    n_estimators: int = 100
):
    # Load data
    load_task = load_data(data_path=data_path)
    
    # Preprocess
    preprocess_task = preprocess(data_path=load_task.output)
    
    # Train
    train_task = train_model(
        data_path=preprocess_task.output,
        n_estimators=n_estimators
    )
    
    # Evaluate
    evaluate_task = evaluate_model(
        model_path=train_task.output,
        test_data_path=data_path
    )

# Compile pipeline
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(ml_pipeline, 'pipeline.yaml')
```

---

## Prefect

```python
# flows/training_flow.py
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(path: str):
    import pandas as pd
    return pd.read_parquet(path)

@task
def preprocess(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['amount', 'age']] = scaler.fit_transform(df[['amount', 'age']])
    return df

@task
def train(df, n_estimators: int = 100):
    from sklearn.ensemble import RandomForestClassifier
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return model

@task
def evaluate(model, df):
    from sklearn.metrics import f1_score
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    predictions = model.predict(X)
    return f1_score(y, predictions)

@task
def deploy(model, f1_score: float, threshold: float = 0.8):
    if f1_score >= threshold:
        save_model(model)
        return "deployed"
    return "rejected"

@flow(name="ML Training Pipeline")
def training_pipeline(
    data_path: str = "s3://bucket/data.parquet",
    n_estimators: int = 100
):
    # Load and preprocess
    df = load_data(data_path)
    df_processed = preprocess(df)
    
    # Train and evaluate
    model = train(df_processed, n_estimators)
    score = evaluate(model, df_processed)
    
    # Deploy if good enough
    status = deploy(model, score)
    
    return {"f1_score": score, "status": status}

# Run
if __name__ == "__main__":
    training_pipeline()
```

---

## Pipeline Comparison

| Feature | Airflow | Kubeflow | Prefect |
|---------|---------|----------|---------|
| Language | Python | Python/YAML | Python |
| Execution | Workers | Kubernetes | Agents |
| Scheduling | Cron-like | Manual/Cron | Flexible |
| UI | Rich | Good | Modern |
| Learning curve | Medium | High | Low |
| ML-specific | No | Yes | No |

---

## Summary

Choose based on:
- **Airflow**: Existing data engineering, mixed workloads
- **Kubeflow**: Kubernetes-native, ML-focused
- **Prefect**: Modern Python, dynamic workflows
- **Dagster**: Data-aware, type safety

---

👉 **[Continue to Module 15: Production Best Practices](../module-15-production/README.md)**
