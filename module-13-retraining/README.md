# Module 13: Continuous Training & Retraining

## Retraining Triggers

```
SCHEDULED RETRAINING:
├── Fixed schedule (daily, weekly, monthly)
├── Simple to implement
├── May retrain unnecessarily or too late
└── Use when: Data changes predictably

PERFORMANCE-BASED:
├── Trigger when metrics drop below threshold
├── Requires ground truth labels
├── Most direct signal
└── Use when: Labels available quickly

DRIFT-BASED:
├── Trigger when data drift detected
├── Proactive - doesn't wait for performance drop
├── May have false positives
└── Use when: Labels delayed or unavailable

HYBRID:
├── Combine multiple triggers
├── Retrain if ANY condition met
├── Most robust approach
└── Use when: Critical applications
```

---

## Retraining Pipeline

```python
# retraining_pipeline.py
from dataclasses import dataclass
from datetime import datetime, timedelta
import mlflow

@dataclass
class RetrainingConfig:
    model_name: str
    min_accuracy: float = 0.85
    min_f1: float = 0.70
    max_psi: float = 0.25
    min_samples_for_retrain: int = 1000
    lookback_days: int = 30

class RetrainingPipeline:
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.mlflow_client = mlflow.tracking.MlflowClient()
    
    def check_retraining_needed(self) -> dict:
        """Check if retraining is needed."""
        checks = {
            "performance_degraded": self._check_performance(),
            "data_drift_detected": self._check_drift(),
            "enough_new_data": self._check_data_volume(),
            "model_age_exceeded": self._check_model_age()
        }
        
        should_retrain = any(checks.values())
        
        return {
            "should_retrain": should_retrain,
            "reasons": [k for k, v in checks.items() if v],
            "checks": checks
        }
    
    def _check_performance(self) -> bool:
        """Check if model performance dropped."""
        recent_metrics = get_recent_metrics(days=7)
        
        accuracy = recent_metrics.get("accuracy", 1.0)
        f1 = recent_metrics.get("f1", 1.0)
        
        return accuracy < self.config.min_accuracy or f1 < self.config.min_f1
    
    def _check_drift(self) -> bool:
        """Check if data drift detected."""
        psi_values = get_feature_psi()
        max_psi = max(psi_values.values())
        
        return max_psi > self.config.max_psi
    
    def _check_data_volume(self) -> bool:
        """Check if enough new data available."""
        new_data_count = get_new_data_count(days=self.config.lookback_days)
        return new_data_count >= self.config.min_samples_for_retrain
    
    def _check_model_age(self, max_age_days: int = 30) -> bool:
        """Check if model is too old."""
        model = self.mlflow_client.get_latest_versions(
            self.config.model_name, stages=["Production"]
        )[0]
        
        model_timestamp = model.creation_timestamp / 1000
        model_age = datetime.now() - datetime.fromtimestamp(model_timestamp)
        
        return model_age.days > max_age_days
    
    def run_retraining(self):
        """Execute retraining pipeline."""
        with mlflow.start_run(run_name=f"retrain_{datetime.now().isoformat()}"):
            # 1. Load new training data
            train_data = self._load_training_data()
            mlflow.log_param("train_samples", len(train_data))
            
            # 2. Feature engineering
            X, y = self._prepare_features(train_data)
            
            # 3. Train model
            model = self._train_model(X, y)
            
            # 4. Evaluate
            metrics = self._evaluate_model(model)
            mlflow.log_metrics(metrics)
            
            # 5. Validate against current production
            if self._validate_improvement(metrics):
                # 6. Register model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=self.config.model_name
                )
                
                # 7. Promote to staging
                self._promote_to_staging()
                
                return {"status": "success", "metrics": metrics}
            else:
                return {"status": "no_improvement", "metrics": metrics}
    
    def _validate_improvement(self, new_metrics: dict) -> bool:
        """Ensure new model is better than current."""
        current_metrics = get_production_model_metrics()
        
        # New model must be at least as good (within 5%)
        return new_metrics["f1"] >= current_metrics["f1"] * 0.95
```

---

## Automated Retraining with Airflow

```python
# dags/retraining_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_retraining_needed(**context):
    pipeline = RetrainingPipeline(config)
    result = pipeline.check_retraining_needed()
    
    if result["should_retrain"]:
        return "run_retraining"
    return "skip_retraining"

def run_retraining(**context):
    pipeline = RetrainingPipeline(config)
    result = pipeline.run_retraining()
    context['task_instance'].xcom_push(key='retrain_result', value=result)

def validate_model(**context):
    result = context['task_instance'].xcom_pull(
        key='retrain_result', task_ids='run_retraining'
    )
    
    if result["status"] == "success":
        # Run validation tests
        tests_passed = run_model_tests()
        if tests_passed:
            return "deploy_staging"
    return "notify_failure"

def deploy_staging(**context):
    deploy_model_to_staging()

def deploy_production(**context):
    promote_model_to_production()

with DAG(
    'model_retraining',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    check_task = BranchPythonOperator(
        task_id='check_retraining',
        python_callable=check_retraining_needed,
    )
    
    skip_task = EmptyOperator(task_id='skip_retraining')
    
    retrain_task = PythonOperator(
        task_id='run_retraining',
        python_callable=run_retraining,
    )
    
    validate_task = BranchPythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
    )
    
    deploy_staging_task = PythonOperator(
        task_id='deploy_staging',
        python_callable=deploy_staging,
    )
    
    integration_test = PythonOperator(
        task_id='integration_test',
        python_callable=run_integration_tests,
    )
    
    deploy_prod_task = PythonOperator(
        task_id='deploy_production',
        python_callable=deploy_production,
    )
    
    notify_task = PythonOperator(
        task_id='notify_failure',
        python_callable=send_failure_notification,
    )
    
    # DAG structure
    check_task >> [skip_task, retrain_task]
    retrain_task >> validate_task
    validate_task >> [deploy_staging_task, notify_task]
    deploy_staging_task >> integration_test >> deploy_prod_task
```

---

## A/B Testing New Models

```python
class ABTestManager:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def setup_test(
        self,
        test_name: str,
        model_a: str,  # Control
        model_b: str,  # Challenger
        traffic_split: float = 0.1  # 10% to challenger
    ):
        config = {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "started_at": datetime.utcnow().isoformat(),
            "status": "running"
        }
        self.redis.hset(f"ab_test:{test_name}", mapping=config)
    
    def get_model_for_request(self, test_name: str, request_id: str) -> str:
        config = self.redis.hgetall(f"ab_test:{test_name}")
        
        # Deterministic assignment based on request_id
        bucket = hash(request_id) % 100
        
        if bucket < config["traffic_split"] * 100:
            return config["model_b"]
        return config["model_a"]
    
    def log_outcome(
        self,
        test_name: str,
        request_id: str,
        model_used: str,
        prediction: int,
        actual: int = None
    ):
        # Log for later analysis
        self.redis.lpush(f"ab_test:{test_name}:outcomes", json.dumps({
            "request_id": request_id,
            "model": model_used,
            "prediction": prediction,
            "actual": actual,
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    def analyze_test(self, test_name: str) -> dict:
        outcomes = self.redis.lrange(f"ab_test:{test_name}:outcomes", 0, -1)
        outcomes = [json.loads(o) for o in outcomes]
        
        results = {"model_a": [], "model_b": []}
        for o in outcomes:
            if o["actual"] is not None:
                results[o["model"]].append(o)
        
        return {
            "model_a": calculate_metrics(results["model_a"]),
            "model_b": calculate_metrics(results["model_b"]),
            "winner": determine_winner(results)
        }
```

---

## Shadow Deployment

```python
class ShadowDeployment:
    """Run new model in shadow mode without affecting production."""
    
    def __init__(self, production_model, shadow_model):
        self.production = production_model
        self.shadow = shadow_model
    
    async def predict(self, features):
        # Production prediction (returned to user)
        prod_start = time.time()
        prod_prediction = self.production.predict(features)
        prod_latency = time.time() - prod_start
        
        # Shadow prediction (logged, not returned)
        asyncio.create_task(self._shadow_predict(features, prod_prediction))
        
        return prod_prediction
    
    async def _shadow_predict(self, features, prod_prediction):
        try:
            shadow_start = time.time()
            shadow_prediction = self.shadow.predict(features)
            shadow_latency = time.time() - shadow_start
            
            # Log comparison
            log_shadow_comparison(
                features=features,
                production=prod_prediction,
                shadow=shadow_prediction,
                prod_latency=prod_latency,
                shadow_latency=shadow_latency
            )
        except Exception as e:
            log_shadow_error(e)
```

---

## Summary

| Strategy | Trigger | Use Case |
|----------|---------|----------|
| Scheduled | Time-based | Stable data patterns |
| Performance | Metric drop | Quick label feedback |
| Drift | Data change | Delayed labels |
| A/B Test | Before full deploy | Validate improvement |
| Shadow | Before A/B | Test in production |

---

👉 **[Continue to Module 14: ML Orchestration](../module-14-orchestration/README.md)**
