# Module 15: Production Best Practices

## MLOps Maturity Assessment

```
LEVEL 0: No MLOps (Ad-hoc)
☐ Manual experiments in notebooks
☐ No version control for data/models
☐ Manual deployment
☐ No monitoring

LEVEL 1: DevOps, No MLOps
☐ CI/CD for application code
☐ Containerized deployment
☐ Basic monitoring
☐ Manual model training

LEVEL 2: Automated Training
☐ Automated training pipelines
☐ Experiment tracking
☐ Model registry
☐ Data versioning

LEVEL 3: Automated Deployment
☐ CI/CD for ML pipelines
☐ Automated model validation
☐ A/B testing capability
☐ Drift monitoring

LEVEL 4: Full MLOps
☐ Automatic retraining
☐ Feature store
☐ Complete lineage
☐ Self-healing systems
```

---

## Security Best Practices

### Model Security

```python
# Secure model serving
from cryptography.fernet import Fernet

class SecureModelServer:
    def __init__(self, model_path: str, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)
        self.model = self._load_encrypted_model(model_path)
    
    def _load_encrypted_model(self, path: str):
        with open(path, 'rb') as f:
            encrypted_model = f.read()
        decrypted = self.cipher.decrypt(encrypted_model)
        return pickle.loads(decrypted)
    
    def predict(self, features: list, api_key: str) -> dict:
        # Validate API key
        if not self._validate_api_key(api_key):
            raise AuthenticationError("Invalid API key")
        
        # Input validation
        validated_features = self._validate_input(features)
        
        # Rate limiting
        if not self._check_rate_limit(api_key):
            raise RateLimitError("Rate limit exceeded")
        
        # Audit log
        self._log_prediction(api_key, validated_features)
        
        return self.model.predict(validated_features)
```

### Data Security

```yaml
# Security checklist
data_security:
  encryption:
    - Encrypt data at rest (S3 SSE, GCS encryption)
    - Encrypt data in transit (TLS/HTTPS)
    - Encrypt model artifacts
  
  access_control:
    - Role-based access (RBAC)
    - Principle of least privilege
    - Separate dev/prod environments
    - Audit all access
  
  pii_handling:
    - Identify PII in training data
    - Anonymize or pseudonymize
    - Implement data retention policies
    - GDPR/CCPA compliance
  
  secrets_management:
    - Use secrets manager (Vault, AWS Secrets Manager)
    - Never hardcode credentials
    - Rotate secrets regularly
```

---

## Cost Optimization

### Training Costs

```python
# Spot instances for training
import boto3

def launch_spot_training():
    ec2 = boto3.client('ec2')
    
    response = ec2.request_spot_instances(
        SpotPrice='0.50',
        InstanceCount=1,
        LaunchSpecification={
            'ImageId': 'ami-xxxxx',
            'InstanceType': 'p3.2xlarge',
            'KeyName': 'my-key',
        }
    )

# Auto-shutdown idle resources
def check_and_shutdown_idle():
    """Shutdown training instances after idle period."""
    for instance in get_training_instances():
        if instance.idle_time > timedelta(hours=1):
            instance.terminate()
```

### Serving Costs

```yaml
# Right-sizing recommendations
serving_optimization:
  autoscaling:
    min_replicas: 1
    max_replicas: 10
    target_cpu_utilization: 70%
    scale_down_delay: 5m
  
  resource_limits:
    development:
      cpu: "500m"
      memory: "1Gi"
    production:
      cpu: "2"
      memory: "4Gi"
  
  cost_saving_strategies:
    - Use spot/preemptible instances for non-critical
    - Scale to zero during off-hours
    - Cache frequent predictions
    - Batch predictions when possible
```

---

## Team Organization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps Team Structure                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  Data Science   │  │   ML Engineering│  │   Platform/Infra│          │
│  │                 │  │                 │  │                 │          │
│  │ • Experiments   │  │ • Pipelines     │  │ • Infrastructure│          │
│  │ • Model dev     │  │ • Training ops  │  │ • Kubernetes    │          │
│  │ • Feature eng   │  │ • Model serving │  │ • CI/CD         │          │
│  │ • Analysis      │  │ • Monitoring    │  │ • Security      │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│           │                   │                   │                     │
│           └───────────────────┼───────────────────┘                     │
│                               │                                         │
│                    ┌──────────▼──────────┐                              │
│                    │    MLOps Platform   │                              │
│                    │                     │                              │
│                    │ Shared tools and    │                              │
│                    │ infrastructure      │                              │
│                    └─────────────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Documentation Standards

## ## Model Details
* **Name**: `fraud_detector_v2`
* **Version**: `2.1.0`
* **Type**: XGBoost Classifier
* **Owner**: ML Team
* **Date**: 2024-01-15

## ## Intended Use
* **Primary use**: Real-time fraud detection
* **Users**: Payment processing service
* **Out of scope**: Credit scoring, identity verification

## ## Training Data
* **Dataset**: `transactions_2023`
* **Size**: 10M records
* **Features**: 25
* **Label distribution**: 1% fraud, 99% legitimate

## ## Performance

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.95 |
| **Precision** | 0.82 |
| **Recall** | 0.75 |
| **F1 Score** | 0.78 |
| **AUC** | 0.92 |



## ## Limitations
* **Environment**: Performance degrades for new merchant categories.
* **Geography**: Higher false positive rate for international transactions.
* **Maintenance**: Requires retraining monthly to combat data drift.

## ## Ethical Considerations
* **Bias**: Monitored for demographic bias via slice-based evaluation.
* **Privacy**: No personally identifiable features (PII) used in training.
* **Audit**: Subject to regular fairness audits and model governance reviews.

## ## Monitoring & Alerts
* **Dashboard**: [Grafana - Fraud Model](https://grafana.company.com/fraud-model)
* **Communication**: `#ml-alerts` Slack channel
* **Retraining Triggers**: 
    * $F1 < 0.70$
    * $PSI > 0.2$

---

## Production Checklist

```yaml
pre_deployment:
  code:
    - [ ] Code reviewed and approved
    - [ ] Unit tests passing
    - [ ] Integration tests passing
    - [ ] Security scan passed
  
  model:
    - [ ] Model registered in registry
    - [ ] Performance validated
    - [ ] Fairness tests passed
    - [ ] Model card documented
  
  infrastructure:
    - [ ] Resources provisioned
    - [ ] Secrets configured
    - [ ] Health checks configured
    - [ ] Autoscaling configured

deployment:
  - [ ] Deployed to staging
  - [ ] Integration tests in staging
  - [ ] Load testing completed
  - [ ] Canary deployment (10%)
  - [ ] Monitoring alerts configured
  - [ ] Runbook documented
  - [ ] Full production rollout

post_deployment:
  - [ ] Verify metrics in dashboard
  - [ ] Check error rates
  - [ ] Confirm alerting works
  - [ ] Update documentation
```

---

## Incident Response

```yaml
runbook:
  model_performance_degradation:
    severity: P2
    symptoms:
      - F1 score dropped below 0.70
      - Prediction latency increased
    
    steps:
      1. Check data drift dashboard
      2. Review recent data changes
      3. Check for upstream data issues
      4. If drift confirmed, trigger retraining
      5. If immediate fix needed, rollback to previous model
    
    rollback:
      command: |
        mlflow models transition \
          --name fraud_detector \
          --version <previous> \
          --stage Production
    
    escalation:
      - 15min: Page ML on-call
      - 30min: Page ML team lead
      - 1hr: Page engineering director

  model_serving_outage:
    severity: P1
    symptoms:
      - 5xx errors from model API
      - Prediction latency > 5s
    
    steps:
      1. Check pod health: kubectl get pods -l app=model
      2. Check logs: kubectl logs -l app=model
      3. Restart pods: kubectl rollout restart deployment/model
      4. If persists, check resource limits
      5. Scale up if needed: kubectl scale deployment/model --replicas=5
```

---

## Summary

### Key Takeaways

1. **Start simple, iterate** - Don't build full MLOps on day one
2. **Automate gradually** - Manual → Automated → Self-healing
3. **Monitor everything** - Data, model, infrastructure
4. **Document thoroughly** - Model cards, runbooks, processes
5. **Security first** - Encryption, access control, audit logs
6. **Optimize costs** - Right-sizing, spot instances, autoscaling

---

**🎉 Congratulations! You've completed the MLOps Course!**

You now have comprehensive knowledge of:
- End-to-end ML pipelines
- Data and model versioning
- Experiment tracking
- Model deployment and serving
- Monitoring and drift detection
- Continuous training and retraining
- Production best practices
