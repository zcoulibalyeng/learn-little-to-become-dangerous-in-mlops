# 🤖 MLOps Course

Master the complete Machine Learning Operations lifecycle from data to production.

## 🎯 What is MLOps?

```
MLOps = DevOps + Machine Learning

THE COMPLETE ML LIFECYCLE:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│   │   DATA   │───►│  TRAIN   │───►│  DEPLOY  │───►│  MONITOR │          │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│        │                                                │               │
│        │                                                │               │
│        └────────────────────────────────────────────────┘               │
│                         CONTINUOUS LOOP                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

MLOps SOLVES:
├── "Model works in notebook but not in production"
├── Data and model versioning
├── Reproducible experiments
├── Automated training pipelines
├── Model deployment and serving
├── Monitoring and drift detection
└── Continuous retraining

USED BY: Google, Netflix, Uber, Airbnb, Spotify, Meta
```

---

## 📚 Course Index

### Part 1: Foundations

#### **[Module 1: Introduction to MLOps](module-01-introduction/README.md)**
- ML lifecycle challenges
- MLOps principles and maturity levels
- DevOps vs MLOps
- MLOps architecture overview

#### **[Module 2: Data Management & Versioning](module-02-data-management/README.md)**
- Data versioning with DVC
- Data pipelines
- Data validation
- Lakehouse architecture

#### **[Module 3: Feature Engineering & Feature Stores](module-03-feature-engineering/README.md)**
- Feature engineering best practices
- Feature stores (Feast, Tecton)
- Online vs offline features
- Feature serving

### Part 2: Experimentation

#### **[Module 4: Experiment Tracking](module-04-experiment-tracking/README.md)**
- MLflow tracking
- Weights & Biases
- Experiment comparison
- Hyperparameter logging

#### **[Module 5: Model Training Pipelines](module-05-model-training/README.md)**
- Training pipelines
- Hyperparameter tuning
- Distributed training
- GPU/TPU utilization

#### **[Module 6: Model Registry & Versioning](module-06-model-registry/README.md)**
- Model versioning strategies
- MLflow Model Registry
- Model artifacts
- Model lineage

### Part 3: Quality & Testing

#### **[Module 7: Testing ML Systems](module-07-testing/README.md)**
- Data testing
- Model testing
- Infrastructure testing
- ML Test Score framework

#### **[Module 8: CI/CD for Machine Learning](module-08-cicd/README.md)**
- Continuous Integration for ML
- Continuous Delivery/Deployment
- GitHub Actions for ML
- Automated model validation

### Part 4: Deployment & Serving

#### **[Module 9: Model Deployment](module-09-deployment/README.md)**
- Deployment patterns
- Containerization with Docker
- Kubernetes deployment
- Serverless ML

#### **[Module 10: Model Serving](module-10-serving/README.md)**
- REST API serving
- Batch inference
- Real-time inference
- Edge deployment

### Part 5: Operations & Maintenance

#### **[Module 11: Monitoring & Observability](module-11-monitoring/README.md)**
- Model performance monitoring
- Infrastructure monitoring
- Logging and alerting
- Dashboards (Grafana, Prometheus)

#### **[Module 12: Data & Model Drift Detection](module-12-drift-detection/README.md)**
- Data drift types
- Concept drift
- Drift detection methods
- Evidently, Alibi Detect

#### **[Module 13: Continuous Training & Retraining](module-13-retraining/README.md)**
- Retraining triggers
- Automated retraining pipelines
- A/B testing models
- Shadow deployment

### Part 6: Infrastructure & Scale

#### **[Module 14: ML Orchestration](module-14-orchestration/README.md)**
- Airflow for ML
- Kubeflow Pipelines
- Prefect, Dagster
- Workflow scheduling

#### **[Module 15: Production Best Practices](module-15-production/README.md)**
- Security and governance
- Cost optimization
- Team structure
- MLOps maturity assessment

---

## 🎯 Learning Objectives

After completing this course, you will be able to:

- ✅ Design end-to-end ML pipelines
- ✅ Version data, code, and models
- ✅ Track and compare experiments
- ✅ Deploy models to production
- ✅ Monitor model performance and detect drift
- ✅ Implement automated retraining
- ✅ Build scalable ML infrastructure

---

## 🛠️ Prerequisites

- Python programming
- Basic machine learning concepts
- Docker fundamentals (see Docker course)
- Git version control

---

## 📈 Difficulty Progression

```
Modules 1-3:   ████░░░░░░  Beginner
Modules 4-6:   ██████░░░░  Intermediate
Modules 7-10:  ████████░░  Advanced
Modules 11-15: ██████████  Expert
```

---

## 🗓️ Suggested Study Plan

| Pace | Duration | Schedule |
|------|----------|----------|
| Intensive | 2 weeks | 1-2 modules/day |
| Balanced | 4 weeks | 1 module/day |
| Relaxed | 6 weeks | 2-3 modules/week |

---

## 🔧 Tools Covered

| Category | Tools |
|----------|-------|
| Data Versioning | DVC, LakeFS |
| Experiment Tracking | MLflow, W&B, Neptune |
| Feature Store | Feast, Tecton |
| Orchestration | Airflow, Kubeflow, Prefect |
| Serving | TensorFlow Serving, TorchServe, Seldon |
| Monitoring | Evidently, Prometheus, Grafana |
| Infrastructure | Docker, Kubernetes, AWS/GCP/Azure |

---

## 📖 Reference Materials

- **[QUICK-REFERENCE.md](QUICK-REFERENCE.md)** - MLOps cheat sheet

---

## 🔗 Related Courses

| Course | Relationship |
|--------|--------------|
| Docker & Kubernetes | Container deployment |
| Python | ML development |
| System Design | Architecture patterns |

---

👉 **[Start with Module 1: Introduction to MLOps](module-01-introduction/README.md)**
