# Enterprise End-to-End MLOps Pipeline: Stock Predictor API

A production-grade, GitOps-driven MLOps pipeline that serves a PyTorch-based sequence inference model via a secure, non-root FastAPI microservice. The infrastructure is entirely declarative, utilizing localized Kubernetes architecture, rigorous DevSecOps scanning, and automated continuous delivery.

## 🏗️ System Architecture & Workflow



1. **Model Engineering:** Python 3.11 code extracts asset features and trains a PyTorch regression model, serializing the weights into a portable `model.pkl` file.
2. **DevSecOps CI:** Code and dependencies are statically analyzed for vulnerabilities (`bandit`, `trivy`).
3. **Containerization:** A multi-stage, slimmed Docker image is built, stripping CUDA overhead to achieve an optimized 299MB network transfer footprint.
4. **GitOps CD:** ArgoCD tracks declarative manifests and synchronizes state natively into a local Minikube cluster.

---

## 🧠 1. Model Training & Serialization (`.pkl`)

The core machine learning engine uses PyTorch (`torch+cpu`) to predict next-day asset close prices based on lagging momentum indicators.

### Training Mechanics
The model maps input vectors containing 3 critical historical features:
* `lag_1`: The previous trading session's close price.
* `daily_return`: The percentage change in asset velocity.
* `ma_5`: The 5-day rolling moving average to capture trend baselines.

The mathematical optimization maps features through hidden layers using a Mean Squared Error (MSE) loss function:

$$\text{Output} = (X_1 \cdot W_1) + (X_2 \cdot W_2) + (X_3 \cdot W_3) + b$$

Once optimization converged, the active state weights were serialized using Python's `pickle` library:
```python
import pickle
# Serializing the model structure and weights matrix
with open("model.pkl", "wb") as f:
    pickle.dump(trained_pytorch_model, f)
