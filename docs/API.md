# API Reference

Complete documentation of all public classes and functions in allocation-decision-framework.

## Quick Reference

```python
# Import main classes
from allocation_framework import AllocationModel, AllocationResults

# Import loss functions
from allocation_framework import (
    AllocationAwareLoss,
    GroupParityLoss,
    EqualityOfOpportunityLoss,
)

# Import metrics
from allocation_framework import (
    allocation_parity,
    equality_of_opportunity,
    allocation_efficiency,
    coverage,
    compute_all_metrics,
)
```

---

## Models

### AllocationModel

Main class for allocation-aware healthcare AI modeling.

```python
class AllocationModel:
    def __init__(
        self,
        fairness_weight: float = 0.5,
        resource_constraint: Optional[int] = None,
        fairness_metric: str = "allocation_parity",
        efficiency_weight: float = 0.3,
        allocation_threshold: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    )
```

#### Parameters

- **fairness_weight** (float, default=0.5)
  - Importance weight for fairness objective (0-1)
  - 0 = pure accuracy maximization
  - 1 = pure fairness maximization
  - 0.3-0.5 = balanced in most applications

- **resource_constraint** (int, optional)
  - Maximum number of allocations available
  - If set, threshold automatically computed to respect constraint
  - Example: 100 patient beds in hospital

- **fairness_metric** (str, default="allocation_parity")
  - Which fairness definition to use
  - Options:
    - "allocation_parity": Equal allocation rates across groups
    - "equality_of_opportunity": Equal TPR (benefit) for those who need it
    - "demographic_parity": Alias for allocation_parity
    - "predictive_parity": Equal precision across groups

- **efficiency_weight** (float, default=0.3)
  - Importance weight for resource efficiency (0-1)
  - Higher = penalize resource waste more

- **allocation_threshold** (float, optional)
  - Fixed decision threshold for binary predictions
  - If None, computed automatically from resource_constraint or defaults to 0.5

- **random_state** (int, optional)
  - Seed for reproducibility

- **verbose** (int, default=0)
  - Verbosity level: 0=silent, 1=info, 2=debug

#### Methods

##### fit(X, y, groups=None, sample_weight=None)

Fit the allocation model on training data.

```python
model.fit(X_train, y_train, groups=groups_train)
```

**Parameters:**
- **X** (array-like, shape=(n_samples, n_features))
  - Feature matrix of training data

- **y** (array-like, shape=(n_samples,))
  - Binary target labels [0, 1]
  - 1 = needs allocation, 0 = doesn't need

- **groups** (array-like, shape=(n_samples,), optional)
  - Group membership for fairness evaluation
  - Example: demographic group, hospital unit, risk category

- **sample_weight** (array-like, shape=(n_samples,), optional)
  - Per-sample weights for imbalanced data

**Returns:**
- **self** - Fitted model instance

**Example:**
```python
import numpy as np
from allocation_framework import AllocationModel

X_train = np.random.randn(100, 10)
y_train = np.random.binomial(1, 0.5, 100)
groups_train = np.random.binomial(1, 0.5, 100)

model = AllocationModel(fairness_weight=0.3)
model.fit(X_train, y_train, groups=groups_train)
```

---

##### predict(X)

Make binary allocation decisions.

```python
allocations = model.predict(X_test)  # Returns [0, 1, 0, 1, ...]
```

**Parameters:**
- **X** (array-like, shape=(n_samples, n_features))
  - Feature matrix to make predictions on

**Returns:**
- **allocations** (ndarray, shape=(n_samples,))
  - Binary predictions: 1 = allocate, 0 = don't allocate

**Example:**
```python
X_test = np.random.randn(50, 10)
allocations = model.predict(X_test)
print(f"Allocations: {allocations}")
print(f"Total allocated: {np.sum(allocations)}")
```

---

##### predict_proba(X)

Get probability predictions.

```python
proba = model.predict_proba(X_test)
```

**Parameters:**
- **X** (array-like, shape=(n_samples, n_features))
  - Feature matrix

**Returns:**
- **proba** (ndarray, shape=(n_samples, 2))
  - Probabilities [P(y=0), P(y=1)] for each sample

**Example:**
```python
proba = model.predict_proba(X_test)
print(f"Allocation probability: {proba[:, 1]}")  # [0.1, 0.8, 0.3, ...]
```

---

##### evaluate(X, y, groups=None)

Comprehensively evaluate model fairness and efficiency.

```python
results = model.evaluate(X_test, y_test, groups=groups_test)
```

**Parameters:**
- **X** (array-like, shape=(n_samples, n_features))
  - Feature matrix

- **y** (array-like, shape=(n_samples,))
  - True labels

- **groups** (array-like, optional)
  - Group membership for stratified evaluation

**Returns:**
- **results** (AllocationResults)
  - predictive_accuracy: float
  - fairness_score: float  
  - allocation_efficiency: float
  - coverage: float
  - group_metrics: dict (per-group performance)
  - metadata: dict (threshold, allocations, etc.)

**Example:**
```python
results = model.evaluate(X_test, y_test, groups=groups_test)

print(f"Accuracy:   {results.predictive_accuracy:.3f}")
print(f"Fairness:   {results.fairness_score:.3f}")
print(f"Efficiency: {results.allocation_efficiency:.3f}")
print(f"Coverage:   {results.coverage:.3f}")

if results.group_metrics:
    for group_id, metrics in results.group_metrics.items():
        print(f"\n{group_id}:")
        print(f"  n_samples: {metrics['n_samples']}")
        print(f"  allocation_rate: {metrics['allocation_rate']:.1%}")
        print(f"  tpr: {metrics['tpr']:.3f}")
```

---

##### get_fairness_accuracy_tradeoff(X_val, y_val, groups_val=None, weight_range=(0.0, 1.0), n_steps=11)

Compute fairness-accuracy trade-off frontier.

```python
tradeoff = model.get_fairness_accuracy_tradeoff(
    X_val, y_val, groups=groups_val,
    weight_range=(0.0, 1.0),
    n_steps=11
)
```

**Parameters:**
- **X_val, y_val** (arrays)
  - Validation feature matrix and labels

- **groups_val** (array, optional)
  - Validation group membership

- **weight_range** (tuple, default=(0.0, 1.0))
  - Range of fairness weights to test

- **n_steps** (int, default=11)
  - Number of points on the frontier

**Returns:**
- **dict** with keys:
  - 'weights': tested fairness weights (ndarray)
  - 'accuracy': accuracy at each weight (ndarray)
  - 'fairness': fairness at each weight (ndarray)

**Example:**
```python
import matplotlib.pyplot as plt

tradeoff = model.get_fairness_accuracy_tradeoff(
    X_val, y_val, groups=groups_val, n_steps=11
)

plt.plot(tradeoff['fairness'], tradeoff['accuracy'], 'o-')
plt.xlabel('Fairness Score')
plt.ylabel('Accuracy')
plt.title('Fairness-Accuracy Trade-Off')
plt.grid(True)
plt.show()
```

---

##### get_params(deep=True)

Get parameters (sklearn compatibility).

```python
params = model.get_params()
# Returns: {'fairness_weight': 0.5, 'resource_constraint': None, ...}
```

##### set_params(**params)

Set parameters (sklearn compatibility).

```python
model.set_params(fairness_weight=0.7, resource_constraint=100)
```

---

### AllocationResults

Dataclass containing evaluation results.

```python
@dataclass
class AllocationResults:
    predictive_accuracy: float
    fairness_score: float
    allocation_efficiency: float
    coverage: float
    group_metrics: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Optional[Dict[str, Any]] = None
```

**Attributes:**

- **predictive_accuracy** (float, 0-1)
  - Standard accuracy: fraction of correct predictions

- **fairness_score** (float, 0-1)
  - Fairness metric value (depends on fairness_metric parameter)
  - 1.0 = perfect fairness, 0.0 = maximum unfairness

- **allocation_efficiency** (float, 0-1)
  - Precision of allocations: TP / (TP + FP)
  - High = allocating only to those who need it

- **coverage** (float, 0-1)
  - Recall: fraction of actual needs met
  - High = catching all those who need help

- **group_metrics** (dict, optional)
  - Per-group performance breakdown
  - Key: group identifier
  - Value: dict with metrics like 'allocation_rate', 'tpr', 'fpr'

- **metadata** (dict, optional)
  - Auxiliary information like threshold, total allocated, etc.

---

## Loss Functions

### AllocationAwareLoss

Combined loss for accuracy + fairness + efficiency.

```python
from allocation_framework import AllocationAwareLoss

loss_fn = AllocationAwareLoss(
    fairness_weight=0.5,
    efficiency_weight=0.3,
)

loss = loss_fn(y_true, y_pred, groups)
```

**Formula:**
```
Loss = accuracy_loss + α·fairness_loss + β·efficiency_loss

Where:
- accuracy_loss: Binary cross-entropy
- fairness_loss: Variance of allocation rates across groups
- efficiency_loss: Deviation from target allocation (50%)
- α: fairness_weight
- β: efficiency_weight
```

---

### GroupParityLoss

Enforce demographic parity (equal allocation across groups).

```python
from allocation_framework import GroupParityLoss

loss_fn = GroupParityLoss(margin=0.1)
loss = loss_fn(y_true, y_pred, groups)
```

**Parameters:**
- **margin** (float, default=0.1)
  - Tolerance for allocation rate differences
  - Loss = max(0, diff - margin)²

---

### EqualityOfOpportunityLoss

Enforce equal benefit for those who need allocation.

```python
from allocation_framework import EqualityOfOpportunityLoss

loss_fn = EqualityOfOpportunityLoss(margin=0.1)
loss = loss_fn(y_true, y_pred, groups)
```

---

## Metrics

### allocation_parity(y_pred, groups)

Score for equal allocation rates across groups.

```python
from allocation_framework import allocation_parity

score = allocation_parity(y_pred, groups)  # Returns: 0.0-1.0
```

**Returns:** float (1.0 = perfect parity)

---

### equality_of_opportunity(y_true, y_pred, groups)

Score for equal TPR (benefit) across groups for those who need it.

```python
from allocation_framework import equality_of_opportunity

score = equality_of_opportunity(y_true, y_pred, groups)
```

**Returns:** float (1.0 = perfect equality)

---

### allocation_efficiency(y_true, y_pred, resource_constraint=None)

Precision: fraction of allocations that are appropriate.

```python
from allocation_framework import allocation_efficiency

eff = allocation_efficiency(y_true, y_pred)  # 0.0-1.0
```

**Formula:** TP / (TP + FP)

---

### coverage(y_true, y_pred)

Recall: fraction of actual needs that are met.

```python
from allocation_framework import coverage

cov = coverage(y_true, y_pred)  # 0.0-1.0
```

**Formula:** TP / (TP + FN)

---

### compute_all_metrics(y_true, y_pred, groups=None, resource_constraint=None)

Compute all metrics at once.

```python
from allocation_framework import compute_all_metrics

metrics = compute_all_metrics(y_true, y_pred, groups)

# Returns dict with keys:
# 'accuracy', 'precision', 'recall', 'allocation_parity',
# 'equality_of_opportunity', 'demographic_parity', 'predictive_parity',
# 'false_positive_rate_parity', 'allocation_efficiency', 'coverage',
# 'resource_utilization' (if resource_constraint provided)
```

---

### get_group_fairness_scores(y_true, y_pred, groups)

Per-group performance breakdown.

```python
from allocation_framework.metrics import get_group_fairness_scores

group_scores = get_group_fairness_scores(y_true, y_pred, groups)

# Returns dict:
# {
#     'group_0': {
#         'n_samples': 50,
#         'accuracy': 0.85,
#         'allocation_rate': 0.6,
#         'tpr': 0.7,
#         'fpr': 0.2,
#         'ppv': 0.75,
#     },
#     'group_1': { ... }
# }
```

---

## Complete Example

```python
from allocation_framework import AllocationModel, compute_all_metrics
import numpy as np
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
groups = np.random.binomial(1, 0.5, 200)

# Split into train/test
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
groups_train, groups_test = groups[:split], groups[split:]

# Train model
model = AllocationModel(
    fairness_weight=0.3,
    resource_constraint=30,
    fairness_metric='allocation_parity'
)
model.fit(X_train, y_train, groups=groups_train)

# Evaluate
results = model.evaluate(X_test, y_test, groups=groups_test)
print(f"Accuracy:   {results.predictive_accuracy:.3f}")
print(f"Fairness:   {results.fairness_score:.3f}")

# Detailed metrics
metrics = compute_all_metrics(y_test, model.predict(X_test), groups_test)
print(f"Efficiency: {metrics['allocation_efficiency']:.3f}")
print(f"Coverage:   {metrics['coverage']:.3f}")

# Trade-off analysis
tradeoff = model.get_fairness_accuracy_tradeoff(
    X_test, y_test, groups=groups_test, n_steps=11
)
print(f"Weights range: {tradeoff['weights'].min():.1f} to {tradeoff['weights'].max():.1f}")
```

---

## Related Documentation

- [Installation Guide](./INSTALL.md)
- [Architecture](./ARCHITECTURE.md)
- [Contributing](../CONTRIBUTING.md)
