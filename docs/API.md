# API reference — experimental release

This page describes **implemented** functionality. See [architecture and limitations](ARCHITECTURE.md). **Research use only; not clinically validated.**

## `AllocationModel`

```python
from allocation_framework import AllocationModel
model = AllocationModel(
    fairness_weight=0.5,
    resource_constraint=None,
    fairness_metric="allocation_parity",
    efficiency_weight=0.3,
    allocation_threshold=None,
    random_state=42,
    verbose=0,
)
```

- `fairness_weight` and `efficiency_weight`: accepted and stored (each between 0 and 1), **not used to optimise the fitted model**.
- `resource_constraint`: non-negative integer or `None`. When supplied, `predict(X)` chooses at most this many highest-scoring samples **per call**, not across calls. With positive capacity, the policy fills available slots up to the batch size; it does not require probabilities to exceed a clinical threshold.
- `fairness_metric`: `allocation_parity`, `equality_of_opportunity`, `demographic_parity` or `predictive_parity`; controls evaluation, not model fitting.
- `allocation_threshold`: threshold used only when `resource_constraint=None`; otherwise top-k selection takes precedence.
- `random_state`: scikit-learn model seed.
- `verbose`: controls diagnostic printing.

### Methods

- `fit(X, y, groups=None, sample_weight=None)`: standardises `X` and fits a balanced logistic regression classifier. `groups` is recorded for metadata and is **not used in fitting**.
- `predict_proba(X)`: two-column classifier-estimated probabilities, not calibrated clinical risk estimates.
- `predict(X)`: binary decisions via threshold or per-call top-k rule. Ranking ties are resolved by input order.
- `evaluate(X, y, groups=None)`: returns an `AllocationResults` dataclass with `predictive_accuracy`, `fairness_score`, `allocation_efficiency`, `coverage`, `group_metrics` and `metadata`.
- `get_params(deep=True)` / `set_params(**params)`: parameter access; changing a parameter after fitting does not retrain the model.
- `get_fairness_accuracy_tradeoff(X_val, y_val, groups_val=None, weight_range=(0,1), n_steps=11)`: **not implemented**; raises `NotImplementedError` on a fitted model. No trade-off frontier is produced.

### Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from allocation_framework import AllocationModel

X, y = make_classification(n_samples=200, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
model = AllocationModel(resource_constraint=10, random_state=42)
model.fit(X_train, y_train)
results = model.evaluate(X_test, y_test)
print(results)
assert model.predict(X_test).sum() <= 10
```

## Standalone loss functions

`AllocationAwareLoss`, `GroupParityLoss`, `EqualityOfOpportunityLoss` and `CalibratedLoss` are available from `allocation_framework.losses.fairness` (the first three also from the package root). They calculate scalar experimental penalties but **are not used in `AllocationModel.fit()`**. The default combined loss uses binary cross-entropy, variance in group allocation rates and a penalty for deviation from an illustrative 50% allocation rate. This 50% target is not a validated policy.

## Metrics

Functions in `allocation_framework.metrics.fairness` include `allocation_parity`, `demographic_parity`, `equality_of_opportunity`, `predictive_parity`, `false_positive_rate_parity`, `allocation_efficiency`, `coverage`, `resource_utilization`, `fairness_accuracy_score` and `compute_all_metrics`.

Group parity scores use `1 - (maximum group rate - minimum group rate)`. `allocation_efficiency` is precision of selected cases; `coverage` is recall of positive labels. Scores may use default values for undefined cases, including no eligible allocations or missing group information. Inspect the source and subgroup counts before interpretation. No metric alone establishes equitable outcomes.

For a runnable demonstration, see [examples/01_introduction.py](../examples/01_introduction.py).
