# Architecture and scientific scope

This repository is an **experimental research implementation**, not a clinical decision support system. It accompanies Ferzana (2026), *Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI*, Springer LNCS 16877, pp. 377–388. [DOI](https://doi.org/10.1007/978-3-032-35393-1_24).

## Implemented pipeline

1. `AllocationModel.fit(X, y, groups=None)` standardises features with `StandardScaler` and fits scikit-learn's `LogisticRegression(class_weight="balanced")`.
2. `predict_proba(X)` returns classifier-estimated class probabilities. These are **not validated clinical risks**.
3. `predict(X)` returns thresholded binary decisions (default 0.5), or selects at most `resource_constraint=k` highest-probability samples **within that prediction call**. Equal scores are broken by input order. No global capacity is tracked between calls.
4. `evaluate(X, y, groups)` reports accuracy, allocation precision (called efficiency), recall (called coverage), selected group disparity scores and per-group summaries.

## Not implemented

- The model's `fairness_weight` and `efficiency_weight` parameters are stored but **do not influence training, allocation or prediction**.
- Standalone `AllocationAwareLoss`, `GroupParityLoss`, `EqualityOfOpportunityLoss` and `CalibratedLoss` are research calculations; **they are not connected to the logistic regression optimiser**.
- `get_fairness_accuracy_tradeoff()` raises `NotImplementedError`; no Pareto frontier is calculated.
- No clinically validated resource policy, outcome model, benefit estimator, fairness guarantee or deployment integration is provided.
- The batch top-k rule may allocate to low-probability cases to fill capacity and overrides `allocation_threshold` when a resource constraint is set.

## Important interpretation limits

Fairness scores measure particular mathematical definitions; they do not establish equitable care. Some metrics return a default score when groups or eligible positive/negative cases are missing. Review the underlying implementation and group support counts before interpreting results. Allocation precision and coverage use labels as a proxy for need, not observed treatment benefit. Results calculated on training data are illustrative, not out-of-sample performance estimates.

## Source layout

- `allocation_framework/models/base.py`: fitted baseline, batch top-k policy and evaluation.
- `allocation_framework/metrics/fairness.py`: parity, opportunity, precision, coverage and other descriptive metrics.
- `allocation_framework/losses/fairness.py`: standalone experimental loss calculations.
- `examples/01_introduction.py`: reproducible synthetic-data tutorial.
- `tests/test_all.py`: unit and regression tests.

See [API reference](API.md), [installation](INSTALL.md) and [contributing](../CONTRIBUTING.md). No empirical scalability or clinical performance benchmark is claimed in this release.
