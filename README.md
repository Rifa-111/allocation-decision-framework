# Allocation Decision Framework

[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)
[![Paper: AIiH 2026](https://img.shields.io/badge/Paper-AIiH%202026-blue.svg)](https://doi.org/10.1007/978-3-032-35393-1_24)

An **experimental, open-source Python research implementation** for exploring healthcare AI allocation decisions, resource limits and group fairness metrics. This software accompanies *Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI* (Ferzana, 2026).

> **Research only.** This is not clinically validated, not production-ready and must not be used to make patient-care or real-world resource-allocation decisions.

## What is implemented

- Logistic regression baseline with optional **per-prediction-batch** top-k resource allocation
- Group fairness metrics, including allocation parity and equality of opportunity
- Allocation efficiency, coverage and group-level evaluation results
- Standalone fairness-aware loss functions for research experimentation
- Automated unit tests and a Python tutorial using synthetic data

## Important limitations

- `AllocationModel.fit()` trains a standard logistic regression baseline. **`fairness_weight` and `efficiency_weight` do not currently alter fitting or predictions.** Fairness is measured after prediction, not optimised by the fitted model.
- `get_fairness_accuracy_tradeoff()` explicitly raises `NotImplementedError` until a valid optimisation approach is implemented. No trade-off curve is claimed.
- `resource_constraint=k` limits allocations to at most `k` cases **within each call to `predict()`**. It is not a persistent capacity ledger across batches, and the allocation policy is not clinically validated.
- The tutorial contains illustrative synthetic scenarios; its outcomes are not evidence of clinical effectiveness or fairness in real populations.

## Installation

Clone the repository and install it in a fresh Python environment:

```bash
git clone https://github.com/Rifa-111/allocation-decision-framework.git
cd allocation-decision-framework
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
```

The package is **not yet advertised as published on PyPI**. Use the GitHub installation above until a release is independently verified.

## Minimal example

```python
import numpy as np
from sklearn.datasets import make_classification
from allocation_framework import AllocationModel

X, y = make_classification(n_samples=100, n_features=10, random_state=42)
groups = np.array([0, 1] * 50)
model = AllocationModel(resource_constraint=20, random_state=42)
model.fit(X, y, groups=groups)
results = model.evaluate(X, y, groups=groups)
print(results)
print("Allocated:", model.predict(X).sum())  # at most 20 in this batch
```

## Tutorial and documentation

- [Python introduction tutorial](examples/01_introduction.py) (synthetic examples)
- [Installation guide](docs/INSTALL.md)
- [API reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)

## Citation

```bibtex
@inproceedings{ferzana2026beyondaccuracy,
  author = {Ferzana, Rifa},
  title = {Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI},
  booktitle = {Artificial Intelligence and Intelligent Healthcare},
  year = {2026},
  series = {Lecture Notes in Computer Science},
  volume = {16877},
  pages = {377--388},
  publisher = {Springer},
  doi = {10.1007/978-3-032-35393-1_24}
}
```

## Licence

MIT — see [LICENSE](LICENSE). Contributions and reproducibility reports are welcome via [GitHub Issues](https://github.com/Rifa-111/allocation-decision-framework/issues).
