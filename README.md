# Allocation Decision Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: AIiH 2026](https://img.shields.io/badge/Paper-AIiH%202026-blue.svg)](https://link.springer.com/chapter/10.1007/978-3-032-35393-1_24)

A decision-theoretic framework for building allocation-aware healthcare AI systems that optimise for fairness and resource efficiency alongside predictive accuracy.

## Overview

This framework implements the research presented in **"Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI"** (Ferzana, 2026), published in the *Artificial Intelligence and Intelligent Healthcare* proceedings at Imperial College London.

### Key Contributions

- **Allocation-aware loss functions** that explicitly model resource constraints and fairness objectives
- **Decision-theoretic evaluation metrics** beyond accuracy
- **Fairness-utility trade-off analysis** for healthcare resource allocation
- **Practical implementations** for common healthcare allocation scenarios

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rifa-111/allocation-decision-framework.git
cd allocation-decision-framework

# Install in development mode
pip install -e ".[dev]"

# Or install from PyPI (once published)
pip install allocation-decision-framework
```

### Basic Usage

```python
from allocation_framework import AllocationModel, FairnessLoss

# Initialise model with fairness constraints
model = AllocationModel(
    fairness_weight=0.3,
    resource_constraint=100,  # e.g., 100 treatment slots
    fairness_metric='allocation_parity'
)

# Train with allocation-aware objectives
model.fit(X_train, y_train, allocation_targets=targets)

# Evaluate beyond accuracy
results = model.evaluate(X_test, y_test)
print(results.fairness_score)
print(results.allocation_efficiency)
print(results.predictive_accuracy)
```

## Documentation

- [Installation Guide](docs/INSTALL.md)
- [Framework Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Tutorials & Examples

Explore jupyter notebooks in `examples/`:

- [`01_introduction.ipynb`](examples/01_introduction.ipynb) — Core concepts and fairness-accuracy trade-offs
- [`02_synthetic_allocation.ipynb`](examples/02_synthetic_allocation.ipynb) — End-to-end example with synthetic data
- [`03_fairness_metrics.ipynb`](examples/03_fairness_metrics.ipynb) — Computing and interpreting fairness metrics
- [`04_resource_constraints.ipynb`](examples/04_resource_constraints.ipynb) — Modeling real-world resource limits
- [`05_comparing_approaches.ipynb`](examples/05_comparing_approaches.ipynb) — Allocation vs. accuracy-only baselines

## Citation

If you use this framework, please cite the original paper:

```bibtex
@inproceedings{ferzana2026beyondaccuracy,
  author = {Ferzana, Rifa},
  title = {Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI},
  booktitle = {Artificial Intelligence and Intelligent Healthcare},
  year = {2026},
  series = {Lecture Notes in Computer Science},
  volume = {16877},
  pages = {XXX--XXX},
  publisher = {Springer},
  url = {https://link.springer.com/chapter/10.1007/978-3-032-35393-1_24}
}
```

## Paper Abstract

> Healthcare AI systems must balance predictive accuracy with fair resource allocation. We present a decision-theoretic framework that explicitly models allocation constraints and fairness objectives, moving beyond single-metric evaluation. Our approach quantifies fairness-accuracy trade-offs and provides principled methods for practitioners to navigate allocation decisions in resource-constrained settings.

## Key Features

✅ **Allocation-aware optimisation** — Loss functions that encode resource and fairness constraints  
✅ **Flexible fairness metrics** — Multiple fairness definitions (parity, equality of opportunity, etc.)  
✅ **Decision support tools** — Trade-off visualization and sensitivity analysis  
✅ **Production-ready** — Type hints, logging, and error handling  
✅ **Reproducible** — Full examples and benchmark datasets included  

## Architecture

```
allocation-decision-framework/
├── allocation_framework/
│   ├── __init__.py
│   ├── models/               # Core allocation models
│   ├── metrics/              # Fairness & allocation metrics
│   ├── losses/               # Custom loss functions
│   ├── visualization/        # Trade-off plots & analysis
│   └── utils/                # Utilities & helpers
├── examples/                 # Jupyter notebooks
├── tests/                    # Unit & integration tests
├── docs/                     # Full documentation
└── setup.py
```

## Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn
- PyTorch (optional, for deep learning variants)
- Jupyter (for tutorials)

See `requirements.txt` for full dependencies.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
pip install -e ".[dev]"
pytest tests/
black allocation_framework/
```

## License

MIT License — see [LICENSE](LICENSE) file for details.

## Contact & Support

- **Author:** Rifa Ferzana
- **GitHub:** [@Rifa-111](https://github.com/Rifa-111)
- **Issues:** [GitHub Issues](https://github.com/Rifa-111/allocation-decision-framework/issues)

---

**Published:** August 2026 at AIiH 2026, Imperial College London  
**Paper DOI:** [10.1007/978-3-032-35393-1_24](https://doi.org/10.1007/978-3-032-35393-1_24)
