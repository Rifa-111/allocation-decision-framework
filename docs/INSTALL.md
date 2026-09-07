# Installation Guide

## Quick Install

### From PyPI (Recommended)

```bash
pip install allocation-decision-framework
```

### From Source

```bash
git clone https://github.com/Rifa-111/allocation-decision-framework.git
cd allocation-decision-framework
pip install -e .
```

## Development Setup

For contributing to the framework:

```bash
git clone https://github.com/Rifa-111/allocation-decision-framework.git
cd allocation-decision-framework

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=allocation_framework

# Format code
black allocation_framework/

# Lint code
flake8 allocation_framework/

# Type check
mypy allocation_framework/
```

## System Requirements

- **Python:** 3.8 or higher
- **Operating System:** Windows, macOS, Linux
- **Disk Space:** ~100 MB for installation + dependencies

## Dependencies

### Core Dependencies

- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.2.0` - Data manipulation
- `scikit-learn>=0.24.0` - Machine learning
- `scipy>=1.6.0` - Scientific computing
- `matplotlib>=3.3.0` - Visualization
- `seaborn>=0.11.0` - Statistical visualization

### Optional Dependencies

```bash
# For deep learning support (PyTorch)
pip install allocation-decision-framework[torch]

# For Jupyter notebook tutorials
pip install allocation-decision-framework[notebooks]

# For development (testing, linting, docs)
pip install allocation-decision-framework[dev]

# All optional features
pip install allocation-decision-framework[all]
```

## Verification

Test that the installation works:

```python
python -c "import allocation_framework; print(allocation_framework.__version__)"
```

Should output: `0.1.0`

Or test with a simple script:

```python
from allocation_framework import AllocationModel, allocation_parity
import numpy as np

# Create sample data
X = np.random.randn(100, 10)
y = np.random.binomial(1, 0.5, 100)
groups = np.random.binomial(1, 0.5, 100)

# Fit model
model = AllocationModel(fairness_weight=0.3)
model.fit(X, y, groups=groups)

# Evaluate
results = model.evaluate(X, y, groups=groups)
print(f"Accuracy: {results.predictive_accuracy:.3f}")
print(f"Fairness: {results.fairness_score:.3f}")
```

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'allocation_framework'`:

1. Ensure installation completed: `pip install allocation-decision-framework`
2. Try reinstalling: `pip install --upgrade --force-reinstall allocation-decision-framework`
3. Check Python version: `python --version` (should be 3.8+)

### Scikit-learn Version Conflict

If you see warnings about scikit-learn version:

```bash
pip install --upgrade scikit-learn numpy scipy
```

### Import in Notebooks

If running in Jupyter and getting import errors, ensure the kernel is using the right Python:

```python
import sys
print(sys.executable)
```

Should point to the Python environment where you installed the package.

## Getting Started

After installation, check out the tutorials:

```bash
# Run the introduction tutorial
python -c "from examples.ex01_introduction import *"

# Or start with a simple example
python
>>> from allocation_framework import AllocationModel
>>> help(AllocationModel)
```

## Next Steps

- [Read the README](../README.md) for overview and quick-start
- [Check out tutorials](../examples/) for detailed examples
- [Review API documentation](./API.md) for complete reference
- [Explore architecture](./ARCHITECTURE.md) for design decisions

## Getting Help

- **Documentation:** https://github.com/Rifa-111/allocation-decision-framework
- **Issues:** https://github.com/Rifa-111/allocation-decision-framework/issues
- **Paper:** https://link.springer.com/chapter/10.1007/978-3-032-35393-1_24

## Citation

If you use this framework in research:

```bibtex
@inproceedings{ferzana2026beyondaccuracy,
  author = {Ferzana, Rifa},
  title = {Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI},
  booktitle = {Artificial Intelligence and Intelligent Healthcare},
  year = {2026},
  series = {Lecture Notes in Computer Science},
  volume = {16877},
  publisher = {Springer},
  url = {https://link.springer.com/chapter/10.1007/978-3-032-35393-1_24}
}
```
