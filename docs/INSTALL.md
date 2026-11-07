# Installation and verification

**Status:** Research prototype. Not clinically validated and not suitable for patient-care or resource-allocation decisions.

## Supported environments

Python **3.10 and 3.11** are tested in GitHub Actions on Ubuntu. Other Python versions and operating systems are not verified by this project's CI. Installation requires an internet connection to obtain dependencies.

## Install from the repository

```bash
git clone https://github.com/Rifa-111/allocation-decision-framework.git
cd allocation-decision-framework
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
python examples/01_introduction.py
```

Until a verified public package release exists, do not assume `pip install allocation-decision-framework` resolves to this repository or that it is published on PyPI.

## Quick smoke test

```bash
python -c "import allocation_framework; print(allocation_framework.__version__)"
```

## Optional extras

The setup file defines `[dev]`, `[torch]` and `[notebooks]` extras. Installing an optional dependency **does not imply a deep-learning model or a notebook is implemented**. The supplied tutorial is a Python script. There is no `[all]` extra or command-line executable in this release.

## Troubleshooting

Use `python -m pip` and `python -m pytest` from the same active virtual environment. If an import fails, check the installation log and confirm the active Python interpreter. Report reproducible issues with the Python version and full traceback in [GitHub Issues](https://github.com/Rifa-111/allocation-decision-framework/issues).

## Citation

Ferzana, R. (2026). *Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI*. Lecture Notes in Computer Science, **16877**, 377–388. Springer. https://doi.org/10.1007/978-3-032-35393-1_24
