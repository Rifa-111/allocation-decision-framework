"""Reproducible synthetic-data tutorial for the experimental allocation framework.

This example illustrates predictive evaluation and per-call top-k allocation.
It does not demonstrate fairness optimisation or clinical effectiveness.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from allocation_framework import AllocationModel, allocation_parity


def main():
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=2, random_state=42,
    )
    # Synthetic group labels for demonstration only. Not protected-attribute data.
    groups = np.random.default_rng(42).integers(0, 2, size=len(y))
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.3, stratify=y, random_state=42
    )
    model = AllocationModel(resource_constraint=20, random_state=42)
    model.fit(X_train, y_train, groups=groups_train)
    allocated = model.predict(X_test)
    results = model.evaluate(X_test, y_test, groups=groups_test)
    assert allocated.sum() <= 20
    print("Experimental allocation model — synthetic hold-out data")
    print(f"Test samples: {len(y_test)}; selected in this batch: {allocated.sum()}")
    print(f"Predictive accuracy: {results.predictive_accuracy:.3f}")
    print(f"Allocation precision: {results.allocation_efficiency:.3f}")
    print(f"Coverage (recall): {results.coverage:.3f}")
    print(f"Allocation parity score (illustrative groups): {allocation_parity(allocated, groups_test):.3f}")
    print("Warning: fairness weights are not used in fitting; no fairness-accuracy frontier is computed.")
    print("Research demonstration only; not a clinical decision tool.")


if __name__ == "__main__":
    main()
