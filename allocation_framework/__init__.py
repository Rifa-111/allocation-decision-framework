"""
Allocation Decision Framework

A decision-theoretic framework for building allocation-aware healthcare AI systems
that optimize for fairness and resource efficiency alongside predictive accuracy.

Example:
    >>> from allocation_framework import AllocationModel, FairnessLoss
    >>> model = AllocationModel(fairness_weight=0.3)
    >>> model.fit(X_train, y_train, groups=groups_train)
    >>> results = model.evaluate(X_test, y_test, groups=groups_test)
    >>> print(f"Accuracy: {results.predictive_accuracy:.3f}")
    >>> print(f"Fairness: {results.fairness_score:.3f}")
"""

__version__ = "0.1.0"
__author__ = "Rifa Ferzana"
__license__ = "MIT"
__all__ = [
    "AllocationModel",
    "FairnessLoss",
    "AllocationAwareLoss",
    "GroupParityLoss",
    "EqualityOfOpportunityLoss",
    "allocation_parity",
    "equality_of_opportunity",
    "demographic_parity",
    "allocation_efficiency",
    "coverage",
    "compute_all_metrics",
    "AllocationResults",
]

from allocation_framework.models.base import AllocationModel, AllocationResults
from allocation_framework.losses.fairness import (
    FairnessLoss,
    AllocationAwareLoss,
    GroupParityLoss,
    EqualityOfOpportunityLoss,
)
from allocation_framework.metrics.fairness import (
    allocation_parity,
    equality_of_opportunity,
    demographic_parity,
    predictive_parity,
    allocation_efficiency,
    coverage,
    resource_utilization,
    compute_all_metrics,
)
