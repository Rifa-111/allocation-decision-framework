"""Experimental allocation evaluation utilities (research use only).

The fitted AllocationModel is a logistic-regression baseline. Fairness and
resource-efficiency weights do not alter its training or predictions. The
package is not validated for clinical decisions.
"""

__version__ = "0.1.0"
__author__ = "Rifa Ferzana"
__license__ = "MIT"

from allocation_framework.models.base import AllocationModel, AllocationResults
from allocation_framework.losses.fairness import (
    FairnessLoss, AllocationAwareLoss, GroupParityLoss, EqualityOfOpportunityLoss,
)
from allocation_framework.metrics.fairness import (
    allocation_parity, equality_of_opportunity, demographic_parity,
    predictive_parity, allocation_efficiency, coverage,
    resource_utilization, compute_all_metrics,
)

__all__ = [
    "AllocationModel", "AllocationResults", "FairnessLoss", "AllocationAwareLoss",
    "GroupParityLoss", "EqualityOfOpportunityLoss", "allocation_parity",
    "equality_of_opportunity", "demographic_parity", "predictive_parity",
    "allocation_efficiency", "coverage", "resource_utilization",
    "compute_all_metrics",
]
