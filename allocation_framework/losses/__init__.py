"""Fairness-aware loss functions for allocation optimization."""

from allocation_framework.losses.fairness import (
    FairnessLoss,
    AllocationAwareLoss,
    GroupParityLoss,
    EqualityOfOpportunityLoss,
    CalibratedLoss,
)

__all__ = [
    "FairnessLoss",
    "AllocationAwareLoss",
    "GroupParityLoss",
    "EqualityOfOpportunityLoss",
    "CalibratedLoss",
]
