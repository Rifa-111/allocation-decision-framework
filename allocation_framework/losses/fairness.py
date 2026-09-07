"""
Fairness-aware loss functions for allocation-aware training.

These loss functions extend standard ML objectives to incorporate
fairness constraints and allocation efficiency goals.
"""

from typing import Optional, Callable
import numpy as np
from abc import ABC, abstractmethod


class FairnessLoss(ABC):
    """
    Abstract base class for fairness-aware loss functions.
    
    Fairness losses encode constraints about how allocations should be
    distributed across groups or individuals.
    """
    
    @abstractmethod
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the fairness loss.
        
        Parameters:
            y_true: True labels (binary)
            y_pred: Predicted labels/probabilities
            groups: Group membership for stratified evaluation
            
        Returns:
            loss: Scalar loss value (non-negative)
        """
        pass


class AllocationAwareLoss(FairnessLoss):
    """
    Combined loss for accuracy + fairness + allocation efficiency.
    
    Total Loss = accuracy_loss + α·fairness_loss + β·efficiency_loss
    
    Where:
    - α (fairness_weight) controls fairness priority (default 0.5)
    - β (efficiency_weight) controls resource efficiency priority (default 0.3)
    """
    
    def __init__(
        self,
        accuracy_fn: Optional[Callable] = None,
        fairness_weight: float = 0.5,
        efficiency_weight: float = 0.3,
        fairness_fn: Optional[Callable] = None,
        efficiency_fn: Optional[Callable] = None,
    ):
        """
        Initialize the combined loss.
        
        Parameters:
            accuracy_fn: Standard ML loss (e.g., cross-entropy).
                        If None, uses binary cross-entropy.
            fairness_weight: Weight for fairness objective (0-1)
            efficiency_weight: Weight for efficiency objective (0-1)
            fairness_fn: Custom fairness loss. If None, uses allocation parity.
            efficiency_fn: Custom efficiency loss. If None, uses default.
        """
        self.accuracy_fn = accuracy_fn or self._default_accuracy_loss
        self.fairness_weight = fairness_weight
        self.efficiency_weight = efficiency_weight
        self.fairness_fn = fairness_fn or self._default_fairness_loss
        self.efficiency_fn = efficiency_fn or self._default_efficiency_loss
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the combined loss."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Ensure predictions are probabilities (between 0 and 1)
        if y_pred.min() < 0 or y_pred.max() > 1:
            # Assume logits, apply sigmoid
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        
        accuracy_loss = self.accuracy_fn(y_true, y_pred)
        fairness_loss = self.fairness_fn(y_true, y_pred, groups)
        efficiency_loss = self.efficiency_fn(y_pred, groups)
        
        total_loss = (
            accuracy_loss
            + self.fairness_weight * fairness_loss
            + self.efficiency_weight * efficiency_loss
        )
        
        return float(total_loss)
    
    @staticmethod
    def _default_accuracy_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Binary cross-entropy loss."""
        y_true = np.clip(y_true, 0, 1)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.mean(bce))
    
    @staticmethod
    def _default_fairness_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """
        Default fairness loss: allocation parity across groups.
        
        Penalizes differences in allocation rates between groups.
        """
        if groups is None:
            return 0.0
        
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return 0.0
        
        allocation_rates = []
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) > 0:
                rate = np.mean(y_pred[mask])
                allocation_rates.append(rate)
        
        # Variance in allocation rates (loss increases with variance)
        return float(np.var(allocation_rates))
    
    @staticmethod
    def _default_efficiency_loss(
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """
        Default efficiency loss: penalize deviation from target allocation.
        
        Targets 50% allocation rate.
        """
        target_rate = 0.5
        actual_rate = np.mean(y_pred)
        return float((actual_rate - target_rate) ** 2)


class GroupParityLoss(FairnessLoss):
    """
    Loss enforcing demographic parity (equal allocation across groups).
    
    Penalizes differences in allocation rates between groups.
    Useful when fair treatment means equal treatment.
    
    Loss = max(0, max_diff - margin)^2
    where max_diff = max allocation rate - min allocation rate
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Parameters:
            margin: Tolerance for allocation rate differences.
                    Difference > margin incurs loss.
        """
        self.margin = margin
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """Compute group parity loss."""
        if groups is None:
            return 0.0
        
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)
        
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return 0.0
        
        allocation_rates = []
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) > 0:
                rate = np.mean(y_pred[mask])
                allocation_rates.append(rate)
        
        # Max difference between groups
        max_diff = np.max(allocation_rates) - np.min(allocation_rates)
        
        # Penalize if difference exceeds margin
        return float(max(0, max_diff - self.margin) ** 2)


class EqualityOfOpportunityLoss(FairnessLoss):
    """
    Loss enforcing equality of opportunity (equal true positive rates).
    
    Ensures that the model makes correct allocations at similar rates
    across groups, even if overall allocation rates differ.
    
    Loss = max(0, max_tpr_diff - margin)^2
    where max_tpr_diff = max TPR across groups - min TPR across groups
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Parameters:
            margin: Tolerance for TPR differences across groups.
        """
        self.margin = margin
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """Compute equality of opportunity loss."""
        if groups is None:
            return 0.0
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)
        
        # Only evaluate on positive class
        positive_mask = y_true == 1
        if not np.any(positive_mask):
            return 0.0
        
        unique_groups = np.unique(groups)
        tpr_by_group = []
        
        for group in unique_groups:
            mask = positive_mask & (groups == group)
            if np.sum(mask) > 0:
                tpr = np.mean(y_pred[mask])
                tpr_by_group.append(tpr)
        
        if len(tpr_by_group) < 2:
            return 0.0
        
        # Max TPR difference across groups
        max_tpr_diff = np.max(tpr_by_group) - np.min(tpr_by_group)
        
        # Penalize if exceeds margin
        return float(max(0, max_tpr_diff - self.margin) ** 2)


class CalibratedLoss(FairnessLoss):
    """
    Loss enforcing calibration across groups.
    
    Among those predicted to receive allocation, similar proportions
    should actually need it across groups.
    
    Loss = max(0, max_ppv_diff - margin)^2
    where PPV = positive predictive value (precision)
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Parameters:
            margin: Tolerance for PPV differences across groups.
        """
        self.margin = margin
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> float:
        """Compute calibration loss."""
        if groups is None:
            return 0.0
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)
        
        unique_groups = np.unique(groups)
        ppv_by_group = []
        
        for group in unique_groups:
            mask = groups == group
            if np.sum(mask) == 0:
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # PPV: among predicted positives, how many are true positives
            predicted_positive = y_pred_group >= 0.5
            if np.sum(predicted_positive) > 0:
                tp = np.sum((y_true_group == 1) & predicted_positive)
                ppv = tp / np.sum(predicted_positive)
                ppv_by_group.append(ppv)
        
        if len(ppv_by_group) < 2:
            return 0.0
        
        max_ppv_diff = np.max(ppv_by_group) - np.min(ppv_by_group)
        return float(max(0, max_ppv_diff - self.margin) ** 2)
