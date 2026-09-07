"""
Fairness and allocation efficiency metrics.

Metrics for evaluating allocation-aware models beyond standard accuracy.
"""

from typing import Optional, Dict, Tuple
import numpy as np


# ============================================================================
# Allocation Fairness Metrics
# ============================================================================


def allocation_parity(
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> float:
    """
    Allocation parity: equal allocation rates across groups (demographic parity).
    
    Returns 1 when allocation rates are identical, 0 when maximally unequal.
    
    Formula: 1 - |allocation_rate_A - allocation_rate_B|
    
    Parameters:
        y_pred: Binary allocation decisions (0 or 1)
        groups: Group membership (e.g., race, gender)
        
    Returns:
        score: Between 0 (no parity) and 1 (perfect parity)
        
    Example:
        >>> y_pred = np.array([1, 1, 0, 0, 1, 1])
        >>> groups = np.array([0, 0, 0, 1, 1, 1])
        >>> score = allocation_parity(y_pred, groups)
        >>> # Group 0: 2/3 = 0.67, Group 1: 2/3 = 0.67
        >>> # Difference: 0, Score: 1.0 (perfect parity)
    """
    if groups is None:
        return 1.0
    
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    
    unique_groups = np.unique(groups)
    if len(unique_groups) == 1:
        return 1.0
    
    allocation_rates = []
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) > 0:
            rate = np.mean(y_pred[mask])
            allocation_rates.append(rate)
    
    if len(allocation_rates) < 2:
        return 1.0
    
    # Convert max difference to parity score (inverse relationship)
    max_diff = np.max(allocation_rates) - np.min(allocation_rates)
    return float(1.0 - max_diff)


def equality_of_opportunity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> float:
    """
    Equality of opportunity: equal true positive rates across groups.
    
    Ensures the model makes correct allocations at similar rates
    for all groups, focusing on those who should receive allocation.
    
    Formula: 1 - |TPR_A - TPR_B| for y_true=1 cases
    
    Parameters:
        y_true: True allocation needs (1 = needs resource, 0 = doesn't)
        y_pred: Predicted allocations
        groups: Group membership
        
    Returns:
        score: Between 0 (no equality) and 1 (perfect equality)
        
    Example:
        >>> y_true = np.array([1, 1, 0, 0, 1, 1])
        >>> y_pred = np.array([1, 0, 0, 0, 1, 1])
        >>> groups = np.array([0, 0, 0, 0, 1, 1])
        >>> score = equality_of_opportunity(y_true, y_pred, groups)
        >>> # Group 0 (needs): [1,1], predicted [1,0], TPR = 0.5
        >>> # Group 1 (needs): [1,1], predicted [1,1], TPR = 1.0
        >>> # Difference: 0.5, Score: 0.5
    """
    if groups is None or len(np.unique(y_true)) < 2:
        return 1.0
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    
    # Only evaluate on positive class (those who need allocation)
    positive_mask = y_true == 1
    if not np.any(positive_mask):
        return 1.0
    
    unique_groups = np.unique(groups)
    tpr_by_group = []
    
    for group in unique_groups:
        mask = positive_mask & (groups == group)
        if np.sum(mask) > 0:
            tpr = np.mean(y_pred[mask])
            tpr_by_group.append(tpr)
    
    if len(tpr_by_group) < 2:
        return 1.0
    
    # Convert max difference to equality score
    max_tpr_diff = np.max(tpr_by_group) - np.min(tpr_by_group)
    return float(1.0 - max_tpr_diff)


def demographic_parity(
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> float:
    """
    Demographic parity: proportion of allocations same across all groups.
    
    Alias for allocation_parity. Ensures equal allocation rate
    regardless of group membership.
    
    Parameters:
        y_pred: Binary allocation decisions
        groups: Group membership
        
    Returns:
        score: Between 0 and 1
    """
    return allocation_parity(y_pred, groups)


def predictive_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> float:
    """
    Predictive parity: equal positive predictive value (precision) across groups.
    
    When the model allocates resources, it should have similar precision
    (likelihood that allocation is appropriate) across groups.
    
    Formula: 1 - |PPV_A - PPV_B|
    where PPV = TP / (TP + FP)
    
    Parameters:
        y_true: True allocation needs
        y_pred: Predicted allocations
        groups: Group membership
        
    Returns:
        score: Between 0 and 1
    """
    if groups is None or len(np.unique(groups)) == 1:
        return 1.0
    
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
        
        tp = np.sum((y_true_group == 1) & (y_pred_group == 1))
        fp = np.sum((y_true_group == 0) & (y_pred_group == 1))
        
        if tp + fp > 0:
            ppv = tp / (tp + fp)
            ppv_by_group.append(ppv)
    
    if len(ppv_by_group) < 2:
        return 1.0
    
    max_ppv_diff = np.max(ppv_by_group) - np.min(ppv_by_group)
    return float(1.0 - max_ppv_diff)


# ============================================================================
# Allocation Efficiency Metrics
# ============================================================================


def allocation_efficiency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resource_constraint: Optional[int] = None,
) -> float:
    """
    Allocation efficiency: how well we match allocations to actual need.
    
    Measures whether we allocate to those who need it, within constraints.
    High efficiency = allocating to the "right" people.
    
    Formula: TP / (TP + FP) = precision of allocations
    
    Parameters:
        y_true: True allocation needs (1 = needs resource)
        y_pred: Predicted allocations (1 = allocate)
        resource_constraint: Max number of allocations available.
                            If None, ignore constraint.
        
    Returns:
        efficiency: Between 0 and 1, where 1 is perfect efficiency
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # True positives divided by total allocations
    tp = np.sum((y_true == 1) & (y_pred == 1))
    total_allocated = np.sum(y_pred)
    
    if total_allocated == 0:
        return 1.0
    
    efficiency = tp / total_allocated
    
    return float(efficiency)


def coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Coverage: proportion of actual needs that are met by allocations.
    
    Measures whether we're catching all those who need allocation.
    Also known as recall or sensitivity.
    
    Formula: TP / (TP + FN) = recall
    
    Parameters:
        y_true: True allocation needs
        y_pred: Predicted allocations
        
    Returns:
        coverage: Between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    total_need = np.sum(y_true)
    
    if total_need == 0:
        return 1.0
    
    return float(tp / total_need)


def resource_utilization(
    y_pred: np.ndarray,
    resource_constraint: int,
) -> float:
    """
    Resource utilization: what fraction of available resources are used.
    
    Parameters:
        y_pred: Allocation decisions
        resource_constraint: Total available resources
        
    Returns:
        utilization: Between 0 and 1
    """
    y_pred = np.asarray(y_pred)
    total_allocated = np.sum(y_pred)
    
    if resource_constraint <= 0:
        return 0.0
    
    return float(min(1.0, total_allocated / resource_constraint))


def false_positive_rate_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> float:
    """
    False positive rate parity: equal false positive rates across groups.
    
    Ensures unfair allocations (giving to those who don't need it)
    happen at similar rates across groups.
    
    Parameters:
        y_true: True allocation needs
        y_pred: Predicted allocations
        groups: Group membership
        
    Returns:
        score: Between 0 and 1 (1 = perfect parity)
    """
    if groups is None or len(np.unique(y_true)) < 2:
        return 1.0
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    
    # Evaluate on negative class (those who don't need allocation)
    negative_mask = y_true == 0
    if not np.any(negative_mask):
        return 1.0
    
    unique_groups = np.unique(groups)
    fpr_by_group = []
    
    for group in unique_groups:
        mask = negative_mask & (groups == group)
        if np.sum(mask) > 0:
            fpr = np.mean(y_pred[mask])  # False positive = allocation when no need
            fpr_by_group.append(fpr)
    
    if len(fpr_by_group) < 2:
        return 1.0
    
    max_fpr_diff = np.max(fpr_by_group) - np.min(fpr_by_group)
    return float(1.0 - max_fpr_diff)


# ============================================================================
# Composite Metrics
# ============================================================================


def fairness_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
    fairness_weight: float = 0.5,
    fairness_metric: str = "allocation_parity",
) -> float:
    """
    Composite score combining accuracy and fairness.
    
    Score = (1 - fairness_weight) * accuracy + fairness_weight * fairness
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        groups: Group membership
        fairness_weight: How much to weight fairness (0-1)
        fairness_metric: Which fairness metric to use
        
    Returns:
        score: Between 0 and 1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Fairness
    if fairness_metric == "allocation_parity":
        fairness = allocation_parity(y_pred, groups)
    elif fairness_metric == "equality_of_opportunity":
        fairness = equality_of_opportunity(y_true, y_pred, groups)
    elif fairness_metric == "demographic_parity":
        fairness = demographic_parity(y_pred, groups)
    elif fairness_metric == "predictive_parity":
        fairness = predictive_parity(y_true, y_pred, groups)
    elif fairness_metric == "false_positive_rate_parity":
        fairness = false_positive_rate_parity(y_true, y_pred, groups)
    else:
        raise ValueError(f"Unknown fairness_metric: {fairness_metric}")
    
    return float(
        (1 - fairness_weight) * accuracy + fairness_weight * fairness
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
    resource_constraint: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute all fairness and efficiency metrics at once.
    
    Parameters:
        y_true: True labels/needs
        y_pred: Predictions/allocations
        groups: Group membership
        resource_constraint: Available resources
        
    Returns:
        Dictionary with all metric names and values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": allocation_efficiency(y_true, y_pred, resource_constraint),
        "recall": coverage(y_true, y_pred),
        "allocation_parity": allocation_parity(y_pred, groups),
        "equality_of_opportunity": equality_of_opportunity(y_true, y_pred, groups),
        "demographic_parity": demographic_parity(y_pred, groups),
        "predictive_parity": predictive_parity(y_true, y_pred, groups),
        "false_positive_rate_parity": false_positive_rate_parity(y_true, y_pred, groups),
        "allocation_efficiency": allocation_efficiency(y_true, y_pred, resource_constraint),
        "coverage": coverage(y_true, y_pred),
    }
    
    if resource_constraint is not None:
        metrics["resource_utilization"] = resource_utilization(y_pred, resource_constraint)
    
    return metrics


def get_group_fairness_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-group performance metrics.
    
    Parameters:
        y_true: True labels
        y_pred: Predictions
        groups: Group membership
        
    Returns:
        Dictionary mapping group IDs to their metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    
    group_scores = {}
    
    for group in np.unique(groups):
        mask = groups == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        n_samples = np.sum(mask)
        accuracy = np.mean(y_true_group == y_pred_group)
        allocation_rate = np.mean(y_pred_group)
        
        # TPR (sensitivity)
        if np.sum(y_true_group) > 0:
            tpr = np.sum((y_true_group == 1) & (y_pred_group == 1)) / np.sum(y_true_group)
        else:
            tpr = 0.0
        
        # FPR
        if np.sum(y_true_group == 0) > 0:
            fpr = np.sum((y_true_group == 0) & (y_pred_group == 1)) / np.sum(y_true_group == 0)
        else:
            fpr = 0.0
        
        # PPV (precision)
        if np.sum(y_pred_group) > 0:
            ppv = np.sum((y_true_group == 1) & (y_pred_group == 1)) / np.sum(y_pred_group)
        else:
            ppv = 0.0
        
        group_scores[str(group)] = {
            "n_samples": int(n_samples),
            "accuracy": float(accuracy),
            "allocation_rate": float(allocation_rate),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "ppv": float(ppv),
        }
    
    return group_scores
