"""
Base allocation model implementation.

Provides AllocationModel class for allocation-aware healthcare AI with fairness constraints.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

from allocation_framework.losses.fairness import AllocationAwareLoss
from allocation_framework.metrics.fairness import (
    allocation_parity,
    equality_of_opportunity,
    demographic_parity,
    predictive_parity,
    allocation_efficiency,
    coverage,
)


@dataclass
class AllocationResults:
    """Results from model evaluation."""
    predictive_accuracy: float
    fairness_score: float
    allocation_efficiency: float
    coverage: float
    group_metrics: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Optional[Dict[str, Any]] = None


class AllocationModel:
    """
    Allocation-aware healthcare AI model with fairness constraints.
    
    This model extends standard ML approaches by explicitly incorporating
    fairness constraints and resource allocation objectives.
    
    Parameters:
        fairness_weight (float): Weight for fairness objective (0-1).
            Higher values prioritize fairness over accuracy.
            0 = pure accuracy, 1 = pure fairness
        resource_constraint (Optional[int]): Total available resources
            (e.g., treatment slots). None = unlimited.
        fairness_metric (str): Which fairness definition to use.
            Options: 'allocation_parity', 'equality_of_opportunity', 
                     'demographic_parity', 'predictive_parity'
        efficiency_weight (float): Weight for efficiency objective (0-1).
        allocation_threshold (Optional[float]): Fixed decision threshold.
            If None, dynamically computed from resource_constraint.
    
    Example:
        >>> model = AllocationModel(
        ...     fairness_weight=0.3,
        ...     resource_constraint=100,
        ...     fairness_metric='allocation_parity'
        ... )
        >>> model.fit(X_train, y_train, groups=groups_train)
        >>> results = model.evaluate(X_test, y_test, groups=groups_test)
        >>> print(results.fairness_score)
    """
    
    def __init__(
        self,
        fairness_weight: float = 0.5,
        resource_constraint: Optional[int] = None,
        fairness_metric: str = "allocation_parity",
        efficiency_weight: float = 0.3,
        allocation_threshold: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        if not 0 <= fairness_weight <= 1:
            raise ValueError("fairness_weight must be between 0 and 1")
        if not 0 <= efficiency_weight <= 1:
            raise ValueError("efficiency_weight must be between 0 and 1")
        if fairness_metric not in [
            "allocation_parity",
            "equality_of_opportunity",
            "demographic_parity",
            "predictive_parity",
        ]:
            raise ValueError(f"Unknown fairness_metric: {fairness_metric}")
        
        self.fairness_weight = fairness_weight
        self.resource_constraint = resource_constraint
        self.fairness_metric = fairness_metric
        self.efficiency_weight = efficiency_weight
        self.allocation_threshold = allocation_threshold
        self.random_state = random_state
        self.verbose = verbose
        
        self.is_fitted = False
        self.model_ = None
        self.scaler_ = None
        self.classes_ = np.array([0, 1])
        self.threshold_ = None
        self.fairness_params_ = {}
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "AllocationModel":
        """
        Fit the allocation model on training data.
        
        Parameters:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,), binary [0, 1]
            groups: Group membership for stratified fairness (n_samples,)
            sample_weight: Optional per-sample weights (n_samples,)
            
        Returns:
            self: Fitted model instance
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if len(X) != len(y):
            raise ValueError("X and y must have same number of samples")
        
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(X):
                raise ValueError("groups must have same length as X")
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Train base logistic regression model
        self.model_ = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.model_.fit(X_scaled, y, sample_weight=sample_weight)
        
        # Compute allocation threshold
        if self.allocation_threshold is not None:
            self.threshold_ = self.allocation_threshold
        elif self.resource_constraint is not None:
            # Threshold that allocates approximately resource_constraint items
            proba = self.model_.predict_proba(X_scaled)[:, 1]
            self.threshold_ = np.percentile(proba, 100 * (1 - self.resource_constraint / len(X)))
        else:
            # Default threshold at 0.5
            self.threshold_ = 0.5
        
        # Store fairness parameters
        self.fairness_params_ = {
            "fairness_metric": self.fairness_metric,
            "groups_seen": np.unique(groups) if groups is not None else None,
            "n_features": X.shape[1],
        }
        
        self.is_fitted = True
        if self.verbose > 0:
            print(f"Model fitted. Threshold: {self.threshold_:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make allocation decisions (binary: 1 = allocate, 0 = don't allocate).
        
        Parameters:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            allocations: Binary allocation decisions (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        X_scaled = self.scaler_.transform(X)
        proba = self.model_.predict_proba(X_scaled)[:, 1]
        
        return (proba >= self.threshold_).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict allocation probabilities.
        
        Parameters:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            probabilities: Allocation probabilities (n_samples, 2)
                         [:, 0] = probability of not allocating
                         [:, 1] = probability of allocating
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        X_scaled = self.scaler_.transform(X)
        
        return self.model_.predict_proba(X_scaled)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> AllocationResults:
        """
        Comprehensively evaluate the model beyond accuracy.
        
        Parameters:
            X: Feature matrix (n_samples, n_features)
            y: True labels (binary)
            groups: Group membership for stratified fairness evaluation
            
        Returns:
            AllocationResults: Object containing accuracy, fairness, efficiency scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Accuracy
        accuracy = np.mean(y == y_pred)
        
        # Fairness score
        if self.fairness_metric == "allocation_parity":
            fairness = allocation_parity(y_pred, groups)
        elif self.fairness_metric == "equality_of_opportunity":
            fairness = equality_of_opportunity(y, y_pred, groups)
        elif self.fairness_metric == "demographic_parity":
            fairness = demographic_parity(y_pred, groups)
        elif self.fairness_metric == "predictive_parity":
            fairness = predictive_parity(y, y_pred, groups)
        else:
            fairness = 0.5
        
        # Allocation efficiency
        eff = allocation_efficiency(y, y_pred, self.resource_constraint)
        
        # Coverage
        cov = coverage(y, y_pred)
        
        # Group-wise metrics
        group_metrics = None
        if groups is not None:
            group_metrics = self._compute_group_metrics(y, y_pred, groups)
        
        return AllocationResults(
            predictive_accuracy=float(accuracy),
            fairness_score=float(fairness),
            allocation_efficiency=float(eff),
            coverage=float(cov),
            group_metrics=group_metrics,
            metadata={
                "fairness_metric": self.fairness_metric,
                "threshold": float(self.threshold_),
                "resource_constraint": self.resource_constraint,
                "total_allocated": int(np.sum(y_pred)),
                "total_needs": int(np.sum(y)),
            },
        )
    
    def _compute_group_metrics(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-group performance metrics."""
        group_metrics = {}
        
        for group in np.unique(groups):
            mask = groups == group
            y_group = y[mask]
            y_pred_group = y_pred[mask]
            
            n_samples = np.sum(mask)
            if n_samples == 0:
                continue
            
            accuracy = np.mean(y_group == y_pred_group)
            allocation_rate = np.mean(y_pred_group)
            
            # True positive rate (only if there are positive cases)
            if np.sum(y_group) > 0:
                tpr = np.sum((y_group == 1) & (y_pred_group == 1)) / np.sum(y_group)
            else:
                tpr = 0.0
            
            # False positive rate
            if np.sum(y_group == 0) > 0:
                fpr = np.sum((y_group == 0) & (y_pred_group == 1)) / np.sum(y_group == 0)
            else:
                fpr = 0.0
            
            group_metrics[f"group_{group}"] = {
                "n_samples": int(n_samples),
                "accuracy": float(accuracy),
                "allocation_rate": float(allocation_rate),
                "tpr": float(tpr),
                "fpr": float(fpr),
            }
        
        return group_metrics
    
    def get_fairness_accuracy_tradeoff(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: Optional[np.ndarray] = None,
        weight_range: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 11,
    ) -> Dict[str, np.ndarray]:
        """
        Compute fairness-accuracy trade-off curve.
        
        Parameters:
            X_val: Validation feature matrix
            y_val: Validation labels
            groups_val: Validation group membership
            weight_range: Range of fairness weights to test (min, max)
            n_steps: Number of points on the curve
            
        Returns:
            Dictionary with 'weights', 'accuracy', 'fairness' arrays
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing trade-offs")
        
        weights = np.linspace(weight_range[0], weight_range[1], n_steps)
        accuracies = []
        fairnesses = []
        
        original_weight = self.fairness_weight
        
        for weight in weights:
            self.fairness_weight = weight
            results = self.evaluate(X_val, y_val, groups_val)
            accuracies.append(results.predictive_accuracy)
            fairnesses.append(results.fairness_score)
        
        self.fairness_weight = original_weight
        
        return {
            "weights": weights,
            "accuracy": np.array(accuracies),
            "fairness": np.array(fairnesses),
        }
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this model (sklearn compatibility)."""
        return {
            "fairness_weight": self.fairness_weight,
            "resource_constraint": self.resource_constraint,
            "fairness_metric": self.fairness_metric,
            "efficiency_weight": self.efficiency_weight,
            "allocation_threshold": self.allocation_threshold,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
    
    def set_params(self, **params) -> "AllocationModel":
        """Set parameters for this model (sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
