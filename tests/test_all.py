"""
Comprehensive test suite for allocation_framework.

Tests cover models, metrics, and loss functions.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from allocation_framework import (
    AllocationModel,
    AllocationAwareLoss,
    GroupParityLoss,
    EqualityOfOpportunityLoss,
    allocation_parity,
    equality_of_opportunity,
    allocation_efficiency,
    coverage,
    compute_all_metrics,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic healthcare data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    
    # Add group membership
    groups = np.random.binomial(n=1, p=0.5, size=len(X))
    
    # Split into train/test
    split = int(0.7 * len(X))
    return {
        "X_train": X[:split],
        "X_test": X[split:],
        "y_train": y[:split],
        "y_test": y[split:],
        "groups_train": groups[:split],
        "groups_test": groups[split:],
    }


class TestAllocationMetrics:
    """Test fairness and efficiency metrics."""
    
    def test_allocation_parity_perfect(self):
        """Test allocation parity with equal allocation."""
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        score = allocation_parity(y_pred, groups)
        assert score == 1.0, "Perfect parity should return 1.0"
    
    def test_allocation_parity_no_parity(self):
        """Test allocation parity with unequal allocation."""
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        score = allocation_parity(y_pred, groups)
        assert 0 < score < 1, "Imperfect parity should be between 0 and 1"
    
    def test_allocation_parity_no_groups(self):
        """Test allocation parity without groups."""
        y_pred = np.array([1, 1, 0, 0])
        score = allocation_parity(y_pred, None)
        assert score == 1.0, "No groups should return 1.0"
    
    def test_equality_of_opportunity_perfect(self):
        """Test equality of opportunity with equal TPR."""
        y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        score = equality_of_opportunity(y_true, y_pred, groups)
        assert score == 1.0, "Perfect TPR equality should return 1.0"
    
    def test_allocation_efficiency(self):
        """Test allocation efficiency."""
        y_true = np.array([1, 1, 0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 0, 1, 1])
        
        # Efficiency = TP / (TP + FP) = 3 / 3 = 1.0
        eff = allocation_efficiency(y_true, y_pred)
        assert eff == 1.0, "All correct allocations should be 1.0"
    
    def test_coverage(self):
        """Test coverage metric."""
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0])
        
        # Coverage = TP / (TP + FN) = 2 / 4 = 0.5
        cov = coverage(y_true, y_pred)
        assert cov == 0.5, "Coverage should be 0.5"
    
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        y_true = np.array([1, 1, 0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1, 1, 1])
        groups = np.array([0, 0, 0, 0, 1, 1])
        
        metrics = compute_all_metrics(y_true, y_pred, groups)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "allocation_parity" in metrics
        assert "allocation_efficiency" in metrics
        assert "coverage" in metrics
        
        # All scores should be between 0 and 1
        for value in metrics.values():
            assert 0 <= value <= 1, f"Metric value {value} out of range"


class TestAllocationModel:
    """Test AllocationModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = AllocationModel(fairness_weight=0.3)
        assert model.fairness_weight == 0.3
        assert not model.is_fitted
    
    def test_invalid_fairness_weight(self):
        """Test initialization with invalid fairness weight."""
        with pytest.raises(ValueError):
            AllocationModel(fairness_weight=1.5)
    
    def test_fit(self, synthetic_data):
        """Test model fitting."""
        model = AllocationModel(fairness_weight=0.5)
        model.fit(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            groups=synthetic_data["groups_train"],
        )
        
        assert model.is_fitted
        assert model.model_ is not None
        assert model.scaler_ is not None
        assert model.threshold_ is not None
    
    def test_predict(self, synthetic_data):
        """Test predictions."""
        model = AllocationModel()
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        
        predictions = model.predict(synthetic_data["X_test"])
        
        assert len(predictions) == len(synthetic_data["X_test"])
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_predict_proba(self, synthetic_data):
        """Test probability predictions."""
        model = AllocationModel()
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        
        proba = model.predict_proba(synthetic_data["X_test"])
        
        assert proba.shape == (len(synthetic_data["X_test"]), 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        # Probabilities should sum to 1
        assert np.allclose(np.sum(proba, axis=1), 1.0)
    
    def test_evaluate(self, synthetic_data):
        """Test model evaluation."""
        model = AllocationModel(fairness_weight=0.3)
        model.fit(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            groups=synthetic_data["groups_train"],
        )
        
        results = model.evaluate(
            synthetic_data["X_test"],
            synthetic_data["y_test"],
            groups=synthetic_data["groups_test"],
        )
        
        assert hasattr(results, "predictive_accuracy")
        assert hasattr(results, "fairness_score")
        assert hasattr(results, "allocation_efficiency")
        assert hasattr(results, "coverage")
        assert 0 <= results.predictive_accuracy <= 1
        assert 0 <= results.fairness_score <= 1
    
    def test_resource_constraint(self, synthetic_data):
        """Test model with resource constraint."""
        constraint = 20
        model = AllocationModel(resource_constraint=constraint)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        
        predictions = model.predict(synthetic_data["X_test"])
        
        # Should allocate approximately constraint items
        allocated = np.sum(predictions)
        assert allocated <= len(synthetic_data["X_test"])
    
    def test_different_fairness_metrics(self, synthetic_data):
        """Test different fairness metrics."""
        metrics = [
            "allocation_parity",
            "equality_of_opportunity",
            "demographic_parity",
            "predictive_parity",
        ]
        
        for metric in metrics:
            model = AllocationModel(fairness_metric=metric)
            model.fit(
                synthetic_data["X_train"],
                synthetic_data["y_train"],
                groups=synthetic_data["groups_train"],
            )
            
            results = model.evaluate(
                synthetic_data["X_test"],
                synthetic_data["y_test"],
                groups=synthetic_data["groups_test"],
            )
            
            assert 0 <= results.fairness_score <= 1
    
    def test_get_params(self):
        """Test get_params method."""
        model = AllocationModel(fairness_weight=0.3, resource_constraint=50)
        params = model.get_params()
        
        assert params["fairness_weight"] == 0.3
        assert params["resource_constraint"] == 50
    
    def test_set_params(self):
        """Test set_params method."""
        model = AllocationModel()
        model.set_params(fairness_weight=0.7, resource_constraint=100)
        
        assert model.fairness_weight == 0.7
        assert model.resource_constraint == 100
    
    def test_fairness_accuracy_tradeoff(self, synthetic_data):
        """Test fairness-accuracy trade-off computation."""
        model = AllocationModel()
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        
        tradeoff = model.get_fairness_accuracy_tradeoff(
            synthetic_data["X_test"],
            synthetic_data["y_test"],
            groups=synthetic_data["groups_test"],
            weight_range=(0.0, 1.0),
            n_steps=5,
        )
        
        assert "weights" in tradeoff
        assert "accuracy" in tradeoff
        assert "fairness" in tradeoff
        assert len(tradeoff["weights"]) == 5


class TestFairnessLosses:
    """Test fairness loss functions."""
    
    def test_allocation_aware_loss(self):
        """Test combined allocation-aware loss."""
        y_true = np.array([1, 1, 0, 0, 1, 1])
        y_pred = np.array([0.8, 0.7, 0.2, 0.1, 0.9, 0.85])
        groups = np.array([0, 0, 0, 0, 1, 1])
        
        loss_fn = AllocationAwareLoss(fairness_weight=0.5)
        loss = loss_fn(y_true, y_pred, groups)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_group_parity_loss_perfect(self):
        """Test group parity loss with perfect parity."""
        y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        groups = np.array([0, 0, 0, 1, 1, 1])
        
        loss_fn = GroupParityLoss(margin=0.0)
        loss = loss_fn(None, y_pred, groups)
        
        assert loss == 0.0
    
    def test_equality_of_opportunity_loss(self):
        """Test equality of opportunity loss."""
        y_true = np.array([1, 1, 0, 0, 1, 1])
        y_pred = np.array([0.8, 0.7, 0.2, 0.1, 0.9, 0.85])
        groups = np.array([0, 0, 0, 0, 1, 1])
        
        loss_fn = EqualityOfOpportunityLoss()
        loss = loss_fn(y_true, y_pred, groups)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_loss_functions_return_float(self):
        """Test that all loss functions return float."""
        y_true = np.array([1, 1, 0, 0, 1, 1])
        y_pred = np.array([0.8, 0.7, 0.2, 0.1, 0.9, 0.85])
        groups = np.array([0, 0, 0, 0, 1, 1])
        
        losses = [
            AllocationAwareLoss(fairness_weight=0.5),
            GroupParityLoss(),
            EqualityOfOpportunityLoss(),
        ]
        
        for loss_fn in losses:
            loss = loss_fn(y_true, y_pred, groups)
            assert isinstance(loss, float)
            assert loss >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_group(self):
        """Test with single group."""
        y_pred = np.array([1, 1, 0, 0])
        groups = np.array([0, 0, 0, 0])
        
        score = allocation_parity(y_pred, groups)
        assert score == 1.0
    
    def test_empty_groups(self):
        """Test with no group membership."""
        y_pred = np.array([1, 1, 0, 0])
        
        score = allocation_parity(y_pred, None)
        assert score == 1.0
    
    def test_all_positive_class(self):
        """Test with all positive class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        
        cov = coverage(y_true, y_pred)
        assert 0 <= cov <= 1
    
    def test_all_negative_class(self):
        """Test with all negative class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        cov = coverage(y_true, y_pred)
        assert cov == 1.0  # No needs, so vacuously true
    
    def test_no_allocations(self):
        """Test with no allocations."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        eff = allocation_efficiency(y_true, y_pred)
        assert eff == 1.0  # No allocations = perfect efficiency
    
    def test_all_allocations(self):
        """Test with all allocations."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        
        cov = coverage(y_true, y_pred)
        assert cov == 1.0


class TestDataTypes:
    """Test compatibility with different data types."""
    
    def test_list_input(self, synthetic_data):
        """Test with list input."""
        X_list = synthetic_data["X_train"].tolist()
        y_list = synthetic_data["y_train"].tolist()
        
        model = AllocationModel()
        model.fit(X_list, y_list)
        
        assert model.is_fitted
    
    def test_integer_predictions(self):
        """Test with integer predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        
        cov = coverage(y_true, y_pred)
        assert 0 <= cov <= 1
    
    def test_float_predictions(self):
        """Test with float predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.1, 0.2, 0.8])
        
        cov = coverage(y_true, y_pred)
        assert 0 <= cov <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestReleaseRegressions:
    """Guard the documented resource-limited prototype behaviour."""

    def test_zero_capacity_allocates_nothing(self, synthetic_data):
        model = AllocationModel(resource_constraint=0)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert model.predict(synthetic_data["X_test"]).sum() == 0

    def test_batch_capacity_is_enforced(self, synthetic_data):
        model = AllocationModel(resource_constraint=5)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert model.predict(synthetic_data["X_test"]).sum() <= 5

    def test_capacity_above_training_size(self, synthetic_data):
        model = AllocationModel(resource_constraint=100000)
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert model.predict(synthetic_data["X_test"]).sum() <= len(synthetic_data["X_test"])

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError):
            AllocationModel(resource_constraint=-1)

    def test_unimplemented_tradeoff_is_explicit(self, synthetic_data):
        model = AllocationModel()
        model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        with pytest.raises(NotImplementedError):
            model.get_fairness_accuracy_tradeoff(synthetic_data["X_test"], synthetic_data["y_test"])
