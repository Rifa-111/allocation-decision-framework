# Architecture Documentation

## Overview

The Allocation Decision Framework implements the research from "Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI" (Ferzana, 2026). It provides a principled, decision-theoretic approach to building ML models for resource allocation in healthcare with explicit fairness constraints.

## Design Philosophy

### Key Principles

1. **Fairness is Not Optional**
   - Fairness is encoded as a first-class objective, not an afterthought
   - Multiple fairness definitions available (parity, equality of opportunity, etc.)
   - Users explicitly specify fairness priorities through weights

2. **Resource Constraints Are Central**
   - Healthcare resources are always limited (beds, staff, equipment, budget)
   - Allocation decisions are zero-sum (one patient's resource = another's denied)
   - Framework explicitly models and optimizes under constraints

3. **Decision-Theoretic Optimization**
   - Rather than single-metric optimization, we optimize multiple objectives
   - Trade-offs are made explicit and quantified
   - Users can explore Pareto frontiers of achievable solutions

4. **Transparency and Explainability**
   - All metrics reported by group (fairness is group-specific)
   - Trade-off analysis shows fairness-accuracy frontier
   - Loss functions and metrics are interpretable

## Architecture

```
allocation-decision-framework/
│
├── models/
│   └── base.py              # AllocationModel (main API)
│       - Wraps base classifier (LogisticRegression)
│       - Handles threshold computation
│       - Orchestrates fairness evaluation
│
├── losses/
│   └── fairness.py          # Fairness-aware loss functions
│       - AllocationAwareLoss (combined objective)
│       - GroupParityLoss
│       - EqualityOfOpportunityLoss
│       - CalibratedLoss
│
├── metrics/
│   └── fairness.py          # Fairness and efficiency metrics
│       - Allocation parity
│       - Equality of opportunity
│       - Coverage, efficiency, utilization
│
└── utils/
    └── helpers.py           # Utility functions
```

## Core Components

### 1. AllocationModel

**Purpose:** Main interface for allocation-aware modeling

**Key Features:**
- Wraps standard classifiers (default: LogisticRegression)
- Computes allocation thresholds from resource constraints or explicit fairness weights
- Evaluates model beyond accuracy (fairness, efficiency, coverage)
- Computes fairness-accuracy trade-off curves

**Parameters:**
```python
AllocationModel(
    fairness_weight=0.5,           # Fairness importance (0-1)
    resource_constraint=None,      # Max allocations available
    fairness_metric='allocation_parity',  # Which fairness to optimize
    efficiency_weight=0.3,         # Efficiency importance (0-1)
    allocation_threshold=None,     # Fixed decision threshold (optional)
    random_state=None,
    verbose=0
)
```

**Workflow:**
```
fit(X_train, y_train, groups=groups_train)
  ↓
  [1] Standardize features
  [2] Train base classifier (LogisticRegression)
  [3] Compute allocation threshold
      - If resource_constraint: use percentile
      - Else if allocation_threshold: use given value
      - Else: use default 0.5
  ↓
predict(X_test)  or  predict_proba(X_test)
  ↓
evaluate(X_test, y_test, groups=groups_test)
  ↓
  [1] Get predictions
  [2] Compute accuracy
  [3] Compute fairness score (based on fairness_metric)
  [4] Compute efficiency and coverage
  [5] Compute per-group metrics
  ↓
returns AllocationResults (accuracy, fairness, efficiency, coverage)
```

### 2. Loss Functions

**Purpose:** Define optimization objectives for training

**AllocationAwareLoss** (Primary)
```
Loss = accuracy_loss + α·fairness_loss + β·efficiency_loss

Where:
- accuracy_loss: Binary cross-entropy (standard ML)
- fairness_loss: Allocation parity across groups (variance of rates)
- efficiency_loss: Deviation from target allocation rate
- α (fairness_weight): How much to prioritize fairness
- β (efficiency_weight): How much to prioritize efficiency
```

**Other Losses:**
- `GroupParityLoss`: Strict demographic parity constraint
- `EqualityOfOpportunityLoss`: Equal TPR across groups
- `CalibratedLoss`: Equal precision across groups

**Note:** Current implementation uses standard LogisticRegression. For custom fairness-aware training, initialize AllocationModel and override `fit()` method.

### 3. Metrics

**Purpose:** Evaluate model fairness and efficiency

**Fairness Metrics:**
- `allocation_parity()`: Equal allocation rates (1 = perfect)
- `equality_of_opportunity()`: Equal TPR for those who need help (1 = perfect)
- `demographic_parity()`: Alias for allocation_parity
- `predictive_parity()`: Equal precision across groups (1 = perfect)
- `false_positive_rate_parity()`: Equal FPR across groups (1 = perfect)

**Efficiency Metrics:**
- `allocation_efficiency()`: Precision of allocations (TP/(TP+FP))
- `coverage()`: Recall (proportion of needs met)
- `resource_utilization()`: Fraction of available resources used

**Composite Metrics:**
- `fairness_accuracy_score()`: Weighted combination of accuracy and fairness
- `compute_all_metrics()`: All metrics at once
- `get_group_fairness_scores()`: Per-group breakdown

## Design Decisions

### Why Standardize Features?

Features are standardized (mean 0, std 1) to:
- Improve logistic regression convergence
- Make threshold selection more robust
- Ensure consistent behavior across datasets

### Why LogisticRegression as Base?

Current implementation uses LogisticRegression because:
- ✓ Interpretable (linear coefficients)
- ✓ Fast training and prediction
- ✓ Built-in class weighting for imbalanced data
- ✓ Good baseline for healthcare (well-studied)

Can be extended to other base learners by subclassing.

### Why Percentile-Based Thresholding?

For resource-constrained settings (most healthcare):
```python
threshold = np.percentile(probabilities, 100 * (1 - K/n))
```

This ensures approximately K allocations are made, addressing:
- Hard constraint: exactly K resources available
- Fairness: threshold is data-driven, not arbitrary

### Why Per-Group Metrics Matter

Single fairness scores hide important variations:
```
Overall Fairness Score: 0.95
But:
  Group A allocation: 95%
  Group B allocation: 05%
  → Not fair despite high overall score!
```

Framework always reports per-group breakdown.

## Fairness Metrics Explained

### Allocation Parity (Demographic Parity)

**Definition:** Each group has equal probability of receiving allocation

**Formula:**
```
Score = 1 - |rate_A - rate_B|

where rate_G = (allocations in group G) / (total in group G)
```

**When to use:**
- Fair treatment = equal treatment
- Trying to increase representation in limited scenarios
- No clear information about differential group needs

**Example:**
```
Group A: 30/50 = 60% allocation
Group B: 30/50 = 60% allocation
Score: 1 - |0.6 - 0.6| = 1.0 (perfect)
```

### Equality of Opportunity

**Definition:** Among those who need allocation, equal proportion receive it across groups

**Formula:**
```
Score = 1 - |TPR_A - TPR_B|

where TPR_G = (correct allocations in group G) / (actual needs in group G)
(only counting true positives)
```

**When to use:**
- Groups may have different prevalence of need
- Want to equally benefit those who actually need help
- More nuanced than demographic parity

**Example:**
```
Among Group A patients needing care:
  30 need care, 21 receive care → TPR = 0.7

Among Group B patients needing care:
  20 need care, 14 receive care → TPR = 0.7

Score: 1 - |0.7 - 0.7| = 1.0 (equal opportunity achieved)
```

### Predictive Parity

**Definition:** Model has similar precision (correctness) across groups

**Formula:**
```
Score = 1 - |PPV_A - PPV_B|

where PPV_G = (true allocations) / (total allocations to group G)
```

**When to use:**
- Concerned about calibration across groups
- Allocations should be equally appropriate across groups
- Fairness = "if you allocate, it's for good reason"

## Trade-Off Analysis

### Fairness-Accuracy Frontier

The framework computes Pareto optimal points:

```python
tradeoff = model.get_fairness_accuracy_tradeoff(
    X_val, y_val, groups=groups_val,
    weight_range=(0.0, 1.0),
    n_steps=11
)

# Returns:
# - weights: fairness weights tested [0, 0.1, ..., 1.0]
# - accuracy: corresponding accuracy scores
# - fairness: corresponding fairness scores
```

**Interpreting the frontier:**

```
         Accuracy
            ↑
          1.0 ├─────────────────────┐
              │    ╱ (accuracy-first)
            0.9│  ╱
              │╱ (balanced)
            0.8├──── (fairness-first)
              │
            0.7└─────────────────────→
              0.5    0.75    1.0    Fairness
```

Points on the frontier cannot be improved on both dimensions simultaneously.

## Extension Points

### Custom Fairness Metrics

Add to `metrics/fairness.py`:

```python
def custom_fairness_metric(y_true, y_pred, groups):
    """Your fairness definition here."""
    # ... computation ...
    return score  # Between 0 and 1
```

### Custom Loss Functions

Subclass `FairnessLoss` in `losses/fairness.py`:

```python
class CustomLoss(FairnessLoss):
    def __call__(self, y_true, y_pred, groups=None):
        return loss_value
```

### Custom Base Learner

Subclass `AllocationModel`:

```python
class CustomAllocationModel(AllocationModel):
    def fit(self, X, y, groups=None, **kwargs):
        # Custom training logic
        self.model_ = YourModel()
        self.model_.fit(X, y)
        return self
```

## Performance Considerations

### Scalability

- **Data:** Tested up to 100K samples
- **Features:** Tested up to 100 features
- **Groups:** Efficient for 2-10 groups; scales to any number

### Computation

```
fit():        O(n·d) - standard logistic regression
predict():    O(n·d) - inference time
evaluate():   O(n·g) - where g = number of groups
```

For large datasets, consider:
- Feature selection/dimensionality reduction
- Batch processing
- Approximate fairness evaluation on sample

## References

**Paper:**
- Ferzana, R. (2026). Beyond Accuracy: A Decision-Theoretic Framework for Allocation-Aware Healthcare AI. In Artificial Intelligence and Intelligent Healthcare, LNCS 16877. Springer.

**Key Concepts:**
- Fairness in machine learning: Barocas et al. (2019)
- Healthcare AI fairness: Mitchell et al. (2021)
- Decision theory: Russell & Norvig (2020)

## Contributing

For architectural changes or new components, see [CONTRIBUTING.md](../CONTRIBUTING.md).
