"""
Tutorial 01: Introduction to Allocation-Aware Healthcare AI

Comprehensive tutorial on allocation-aware AI, fairness metrics, and the 
fairness-accuracy trade-off. Based on "Beyond Accuracy: A Decision-Theoretic 
Framework for Allocation-Aware Healthcare AI" (Ferzana, 2026).

Key Learning Outcomes:
- Why accuracy alone is insufficient for healthcare resource allocation
- Multiple fairness definitions and when to use each
- The fairness-accuracy trade-off and how to navigate it
- Using AllocationModel for decision-theoretic optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# For real usage: from allocation_framework import AllocationModel, allocation_parity
# For this demo, we'll show concepts with direct implementations


print("=" * 80)
print("TUTORIAL 01: Introduction to Allocation-Aware Healthcare AI")
print("=" * 80)

# =============================================================================
# SECTION 1: The Problem - Why Accuracy Isn't Enough
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 1: The Problem - Why Accuracy Isn't Enough")
print("=" * 80)

print("""
Healthcare AI for resource allocation requires balancing multiple objectives:

1. ACCURACY: Model predictions match expert judgment (standard ML metric)
2. FAIRNESS: Resource allocation is equitable across demographic groups
3. EFFICIENCY: Limited resources are allocated to those who need them most

PROBLEM: Standard accuracy-first models can achieve high accuracy while 
being deeply unfair. Both models below have 85% accuracy but very different 
fairness properties.

EXAMPLE: Emergency Department Triage (100 patients, 50 available beds)
""")

# Simulated ER allocation scenarios
print("\n" + "-" * 80)
print("Scenario 1: Accuracy-First Allocation (85% accuracy)")
print("-" * 80)

scenario_1 = {
    "accuracy": 0.85,
    "group_a_allocation": 48,  # 96% of Group A
    "group_b_allocation": 2,   # 4% of Group B
    "group_a_beds": 50,
    "group_b_beds": 50,
}

print(f"✓ Overall accuracy: {scenario_1['accuracy']:.0%}")
print(f"  Beds allocated: {scenario_1['group_a_allocation']} to Group A, "
      f"{scenario_1['group_b_allocation']} to Group B")
print(f"  Group A allocation rate: {scenario_1['group_a_allocation']/scenario_1['group_a_beds']:.0%}")
print(f"  Group B allocation rate: {scenario_1['group_b_allocation']/scenario_1['group_b_beds']:.0%}")
print(f"  ✗ Fairness Problem: Only {scenario_1['group_b_allocation']/scenario_1['group_b_beds']:.0%} "
      f"of Group B get beds vs "
      f"{scenario_1['group_a_allocation']/scenario_1['group_a_beds']:.0%} of Group A")

print("\n" + "-" * 80)
print("Scenario 2: Allocation-Aware Model (82% accuracy)")
print("-" * 80)

scenario_2 = {
    "accuracy": 0.82,
    "group_a_allocation": 25,  # 50% of Group A
    "group_b_allocation": 25,  # 50% of Group B
    "group_a_beds": 50,
    "group_b_beds": 50,
}

print(f"✓ Overall accuracy: {scenario_2['accuracy']:.0%}")
print(f"  Beds allocated: {scenario_2['group_a_allocation']} to Group A, "
      f"{scenario_2['group_b_allocation']} to Group B")
print(f"  Group A allocation rate: {scenario_2['group_a_allocation']/scenario_2['group_a_beds']:.0%}")
print(f"  Group B allocation rate: {scenario_2['group_b_allocation']/scenario_2['group_b_beds']:.0%}")
print(f"  ✓ Fairness Benefit: Equal allocation across groups")

print("\n" + "-" * 80)
print("The Trade-Off")
print("-" * 80)

accuracy_delta = (scenario_1['accuracy'] - scenario_2['accuracy']) * 100
fairness_delta = ((scenario_2['group_b_allocation']/scenario_2['group_b_beds']) - 
                  (scenario_1['group_b_allocation']/scenario_1['group_b_beds'])) * 100

print(f"Cost: {accuracy_delta:.1f}pp decrease in accuracy")
print(f"Benefit: {fairness_delta:.0f}pp increase in allocation equity for disadvantaged group")
print(f"\nThis is the decision practitioners must make explicitly.")


# =============================================================================
# SECTION 2: Create Synthetic Healthcare Data
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 2: Creating Synthetic Healthcare Data")
print("=" * 80)

np.random.seed(42)
n_samples = 500

print("\nGenerating synthetic patient data...")
X, y = make_classification(
    n_samples=n_samples,
    n_features=10,
    n_informative=5,
    n_redundant=3,
    n_clusters_per_class=2,
    random_state=42
)

# Create demographic groups (simplified for demo)
groups = np.random.binomial(n=1, p=0.5, size=n_samples)

# Split data
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.3, random_state=42
)

print(f"\nDataset Statistics:")
print(f"  Total samples: {n_samples}")
print(f"  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")
print(f"  Features: {X.shape[1]} patient characteristics")
print(f"  Group distribution (training):")
print(f"    Group 0: {np.sum(groups_train == 0)} ({np.mean(groups_train == 0):.0%})")
print(f"    Group 1: {np.sum(groups_train == 1)} ({np.mean(groups_train == 1):.0%})")
print(f"  Positive class rate: {np.mean(y_train):.0%}")


# =============================================================================
# SECTION 3: Fairness Metrics Explained
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 3: Fairness Metrics for Allocation")
print("=" * 80)

def allocation_parity(y_pred, groups):
    """Equal allocation rates across groups."""
    if groups is None:
        return 1.0
    unique_groups = np.unique(groups)
    rates = [np.mean(y_pred[groups == g]) for g in unique_groups if np.sum(groups == g) > 0]
    if len(rates) < 2:
        return 1.0
    return 1.0 - (np.max(rates) - np.min(rates))


def equality_of_opportunity(y_true, y_pred, groups):
    """Equal TPR across groups for those who need allocation."""
    if groups is None:
        return 1.0
    positive_mask = y_true == 1
    if not np.any(positive_mask):
        return 1.0
    unique_groups = np.unique(groups)
    tprs = []
    for g in unique_groups:
        mask = (groups == g) & positive_mask
        if np.sum(mask) > 0:
            tpr = np.mean(y_pred[mask])
            tprs.append(tpr)
    if len(tprs) < 2:
        return 1.0
    return 1.0 - (np.max(tprs) - np.min(tprs))


print("""
FAIRNESS METRIC 1: Allocation Parity (Demographic Parity)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Each group has equal probability of receiving allocation
Formula: 1 - |allocation_rate_Group_A - allocation_rate_Group_B|

When to use:
  ✓ Fair treatment means "equal treatment" 
  ✓ Trying to increase representation in limited resource scenarios
  ✗ Not suitable when groups have different actual needs

Example:
  Group A allocation: 60% of patients get beds
  Group B allocation: 60% of patients get beds
  → Parity score: 1.0 (perfect equality)


FAIRNESS METRIC 2: Equality of Opportunity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: Equal success rates for those who actually need allocation
Formula: 1 - |TPR_Group_A - TPR_Group_B| (where y_true=1)

When to use:
  ✓ Groups may have different prevalence (e.g., disease rates)
  ✓ Focus on equally benefiting those who need help
  ✓ More nuanced fairness when needs differ

Example:
  Among Group A patients who need beds: 70% get them (TPR=0.7)
  Among Group B patients who need beds: 70% get them (TPR=0.7)
  → Equal opportunity score: 1.0 (fair!)
""")


# =============================================================================
# SECTION 4: Simulate Allocation Strategies
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 4: Comparing Allocation Strategies")
print("=" * 80)

from sklearn.linear_model import LogisticRegression

print("\nTraining base model...")
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Get probability predictions
y_proba_test = clf.predict_proba(X_test)[:, 1]

print("\n" + "-" * 80)
print("STRATEGY 1: Accuracy-First (threshold at 0.5)")
print("-" * 80)

y_pred_acc = (y_proba_test >= 0.5).astype(int)

acc_1 = np.mean(y_test == y_pred_acc)
parity_1 = allocation_parity(y_pred_acc, groups_test)
eq_opp_1 = equality_of_opportunity(y_test, y_pred_acc, groups_test)

# Group rates
group_0_rate_1 = np.mean(y_pred_acc[groups_test == 0])
group_1_rate_1 = np.mean(y_pred_acc[groups_test == 1])

print(f"Predictive Accuracy:   {acc_1:.3f}")
print(f"Allocation Parity:     {parity_1:.3f}")
print(f"Equality of Opportunity: {eq_opp_1:.3f}")
print(f"Allocation rates:")
print(f"  Group 0: {group_0_rate_1:.1%}")
print(f"  Group 1: {group_1_rate_1:.1%}")
print(f"  Difference: {abs(group_0_rate_1 - group_1_rate_1):.1%}")

print("\n" + "-" * 80)
print("STRATEGY 2: Fair Allocation (50th percentile threshold)")
print("-" * 80)

# Allocate top 50% of patients (matches resource constraint)
percentile_threshold = np.percentile(y_proba_test, 50)
y_pred_fair = (y_proba_test >= percentile_threshold).astype(int)

acc_2 = np.mean(y_test == y_pred_fair)
parity_2 = allocation_parity(y_pred_fair, groups_test)
eq_opp_2 = equality_of_opportunity(y_test, y_pred_fair, groups_test)

# Group rates
group_0_rate_2 = np.mean(y_pred_fair[groups_test == 0])
group_1_rate_2 = np.mean(y_pred_fair[groups_test == 1])

print(f"Predictive Accuracy:   {acc_2:.3f}")
print(f"Allocation Parity:     {parity_2:.3f}")
print(f"Equality of Opportunity: {eq_opp_2:.3f}")
print(f"Allocation rates:")
print(f"  Group 0: {group_0_rate_2:.1%}")
print(f"  Group 1: {group_1_rate_2:.1%}")
print(f"  Difference: {abs(group_0_rate_2 - group_1_rate_2):.1%}")

print("\n" + "-" * 80)
print("SUMMARY: The Trade-Off")
print("-" * 80)

print(f"Accuracy Delta:        {acc_1 - acc_2:+.3f} ({(acc_1-acc_2)*100:+.1f}pp)")
print(f"Parity Improvement:    {parity_2 - parity_1:+.3f}")
print(f"EqOpp Improvement:     {eq_opp_2 - eq_opp_1:+.3f}")
print(f"Fairness Gain:         {abs(group_0_rate_2 - group_1_rate_2) - abs(group_0_rate_1 - group_1_rate_1):.1%} "
      f"decrease in group difference")


# =============================================================================
# SECTION 5: Key Insights
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 5: Key Insights from 'Beyond Accuracy'")
print("=" * 80)

print("""
1. STANDARD ACCURACY IS INSUFFICIENT
   High accuracy doesn't guarantee fair allocation. The model can correctly
   predict who needs treatment but systematically deny certain groups.

2. MULTIPLE FAIRNESS DEFINITIONS EXIST
   • Allocation Parity: Equal treatment across groups
   • Equality of Opportunity: Equal benefit for those who need help
   • Predictive Parity: Equal precision across groups
   • False Positive Rate Parity: Equal "false alarms" across groups
   
   The "right" fairness metric depends on your values and constraints.

3. FAIRNESS-ACCURACY TRADE-OFFS ARE REAL BUT MANAGEABLE
   In real healthcare:
   • Accuracy losses from fairness constraints are often small
   • The fairness gains are substantial for disadvantaged groups
   • The trade-off is worth making explicitly

4. DECISION-THEORETIC APPROACH IS PRINCIPLED
   Instead of ad-hoc threshold tweaking, we should:
   • Define explicit objectives (accuracy, fairness, efficiency)
   • Assign weights based on values and constraints
   • Compute Pareto frontiers of achievable solutions
   • Choose based on informed trade-offs

5. RESOURCE CONSTRAINTS MAKE FAIRNESS CRITICAL
   When resources are limited (which they always are in healthcare):
   • Allocation decisions are zero-sum (one patient's bed = another's denied)
   • Unfair allocation can cause real harm
   • Fairness-aware optimization is not a luxury, it's an ethical necessity
""")


# =============================================================================
# SECTION 6: Using AllocationModel
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 6: Using the AllocationModel Framework")
print("=" * 80)

print("""
The allocation-decision-framework provides an integrated approach:

from allocation_framework import AllocationModel

# Initialize with fairness objective
model = AllocationModel(
    fairness_weight=0.3,           # 30% fairness, 70% accuracy
    resource_constraint=50,        # Only 50 allocations available
    fairness_metric='allocation_parity'
)

# Train with your data
model.fit(X_train, y_train, groups=groups_train)

# Evaluate comprehensively
results = model.evaluate(X_test, y_test, groups=groups_test)

print(f"Accuracy:       {results.predictive_accuracy:.3f}")
print(f"Fairness:       {results.fairness_score:.3f}")
print(f"Efficiency:     {results.allocation_efficiency:.3f}")
print(f"Coverage:       {results.coverage:.3f}")

# Explore fairness-accuracy trade-off
tradeoff = model.get_fairness_accuracy_tradeoff(
    X_test, y_test, groups=groups_test,
    weight_range=(0.0, 1.0), n_steps=11
)

# Plot to see the frontier
plt.plot(tradeoff['fairness'], tradeoff['accuracy'], 'o-')
plt.xlabel('Fairness Score')
plt.ylabel('Accuracy')
plt.title('Fairness-Accuracy Trade-Off')
plt.grid(True)
plt.show()
""")


# =============================================================================
# SECTION 7: Next Steps
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 7: Next Steps - Recommended Learning Path")
print("=" * 80)

print("""
📚 TUTORIAL SEQUENCE:

Tutorial 02: Synthetic Allocation Example (02_synthetic_allocation.ipynb)
  → Build a complete allocation model from scratch
  → Work with synthetic healthcare data
  → Visualize predictions and fairness metrics
  
Tutorial 03: Fairness Metrics Deep Dive (03_fairness_metrics.ipynb)
  → Understand each fairness metric in detail
  → Compute group-specific metrics
  → Decide which fairness definition matches your values
  
Tutorial 04: Resource Constraints (04_resource_constraints.ipynb)
  → Model real-world constraints (e.g., limited bed capacity)
  → Optimize under scarcity
  → Compare strategies for limited resources
  
Tutorial 05: Comparison Study (05_comparing_approaches.ipynb)
  → Compare allocation-aware vs. accuracy-first approaches
  → Sensitivity analysis: how much does fairness weight matter?
  → Deployment recommendations

🔬 RESEARCH WORK:

Citation:
  Ferzana, R. (2026). Beyond Accuracy: A Decision-Theoretic Framework for 
  Allocation-Aware Healthcare AI. In Artificial Intelligence and Intelligent 
  Healthcare, LNCS 16877. Springer.
  DOI: https://doi.org/10.1007/978-3-032-35393-1_24

📦 INSTALLATION:

  pip install allocation-decision-framework

🐙 GITHUB:

  https://github.com/Rifa-111/allocation-decision-framework
  
🤝 CONTRIBUTE:

  Found a bug? Have a feature request? Open an issue on GitHub!
""")

print("\n" + "=" * 80)
print("End of Tutorial 01")
print("=" * 80)
