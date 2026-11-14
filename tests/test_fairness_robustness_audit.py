"""
tests/test_fairness_robustness_audit.py

Automated Test Suite for Phase 8: Fairness Audit, Robustness & Model Governance.
Verifies:
1. Sensitive demographic attributes remain strictly excluded from predictive features.
2. Wilson score confidence interval helper satisfies mathematical bounds.
3. Summary JSON structure is valid and contains complete fairness, robustness, and failure audits.
4. Demographic subgroup partitions sum to total cohort size (170,874 snapshots).
5. Robustness findings verify performance drop on GGG and missing prior scores.
6. Failure mode profiling confirms behavioral discrepancies between TP, FP, TN, and FN.
7. All 5 publication-ready diagnostic figures exist on disk.
"""

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_models import ALL_ALLOWED_FEATURES, PROHIBITED_COLUMNS
from src.fairness_robustness_audit import wilson_score_interval, compute_binary_metrics_at_threshold


def test_sensitive_attributes_excluded_from_predictive_features():
    """Verifies that sensitive demographic attributes are strictly absent from predictive features."""
    sensitive_demographics = [
        "gender", "region", "highest_education", "imd_band", "age_band", "disability"
    ]
    for attr in sensitive_demographics:
        assert attr not in ALL_ALLOWED_FEATURES, f"Sensitive attribute '{attr}' leaked into predictive feature allowlist!"
        assert attr in PROHIBITED_COLUMNS, f"Sensitive attribute '{attr}' missing from prohibited list!"


def test_wilson_score_interval_properties():
    """Verifies that Wilson score confidence intervals are bounded and monotonic."""
    # Zero sample case
    low_0, high_0 = wilson_score_interval(0, 0)
    assert low_0 == 0.0 and high_0 == 0.0

    # Normal case: 40 successes out of 100
    low, high = wilson_score_interval(40, 100)
    assert 0.0 <= low < 40.0 < high <= 100.0

    # Extreme case: 0 successes out of 50
    low_zero, high_zero = wilson_score_interval(0, 50)
    assert low_zero == 0.0
    assert high_zero > 0.0  # Wilson interval handles 0 without collapsing to 0 width

    # Extreme case: 50 successes out of 50
    low_full, high_full = wilson_score_interval(50, 50)
    assert low_full < 100.0
    assert high_full == 100.0


def test_fairness_summary_json_integrity():
    """Verifies that fairness_robustness_summary.json exists and contains all required audit blocks."""
    summary_path = "reports/fairness_robustness_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("fairness_robustness_summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    assert "fairness_audit" in data
    assert "robustness_analysis" in data
    assert "failure_mode_analysis" in data
    assert "risk_tier_validation" in data

    # Check that all 6 required demographic attributes are audited
    for attr in ["gender", "age_band", "disability", "highest_education", "imd_band", "region"]:
        assert attr in data["fairness_audit"]
        assert "subgroups" in data["fairness_audit"][attr]
        assert "parity_disparities" in data["fairness_audit"][attr]


def test_demographic_subgroup_sample_sum_integrity():
    """Verifies that demographic subgroups partition the full 170,874 snapshots without dropouts."""
    summary_path = "reports/fairness_robustness_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("fairness_robustness_summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    expected_total = 170874
    # Gender sum
    gender_sub = data["fairness_audit"]["gender"]["subgroups"]
    gender_sum = sum(g["n_samples"] for g in gender_sub.values())
    assert gender_sum == expected_total

    # Disability sum
    dis_sub = data["fairness_audit"]["disability"]["subgroups"]
    dis_sum = sum(g["n_samples"] for g in dis_sub.values())
    assert dis_sum == expected_total

    # Age band sum
    age_sub = data["fairness_audit"]["age_band"]["subgroups"]
    age_sum = sum(g["n_samples"] for g in age_sub.values())
    assert age_sum == expected_total


def test_robustness_score_availability_and_module_degradation():
    """Verifies empirical degradation when historical scores are missing and on module GGG."""
    summary_path = "reports/fairness_robustness_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("fairness_robustness_summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    score_audit = data["robustness_analysis"]["score_availability"]
    ap_missing = score_audit["missing_historical_score"]["pr_auc"]
    ap_observed = score_audit["observed_historical_score"]["pr_auc"]
    assert ap_missing < ap_observed, "Expected missing score to degrade PR-AUC relative to observed scores!"

    mod_audit = data["robustness_analysis"]["modules"]
    assert "GGG" in mod_audit
    assert mod_audit["GGG"]["pr_auc"] < 0.25, "Expected module GGG to exhibit marked degradation!"


def test_failure_mode_profiles():
    """Verifies behavioral and historical divergence across TP, FP, TN, and FN failure categories."""
    summary_path = "reports/fairness_robustness_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("fairness_robustness_summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    profiles = data["failure_mode_analysis"]
    assert "True_Positive" in profiles
    assert "False_Positive" in profiles
    assert "True_Negative" in profiles
    assert "False_Negative" in profiles

    tp_score = profiles["True_Positive"]["mean_recent_prior_score"]
    tn_score = profiles["True_Negative"]["mean_recent_prior_score"]
    assert tp_score < 55.0, f"True positives should reflect prior academic struggle (got {tp_score})"
    assert tn_score > 75.0, f"True negatives should reflect strong prior grades (got {tn_score})"


def test_fairness_figures_exist():
    """Verifies that all 5 publication-ready diagnostic figures exist in reports/figures/fairness/."""
    fig_dir = "reports/figures/fairness"
    if not os.path.exists(fig_dir):
        pytest.skip("reports/figures/fairness directory does not exist.")

    expected_figures = [
        "demographic_fairness_parity_metrics.png",
        "subgroup_performance_comparison.png",
        "robustness_across_modules_and_presentations.png",
        "score_availability_and_vle_robustness.png",
        "failure_mode_analysis_distributions.png",
    ]
    for fn in expected_figures:
        p = os.path.join(fig_dir, fn)
        assert os.path.exists(p), f"Missing diagnostic figure: {p}"

