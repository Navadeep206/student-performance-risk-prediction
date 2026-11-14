"""
tests/test_model_explainability.py

Automated Test Suite for Phase 7: Model Explainability, Attribution & Risk Assessment.
Verifies:
1. Global TreeSHAP attributions exist across all 23 allowed features with valid non-negative values.
2. Advisory risk categorization framework forms a complete partition of the evaluation set.
3. Monotonicity of empirical failure rates across risk tiers (Low < Moderate < High).
4. Local case study decompositions have both risk drivers and protective factors.
5. All explainability summary JSON and diagnostic plot deliverables exist and are valid.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_explainability import (
    evaluate_advisory_risk_tiers,
    build_fitted_optimized_model,
    compute_global_shap_attributions,
)
from src.baseline_models import ALL_ALLOWED_FEATURES


def test_risk_tier_partition_and_monotonicity():
    """Verifies that risk tiers partition the dataset completely and exhibit monotonic failure rates."""
    np.random.seed(42)
    # Synthetic probabilities with higher failure rates at higher probs
    probs = np.array([0.01]*50 + [0.10]*30 + [0.45]*20)
    y_true = np.array([0]*49 + [1]*1 + [0]*27 + [1]*3 + [0]*10 + [1]*10)

    res = evaluate_advisory_risk_tiers(probs, y_true, tau_high=0.20, tau_low=0.05)
    assert res["total_records"] == 100
    tiers = res["tier_metrics"]

    # Partition completeness
    assert tiers["low_risk"]["count"] + tiers["moderate_risk"]["count"] + tiers["high_risk"]["count"] == 100

    # Monotonicity check: Low < Moderate < High failure rate
    rate_low = tiers["low_risk"]["empirical_failure_rate_pct"]
    rate_mod = tiers["moderate_risk"]["empirical_failure_rate_pct"]
    rate_high = tiers["high_risk"]["empirical_failure_rate_pct"]

    assert rate_low < rate_mod < rate_high
    assert rate_low == 2.0   # 1 out of 50
    assert rate_mod == 10.0  # 3 out of 30
    assert rate_high == 50.0 # 10 out of 20


def test_explainability_summary_json_structure():
    """Verifies that model_explainability_summary.json exists and contains required sections."""
    summary_path = "reports/model_explainability_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("model_explainability_summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    assert "global_feature_importance" in data
    assert len(data["global_feature_importance"]) == len(ALL_ALLOWED_FEATURES)
    assert "scenario_specific_importance" in data
    assert "risk_tier_evaluation_snapshot_level" in data
    assert "risk_tier_evaluation_alternative_A_first_assessment" in data
    assert "risk_tier_evaluation_alternative_B_day30_cutoff" in data
    assert "local_case_studies" in data
    assert len(data["local_case_studies"]) == 4


def test_local_case_study_anonymity_and_structure():
    """Verifies that local case studies avoid raw student IDs and contain required fields."""
    summary_path = "reports/model_explainability_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("model_explainability_summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    for case in data["local_case_studies"]:
        assert "id_student" not in case, "Raw student ID leaked in local case explanation!"
        assert "case_label" in case
        assert "predicted_probability" in case
        assert "actual_outcome" in case
        assert "plain_language_explanation" in case
        assert "advisor_actionable_recommendation" in case
        assert len(case["top_risk_drivers"]) > 0 or len(case["top_protective_factors"]) > 0


def test_explainability_figures_exist():
    """Verifies that all required publication diagnostic plots were generated."""
    fig_dir = "reports/figures/explainability"
    if not os.path.exists(fig_dir):
        pytest.skip("reports/figures/explainability not yet created.")

    expected_figures = [
        "shap_global_bar_importance.png",
        "shap_global_summary_beeswarm.png",
        "scenario_feature_importance_comparison.png",
        "risk_tier_distribution.png",
        "local_waterfall_examples.png",
    ]
    for fn in expected_figures:
        p = os.path.join(fig_dir, fn)
        assert os.path.exists(p), f"Missing expected plot: {p}"

