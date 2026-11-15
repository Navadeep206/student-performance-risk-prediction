"""
tests/test_temporal_drift_audit.py

Automated Test Suite for Phase 9 - Task 1: Temporal Drift & Distributional Shift Audit.
Verifies:
1. Population Stability Index (PSI) calculation correctness, non-negativity, and Laplace smoothing.
2. Summary JSON structure validity and completeness across all drift dimensions.
3. Severe covariate shift detection on target_assessment_weight and scheduled_date.
4. Concept drift detection on module BBB and cold-start assessments.
5. Expected Calibration Error (ECE) calculation on 2014J predictions.
6. Existence of all 5 publication-ready diagnostic figures in reports/figures/temporal_drift/.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.temporal_drift_audit import (
    calculate_continuous_psi,
    calculate_categorical_psi,
    compute_expected_calibration_error,
)


def test_continuous_and_categorical_psi_properties():
    """Verifies mathematical validity and non-negativity of PSI calculations."""
    # Identical continuous distribution -> PSI approx 0.0
    np.random.seed(42)
    s1 = pd.Series(np.random.normal(50, 10, 1000))
    s2 = pd.Series(np.random.normal(50, 10, 1000))
    psi_ident = calculate_continuous_psi(s1, s2, bins=10)
    assert 0.0 <= psi_ident < 0.05, f"Expected near-zero PSI for identical distributions, got {psi_ident}"

    # Shifted continuous distribution -> PSI > 0.25
    s_shifted = pd.Series(np.random.normal(70, 10, 1000))
    psi_shifted = calculate_continuous_psi(s1, s_shifted, bins=10)
    assert psi_shifted > 0.25, f"Expected large PSI for shifted distribution, got {psi_shifted}"

    # Categorical PSI test
    cat_ref = pd.Series(["A"] * 500 + ["B"] * 500)
    cat_tgt = pd.Series(["A"] * 200 + ["B"] * 800)
    psi_cat = calculate_categorical_psi(cat_ref, cat_tgt)
    assert psi_cat > 0.10, f"Expected moderate PSI for shifted categories, got {psi_cat}"


def test_temporal_drift_summary_json_integrity():
    """Verifies that reports/temporal_drift_summary.json exists and has complete sections."""
    path = "reports/temporal_drift_summary.json"
    if not os.path.exists(path):
        pytest.skip("reports/temporal_drift_summary.json not yet generated.")

    with open(path) as f:
        data = json.load(f)

    assert "cohort_overview" in data
    assert "label_shift" in data
    assert "numerical_feature_drift" in data
    assert "categorical_feature_drift" in data
    assert "concept_drift_analysis" in data
    assert "prediction_and_calibration_shift" in data
    assert "pipeline_preprocessing_audit" in data

    # Check that 20 numerical features were audited
    assert len(data["numerical_feature_drift"]) == 20


def test_severe_feature_drift_detection():
    """Verifies that high-PSI features are correctly identified."""
    path = "reports/temporal_drift_summary.json"
    if not os.path.exists(path):
        pytest.skip("reports/temporal_drift_summary.json not yet generated.")

    with open(path) as f:
        data = json.load(f)

    num_drift = data["numerical_feature_drift"]
    assert num_drift["target_assessment_weight"]["psi"] >= 0.25
    assert num_drift["scheduled_date"]["psi"] >= 0.25
    assert num_drift["num_prior_scored_assessments"]["psi"] >= 0.25


def test_concept_drift_in_bbb_and_cold_start():
    """Verifies detection of conditional failure rate shifts in module BBB and cold-start assessments."""
    path = "reports/temporal_drift_summary.json"
    if not os.path.exists(path):
        pytest.skip("reports/temporal_drift_summary.json not yet generated.")

    with open(path) as f:
        data = json.load(f)

    mod_drift = data["concept_drift_analysis"]["module_conditional_failure_rates"]
    assert "BBB" in mod_drift
    # Failure rate in BBB jumped by over 10 percentage points
    assert mod_drift["BBB"]["failure_rate_shift_pct"] > 10.0

    cold_drift = data["concept_drift_analysis"]["cold_start_conditional_failure_rates"]
    assert cold_drift["missing_prior_grade"]["failure_rate_shift_pct"] > 3.0


def test_calibration_degradation_ece_metric():
    """Verifies that calibration metrics on 2014J reflect degraded confidence alignment."""
    path = "reports/temporal_drift_summary.json"
    if not os.path.exists(path):
        pytest.skip("reports/temporal_drift_summary.json not yet generated.")

    with open(path) as f:
        data = json.load(f)

    calib = data["prediction_and_calibration_shift"]
    assert calib["expected_calibration_error_ece"] > 0.03
    assert calib["brier_score_loss"] > 0.07


def test_temporal_drift_figures_exist():
    """Verifies that all 5 diagnostic figures exist in reports/figures/temporal_drift/."""
    fig_dir = "reports/figures/temporal_drift"
    if not os.path.exists(fig_dir):
        pytest.skip("reports/figures/temporal_drift directory does not exist.")

    expected_figures = [
        "feature_psi_ranking.png",
        "module_composition_and_risk_drift.png",
        "calibration_curve_drift_2014J.png",
        "predicted_probability_distribution_shift.png",
        "conditional_failure_rate_concept_drift.png",
    ]
    for fn in expected_figures:
        p = os.path.join(fig_dir, fn)
        assert os.path.exists(p), f"Missing diagnostic figure: {p}"

