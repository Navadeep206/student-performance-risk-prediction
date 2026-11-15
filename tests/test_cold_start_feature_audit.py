"""
tests/test_cold_start_feature_audit.py

Automated Test Suite for Phase 9 - Task 2, Experiment B: Feature Availability & Leakage Audit.
Verifies:
1. Feature availability specifications correctly reflect cold-start unobservability.
2. Prior score features (recent_prior_score, mean_prior_score, std_prior_score) are 100% missing in cold-start snapshots.
3. Prohibited targets, future outcomes, and sensitive demographics are strictly excluded.
4. Feature availability report exists and is populated.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_models import ALL_ALLOWED_FEATURES, PROHIBITED_COLUMNS
from src.cold_start_feature_audit import (
    FEATURE_AUDIT_SPECIFICATIONS,
    audit_feature_availability_and_leakage,
)


def test_feature_audit_specifications_completeness():
    """Verifies that all 23 allowed features are formally audited."""
    audited_names = {s["feature_name"] for s in FEATURE_AUDIT_SPECIFICATIONS}
    assert audited_names == set(ALL_ALLOWED_FEATURES), "Mismatch between audited features and allowlist!"


def test_prior_scores_strictly_unobserved_in_cold_start():
    """Verifies that prior score features are unobserved during cold-start predictions."""
    df = pd.read_csv("data/processed/prediction_dataset.csv")
    sub7 = df[df["cutoff_scenario"] == "7_day_before"].reset_index(drop=True)
    audit_res = audit_feature_availability_and_leakage(sub7)

    # Check that cold-start snapshots were identified
    assert audit_res["total_cold_start_snapshots"] > 25000

    # Verify prior scores have 100% missingness in cold-start
    for r in audit_res["feature_availability_matrix"]:
        if r["feature_name"] in ["recent_prior_score", "mean_prior_score", "std_prior_score"]:
            assert r["empirical_cold_start_missingness_pct"] == 100.0, f"Expected 100% missingness for {r['feature_name']} in cold start!"
            assert not r["cold_start_available"]


def test_prohibited_columns_exclusion_compliance():
    """Verifies that zero prohibited targets, future fields, or demographics appear in allowlist."""
    df = pd.read_csv("data/processed/prediction_dataset.csv")
    sub7 = df[df["cutoff_scenario"] == "7_day_before"].reset_index(drop=True)
    audit_res = audit_feature_availability_and_leakage(sub7)

    assert audit_res["all_compliant"], "Detected violation in prohibited columns compliance check!"
    for p_col in PROHIBITED_COLUMNS:
        assert p_col not in ALL_ALLOWED_FEATURES


def test_feature_availability_report_exists():
    """Verifies that reports/cold_start_feature_availability.md was generated."""
    path = "reports/cold_start_feature_availability.md"
    assert os.path.exists(path), f"Missing expected audit report: {path}"
    with open(path) as f:
        content = f.read()
    assert "Feature Availability Matrix" in content
    assert "Zero Leakage Verified" in content
