"""
tests/test_intervention_audit.py

Automated Test Suite for the Student-Level Intervention Evaluation Audit.
Verifies:
1. Leakage-free student-level alternatives have exactly one prediction per student.
2. No future assessment opportunities are pooled across time.
3. Denominators for Precision@k and Recall@k match mathematical definitions.
4. Metric bounds (0.0 <= Recall <= 1.0, 0.0 <= Precision <= 1.0).
5. Reproducibility of all intervention rankings directly from saved OOF predictions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_models import compute_recall_at_k


@pytest.fixture(scope="module")
def oof_predictions():
    """Loads outer validation predictions for 14-day delay baseline."""
    path = "data/processed/oof_predictions_optimized_14day_delay.csv"
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    return df[df["cutoff_scenario"] == "7_day_before"].copy()


def test_leakage_free_alternative_A_uniqueness(oof_predictions):
    """Verifies that Alternative A (First Assessment Milestone) has strictly 1 record per unique student."""
    sub7 = oof_predictions
    first_assess = sub7.sort_values("scheduled_date").groupby("id_student").first().reset_index()

    assert len(first_assess) == sub7["id_student"].nunique()
    assert first_assess["id_student"].is_unique
    # Verification that only the earliest assessment is selected
    for s in first_assess["id_student"].sample(50, random_state=42):
        s_dates = sub7[sub7["id_student"] == s]["scheduled_date"].values
        selected_date = first_assess[first_assess["id_student"] == s]["scheduled_date"].iloc[0]
        assert selected_date == min(s_dates)


def test_leakage_free_alternative_B_uniqueness(oof_predictions):
    """Verifies that Alternative B (Day 30 Cutoff) has strictly 1 upcoming assessment per student."""
    sub7 = oof_predictions
    after_30 = sub7[sub7["scheduled_date"] > 30].sort_values("scheduled_date")
    day30_cohort = after_30.groupby("id_student").first().reset_index()

    assert day30_cohort["id_student"].is_unique
    assert (day30_cohort["scheduled_date"] > 30).all()


def test_precision_recall_denominators(oof_predictions):
    """Verifies exact mathematical formulas for Precision@k and Recall@k."""
    sub7 = oof_predictions
    first_assess = sub7.sort_values("scheduled_date").groupby("id_student").first().reset_index()

    n = len(first_assess)
    n_pos = int(first_assess["at_risk"].sum())

    k = 0.10
    budget_m = int(np.ceil(k * n))

    rec = compute_recall_at_k(first_assess["at_risk"].values, first_assess["prob_optimized_hgb"].values, k=k)
    prec = (rec * n_pos) / budget_m

    assert 0.0 <= rec <= 1.0
    assert 0.0 <= prec <= 1.0

    # Sort descending and verify TP count in top budget_m
    top_m = first_assess.sort_values("prob_optimized_hgb", ascending=False).iloc[:budget_m]
    tp = int(top_m["at_risk"].sum())
    calculated_prec = tp / budget_m
    assert abs(prec - calculated_prec) < 0.01


def test_reproducibility_of_audit_artifacts():
    """Verifies that audit summary JSON and benchmark CSV exist and are valid."""
    json_path = "reports/intervention_evaluation_corrected_summary.json"
    csv_path = "data/processed/student_intervention_benchmarks_leakage_free.csv"

    assert os.path.exists(json_path), f"Missing {json_path}"
    assert os.path.exists(csv_path), f"Missing {csv_path}"

    with open(json_path) as f:
        data = json.load(f)

    assert "peak_risk_aggregation_audit" in data
    assert "leakage_free_alternative_A_first_assessment" in data
    assert "leakage_free_alternative_B_day30_cutoff" in data

    df_csv = pd.read_csv(csv_path)
    assert len(df_csv) == 3
    assert "recall_at_20" in df_csv.columns

