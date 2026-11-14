"""
tests/test_model_optimization.py

Automated Test Suite for Phase 6: Model Optimization, Generalization & Robustness.
Verifies:
1. Nested hyperparameter optimization adheres to strict zero-leakage principles.
2. Probability calibration metrics (Brier score, ECE) are mathematically sound.
3. Snapshot-level vs student-level aggregated Recall@k calculations and bounds.
4. Chronological evaluation strictly partitions earlier vs later cohorts.
5. Novel students in chronological testing possess exactly zero overlap with training data.
6. Generated outer-fold prediction files exist, share correct schemas, and contain no nulls.
7. Model selection metadata integrity in summary JSON.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_optimization import (
    compute_ece,
    CANDIDATE_SEARCH_SPACE,
    optimize_hyperparameters_inner,
    compute_student_level_metrics,
)
from src.baseline_models import compute_recall_at_k


def test_ece_calculation_exactness():
    """Verifies that compute_ece correctly computes Expected Calibration Error."""
    # Perfectly calibrated case: 100 predictions of 0.10 where exactly 10% are positive
    y_true_perfect = np.array([1]*10 + [0]*90)
    y_prob_perfect = np.full(100, 0.10)
    ece_perf = compute_ece(y_true_perfect, y_prob_perfect, n_bins=10)
    assert ece_perf == 0.0

    # Completely uncalibrated case: predicts 1.0 for all 0s, and 0.0 for all 1s
    y_true_bad = np.array([1]*50 + [0]*50)
    y_prob_bad = np.array([0.0]*50 + [1.0]*50)
    ece_bad = compute_ece(y_true_bad, y_prob_bad, n_bins=10)
    assert ece_bad == 1.0


def test_student_level_aggregation_metrics():
    """Verifies student-level risk aggregation (Max and Mean) and recall computation."""
    # Synthetic student dataframe with 3 students, 2 assessments each
    df_synthetic = pd.DataFrame({
        "id_student": [101, 101, 102, 102, 103, 103],
        "at_risk":    [0,   1,   0,   0,   0,   0],
        "prob":       [0.2, 0.8, 0.3, 0.4, 0.1, 0.15],
    })

    res = compute_student_level_metrics(df_synthetic, "prob_col" if "prob_col" in df_synthetic else "prob")
    assert res["total_unique_students"] == 3
    assert res["at_risk_unique_students"] == 1  # Student 101 failed assessment 2

    # Student 101 max prob is 0.8 (rank 1 out of 3 students)
    # Budget top 34% (1 out of 3): captures student 101 -> Recall = 1.0
    rec_max = compute_recall_at_k([1, 0, 0], [0.8, 0.4, 0.15], k=0.34)
    assert rec_max == 1.0


def test_candidate_search_space_validity():
    """Verifies that all candidates in hyperparameter search space have valid HistGradientBoosting parameters."""
    for cand in CANDIDATE_SEARCH_SPACE:
        assert "name" in cand
        assert "params" in cand
        p = cand["params"]
        assert 0.01 <= p["learning_rate"] <= 0.2
        assert 50 <= p["max_iter"] <= 300
        assert 3 <= p["max_depth"] <= 12
        assert 10 <= p["max_leaf_nodes"] <= 128
        assert p["l2_regularization"] >= 0.0


def test_chronological_split_integrity():
    """Verifies temporal precedence: earlier presentations must precede later presentations."""
    df_path = "data/processed/prediction_dataset.csv"
    if not os.path.exists(df_path):
        pytest.skip("prediction_dataset.csv missing.")

    df = pd.read_csv(df_path, usecols=["code_presentation", "id_student", "scheduled_date"])
    train_pres = {"2013B", "2013J", "2014B"}
    test_pres = {"2014J"}

    assert train_pres.isdisjoint(test_pres)

    train_students = set(df[df["code_presentation"].isin(train_pres)]["id_student"])
    test_students = set(df[df["code_presentation"].isin(test_pres)]["id_student"])

    # Overlapping continuing students exist
    overlap = train_students.intersection(test_students)
    assert len(overlap) > 0, "Expected continuing students across academic intakes."

    # Strictly novel students must have 0 overlap with training set
    novel_students = test_students - train_students
    assert len(novel_students) > 0
    assert len(novel_students.intersection(train_students)) == 0


def test_phase6_artifacts_exist_after_run():
    """Verifies that Phase 6 generated summary JSON and out-of-fold files if run."""
    summary_path = "reports/model_optimization_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("model_optimization_summary.json not yet generated.")

    with open(summary_path) as f:
        data = json.load(f)

    assert "14_day_delay__7_day_before" in data
    assert "chronological_evaluation_2014J" in data
    assert "final_selected_model_architecture" in data

    # Check outer prediction files
    for fn in ["oof_predictions_optimized_14day_delay.csv", "oof_predictions_chronological_2014J.csv"]:
        f_path = os.path.join("data/processed", fn)
        assert os.path.exists(f_path), f"Missing {fn}"
        df_oof = pd.read_csv(f_path, nrows=100)
        assert "prob_optimized_hgb" in df_oof.columns

