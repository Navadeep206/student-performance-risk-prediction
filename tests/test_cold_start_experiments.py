"""
tests/test_cold_start_experiments.py

Automated unit and integration test suite for Phase 9 Task 2:
Cold-Start Prediction Investigation & Experimentation Engine.
"""

import os
import json
import numpy as np
import pytest

from src.cold_start_experiments import compute_metrics_bundle


REPORTS_DIR = "reports"


def test_metrics_bundle_calculation():
    """Validates compute_metrics_bundle function on controlled synthetic predictions."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.1, 0.05, 0.85, 0.15])

    metrics = compute_metrics_bundle(y_true, y_prob, tau=0.20)

    assert metrics["n_samples"] == 10
    assert metrics["n_positives"] == 4
    assert metrics["prevalence_pct"] == 40.0
    assert metrics["pr_auc"] > 0.90
    assert metrics["roc_auc"] > 0.90
    assert "brier_score" in metrics
    assert "ece" in metrics


def test_experiment_a_cohort_summary_exists_and_valid():
    """Verifies that Experiment A output JSON exists and matches verified OULAD cohort parameters."""
    path = os.path.join(REPORTS_DIR, "cold_start_cohort_summary.json")
    assert os.path.exists(path), f"File {path} does not exist."

    with open(path, "r") as f:
        data = json.load(f)

    assert "cold_start_population" in data
    assert "non_cold_start_population" in data
    assert "disparity_tests" in data

    cs = data["cold_start_population"]
    ncs = data["non_cold_start_population"]

    # Cold-start snapshot count: 29,478 (17.25% of benchmark)
    assert cs["n_snapshots"] == 29478
    assert cs["n_unique_learners"] == 23351
    assert 17.0 <= cs["share_of_total_snapshots_pct"] <= 17.5

    # Failure rate comparison: Cold-start failure rate is significantly higher (10.49% vs 8.16%)
    assert 10.0 <= cs["failure_rate_pct"] <= 11.0
    assert 7.8 <= ncs["failure_rate_pct"] <= 8.5
    assert cs["failure_rate_pct"] > ncs["failure_rate_pct"]

    # Scheduled date comparison (Day 26.6 vs Day 152.3)
    assert cs["mean_scheduled_date"] < ncs["mean_scheduled_date"]

    # VLE activity comparison
    assert cs["mean_vle_total_clicks"] < ncs["mean_vle_total_clicks"]


def test_experiment_c_feature_strategies_benchmark():
    """Verifies Experiment C 5-fold CV strategy evaluation results."""
    path = os.path.join(REPORTS_DIR, "cold_start_experiments.json")
    assert os.path.exists(path), f"File {path} does not exist."

    with open(path, "r") as f:
        data = json.load(f)

    assert "cv_strategy_benchmarks" in data
    assert "holdout_2014j_comparison" in data

    cv = data["cv_strategy_benchmarks"]

    # Verify all 6 strategies are present
    expected_strategies = [
        "strategy_1_baseline",
        "strategy_2_indicators",
        "strategy_3_engagement_only",
        "strategy_4_academic_independent",
        "strategy_5_reduced_early",
        "strategy_6_segmented_specialist"
    ]
    for strat in expected_strategies:
        assert strat in cv, f"Strategy {strat} missing from CV results."
        assert "cold_start_metrics" in cv[strat]
        assert "pr_auc" in cv[strat]["cold_start_metrics"]

    s1_cs_ap = cv["strategy_1_baseline"]["cold_start_metrics"]["pr_auc"]
    s2_cs_ap = cv["strategy_2_indicators"]["cold_start_metrics"]["pr_auc"]
    s6_cs_ap = cv["strategy_6_segmented_specialist"]["cold_start_metrics"]["pr_auc"]

    # Baseline PR-AUC on cold start should be > 0.25
    assert s1_cs_ap >= 0.25

    # Strategy 2 (missingness indicators) does not improve upon native tree routing
    assert abs(s1_cs_ap - s2_cs_ap) < 0.005

    # Strategy 6 (dedicated specialist) does not outperform unified baseline model
    assert s6_cs_ap <= s1_cs_ap + 0.005


def test_experiment_d_model_architecture_comparison():
    """Verifies Experiment D architecture comparison on 2014J holdout."""
    path = os.path.join(REPORTS_DIR, "cold_start_model_comparison.json")
    assert os.path.exists(path), f"File {path} does not exist."

    with open(path, "r") as f:
        data = json.load(f)

    assert "hist_gradient_boosting" in data
    assert "random_forest" in data
    assert "logistic_regression" in data

    hgb_cs_ap = data["hist_gradient_boosting"]["cold_start_2014j"]["pr_auc"]
    lr_cs_ap = data["logistic_regression"]["cold_start_2014j"]["pr_auc"]

    # HistGradientBoosting significantly outperforms Logistic Regression on cold start
    assert hgb_cs_ap > lr_cs_ap
    # Logistic regression suffers catastrophic drop due to median imputation
    assert lr_cs_ap < 0.20


def test_experiment_e_recommendation_and_governance():
    """Verifies Experiment E practical recommendation and advisory policies."""
    path = os.path.join(REPORTS_DIR, "cold_start_recommendation.json")
    assert os.path.exists(path), f"File {path} does not exist."

    with open(path, "r") as f:
        data = json.load(f)

    assert "primary_architectural_recommendation" in data
    assert "justification" in data
    assert "operational_advisory_guidelines" in data
    assert "separate_model_justified" in data

    # Separate model is explicitly not justified
    assert data["separate_model_justified"] is False
    assert len(data["operational_advisory_guidelines"]) >= 3


def test_all_markdown_reports_exist():
    """Verifies that all Phase 9 Task 2 markdown reports are generated and non-empty."""
    expected_reports = [
        "reports/cold_start_feature_availability.md",
        "reports/cold_start_cohort_analysis.md",
        "reports/cold_start_experiments.md",
        "reports/cold_start_model_comparison.md",
        "reports/cold_start_recommendation.md",
        "reports/phase9_task2_completion_report.md"
    ]
    for r in expected_reports:
        assert os.path.exists(r), f"Report {r} was not created."
        assert os.path.getsize(r) > 200, f"Report {r} is empty or suspiciously small."

