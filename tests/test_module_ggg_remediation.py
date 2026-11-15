"""
tests/test_module_ggg_remediation.py

Automated unit, regression, and integration test suite for Phase 9 Task 3:
Module GGG Failure Investigation, Root Cause Verification, and Governance Remediation.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.module_ggg_remediation import (
    FeatureRemediator,
    ModuleGGGGovernanceRouter,
    ModuleGGGRemediationEvaluator
)


REPORTS_DIR = "reports"


def test_feature_remediator_clean_previous_scheduled_count():
    """Regression test for FAIL-GGG-03: Verifies deterministic tie-breaking for clustered dates."""
    # Module GGG schedule: 3 TMAs at 61, 124, 173; 6 CMAs at 229
    ggg_schedule = [61, 124, 173, 229, 229, 229, 229, 229, 229]

    # TMA 1 has 0 prior scheduled dates
    assert FeatureRemediator.compute_clean_previous_scheduled_count(ggg_schedule, 61) == 0
    # TMA 2 has 1 prior scheduled date
    assert FeatureRemediator.compute_clean_previous_scheduled_count(ggg_schedule, 124) == 1
    # TMA 3 has 2 prior scheduled dates
    assert FeatureRemediator.compute_clean_previous_scheduled_count(ggg_schedule, 173) == 2

    # All 6 CMAs scheduled at Day 229 MUST receive exactly 3 (the 3 TMAs)
    for _ in range(6):
        count = FeatureRemediator.compute_clean_previous_scheduled_count(ggg_schedule, 229)
        assert count == 3, f"Expected 3 prior assessments for Day 229, got {count}"


def test_feature_remediator_filter_prior_submissions():
    """Regression test for FAIL-GGG-04: Verifies completion-based filtering prevents peer blindness."""
    sample_subs = pd.DataFrame([
        {"id_assessment": 37418, "id_student": 100, "date_submitted": 80, "score": 85.0},
        {"id_assessment": 37419, "id_student": 100, "date_submitted": 110, "score": 90.0},
        {"id_assessment": 37420, "id_student": 100, "date_submitted": 225, "score": 75.0}, # Submitted after cutoff
        {"id_assessment": 37421, "id_student": 100, "date_submitted": 150, "score": None}, # Unscored
    ])

    cutoff = 222
    delay = 14
    # Evaluating target assessment 37419 at cutoff 222
    valid = FeatureRemediator.filter_prior_submissions_by_completion(
        sample_subs, target_assessment_id=37419, cutoff_date=cutoff, delay_days=delay
    )

    # 37418 was submitted at day 80 (80 + 14 = 94 <= 222) and has a valid score -> Included
    # 37419 is the target -> Excluded
    # 37420 was submitted at day 225 (225 + 14 = 239 > 222) -> Excluded
    # 37421 has score None -> Excluded
    assert len(valid) == 1
    assert valid.iloc[0]["id_assessment"] == 37418


def test_governance_router_module_routing():
    """Integration test for FAIL-GGG-01, 02, 05: Verifies operational guardrail routing."""
    router = ModuleGGGGovernanceRouter(tma_calibrated_threshold=0.10, global_threshold=0.20)

    # 1. Module BBB TMA (Standard high confidence)
    res_bbb = router.route_prediction({"code_module": "BBB", "target_assessment_type": "TMA", "scheduled_date": 100}, raw_prob=0.25)
    assert res_bbb["confidence_tier"] == "HIGH_CONFIDENCE_STANDARD"
    assert res_bbb["suppress_automated_outreach"] is False
    assert res_bbb["predicted_at_risk"] is True

    # 2. Module GGG CMA (Decoupled formative quiz -> Automated outreach suppressed)
    res_ggg_cma = router.route_prediction({"code_module": "GGG", "target_assessment_type": "CMA", "scheduled_date": 229}, raw_prob=0.35)
    assert res_ggg_cma["confidence_tier"] == "UNRELIABLE_DECOUPLED_QUIZ"
    assert res_ggg_cma["suppress_automated_outreach"] is True
    assert res_ggg_cma["predicted_at_risk"] is False

    # 3. Module GGG TMA 1 (Onboarding check-in)
    res_ggg_tma1 = router.route_prediction({"code_module": "GGG", "target_assessment_type": "TMA", "scheduled_date": 61}, raw_prob=0.12)
    assert res_ggg_tma1["confidence_tier"] == "LOW_CONFIDENCE_ONBOARDING"
    assert res_ggg_tma1["governance_action"] == "SUPPORTIVE_CHECK_IN_ONLY"
    assert res_ggg_tma1["predicted_at_risk"] is True

    # 4. Module GGG TMA 3 (Synchronous advisor review)
    res_ggg_tma3 = router.route_prediction({"code_module": "GGG", "target_assessment_type": "TMA", "scheduled_date": 173}, raw_prob=0.15)
    assert res_ggg_tma3["confidence_tier"] == "MODERATE_CONFIDENCE_MILESTONE"
    assert res_ggg_tma3["governance_action"] == "SYNCHRONOUS_ADVISOR_REVIEW"
    assert res_ggg_tma3["predicted_at_risk"] is True


def test_remediation_summary_json_integrity():
    """Verifies that remediation summary JSON exists and contains positive performance improvements."""
    path = os.path.join(REPORTS_DIR, "module_ggg_remediation_summary.json")
    assert os.path.exists(path), f"File {path} does not exist."

    with open(path, "r") as f:
        data = json.load(f)

    assert "baseline_unremediated_performance" in data
    assert "remediated_governance_performance" in data
    assert "tma3_capstone_milestone_validation" in data

    base = data["baseline_unremediated_performance"]
    rem = data["remediated_governance_performance"]
    tma3 = data["tma3_capstone_milestone_validation"]

    # Remediated governance improves recall by > 20%
    assert rem["recall_gain_pct"] > 20.0
    # Remediated governance reduces false negatives by > 20%
    assert rem["fnr_reduction_pct"] > 20.0
    # TMA 3 discrimination remains strong
    assert tma3["pr_auc"] >= 0.35
    assert tma3["roc_auc"] >= 0.80


def test_failure_investigation_json_integrity():
    """Verifies that failure investigation JSON exists and contains 5 confirmed root causes."""
    path = os.path.join(REPORTS_DIR, "module_ggg_failure_investigation.json")
    assert os.path.exists(path), f"File {path} does not exist."

    with open(path, "r") as f:
        data = json.load(f)

    assert "curricular_audit" in data
    assert "temporal_feature_audit" in data
    assert "confirmed_root_causes" in data

    causes = data["confirmed_root_causes"]
    assert len(causes) == 5

    severities = [c["severity"] for c in causes]
    assert "Critical" in severities
    assert "High" in severities
    assert "Medium" in severities


def test_matched_evaluation_artifacts_and_metrics():
    """Verifies matched-population evaluation artifacts, sample matching, and threshold metrics."""
    json_path = os.path.join(REPORTS_DIR, "module_ggg_matched_evaluation.json")
    md_path = os.path.join(REPORTS_DIR, "module_ggg_matched_evaluation.md")

    assert os.path.exists(json_path), f"File {json_path} does not exist."
    assert os.path.exists(md_path), f"File {md_path} does not exist."

    with open(json_path, "r") as f:
        data = json.load(f)

    # 1. Exact Population & Label Matching
    pop = data["evaluation_population"]
    assert pop["course_module"] == "GGG"
    assert pop["assessment_type"] == "TMA"
    assert pop["total_matched_snapshots"] == 5656
    assert pop["total_at_risk_failures"] == 411
    assert pop["overall_prevalence_pct"] == 7.27

    # 2. Overall Metrics & Separate Threshold Evaluation
    comp = data["overall_baseline_vs_remediated"]
    assert comp["pr_auc"]["baseline"] == 0.2508
    assert comp["roc_auc"]["baseline"] == 0.7510

    # tau = 0.20
    t20 = comp["tau_020_comparison"]
    assert t20["recall"]["baseline"] == 24.09
    assert t20["fnr"]["baseline"] == 75.91
    cm_base_20 = t20["confusion_matrix"]["baseline"]
    assert cm_base_20["tn"] + cm_base_20["fp"] + cm_base_20["fn"] + cm_base_20["tp"] == 5656
    assert cm_base_20["tp"] + cm_base_20["fn"] == 411

    # tau = 0.10
    t10 = comp["tau_010_comparison"]
    assert t10["recall"]["baseline"] == 45.01
    assert t10["fnr"]["baseline"] == 54.99
    cm_base_10 = t10["confusion_matrix"]["baseline"]
    assert cm_base_10["tn"] + cm_base_10["fp"] + cm_base_10["fn"] + cm_base_10["tp"] == 5656
    assert cm_base_10["tp"] + cm_base_10["fn"] == 411

    # 3. Fold-wise evaluations present for Folds 0..4
    assert len(data["fold_evaluations"]) == 5
    for f in range(5):
        assert f"fold_{f}" in data["fold_evaluations"]

    # 4. Presentation-wise evaluations present
    assert "2013J" in data["presentation_evaluations"]
    assert "2014B" in data["presentation_evaluations"]
    assert "2014J" in data["presentation_evaluations"]


