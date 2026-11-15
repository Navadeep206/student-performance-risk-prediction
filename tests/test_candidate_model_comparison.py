"""
Unit and integration tests for Phase 9 Task 4: Candidate Model Comparison & Selection.
"""

import json
import os
import numpy as np
import pytest

from src.candidate_model_comparison import compute_metrics_bundle, CandidateModelComparisonEngine


def test_compute_metrics_bundle_synthetic():
    """Verify metrics bundle calculation on known synthetic predictions."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    bundle = compute_metrics_bundle(y_true, y_prob, tau=0.5)
    
    assert bundle["n_samples"] == 8
    assert bundle["n_positives"] == 4
    assert bundle["prevalence_pct"] == 50.0
    assert bundle["threshold"] == 0.5
    assert bundle["pr_auc"] > 0.9
    assert bundle["roc_auc"] == 1.0
    
    cm = bundle["confusion_matrix"]
    assert cm["tp"] == 4
    assert cm["fp"] == 0
    assert cm["tn"] == 4
    assert cm["fn"] == 0
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 8
    
    assert bundle["precision_pct"] == 100.0
    assert bundle["recall_pct"] == 100.0
    assert bundle["f1_pct"] == 100.0
    assert bundle["fnr_pct"] == 0.0
    assert bundle["fpr_pct"] == 0.0
    
    # Recall at budget
    assert 0.0 <= bundle["recall_at_10_pct"] <= 100.0
    assert 0.0 <= bundle["recall_at_20_pct"] <= 100.0
    
    # Brier and ECE
    assert 0.0 <= bundle["brier_score"] <= 1.0
    assert 0.0 <= bundle["ece"] <= 1.0


def test_summary_json_structure_and_counts():
    """Verify the candidate model comparison summary JSON exists and has exact counts."""
    summary_path = "reports/candidate_model_comparison_summary.json"
    assert os.path.exists(summary_path), f"Missing summary artifact: {summary_path}"
    
    with open(summary_path, "r") as f:
        data = json.load(f)
        
    assert "candidate_models" in data
    assert sorted(data["candidate_models"]) == ["hist_gradient_boosting", "logistic_regression", "random_forest"]
    
    # Dimension 1: Grouped CV (exact benchmark size 170,874)
    cv_data = data["dimension_1_grouped_cv"]["overall_cv_benchmarks"]
    for model_name in ["logistic_regression", "random_forest", "hist_gradient_boosting"]:
        assert model_name in cv_data
        assert cv_data[model_name]["tau_020"]["n_samples"] == 170874
        assert cv_data[model_name]["tau_010"]["n_samples"] == 170874
        assert cv_data[model_name]["tau_020"]["n_positives"] == 14623
        
    # Dimension 2: Prospective 2014J Holdout (exact holdout size 50,014)
    holdout_data = data["dimension_2_chronological_holdout"]
    for model_name in ["logistic_regression", "random_forest", "hist_gradient_boosting"]:
        assert model_name in holdout_data
        assert holdout_data[model_name]["overall_2014j"]["tau_020"]["n_samples"] == 50014
        assert holdout_data[model_name]["overall_2014j"]["tau_020"]["n_positives"] == 4637
        
    # Novel vs Returning split sums to 50,014
    novel_n = holdout_data["hist_gradient_boosting"]["novel_students_2014j"]["tau_020"]["n_samples"]
    returning_n = holdout_data["hist_gradient_boosting"]["returning_students_2014j"]["tau_020"]["n_samples"]
    assert novel_n + returning_n == 50014
    assert novel_n == 43698
    assert returning_n == 6316
    
    # Dimension 3: Cold-Start Missing vs Observed split sums to 50,014
    cold_start_data = data["dimension_3_cold_start"]
    missing_n = cold_start_data["hist_gradient_boosting"]["cold_start_unobserved_grades"]["n_samples"]
    observed_n = cold_start_data["hist_gradient_boosting"]["non_cold_start_observed_grades"]["n_samples"]
    assert missing_n + observed_n == 50014
    assert missing_n == 10407
    assert observed_n == 39607
    
    # Dimension 4: All 7 modules sum to 50,014
    module_data = data["dimension_4_by_module"]
    assert len(module_data) == 7
    total_mod_samples = sum(module_data[m]["n_samples"] for m in module_data)
    assert total_mod_samples == 50014
    for m in module_data:
        assert "models" in module_data[m]
        assert "hist_gradient_boosting" in module_data[m]["models"]
    
    # Dimension 5: Assessment milestones sum to 50,014
    milestones = data["dimension_5_by_milestone"]
    assert len(milestones) == 4
    total_milestone_samples = sum(milestones[m]["n_samples"] for m in milestones)
    assert total_milestone_samples == 50014
    for m in milestones:
        assert "models" in milestones[m]
        assert "hist_gradient_boosting" in milestones[m]["models"]
    
    # Dimension 6: Prediction Horizons
    horiz_data = data["dimension_6_by_horizon"]
    assert "7_day_before" in horiz_data
    assert "1_day_before" in horiz_data
    assert horiz_data["7_day_before"]["n_samples"] == 170874
    assert horiz_data["1_day_before"]["n_samples"] == 170874


def test_candidate_ranking_and_verdict():
    """Verify empirical relationships between candidate models and selection rationale."""
    summary_path = "reports/candidate_model_comparison_summary.json"
    with open(summary_path, "r") as f:
        data = json.load(f)
        
    synthesis = data["dimension_7_selection_verdict"]
    matrix = synthesis["comparative_audit_matrix"]
    
    # HGB dominates grouped-CV PR-AUC (> 0.45)
    assert matrix["grouped_cv_pr_auc"]["hist_gradient_boosting"] > 0.45
    assert matrix["grouped_cv_pr_auc"]["hist_gradient_boosting"] > matrix["grouped_cv_pr_auc"]["random_forest"]
    assert matrix["grouped_cv_pr_auc"]["random_forest"] > matrix["grouped_cv_pr_auc"]["logistic_regression"]
    
    # RF has slightly higher prospective 2014J PR-AUC and lower ECE
    assert matrix["prospective_holdout_pr_auc"]["random_forest"] >= matrix["prospective_holdout_pr_auc"]["hist_gradient_boosting"]
    assert matrix["prospective_ece"]["random_forest"] < matrix["prospective_ece"]["hist_gradient_boosting"]
    
    # Logistic regression fails cold-start (Recall@0.20 < 1.0%)
    assert matrix["cold_start_recall_at_020"]["logistic_regression"] < 1.0
    
    # HGB cold-start recall is substantially higher than Random Forest (> 25% vs < 20%)
    assert matrix["cold_start_recall_at_020"]["hist_gradient_boosting"] > 25.0
    assert matrix["cold_start_recall_at_020"]["random_forest"] < 20.0
    
    # Synthesis flags
    assert synthesis["universal_winner_exists"] is False
    assert synthesis["selected_model"] == "HistGradientBoostingClassifier"
    assert "Governed" in synthesis["selection_status"]


def test_report_markdown_integrity():
    """Verify candidate model comparison report markdown contains all required analytical sections."""
    report_path = "reports/candidate_model_comparison.md"
    assert os.path.exists(report_path), f"Missing report artifact: {report_path}"
    
    with open(report_path, "r") as f:
        content = f.read()
        
    assert "# Candidate Model Comparison & Selection Report" in content
    assert "## 1. Executive Summary & Selection Verdict" in content
    assert "## 2. Master Candidate Comparison Table Across Evaluation Dimensions" in content
    assert "## 3. Disciplinary Module-Level Performance in 2014J" in content
    assert "## 4. Assessment Milestone Progression" in content
    assert "## 5. Prediction Horizon Sensitivity" in content
    assert "## 6. Comprehensive Trade-off Analysis & Final Recommendation" in content
    assert "HistGradientBoostingClassifier" in content
    assert "RandomForestClassifier" in content
    assert "LogisticRegression" in content
    assert "ModuleGGGGovernanceRouter" in content


def test_model_selection_policy_audit_artifacts():
    """Verify Phase 9 Task 4 Model Selection Policy Audit summary JSON and Markdown report."""
    summary_path = "reports/phase9_task4_model_selection_policy_summary.json"
    report_path = "reports/phase9_task4_model_selection_policy_audit.md"
    
    assert os.path.exists(summary_path), f"Missing policy summary: {summary_path}"
    assert os.path.exists(report_path), f"Missing policy report: {report_path}"
    
    with open(summary_path, "r") as f:
        data = json.load(f)
        
    # Section 1: Calibration
    sec1 = data["section_1_calibration_audit"]
    assert sec1["phase6_training_isolation"]["outer_validation_leakage"] is False
    assert sec1["phase6_training_isolation"]["chronological_2014j_labels_used"] is False
    assert sec1["policy_verdict"]["production_calibration_recommended"] is False
    assert sec1["ranking_degradation_mechanism"]["isotonic_pr_auc_drop"] < -0.10
    
    # Section 2: Thresholds
    sec2 = data["section_2_threshold_governance"]
    assert sec2["module_generalization_guardrail"]["prohibit_broad_generalization"] is True
    assert sec2["module_ggg_specific_justification"]["tau_020_performance"]["fnr_pct"] > 75.0
    assert sec2["module_ggg_specific_justification"]["tau_010_performance"]["recall_pct"] > 40.0
    
    # Section 3: CMAs
    sec3 = data["section_3_cma_governance"]
    assert sec3["cma_population_2014j"]["n_snapshots"] == 31894
    assert any("Research-Only" in m for m in sec3["governance_mandates"])
    assert any("suppress alerts on decoupled GGG CMAs" in m for m in sec3["governance_mandates"])
    
    # Section 4: Final Selection & Production Boundary
    sec4 = data["section_4_final_model_selection"]
    assert sec4["governed_model_roles"]["primary_candidate"]["model"] == "HistGradientBoostingClassifier"
    assert sec4["governed_model_roles"]["challenger_candidate"]["model"] == "RandomForestClassifier"
    assert sec4["governed_model_roles"]["baseline_candidate"]["model"] == "LogisticRegression"
    assert "WITHHELD" in sec4["production_approval_boundary"]["production_deployment_approval"]
    
    # Markdown report contents
    with open(report_path, "r") as f:
        md = f.read()
    assert "## 1. Probability Calibration Audit" in md
    assert "## 2. Threshold Governance Audit" in md
    assert "## 3. CMA (Computer-Marked Assessment) Operational Governance" in md
    assert "## 4. Final Governed Model Selection Statement" in md
    assert "Production Deployment Approval is WITHHELD" in md

