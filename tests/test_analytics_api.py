"""
Phase 11 Unit and Integration Tests for Dashboard Analytics & Governance API Routes.
Verifies all read-only analytics endpoints, data provenance, pagination,
HMAC pseudonymization safety, and strict shadow-mode constraints.
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.api import app


TEST_SECRET = "pytest-isolated-test-secret-32bytes!!"


@pytest.fixture(autouse=True)
def setup_test_environment():
    old_secret = os.environ.get("STUDENT_PSEUDONYMIZATION_SECRET")
    os.environ["STUDENT_PSEUDONYMIZATION_SECRET"] = TEST_SECRET
    yield
    if old_secret is None:
        os.environ.pop("STUDENT_PSEUDONYMIZATION_SECRET", None)
    else:
        os.environ["STUDENT_PSEUDONYMIZATION_SECRET"] = old_secret


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_cohort_summary_endpoint(client):
    """Verify 2014J cohort summary returns certified provenance numbers."""
    res = client.get("/api/v1/analytics/cohort-summary")
    assert res.status_code == 200
    data = res.json()
    
    assert data["enrolled_learners_2014j"] == 8607
    assert data["total_prediction_snapshots"] == 50014
    assert data["actionable_tma_alerts"] == 7130
    assert data["unique_learners_alerted"] == 3299
    assert data["alert_coverage_pct"] == 38.33
    assert data["mean_alerts_per_alerted_learner"] == 2.16
    assert data["estimated_weekly_alert_load"] == 209.7
    assert data["raw_cma_alerts_suppressed"] == 2243
    assert data["tma_alert_rate_pct"] == 21.93
    assert data["outreach_permitted"] is False
    assert "SHADOW_MODE" in data["operational_mode"]


def test_risk_distribution_endpoint(client):
    """Verify binned risk distribution and calibrated threshold markers."""
    res = client.get("/api/v1/analytics/risk-distribution")
    assert res.status_code == 200
    data = res.json()
    
    assert data["total_snapshots"] == 50014
    assert data["tma_snapshots"] == 32506
    assert data["cma_snapshots"] == 17508
    assert len(data["bins"]) == 10
    assert data["threshold_markers"]["module_ggg_calibrated_tma"] == 0.10
    assert data["threshold_markers"]["standard_modules_tma"] == 0.20
    assert "HIGH_RISK" in data["tier_counts"]
    assert "SUPPRESSED" in data["tier_counts"]


def test_student_explorer_pagination_and_sorting(client):
    """Verify student explorer returns paginated results with proper sorting and pseudonymization."""
    res = client.get("/api/v1/analytics/student-explorer?page=1&page_size=10&sort_by=predicted_risk_score&sort_dir=desc")
    assert res.status_code == 200
    data = res.json()
    
    assert data["total_records"] == 50014
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["records"]) == 10
    
    # Check that highest score is first
    first_score = data["records"][0]["predicted_risk_score"]
    second_score = data["records"][1]["predicted_risk_score"]
    assert first_score >= second_score
    
    # Check pseudonymization integrity
    for rec in data["records"]:
        assert rec["pseudonymized_student_id"].startswith("#")
        assert len(rec["pseudonymized_student_id"]) >= 8
        assert rec["student_outreach_enabled"] is False
        assert rec["suppress_automated_outreach"] is True


def test_student_explorer_module_and_tier_filtering(client):
    """Verify filtering by course module and risk tier."""
    # Filter by Module GGG
    res = client.get("/api/v1/analytics/student-explorer?module=GGG&page_size=20")
    assert res.status_code == 200
    data = res.json()
    assert all(r["code_module"] == "GGG" for r in data["records"])
    
    # Filter by HIGH_RISK
    res_high = client.get("/api/v1/analytics/student-explorer?risk_tier=HIGH_RISK&page_size=20")
    assert res_high.status_code == 200
    data_high = res_high.json()
    assert all(r["risk_tier"] == "HIGH_RISK" for r in data_high["records"])
    assert all(r["predicted_at_risk"] is True for r in data_high["records"])


def test_modules_comparison_endpoint(client):
    """Verify all 7 modules have certified benchmark, holdout, and alert counts."""
    res = client.get("/api/v1/analytics/modules-comparison")
    assert res.status_code == 200
    data = res.json()
    
    assert data["benchmark_total_snapshots"] == 170874
    assert data["holdout_2014j_total_snapshots"] == 50014
    assert data["holdout_2014j_actionable_tma_snapshots"] == 32506
    assert data["holdout_2014j_governed_tma_alerts"] == 7130
    
    modules = {m["code_module"]: m for m in data["modules"]}
    assert set(modules.keys()) == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}
    
    # Check exact certified values
    assert modules["BBB"]["governed_tma_alerts_2014j"] == 2501
    assert modules["DDD"]["governed_tma_alerts_2014j"] == 1752
    assert modules["CCC"]["governed_tma_alerts_2014j"] == 1532
    assert modules["FFF"]["governed_tma_alerts_2014j"] == 772
    assert modules["GGG"]["governed_tma_alerts_2014j"] == 239
    assert modules["AAA"]["governed_tma_alerts_2014j"] == 187
    assert modules["EEE"]["governed_tma_alerts_2014j"] == 147
    
    # Check GGG matched evaluation panel
    ggg = data["module_ggg_matched_evaluation"]
    assert ggg["matched_snapshots_population"] == 5656
    assert ggg["unique_learners_count"] == 2097
    assert ggg["positive_failures_count"] == 411
    assert ggg["baseline_pr_auc"] == 0.2508
    assert ggg["remediated_pr_auc"] == 0.2353
    assert ggg["calibrated_tau_010_baseline_recall"] == 45.01
    assert ggg["calibrated_tau_010_remediated_recall"] == 45.74
    assert ggg["baseline_additional_detected_cases"] == 86
    assert ggg["remediated_additional_detected_cases"] == 107
    assert ggg["cma_quizzes_total_count"] == 18
    assert ggg["cma_quizzes_weight"] == 0.0


def test_explainability_global_importance_endpoint(client):
    """Verify TreeSHAP global feature importance metrics and methodological guidance."""
    res = client.get("/api/v1/explainability/global-importance")
    assert res.status_code == 200
    data = res.json()
    
    features = {f["feature_name"]: f["mean_abs_shap"] for f in data["features"]}
    assert features["recent_prior_score"] == 0.7223
    assert features["target_assessment_type"] == 0.6836
    assert features["registration_timing_days"] == 0.4473
    assert features["vle_active_days_to_date"] == 0.4226
    assert features["target_assessment_weight"] == 0.2601
    assert features["prior_submission_lateness_mean"] == 0.1873
    assert len(data["methodological_guidance"]) >= 3


def test_governance_audit_endpoint(client):
    """Verify governance rules, threshold policies, and prohibited attributes."""
    res = client.get("/api/v1/governance/audit-records")
    assert res.status_code == 200
    data = res.json()
    
    assert data["operational_mode"] == "READ_ONLY_SHADOW_MODE"
    assert data["allow_student_outreach"] is False
    assert data["cma_telemetry_suppressed"] is True
    assert data["threshold_policy"]["standard_modules_tma"]["operational_threshold"] == 0.20
    assert data["threshold_policy"]["module_ggg_calibrated_tma"]["operational_threshold"] == 0.10
    assert "gender" in data["prohibited_features_enforced"]
    assert "final_result" in data["prohibited_features_enforced"]


def test_monitoring_psi_feature_drift_endpoint(client):
    """Verify generalization gap, PSI drift, and calibration metrics."""
    res = client.get("/api/v1/monitoring/psi-feature-drift")
    assert res.status_code == 200
    data = res.json()
    
    gap_metrics = {m["metric_name"]: m for m in data["performance_generalization_gap"]}
    pr_auc_metric = [m for k, m in gap_metrics.items() if "PR-AUC" in k][0]
    assert pr_auc_metric["grouped_cv_value"] == 0.4606
    assert pr_auc_metric["chronological_2014j_value"] == 0.3005
    
    ece_metric = [m for k, m in gap_metrics.items() if "ECE" in k][0]
    assert ece_metric["grouped_cv_value"] == 0.0163
    assert ece_metric["chronological_2014j_value"] == 0.0420
    
    calib = data["calibration_metrics"]
    assert calib["expected_calibration_error_ece"] == 0.0420
    assert calib["brier_score_loss"] == 0.0812
    
    psi_features = {p["feature_name"]: p["psi_value"] for p in data["top_feature_psi_drift"]}
    assert psi_features["target_assessment_type"] == 0.4077
    assert psi_features["registration_timing_days"] == 0.2191
    assert psi_features["vle_active_days_to_date"] == 0.1102


def test_serve_frontend_dashboard_and_html(client):
    """Verify that root / and /dashboard deliver the compiled React enterprise dashboard HTML."""
    for path in ["/", "/dashboard"]:
        res = client.get(path)
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Student Performance Analytics Portal" in res.text
        assert '<div id="root"></div>' in res.text

