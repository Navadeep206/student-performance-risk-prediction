"""
Unit and integration tests for Phase 9 Task 5: Operational Readiness & Deployment Feasibility Audit.
"""

import json
import os
import pytest
from src.operational_readiness_audit import OperationalReadinessAuditEngine


@pytest.fixture
def audit_engine():
    return OperationalReadinessAuditEngine()


def test_calibration_monotonic_invariance(audit_engine):
    """Verify that strictly monotonic sigmoid preserves AP and ROC-AUC exactly."""
    res = audit_engine.run_calibration_consistency_check()
    assert res["mathematical_monotonic_invariance_verified"] is True
    assert res["synthetic_test_metrics"]["ap_difference"] < 1e-9
    assert res["synthetic_test_metrics"]["roc_difference"] < 1e-9
    assert res["calibrated_classifier_cv_mechanism"]["refits_base_estimators"] is True
    assert "refits" in res["calibrated_classifier_cv_mechanism"]["explanation_of_phase6_ap_drop"].lower()


def test_counselor_workload_metrics_2014j(audit_engine):
    """Verify counselor workload metrics on 2014J prospective holdout."""
    res = audit_engine.run_counselor_workload_audit()
    
    # Cohort checks
    assert res["cohort_parameters"]["total_snapshots"] == 50014
    assert res["cohort_parameters"]["total_unique_students"] == 8607
    
    # Alert volumes
    raw = res["alert_volume_comparison"]["raw_unconstrained_tau_020"]
    assert raw["total_alerts"] == 9227
    assert raw["tma_alerts"] == 6984
    assert raw["cma_alerts"] == 2243
    
    gov = res["alert_volume_comparison"]["governed_tma_only_policy"]
    assert gov["total_actionable_alerts"] == 7130
    assert gov["cma_alerts_suppressed"] == 2243
    
    # Student-level coverage
    cov = res["student_coverage_governed"]
    assert cov["unique_students_alerted"] == 3299
    assert cov["pct_eligible_students_alerted"] == 38.33
    assert cov["mean_alerts_per_alerted_student"] == 2.16
    assert cov["max_alerts_per_student"] == 9
    
    # Milestone sum
    milestones = res["milestone_level_alert_distribution"]
    assert sum(milestones.values()) == 7130
    assert milestones["TMA_1_Onboarding"] == 2280
    assert milestones["TMA_2_Midterm"] == 1801
    assert milestones["TMA_3_Capstone"] == 3049
    
    # Module sum
    modules = res["module_level_alert_distribution"]
    assert sum(modules.values()) == 7130
    assert modules["BBB"] == 2501
    assert modules["DDD"] == 1752
    assert modules["CCC"] == 1532
    assert modules["FFF"] == 772
    assert modules["GGG"] == 239
    assert modules["AAA"] == 187
    assert modules["EEE"] == 147


def test_governance_router_audit_and_gap_detection(audit_engine):
    """Verify router core behavior and identification of gaps."""
    res = audit_engine.run_governance_router_audit()
    
    # Core checks
    core = res["router_verification_results"]
    assert core["ggg_cma_suppressed"] is True
    assert core["ggg_tma_onboarding_calibrated_threshold"] is True
    assert core["ggg_tma_midterm_calibrated_threshold"] is True
    assert core["other_modules_retain_020_default"] is True
    assert core["invalid_metadata_safe_handling"] is True
    
    # Integration gap
    gap = res["integration_gap_analysis"]
    assert gap["cma_general_suppression_gap"] is True
    assert "suppresses CMAs exclusively for Module GGG" in gap["gap_description"]


def test_operational_readiness_deliverables_and_classifications():
    """Verify existence and schema of Task 5 deliverables and requirement status tags."""
    json_path = "reports/phase9_task5_operational_readiness_summary.json"
    md_path = "reports/phase9_task5_operational_readiness.md"
    
    assert os.path.exists(json_path), f"Missing summary: {json_path}"
    assert os.path.exists(md_path), f"Missing report: {md_path}"
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert "operational_requirements_status" in data
    status = data["operational_requirements_status"]
    
    valid_classifications = {"VERIFIED", "PARTIALLY VERIFIED", "NOT VERIFIED", "REQUIRES INSTITUTIONAL INPUT"}
    for req, val in status.items():
        assert val in valid_classifications, f"Invalid classification {val} for {req}"
        
    assert status["1. Calibration Methodology & Ranking Integrity"] == "VERIFIED"
    assert status["2. Counselor Workload & Alert Volume Viability"] == "REQUIRES INSTITUTIONAL INPUT"
    assert status["3. Module GGG Governance Router Core Functionality"] == "VERIFIED"
    assert status["4. Universal CMA Outreach Suppression Integration"] == "PARTIALLY VERIFIED"
    assert status["5. Static Registration & Coursework Telemetry"] == "VERIFIED"
    assert status["6. Dynamic VLE Clickstream Ingestion at 7-Day Horizon"] == "VERIFIED"
    assert status["7. Real-Time VLE Ingestion at 1-Day Horizon"] == "NOT VERIFIED"
    assert status["8. Actual Grade Publication Delay (14-Day Simulation)"] == "REQUIRES INSTITUTIONAL INPUT"
    assert status["9. Shadow-Mode Evaluation Protocol & Governance Gates"] == "VERIFIED"
    
    assert "DENIED" in data["audit_metadata"]["deployment_permission"]
    
    with open(md_path, "r") as f:
        md = f.read()
    assert "# Phase 9 Task 5: Operational Readiness & Deployment Feasibility Audit Report" in md
    assert "Production Approval Remains WITHHELD" in md
    assert "Operational Requirements Status Matrix" in md

