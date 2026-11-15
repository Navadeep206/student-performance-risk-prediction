"""
Unit and integration tests for Phase 9 Operational Gap Closure (Tasks A1 - A6).
"""

import os
import pytest
import math
from datetime import datetime, timezone

from src.governance_router import InstitutionalGovernanceRouter
from src.decision_audit import (
    pseudonymize_student_id,
    DecisionAuditRecord,
    DecisionAuditLogger,
    get_pseudonymization_secret
)
from src.incident_review import (
    ThresholdAwareIncidentReviewEngine,
    IncidentReviewRecord
)
from src.telemetry_interfaces import (
    CutoffEnforcementPolicy,
    VLEEventRecord,
    GradebookReleaseRecord,
    RegistrationRecord,
    CourseScheduleRecord,
    HorizonFeasibilityStatus
)
from src.shadow_mode import (
    ShadowModeManager,
    SecurityViolationError,
    GovernanceConfigurationError
)

TEST_SECRET = "test-cryptographic-secret-key-32bytes!!"


# =============================================================================
# TASK A1: UNIVERSAL CMA OUTREACH SUPPRESSION TESTS
# =============================================================================

def test_universal_cma_suppression_all_modules():
    """Verify that CMAs across all 7 modules (AAA-GGG) are strictly suppressed from automated outreach."""
    router = InstitutionalGovernanceRouter(universal_cma_suppression=True)
    all_modules = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    
    for mod in all_modules:
        res = router.route_prediction({
            "code_module": mod,
            "target_assessment_type": "CMA",
            "target_assessment_weight": 5.0,
            "scheduled_date": 54
        }, raw_prob=0.85)
        
        # Must suppress automated outreach regardless of probability
        assert res["suppress_automated_outreach"] is True, f"Module {mod} CMA was not suppressed!"
        assert res["student_outreach_enabled"] is False
        assert res["predicted_at_risk"] is False
        assert res["operational_threshold_applied"] is None
        assert res["governance_action"] == "AUTOMATED_OUTREACH_SUPPRESSED"
        
        if mod == "GGG":
            assert res["confidence_tier"] == "UNRELIABLE_DECOUPLED_QUIZ"
        else:
            assert res["confidence_tier"] == "RESEARCH_ONLY_TELEMETRY"


def test_tma_threshold_routing_all_modules():
    """Verify shadow-mode communication prohibition and threshold routing for GGG (0.10) and AAA-FFF (0.20)."""
    router = InstitutionalGovernanceRouter(tma_calibrated_threshold=0.10, global_threshold=0.20, shadow_mode=True)
    
    # 1. GGG TMA Onboarding (Day <= 70, prob = 0.12 >= tau = 0.10)
    res_ggg_early = router.route_prediction({"code_module": "GGG", "target_assessment_type": "TMA", "scheduled_date": 60}, 0.12)
    assert res_ggg_early["operational_threshold_applied"] == 0.10
    assert res_ggg_early["predicted_at_risk"] is True
    assert res_ggg_early["suppress_automated_outreach"] is True
    assert res_ggg_early["student_outreach_enabled"] is False
    assert res_ggg_early["governance_action"] == "SHADOW_MODE_ADVISORY_REVIEW"
    
    # 2. GGG TMA Capstone (Day > 70, prob = 0.12 >= tau = 0.10)
    res_ggg_late = router.route_prediction({"code_module": "GGG", "target_assessment_type": "TMA", "scheduled_date": 173}, 0.12)
    assert res_ggg_late["operational_threshold_applied"] == 0.10
    assert res_ggg_late["predicted_at_risk"] is True
    assert res_ggg_late["suppress_automated_outreach"] is True
    assert res_ggg_late["student_outreach_enabled"] is False
    assert res_ggg_late["governance_action"] == "SHADOW_MODE_ADVISORY_REVIEW"
    
    # 3. GGG TMA Below Threshold (prob = 0.08 < tau = 0.10)
    res_ggg_low = router.route_prediction({"code_module": "GGG", "target_assessment_type": "TMA", "scheduled_date": 173}, 0.08)
    assert res_ggg_low["operational_threshold_applied"] == 0.10
    assert res_ggg_low["predicted_at_risk"] is False
    assert res_ggg_low["suppress_automated_outreach"] is True
    assert res_ggg_low["student_outreach_enabled"] is False
    assert res_ggg_low["governance_action"] == "SHADOW_MODE_MONITORING"
    
    # 4. Modules AAA-FFF standard TMAs
    for mod in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]:
        # Below 0.20
        res_low = router.route_prediction({"code_module": mod, "target_assessment_type": "TMA", "scheduled_date": 100}, 0.15)
        assert res_low["operational_threshold_applied"] == 0.20
        assert res_low["predicted_at_risk"] is False
        assert res_low["suppress_automated_outreach"] is True
        assert res_low["student_outreach_enabled"] is False
        assert res_low["governance_action"] == "SHADOW_MODE_MONITORING"
        
        # Above 0.20
        res_high = router.route_prediction({"code_module": mod, "target_assessment_type": "TMA", "scheduled_date": 100}, 0.25)
        assert res_high["operational_threshold_applied"] == 0.20
        assert res_high["predicted_at_risk"] is True
        assert res_high["suppress_automated_outreach"] is True
        assert res_high["student_outreach_enabled"] is False
        assert res_high["governance_action"] == "SHADOW_MODE_ADVISORY_REVIEW"

    # 5. Outbound outreach attempt raises SecurityViolationError
    with pytest.raises(SecurityViolationError, match="External student outreach is strictly prohibited"):
        router.attempt_outreach(999, "Automated alert")

    # 6. Unconstrained simulation mode verification (when shadow_mode=False)
    router_unconstrained = InstitutionalGovernanceRouter(tma_calibrated_threshold=0.10, global_threshold=0.20, shadow_mode=False)
    res_sim = router_unconstrained.route_prediction({"code_module": "BBB", "target_assessment_type": "TMA", "scheduled_date": 100}, 0.25)
    assert res_sim["governance_action"] == "AUTOMATED_OUTREACH_ENABLED"
    assert res_sim["student_outreach_enabled"] is True


def test_metadata_validation_and_safe_handling():
    """Verify that unverified, missing, or invalid assessment metadata routes safely to manual review."""
    router = InstitutionalGovernanceRouter()
    
    # Invalid type
    res_invalid = router.route_prediction({"code_module": "BBB", "target_assessment_type": "INVALID_TYPE"}, 0.50)
    assert res_invalid["confidence_tier"] == "UNVERIFIED_ASSESSMENT_TYPE"
    assert res_invalid["governance_action"] == "MANUAL_ADVISOR_REVIEW"
    assert res_invalid["suppress_automated_outreach"] is True
    assert res_invalid["student_outreach_enabled"] is False
    assert res_invalid["predicted_at_risk"] is False
    
    # Missing type
    res_missing = router.route_prediction({"code_module": "CCC"}, 0.40)
    assert res_missing["confidence_tier"] == "UNVERIFIED_ASSESSMENT_TYPE"
    assert res_missing["suppress_automated_outreach"] is True
    assert res_missing["student_outreach_enabled"] is False


# =============================================================================
# TASK A2 & A3: PSEUDONYMIZATION & DECISION AUDIT LOGGING TESTS
# =============================================================================

def test_keyed_hmac_pseudonymization_properties():
    """Verify keyed HMAC-SHA256 deterministic behavior and secret-dependence."""
    student_id = 1234567
    
    # Deterministic with identical key
    p1 = pseudonymize_student_id(student_id, secret_key=TEST_SECRET)
    p2 = pseudonymize_student_id(student_id, secret_key=TEST_SECRET)
    assert p1 == p2
    assert len(p1) == 64
    assert str(student_id) not in p1
    
    # Secret-dependent (re-keying / rotation yields distinct hash)
    p3 = pseudonymize_student_id(student_id, secret_key="completely-different-secret-rotation-key!")
    assert p1 != p3
    
    # Short key rejection (< 16 chars)
    with pytest.raises(ValueError, match="at least 16 characters"):
        pseudonymize_student_id(student_id, secret_key="too-short")
        
    # Empty student ID rejection
    with pytest.raises(ValueError, match="cannot be empty"):
        pseudonymize_student_id("", secret_key=TEST_SECRET)


def test_decision_audit_record_validation():
    """Verify strict validation and immutability of DecisionAuditRecord."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pseudonym = pseudonymize_student_id(999888, secret_key=TEST_SECRET)
    
    # Valid record
    rec = DecisionAuditRecord(
        prediction_id="pred-uuid-1234",
        timestamp_utc=now_utc,
        pseudonymized_student_id=pseudonym,
        code_module="BBB",
        code_presentation="2014J",
        id_assessment=14990,
        assessment_type="TMA",
        scheduled_date=54,
        model_name="HistGradientBoostingClassifier",
        model_version="v1.2.0-phase9",
        feature_pipeline_version="v1.2.0-remediated",
        input_feature_hash="hash-12345678",
        raw_risk_probability=0.35,
        operational_threshold=0.20,
        predicted_at_risk=True,
        governance_routing_action="AUTOMATED_OUTREACH_ENABLED",
        suppress_automated_outreach=False,
        routing_reason="Standard TMA protocol"
    )
    assert rec.raw_risk_probability == 0.35
    
    # Immutability check
    with pytest.raises(Exception):
        rec.raw_risk_probability = 0.50
        
    # Rejection of invalid probability (< 0 or > 1 or NaN)
    with pytest.raises(ValueError, match="out of bounds"):
        DecisionAuditRecord(
            prediction_id="pred-uuid-1", timestamp_utc=now_utc, pseudonymized_student_id=pseudonym,
            code_module="BBB", code_presentation="2014J", id_assessment=14990, assessment_type="TMA",
            scheduled_date=54, model_name="HGB", model_version="v1", feature_pipeline_version="v1",
            input_feature_hash="h1", raw_risk_probability=1.5, operational_threshold=0.20,
            predicted_at_risk=True, governance_routing_action="ACTION", suppress_automated_outreach=False,
            routing_reason="reason"
        )
        
    with pytest.raises(ValueError, match="must be a valid float"):
        DecisionAuditRecord(
            prediction_id="pred-uuid-1", timestamp_utc=now_utc, pseudonymized_student_id=pseudonym,
            code_module="BBB", code_presentation="2014J", id_assessment=14990, assessment_type="TMA",
            scheduled_date=54, model_name="HGB", model_version="v1", feature_pipeline_version="v1",
            input_feature_hash="h1", raw_risk_probability=float("nan"), operational_threshold=0.20,
            predicted_at_risk=True, governance_routing_action="ACTION", suppress_automated_outreach=False,
            routing_reason="reason"
        )

    # Rejection of missing model metadata
    with pytest.raises(ValueError, match="model_name must be specified"):
        DecisionAuditRecord(
            prediction_id="pred-uuid-1", timestamp_utc=now_utc, pseudonymized_student_id=pseudonym,
            code_module="BBB", code_presentation="2014J", id_assessment=14990, assessment_type="TMA",
            scheduled_date=54, model_name="", model_version="v1", feature_pipeline_version="v1",
            input_feature_hash="h1", raw_risk_probability=0.3, operational_threshold=0.20,
            predicted_at_risk=True, governance_routing_action="ACTION", suppress_automated_outreach=False,
            routing_reason="reason"
        )


def test_concurrent_decision_audit_logging():
    """Verify that concurrent writes to DecisionAuditLogger do not drop or corrupt records."""
    import concurrent.futures
    logger = DecisionAuditLogger()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pseudonym = pseudonymize_student_id(999888, secret_key=TEST_SECRET)
    
    def log_item(idx):
        rec = DecisionAuditRecord(
            prediction_id=f"concurrent-uuid-{idx}",
            timestamp_utc=now_utc,
            pseudonymized_student_id=pseudonym,
            code_module="BBB",
            code_presentation="2014J",
            id_assessment=14990,
            assessment_type="TMA",
            scheduled_date=54,
            model_name="HistGradientBoostingClassifier",
            model_version="1.0.0",
            feature_pipeline_version="1.0.0",
            input_feature_hash=f"hash-{idx}",
            raw_risk_probability=0.25,
            operational_threshold=0.20,
            predicted_at_risk=True,
            governance_routing_action="SHADOW_MODE_ADVISORY_REVIEW",
            suppress_automated_outreach=True,
            routing_reason="Concurrent test",
            student_outreach_enabled=False
        )
        logger.log(rec)
        return idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(log_item, i) for i in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    assert len(results) == 50
    assert len(logger.get_records()) == 50
    # Verify no raw student IDs or secrets logged
    for r in logger.get_records():
        assert str(999888) not in r.pseudonymized_student_id
        assert TEST_SECRET not in r.to_json()


# =============================================================================
# TASK A4: THRESHOLD-AWARE INCIDENT REVIEW TESTS
# =============================================================================

def test_threshold_aware_incident_review():
    """Verify threshold-aware incident review for standard TMA, GGG TMA, suppressed CMA, and unknown metadata."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pseudonym = pseudonymize_student_id(111222, secret_key=TEST_SECRET)
    
    # 1. Standard BBB TMA True Positive (prob=0.25 >= tau=0.20, actual_score=40)
    rec_bbb_tp = DecisionAuditRecord(
        prediction_id="p-1", timestamp_utc=now_utc, pseudonymized_student_id=pseudonym,
        code_module="BBB", code_presentation="2014J", id_assessment=14990, assessment_type="TMA",
        scheduled_date=54, model_name="HGB", model_version="v1", feature_pipeline_version="v1",
        input_feature_hash="h1", raw_risk_probability=0.25, operational_threshold=0.20,
        predicted_at_risk=True, governance_routing_action="AUTOMATED_OUTREACH_ENABLED",
        suppress_automated_outreach=False, routing_reason="Standard protocol"
    )
    rev_bbb_tp = ThresholdAwareIncidentReviewEngine.review_decision(rec_bbb_tp, actual_score=40.0)
    assert rev_bbb_tp.outcome_classification == "TRUE_POSITIVE"
    assert rev_bbb_tp.is_incident_eligible is True
    assert rev_bbb_tp.operational_threshold_applied == 0.20
    
    # 2. Standard BBB TMA False Positive (prob=0.25, actual_score=80)
    rev_bbb_fp = ThresholdAwareIncidentReviewEngine.review_decision(rec_bbb_tp, actual_score=80.0)
    assert rev_bbb_fp.outcome_classification == "FALSE_POSITIVE"
    assert "Does NOT inherently represent a model defect" in rev_bbb_fp.investigation_notes
    
    # 3. Module GGG TMA Flagged at 0.12 (prob=0.12 >= tau=0.10, actual_score=35)
    rec_ggg_tp = DecisionAuditRecord(
        prediction_id="p-2", timestamp_utc=now_utc, pseudonymized_student_id=pseudonym,
        code_module="GGG", code_presentation="2014J", id_assessment=37415, assessment_type="TMA",
        scheduled_date=61, model_name="HGB", model_version="v1", feature_pipeline_version="v1",
        input_feature_hash="h1", raw_risk_probability=0.12, operational_threshold=0.10,
        predicted_at_risk=True, governance_routing_action="SUPPORTIVE_CHECK_IN_ONLY",
        suppress_automated_outreach=False, routing_reason="GGG TMA calibrated"
    )
    rev_ggg_tp = ThresholdAwareIncidentReviewEngine.review_decision(rec_ggg_tp, actual_score=35.0)
    assert rev_ggg_tp.outcome_classification == "TRUE_POSITIVE"
    assert rev_ggg_tp.operational_threshold_applied == 0.10
    
    # 4. CMA Prediction (prob=0.45, actual_score=20) -> Must be EXCLUDED from outreach incidents
    rec_cma = DecisionAuditRecord(
        prediction_id="p-3", timestamp_utc=now_utc, pseudonymized_student_id=pseudonym,
        code_module="CCC", code_presentation="2014J", id_assessment=24290, assessment_type="CMA",
        scheduled_date=32, model_name="HGB", model_version="v1", feature_pipeline_version="v1",
        input_feature_hash="h1", raw_risk_probability=0.45, operational_threshold=None,
        predicted_at_risk=False, governance_routing_action="AUTOMATED_OUTREACH_SUPPRESSED",
        suppress_automated_outreach=True, routing_reason="Universal CMA suppression"
    )
    rev_cma = ThresholdAwareIncidentReviewEngine.review_decision(rec_cma, actual_score=20.0)
    assert rev_cma.is_incident_eligible is False
    assert rev_cma.outcome_classification == "CMA_RESEARCH_TELEMETRY_EXCLUDED"


# =============================================================================
# TASK A5: TELEMETRY & GRADEBOOK INTEGRATION READINESS TESTS
# =============================================================================

def test_telemetry_and_gradebook_interfaces():
    """Verify temporal cutoff enforcement, late event rejection, and production grade release checks."""
    # 1. Eligible VLE event (date=40 <= cutoff=47)
    vle_ok = VLEEventRecord(
        student_id=123, code_module="BBB", code_presentation="2014J", id_site=500,
        activity_type="oucontent", date=40, sum_click=5,
        event_timestamp_utc="2014-11-01T10:00:00Z", ingestion_timestamp_utc="2014-11-01T23:00:00Z"
    )
    ok, msg = CutoffEnforcementPolicy.evaluate_vle_event(vle_ok, cutoff_date=47, batch_execution_timestamp_utc="2014-11-02T04:00:00Z")
    assert ok is True
    assert "ELIGIBLE" in msg
    
    # 2. Late VLE event (date=50 > cutoff=47)
    vle_late = VLEEventRecord(
        student_id=123, code_module="BBB", code_presentation="2014J", id_site=500,
        activity_type="oucontent", date=50, sum_click=5,
        event_timestamp_utc="2014-11-10T10:00:00Z", ingestion_timestamp_utc="2014-11-10T23:00:00Z"
    )
    ok_late, msg_late = CutoffEnforcementPolicy.evaluate_vle_event(vle_late, cutoff_date=47)
    assert ok_late is False
    assert "REJECTED" in msg_late
    
    # 3. Late Ingestion VLE event (event date is ok, but ingestion timestamp is after batch execution)
    ok_ingest, msg_ingest = CutoffEnforcementPolicy.evaluate_vle_event(
        vle_ok, cutoff_date=47, batch_execution_timestamp_utc="2014-11-01T12:00:00Z"
    )
    assert ok_ingest is False
    assert "ingestion timestamp" in msg_ingest
    
    # 4. Production Gradebook Interface: Unpublished grade
    grade_unpub = GradebookReleaseRecord(
        student_id=123, id_assessment=14990, date_submitted=30, score=75.0,
        publication_timestamp_utc=None, ingestion_timestamp_utc="2014-11-05T00:00:00Z"
    )
    avail_unpub, msg_unpub = CutoffEnforcementPolicy.is_grade_available_production(grade_unpub, cutoff_timestamp_utc="2014-11-06T00:00:00Z")
    assert avail_unpub is False
    assert "UNAVAILABLE" in msg_unpub
    
    # 5. Production Gradebook Interface: Published before cutoff
    grade_pub = GradebookReleaseRecord(
        student_id=123, id_assessment=14990, date_submitted=30, score=75.0,
        publication_timestamp_utc="2014-11-04T12:00:00Z", ingestion_timestamp_utc="2014-11-04T18:00:00Z"
    )
    avail_pub, msg_pub = CutoffEnforcementPolicy.is_grade_available_production(grade_pub, cutoff_timestamp_utc="2014-11-06T00:00:00Z")
    assert avail_pub is True
    assert "AVAILABLE" in msg_pub
    
    # 6. Horizon feasibility status
    assert "Candidate" in HorizonFeasibilityStatus.RESEARCH_TO_OPERATIONAL_CANDIDATE.value
    assert "unverified" in HorizonFeasibilityStatus.OPERATIONALLY_UNVERIFIED_FRAGILE.value.lower()


# =============================================================================
# TASK A6: SHADOW-MODE SAFETY GATES TESTS
# =============================================================================

def test_shadow_mode_safety_gates():
    """Verify shadow-mode hard-coded outreach locks, configuration fail-safes, and drift monitoring."""
    # 1. Outreach lock
    shadow_mgr = ShadowModeManager(
        model_name="HistGradientBoostingClassifier",
        model_version="v1.2.0-phase9",
        feature_pipeline_version="v1.2.0-remediated",
        secret_key=TEST_SECRET
    )
    assert shadow_mgr.ALLOW_STUDENT_OUTREACH is False
    with pytest.raises(SecurityViolationError, match="External student outreach is strictly prohibited"):
        shadow_mgr.attempt_outreach(123456, "Outreach message")
        
    # 2. Missing configuration fail-safe
    with pytest.raises(GovernanceConfigurationError, match="requires explicit model_name"):
        ShadowModeManager(model_name="", model_version="v1", feature_pipeline_version="v1", secret_key=TEST_SECRET)
        
    # 3. Missing secret fail-safe
    old_env = os.environ.get("STUDENT_PSEUDONYMIZATION_SECRET")
    try:
        os.environ["STUDENT_PSEUDONYMIZATION_SECRET"] = ""
        with pytest.raises(GovernanceConfigurationError, match="Missing STUDENT_PSEUDONYMIZATION_SECRET"):
            ShadowModeManager(model_name="HGB", model_version="v1", feature_pipeline_version="v1", secret_key=None)
    finally:
        if old_env is None:
            os.environ.pop("STUDENT_PSEUDONYMIZATION_SECRET", None)
        else:
            os.environ["STUDENT_PSEUDONYMIZATION_SECRET"] = old_env


# =============================================================================
# TASK A7: REPORT DELIVERABLES VERIFICATION
# =============================================================================

def test_operational_gap_closure_artifacts():
    """Verify that Phase 9 Operational Gap Closure deliverables exist and confirm all gaps closed."""
    json_path = "reports/phase9_operational_gap_closure_summary.json"
    md_path = "reports/phase9_operational_gap_closure.md"
    
    assert os.path.exists(json_path), f"Missing summary: {json_path}"
    assert os.path.exists(md_path), f"Missing report: {md_path}"
    
    import json
    with open(json_path, "r") as f:
        data = json.load(f)
        
    gaps = data["closed_gaps"]
    assert gaps["gap_1_universal_cma_suppression"]["verification_all_modules_suppressed"] is True
    assert gaps["gap_2_complete_decision_audit_logging"]["record_validated"] is True
    assert gaps["gap_3_privacy_safe_pseudonymization"]["deterministic_with_same_key"] is True
    assert gaps["gap_3_privacy_safe_pseudonymization"]["distinct_with_different_keys"] is True
    assert gaps["gap_4_threshold_aware_incident_review"]["cma_excluded_from_incidents"] is True
    assert gaps["gap_6_shadow_mode_safety_locks"]["outreach_safety_lock_active"] is True
    assert "WITHHELD" in data["metadata"]["production_approval_status"]
    
    with open(md_path, "r") as f:
        md = f.read()
    assert "# Phase 9: Operational Gap Closure Report" in md
    assert "Universal CMA Outreach Suppression" in md
    assert "Production Deployment Approval Remains WITHHELD" in md
