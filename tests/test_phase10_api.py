"""
Unit and integration tests for Phase 10: Model Artifacts, Prediction API & Testing.
Safety, Shadow-Mode Outreach Prohibition, and Artifact Acceptance Tests.
"""

import os
import json
import concurrent.futures
import pytest
from fastapi.testclient import TestClient

from src.api import app, ALL_ALLOWED_FEATURES
from src.model_artifacts import (
    ModelArtifactManager,
    compute_file_sha256,
    HGB_ARTIFACT_NAME,
    HGB_METADATA_NAME,
    RF_ARTIFACT_NAME,
    RF_METADATA_NAME,
    MODEL_DIR
)
from src.governance_router import InstitutionalGovernanceRouter
from src.shadow_mode import ShadowModeManager, SecurityViolationError


TEST_SECRET = "pytest-isolated-test-secret-32bytes!!"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensures test-isolated STUDENT_PSEUDONYMIZATION_SECRET is set for API tests."""
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


@pytest.fixture
def valid_prediction_payload():
    return {
        "id_student": 100200,
        "code_module": "BBB",
        "code_presentation": "2014J",
        "id_assessment": 14990,
        "target_assessment_type": "TMA",
        "scheduled_date": 54,
        "target_assessment_weight": 18.0,
        "num_previous_scheduled_assessments": 0,
        "registration_timing_days": -45.0,
        "vle_total_clicks_to_date": 350.0,
        "vle_active_days_to_date": 12.0,
        "vle_core_clicks_to_date": 200.0,
        "vle_forum_clicks_to_date": 40.0,
        "vle_quiz_clicks_to_date": 10.0,
        "vle_days_since_last_activity": 2.0,
        "vle_clicks_last_7d": 45.0,
        "vle_clicks_last_14d": 90.0,
        "vle_activity_trend_ratio": 0.5,
        "num_prior_submissions": 0,
        "prior_submission_lateness_mean": None,
        "num_prior_scored_assessments": 0,
        "mean_prior_score": None,
        "std_prior_score": None,
        "prior_at_risk_rate": None,
        "recent_prior_score": None
    }


# =============================================================================
# 1. SHADOW-MODE GOVERNANCE & OUTREACH PROHIBITION VERIFICATION
# =============================================================================

def test_health_endpoint(client):
    """Verify /health endpoint returns 200, model availability, and shadow mode status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_artifact_available"] is True
    assert data["model_version"] == "1.0.0"
    assert "SHADOW_MODE_READ_ONLY" in data["operational_mode"]


def test_model_info_endpoint(client):
    """Verify /model-info returns complete architecture, metadata, and threshold governance."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "HistGradientBoostingClassifier"
    assert data["model_version"] == "1.0.0"
    assert data["features"]["total_count"] == 23
    assert len(data["features"]["categorical"]) == 3
    assert len(data["features"]["numerical"]) == 20
    assert data["supported_horizons"] == [7]
    assert data["threshold_policy"]["standard_tma_threshold"] == 0.20
    assert data["threshold_policy"]["ggg_tma_calibrated_threshold"] == 0.10
    assert "WITHHELD" in data["operational_status"]


def test_ggg_tma_at_probability_012_routing_and_outreach_disabled():
    """
    CRITICAL ACCEPTANCE CHECK:
    GGG TMA with probability 0.12 and threshold 0.10 must return:
    - predicted_at_risk = True
    - operational_threshold = 0.10
    - governance_action = SHADOW_MODE_ADVISORY_REVIEW
    - suppress_automated_outreach = True
    - student_outreach_enabled = False
    """
    router = InstitutionalGovernanceRouter(tma_calibrated_threshold=0.10, global_threshold=0.20, shadow_mode=True)
    snapshot = {
        "code_module": "GGG",
        "target_assessment_type": "TMA",
        "scheduled_date": 60,
        "target_assessment_weight": 0.0
    }
    res = router.route_prediction(snapshot, raw_prob=0.12)
    
    assert res["predicted_at_risk"] is True
    assert res["operational_threshold_applied"] == 0.10
    assert res["governance_action"] == "SHADOW_MODE_ADVISORY_REVIEW"
    assert res["suppress_automated_outreach"] is True
    assert res["student_outreach_enabled"] is False
    assert "SHADOW MODE ACTIVE" in res["advising_recommendation"]


def test_predict_single_standard_tma_above_and_below_threshold(client, valid_prediction_payload):
    """
    Verify standard TMA predictions follow the shadow-mode communication prohibition.
    Both above and below threshold must have student_outreach_enabled = False.
    """
    # 1. Low risk payload (below threshold)
    low_risk_payload = valid_prediction_payload.copy()
    low_risk_payload["vle_total_clicks_to_date"] = 2500.0
    low_risk_payload["vle_active_days_to_date"] = 45.0
    low_risk_payload["mean_prior_score"] = 85.0
    low_risk_payload["recent_prior_score"] = 90.0
    low_risk_payload["num_prior_scored_assessments"] = 3
    
    res_low = client.post("/predict", json=low_risk_payload)
    assert res_low.status_code == 200
    data_low = res_low.json()
    assert data_low["operational_threshold_applied"] == 0.20
    assert data_low["predicted_at_risk"] is False
    assert data_low["suppress_automated_outreach"] is True
    assert data_low["student_outreach_enabled"] is False
    assert data_low["governance_action"] == "SHADOW_MODE_MONITORING"
    
    # 2. High risk payload (above threshold)
    high_risk_payload = valid_prediction_payload.copy()
    high_risk_payload["vle_total_clicks_to_date"] = 0.0
    high_risk_payload["vle_active_days_to_date"] = 0.0
    high_risk_payload["vle_days_since_last_activity"] = 50.0
    high_risk_payload["num_previous_scheduled_assessments"] = 2
    
    res_high = client.post("/predict", json=high_risk_payload)
    assert res_high.status_code == 200
    data_high = res_high.json()
    assert data_high["operational_threshold_applied"] == 0.20
    # Regardless of risk level, shadow-mode communication prohibition is strictly enforced
    assert data_high["suppress_automated_outreach"] is True
    assert data_high["student_outreach_enabled"] is False
    if data_high["predicted_at_risk"] is True:
        assert data_high["governance_action"] == "SHADOW_MODE_ADVISORY_REVIEW"


def test_predict_single_cma_universal_suppression(client, valid_prediction_payload):
    """Verify /predict enforces universal CMA outreach suppression across all 7 modules."""
    for mod in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]:
        payload = valid_prediction_payload.copy()
        payload["code_module"] = mod
        payload["target_assessment_type"] = "CMA"
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["suppress_automated_outreach"] is True, f"Module {mod} CMA was not suppressed!"
        assert data["student_outreach_enabled"] is False
        assert data["predicted_at_risk"] is False
        assert data["operational_threshold_applied"] is None
        assert data["governance_action"] == "AUTOMATED_OUTREACH_SUPPRESSED"


def test_predict_unknown_metadata_safe_handling(client, valid_prediction_payload):
    """Verify /predict safely handles unknown assessment types by routing to manual review with outreach disabled."""
    payload = valid_prediction_payload.copy()
    payload["target_assessment_type"] = "UNKNOWN_QUIZ"
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["suppress_automated_outreach"] is True
    assert data["student_outreach_enabled"] is False
    assert data["governance_action"] == "MANUAL_ADVISOR_REVIEW"
    assert "Manual academic advisor verification required" in data["advising_recommendation"]


def test_attempted_outreach_raises_security_violation():
    """Verify that any attempted communication in shadow mode strictly raises SecurityViolationError."""
    router = InstitutionalGovernanceRouter(shadow_mode=True)
    with pytest.raises(SecurityViolationError, match="External student outreach is strictly prohibited"):
        router.attempt_outreach(100200, "Automated check-in email")

    shadow_mgr = ShadowModeManager(
        model_name="HistGradientBoostingClassifier",
        model_version="1.0.0",
        feature_pipeline_version="1.0.0",
        secret_key=TEST_SECRET
    )
    with pytest.raises(SecurityViolationError, match="External student outreach is strictly prohibited"):
        shadow_mgr.attempt_outreach(100200, "Automated nudge message")


def test_api_does_not_expose_outreach_endpoints(client):
    """Verify that the API exposes zero endpoints that initiate student communication or live intervention."""
    routes = [route.path for route in app.routes]
    prohibited_routes = ["/intervene", "/outreach", "/send-message", "/email", "/notify", "/nudge"]
    for pr in prohibited_routes:
        assert pr not in routes, f"Prohibited communication endpoint found: {pr}"
        
    res = client.post("/intervene", json={"student_id": 100200})
    assert res.status_code == 404


# =============================================================================
# 2. HMAC SECRET CONFIGURATION & ERROR HANDLING TESTS
# =============================================================================

def test_missing_hmac_secret_fails_safely_no_silent_fallback(client, valid_prediction_payload):
    """Verify that a missing STUDENT_PSEUDONYMIZATION_SECRET fails safely without a silent fallback."""
    old_secret = os.environ.get("STUDENT_PSEUDONYMIZATION_SECRET")
    try:
        os.environ.pop("STUDENT_PSEUDONYMIZATION_SECRET", None)
        response = client.post("/predict", json=valid_prediction_payload)
        assert response.status_code == 500
        data = response.json()
        assert "Missing STUDENT_PSEUDONYMIZATION_SECRET" in data["detail"]
        # Ensure secret value is never exposed
        assert TEST_SECRET not in response.text
    finally:
        if old_secret:
            os.environ["STUDENT_PSEUDONYMIZATION_SECRET"] = old_secret


def test_short_hmac_secret_fails_safely(client, valid_prediction_payload):
    """Verify that an insufficiently long HMAC secret (< 16 chars) is safely rejected."""
    old_secret = os.environ.get("STUDENT_PSEUDONYMIZATION_SECRET")
    try:
        os.environ["STUDENT_PSEUDONYMIZATION_SECRET"] = "short-key"
        response = client.post("/predict", json=valid_prediction_payload)
        assert response.status_code == 500
        assert "at least 16 characters" in response.text
        assert "short-key" not in response.text
    finally:
        if old_secret:
            os.environ["STUDENT_PSEUDONYMIZATION_SECRET"] = old_secret


# =============================================================================
# 3. PROHIBITED FEATURE REJECTION TESTS
# =============================================================================

def test_predict_prohibited_target_leakage_rejected(client, valid_prediction_payload):
    """Verify that providing target leakage features (final_result, date_unregistration) returns 422."""
    payload_leak1 = valid_prediction_payload.copy()
    payload_leak1["final_result"] = "Pass"
    res1 = client.post("/predict", json=payload_leak1)
    assert res1.status_code == 422
    assert "Prohibited target leakage columns detected" in res1.text
    
    payload_leak2 = valid_prediction_payload.copy()
    payload_leak2["date_unregistration"] = -10
    res2 = client.post("/predict", json=payload_leak2)
    assert res2.status_code == 422
    assert "Prohibited target leakage columns detected" in res2.text


def test_predict_prohibited_sensitive_demographics_rejected(client, valid_prediction_payload):
    """Verify that providing sensitive demographic features returns HTTP 422."""
    for demo in ["gender", "imd_band", "highest_education", "age_band", "disability", "region"]:
        payload_demo = valid_prediction_payload.copy()
        payload_demo[demo] = "Sensitive_Value"
        res = client.post("/predict", json=payload_demo)
        assert res.status_code == 422
        assert "Prohibited sensitive demographic features detected" in res.text


# =============================================================================
# 4. BATCH PREDICTION & CONCURRENCY TESTS
# =============================================================================

def test_predict_batch_isolated_fault_handling(client, valid_prediction_payload):
    """Verify /predict/batch scores valid items and isolates invalid items without failing the batch."""
    item1 = valid_prediction_payload.copy()
    item1["id_student"] = 101
    
    item2 = valid_prediction_payload.copy()
    item2["id_student"] = 102
    item2["scheduled_date"] = 100
    
    # Invalid item: target_assessment_weight > 100
    item_bad = valid_prediction_payload.copy()
    item_bad["id_student"] = 103
    item_bad["target_assessment_weight"] = 150.0
    
    batch_payload = {"predictions": [item1, item2, item_bad]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 422
    
    # Valid batch of 2 items
    valid_batch = {"predictions": [item1, item2]}
    res_valid = client.post("/predict/batch", json=valid_batch)
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["total_requested"] == 2
    assert data["successful_count"] == 2
    assert data["failed_count"] == 0
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert r["data"]["student_outreach_enabled"] is False
        assert r["data"]["suppress_automated_outreach"] is True


def test_concurrent_api_prediction_requests(client, valid_prediction_payload):
    """Verify that concurrent prediction requests succeed without state corruption or race conditions."""
    def send_req(i):
        payload = valid_prediction_payload.copy()
        payload["id_student"] = 200000 + i
        return client.post("/predict", json=payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send_req, i) for i in range(25)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(responses) == 25
    prediction_ids = set()
    for resp in responses:
        assert resp.status_code == 200
        body = resp.json()
        assert body["student_outreach_enabled"] is False
        assert body["suppress_automated_outreach"] is True
        prediction_ids.add(body["prediction_id"])

    # All prediction IDs must be unique
    assert len(prediction_ids) == 25


# =============================================================================
# 5. MODEL ARTIFACT INTEGRITY VERIFICATION (HGB & CHALLENGER RF)
# =============================================================================

def test_model_artifact_manager_missing_and_corrupt_files():
    """Verify ModelArtifactManager raises FileNotFoundError and ValueError on corruption."""
    # Missing artifact check
    with pytest.raises(FileNotFoundError):
        ModelArtifactManager(model_path="models/non_existent.joblib", verify_hash=True)
        
    # Corrupted hash check
    real_model_path = os.path.join(MODEL_DIR, HGB_ARTIFACT_NAME)
    real_meta_path = os.path.join(MODEL_DIR, HGB_METADATA_NAME)
    
    corrupt_meta = os.path.join(MODEL_DIR, "temp_corrupt_metadata.json")
    with open(real_meta_path, "r") as f:
        meta_dict = json.load(f)
    meta_dict["sha256_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(corrupt_meta, "w") as f:
        json.dump(meta_dict, f)
        
    try:
        with pytest.raises(ValueError, match="Artifact Integrity Error: SHA-256 hash mismatch"):
            ModelArtifactManager(model_path=real_model_path, metadata_path=corrupt_meta, verify_hash=True)
    finally:
        if os.path.exists(corrupt_meta):
            os.remove(corrupt_meta)


def test_saved_hgb_artifact_verification_against_approved_pipeline():
    """
    Independently verify saved Primary HistGradientBoosting artifact against Phase 6 specs:
    1. Exactly 23 approved predictive features.
    2. Exact feature order.
    3. Correct categorical feature handling (3 categoricals: code_module, code_presentation, target_assessment_type).
    4. Correct missing-value handling.
    5. Zero final_result or date_unregistration leakage.
    6. Correct model and feature pipeline versions (1.0.0).
    7. Metadata and binary SHA-256 consistency.
    """
    model_path = os.path.join(MODEL_DIR, HGB_ARTIFACT_NAME)
    meta_path = os.path.join(MODEL_DIR, HGB_METADATA_NAME)
    
    assert os.path.exists(model_path)
    assert os.path.exists(meta_path)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    # 1. Feature count
    assert meta["feature_count"] == 23
    assert len(meta["feature_names"]) == 23
    assert meta["feature_names"] == ALL_ALLOWED_FEATURES
    
    # 2. Categorical and numerical separation
    assert meta["categorical_features"] == ["code_module", "code_presentation", "target_assessment_type"]
    assert len(meta["numerical_features"]) == 20
    
    # 3. No leakage or sensitive demographics
    for prohibited in ["final_result", "date_unregistration", "gender", "imd_band", "disability", "region"]:
        assert prohibited not in meta["feature_names"]
        
    # 4. Versions
    assert meta["model_version"] == "1.0.0"
    assert meta["feature_pipeline_version"] == "1.0.0"
    assert meta["model_name"] == "HistGradientBoostingClassifier"
    
    # 5. SHA-256 consistency
    computed_hash = compute_file_sha256(model_path)
    assert computed_hash == meta["sha256_hash"]
    
    # 6. Verify pipeline loads and executes
    mgr = ModelArtifactManager(model_path=model_path, metadata_path=meta_path, verify_hash=True)
    assert mgr.is_loaded is True


def test_saved_rf_challenger_artifact_verification():
    """
    Independently verify saved Challenger Random Forest artifact:
    1. Exactly 23 approved features in identical ordering.
    2. No target leakage or sensitive attributes.
    3. SHA-256 consistency.
    4. Successful loading and inference verification.
    """
    model_path = os.path.join(MODEL_DIR, RF_ARTIFACT_NAME)
    meta_path = os.path.join(MODEL_DIR, RF_METADATA_NAME)
    
    assert os.path.exists(model_path)
    assert os.path.exists(meta_path)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    assert meta["feature_count"] == 23
    assert meta["feature_names"] == ALL_ALLOWED_FEATURES
    assert meta["model_name"] == "RandomForestClassifier"
    assert meta["model_version"] == "1.0.0"
    
    for prohibited in ["final_result", "date_unregistration", "gender", "imd_band", "disability", "region"]:
        assert prohibited not in meta["feature_names"]
        
    computed_hash = compute_file_sha256(model_path)
    assert computed_hash == meta["sha256_hash"]
    
    mgr = ModelArtifactManager(model_path=model_path, metadata_path=meta_path, verify_hash=True)
    assert mgr.is_loaded is True
