"""
Student At-Risk Prediction Inference API.
Phase 10 (Tasks 10.3 and 10.4).

Provides:
- GET /health: Health check, model loaded status, and version.
- GET /model-info: Model metadata, provenance, thresholds, and governance status.
- POST /predict: Single student assessment prediction with governance routing & HMAC pseudonymization.
- POST /predict/batch: Batch prediction with isolated record validation.
- Enforces strict rejection of prohibited features (target leakage and sensitive demographics).
- Implements universal CMA outreach suppression across all modules.
- Formulates responses in advisory, non-punitive language.
"""

import os
import sys
import uuid
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_artifacts import ModelArtifactManager
from src.governance_router import InstitutionalGovernanceRouter
from src.decision_audit import pseudonymize_student_id, DecisionAuditRecord, DecisionAuditLogger
from src.baseline_models import PROHIBITED_COLUMNS, ALL_ALLOWED_FEATURES
from src.analytics_routes import router as analytics_router


app = FastAPI(
    title="Student At-Risk Prediction API",
    description="Operational inference service for prospective academic early warning and risk assessment.",
    version="1.0.0"
)

# Mount analytics routes directly to preserve APIRoute list compatibility
for _r in analytics_router.routes:
    app.router.routes.append(_r)

# Global Manager Instances
model_manager: Optional[ModelArtifactManager] = None
governance_router: Optional[InstitutionalGovernanceRouter] = None
audit_logger = DecisionAuditLogger()


def get_model_manager() -> ModelArtifactManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelArtifactManager(verify_hash=True)
    return model_manager


def get_governance_router() -> InstitutionalGovernanceRouter:
    global governance_router
    if governance_router is None:
        governance_router = InstitutionalGovernanceRouter(universal_cma_suppression=True)
    return governance_router


@app.on_event("startup")
def startup_event():
    get_model_manager()
    get_governance_router()


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

SENSITIVE_DEMOGRAPHIC_COLUMNS = {
    "gender", "region", "highest_education", "imd_band", "age_band", "disability"
}
LEAKAGE_COLUMNS = {
    "final_result", "date_unregistration", "target_assessment_score", "next_assessment_score", "at_risk"
}


class PredictionItem(BaseModel):
    id_student: Union[int, str] = Field(..., description="Student identifier (pseudonymized on ingestion)")
    code_module: str = Field(..., description="Course module code (e.g., 'BBB', 'GGG')")
    code_presentation: str = Field(..., description="Semester presentation code (e.g., '2014J')")
    id_assessment: int = Field(..., description="Target assessment ID")
    target_assessment_type: str = Field(..., description="Assessment type: 'TMA', 'CMA', or 'Exam'")
    scheduled_date: int = Field(..., description="Assessment submission deadline relative day")
    target_assessment_weight: float = Field(0.0, ge=0.0, le=100.0, description="Assessment weight percentage")
    
    # 20 Numerical Features
    num_previous_scheduled_assessments: int = Field(0, ge=0)
    registration_timing_days: Optional[float] = Field(None, description="Days before course start student registered")
    vle_total_clicks_to_date: float = Field(0.0, ge=0.0)
    vle_active_days_to_date: float = Field(0.0, ge=0.0)
    vle_core_clicks_to_date: float = Field(0.0, ge=0.0)
    vle_forum_clicks_to_date: float = Field(0.0, ge=0.0)
    vle_quiz_clicks_to_date: float = Field(0.0, ge=0.0)
    vle_days_since_last_activity: Optional[float] = Field(None, ge=0.0)
    vle_clicks_last_7d: float = Field(0.0, ge=0.0)
    vle_clicks_last_14d: float = Field(0.0, ge=0.0)
    vle_activity_trend_ratio: Optional[float] = Field(None, ge=0.0)
    num_prior_submissions: int = Field(0, ge=0)
    prior_submission_lateness_mean: Optional[float] = Field(None)
    num_prior_scored_assessments: int = Field(0, ge=0)
    mean_prior_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    std_prior_score: Optional[float] = Field(None, ge=0.0)
    prior_at_risk_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    recent_prior_score: Optional[float] = Field(None, ge=0.0, le=100.0)

    # Optional dictionary to intercept prohibited features
    extra_attributes: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    def reject_prohibited_and_sensitive_features(cls, values):
        if not isinstance(values, dict):
            return values
            
        keys = set(values.keys())
        
        # Check leakage
        leakage_present = keys.intersection(LEAKAGE_COLUMNS)
        if leakage_present:
            raise ValueError(f"Prohibited target leakage columns detected: {leakage_present}. These cannot be provided to the model.")
            
        # Check sensitive demographics
        sensitive_present = keys.intersection(SENSITIVE_DEMOGRAPHIC_COLUMNS)
        if sensitive_present:
            raise ValueError(f"Prohibited sensitive demographic features detected: {sensitive_present}. Demographic attributes are strictly excluded from prediction.")

        return values


class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp_utc: str
    pseudonymized_student_id: str
    code_module: str
    code_presentation: str
    id_assessment: int
    assessment_type: str
    scheduled_date: int
    raw_risk_probability: float
    risk_tier: str
    operational_threshold_applied: Optional[float]
    predicted_at_risk: bool
    governance_action: str
    suppress_automated_outreach: bool
    student_outreach_enabled: bool = False
    advising_recommendation: str
    model_name: str
    model_version: str
    feature_pipeline_version: str
    evaluation_status: str


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionItem]


class BatchPredictionItemResult(BaseModel):
    item_index: int
    success: bool
    data: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_requested: int
    successful_count: int
    failed_count: int
    results: List[BatchPredictionItemResult]


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint confirming API status, model loading, and version."""
    mgr = get_model_manager()
    is_loaded = bool(mgr and mgr.is_loaded)
    version = mgr.metadata.get("model_version", "unknown") if is_loaded else "not_loaded"
    return {
        "status": "healthy",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_artifact_available": is_loaded,
        "model_version": version,
        "operational_mode": "SHADOW_MODE_READ_ONLY (Student Outreach Disabled)"
    }


@app.get("/model-info", status_code=status.HTTP_200_OK)
def model_info():
    """Returns complete model metadata, feature set, thresholds, and provenance."""
    mgr = get_model_manager()
    if not mgr or not mgr.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact is not loaded.")
    
    meta = mgr.metadata
    return {
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "feature_pipeline_version": meta.get("feature_pipeline_version"),
        "supported_horizons": meta.get("supported_prediction_horizons"),
        "assumed_grading_delay_days": meta.get("assumed_grading_delay_days"),
        "threshold_policy": meta.get("decision_threshold_policy"),
        "features": {
            "total_count": meta.get("feature_count"),
            "categorical": meta.get("categorical_features"),
            "numerical": meta.get("numerical_features")
        },
        "training_data_provenance": meta.get("training_data_provenance"),
        "sha256_hash": meta.get("sha256_hash"),
        "operational_status": meta.get("operational_status")
    }


def _process_single_prediction(item: PredictionItem) -> PredictionResponse:
    """Helper to process a single PredictionItem through pipeline, router, and audit logger."""
    mgr = get_model_manager()
    router = get_governance_router()
    if not mgr or not mgr.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact is not loaded.")
    
    item_dict = item.model_dump()
    
    # 1. Prepare feature row
    row_dict = {}
    for col in ALL_ALLOWED_FEATURES:
        row_dict[col] = item_dict.get(col)
        
    df_row = pd.DataFrame([row_dict])
    
    # Compute feature vector hash
    row_json = json.dumps(row_dict, sort_keys=True, default=str)
    feat_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
    
    # 2. Model Prediction
    raw_prob = float(mgr.predict_proba(df_row)[0])
    
    # 3. Governance Routing
    route = router.route_prediction({
        "code_module": item.code_module,
        "target_assessment_type": item.target_assessment_type,
        "scheduled_date": item.scheduled_date,
        "target_assessment_weight": item.target_assessment_weight
    }, raw_prob=raw_prob)
    
    # 4. Pseudonymize student ID safely
    # Fail safely if STUDENT_PSEUDONYMIZATION_SECRET is missing or empty - NO silent fallback!
    secret_key = os.environ.get("STUDENT_PSEUDONYMIZATION_SECRET", "").strip()
    if not secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security Configuration Error: Missing STUDENT_PSEUDONYMIZATION_SECRET. Student identifiers cannot be processed without keyed pseudonymization."
        )
    try:
        pseudonym = pseudonymize_student_id(item.id_student, secret_key=secret_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pseudonymization error: {str(e)}"
        )
    
    # 5. Determine Risk Tier
    eff_thresh = route["operational_threshold_applied"] or 0.20
    if raw_prob >= eff_thresh:
        risk_tier = "HIGH_RISK"
    elif raw_prob >= 0.05:
        risk_tier = "MEDIUM_RISK"
    else:
        risk_tier = "LOW_RISK"
        
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pred_id = str(uuid.uuid4())
    
    # 6. Audit Logging
    audit_rec = DecisionAuditRecord(
        prediction_id=pred_id,
        timestamp_utc=now_utc,
        pseudonymized_student_id=pseudonym,
        code_module=route["code_module"],
        code_presentation=item.code_presentation,
        id_assessment=item.id_assessment,
        assessment_type=route["target_assessment_type"],
        scheduled_date=item.scheduled_date,
        model_name=mgr.metadata["model_name"],
        model_version=mgr.metadata["model_version"],
        feature_pipeline_version=mgr.metadata["feature_pipeline_version"],
        input_feature_hash=feat_hash,
        raw_risk_probability=route["raw_risk_probability"],
        operational_threshold=route["operational_threshold_applied"],
        predicted_at_risk=route["predicted_at_risk"],
        governance_routing_action=route["governance_action"],
        suppress_automated_outreach=route["suppress_automated_outreach"],
        student_outreach_enabled=route.get("student_outreach_enabled", False),
        routing_reason=route["advising_recommendation"]
    )
    audit_logger.log(audit_rec)
    
    return PredictionResponse(
        prediction_id=pred_id,
        timestamp_utc=now_utc,
        pseudonymized_student_id=pseudonym,
        code_module=route["code_module"],
        code_presentation=item.code_presentation,
        id_assessment=item.id_assessment,
        assessment_type=route["target_assessment_type"],
        scheduled_date=item.scheduled_date,
        raw_risk_probability=route["raw_risk_probability"],
        risk_tier=risk_tier,
        operational_threshold_applied=route["operational_threshold_applied"],
        predicted_at_risk=route["predicted_at_risk"],
        governance_action=route["governance_action"],
        suppress_automated_outreach=route["suppress_automated_outreach"],
        student_outreach_enabled=route.get("student_outreach_enabled", False),
        advising_recommendation=route["advising_recommendation"],
        model_name=mgr.metadata["model_name"],
        model_version=mgr.metadata["model_version"],
        feature_pipeline_version=mgr.metadata["feature_pipeline_version"],
        evaluation_status="OFFLINE_RESEARCH_SHADOW_MODE"
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_single(item: PredictionItem):
    """
    Generates a prospective risk prediction for an individual student assessment snapshot.
    Enforces governance routing, threshold policy, and audit logging.
    """
    try:
        return _process_single_prediction(item)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(payload: BatchPredictionRequest):
    """
    Processes a batch of student prediction items with isolated record validation.
    Validation failures on one record do not corrupt or abort the remaining batch.
    """
    results: List[BatchPredictionItemResult] = []
    success_count = 0
    fail_count = 0
    
    for idx, item in enumerate(payload.predictions):
        try:
            res = _process_single_prediction(item)
            results.append(BatchPredictionItemResult(
                item_index=idx,
                success=True,
                data=res,
                error=None
            ))
            success_count += 1
        except Exception as e:
            results.append(BatchPredictionItemResult(
                item_index=idx,
                success=False,
                data=None,
                error=str(e)
            ))
            fail_count += 1
            
    return BatchPredictionResponse(
        total_requested=len(payload.predictions),
        successful_count=success_count,
        failed_count=fail_count,
        results=results
    )


# =============================================================================
# FRONTEND STATIC FILES & SINGLE-PAGE APPLICATION SERVING
# =============================================================================

_frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(_frontend_dist):
    _assets_dir = os.path.join(_frontend_dist, "assets")
    if os.path.exists(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def serve_frontend_dashboard():
        """Serves the compiled production React academic analytics dashboard."""
        return FileResponse(os.path.join(_frontend_dist, "index.html"))
