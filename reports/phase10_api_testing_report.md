# Phase 10: API Implementation & Safety Testing Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 2.0 (Post-Safety & Artifact Acceptance Audit)  
- **Completion Timestamp:** 2026-11-15T07:25:00Z  
- **Framework:** FastAPI (Uvicorn / Pydantic v2 / Starlette)  
- **Test Status:** **17/17 API Tests Passed (100% Pass Rate)**  
- **Production Approval Status:** **WITHHELD (Mandatory Shadow Mode Enforced)**  

---

## 1. API Architecture & Governance Controls

The Prediction API provides clean, low-latency, auditable inference endpoints with non-negotiable shadow-mode protections:

| Endpoint | Method | Operational Purpose | Mandatory Safety & Governance Controls |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | Service health & model state | Returns `200 OK`, `model_artifact_available=True`, and certifies `SHADOW_MODE_READ_ONLY`. |
| `/model-info` | `GET` | Architecture, features, and thresholds | Exposes 23 features, thresholds ($\tau=0.20$ default, $\tau=0.10$ GGG TMA), and flags `WITHHELD`. |
| `/predict` | `POST` | Single snapshot inference | Enforces keyed HMAC pseudonymization, shadow-mode communication prohibition (`student_outreach_enabled=False`), universal CMA suppression, and audit logging. |
| `/predict/batch` | `POST` | Multi-student batch inference | Schema-level validation and isolated record processing (faults in one item do not abort the batch). |

### Strict Shadow-Mode Prohibitions:
1. **Zero Active Communication:** The system strictly separates the predictive risk classification ($\hat{y} \in \{0, 1\}$) from communication permissions. In shadow mode, `student_outreach_enabled = False` and `suppress_automated_outreach = True` for **all** predictions regardless of risk probability or module.
2. **Zero Messaging Endpoints:** The API exposes no endpoints for sending emails, SMS, LMS nudges, or interventions (`/intervene`, `/outreach`, `/send-message` return HTTP 404).
3. **Security Lockout:** Direct calls to internal outreach methods (`router.attempt_outreach` or `shadow_mgr.attempt_outreach`) throw `SecurityViolationError`.
4. **Safe Failover on Missing Secret:** Silent production fallbacks are eliminated. If `STUDENT_PSEUDONYMIZATION_SECRET` is unset or empty, the API safely fails with HTTP 500 without logging or exposing secrets.

---

## 2. API Automated Test Suite Results (17 Tests)

Execution verified via `.venv/bin/pytest tests/test_phase10_api.py -v`:

| # | Test Name | Target Scope | Key Verification Assertions | Status |
| :-: | :--- | :--- | :--- | :---: |
| 1 | `test_health_endpoint` | `GET /health` | Validates 200 OK, `model_artifact_available=True`, `operational_mode="SHADOW_MODE_READ_ONLY"`. | **PASSED** |
| 2 | `test_model_info_endpoint` | `GET /model-info` | Validates 200 OK, 23 features (20 num, 3 cat), threshold policies, `operational_status="WITHHELD"`. | **PASSED** |
| 3 | `test_ggg_tma_at_probability_012_routing_and_outreach_disabled` | Governance Router | For GGG TMA with raw prob 0.12: asserts `predicted_at_risk=True`, `operational_threshold=0.10`, `governance_action="SHADOW_MODE_ADVISORY_REVIEW"`, `suppress_automated_outreach=True`, `student_outreach_enabled=False`. | **PASSED** |
| 4 | `test_predict_single_standard_tma_above_and_below_threshold` | `POST /predict` | Validates BBB TMA: prob $\ge 0.20$ yields `SHADOW_MODE_ADVISORY_REVIEW`; prob $< 0.20$ yields `SHADOW_MODE_MONITORING`. Both enforce `student_outreach_enabled=False` and `suppress_automated_outreach=True`. | **PASSED** |
| 5 | `test_predict_single_cma_universal_suppression` | `POST /predict` | Iterates through AAA–GGG: asserts `predicted_at_risk=False`, `operational_threshold=None`, `governance_action="AUTOMATED_OUTREACH_SUPPRESSED"`, `student_outreach_enabled=False`. | **PASSED** |
| 6 | `test_predict_unknown_metadata_safe_handling` | `POST /predict` | Unknown type routes to `MANUAL_ADVISOR_REVIEW` with `suppress_automated_outreach=True` and `student_outreach_enabled=False`. | **PASSED** |
| 7 | `test_attempted_outreach_raises_security_violation` | Security Lock | Calling `attempt_outreach()` on router or shadow manager raises `SecurityViolationError`. | **PASSED** |
| 8 | `test_api_does_not_expose_outreach_endpoints` | API Inspection | Confirms no communication/outreach routes exist; calling `/intervene` returns 404. | **PASSED** |
| 9 | `test_missing_hmac_secret_fails_safely_no_silent_fallback` | HMAC Fail-Safe | Unsetting `STUDENT_PSEUDONYMIZATION_SECRET` causes `/predict` to fail safely with HTTP 500. Zero secret leakage. | **PASSED** |
| 10 | `test_short_hmac_secret_fails_safely` | HMAC Key Check | Secret $< 16$ characters rejected safely with HTTP 500. | **PASSED** |
| 11 | `test_predict_prohibited_target_leakage_rejected` | Leakage Gate | Providing `final_result` or `date_unregistration` immediately returns HTTP 422. | **PASSED** |
| 12 | `test_predict_prohibited_sensitive_demographics_rejected` | Fairness Gate | Providing `gender`, `imd_band`, `highest_education`, `age_band`, `disability`, `region` returns HTTP 422. | **PASSED** |
| 13 | `test_predict_batch_isolated_fault_handling` | `POST /predict/batch` | Validates multi-item batch scoring with individual record isolation and schema enforcement. | **PASSED** |
| 14 | `test_concurrent_api_prediction_requests` | Thread Safety | 25 concurrent requests via `ThreadPoolExecutor(max_workers=8)` succeed with 200 OK, unique UUIDs, and immutable logging. | **PASSED** |
| 15 | `test_model_artifact_manager_missing_and_corrupt_files` | Artifact Integrity | Missing file throws `FileNotFoundError`; altered hash throws `ValueError` ("SHA-256 hash mismatch"). | **PASSED** |
| 16 | `test_saved_hgb_artifact_verification_against_approved_pipeline` | Pipeline Audit | Primary HGB binary matches Phase 6 specs: 23 features, exact ordering, native NaN handling, valid SHA-256. | **PASSED** |
| 17 | `test_saved_rf_challenger_artifact_verification` | Pipeline Audit | Challenger RF binary matches specs: 23 features, median/OHE imputers, valid SHA-256. | **PASSED** |

---

## 3. Sample API Payloads Under Shadow-Mode Enforcement

### 3.1 Request (`POST /predict`):
```json
{
  "id_student": 100200,
  "code_module": "GGG",
  "code_presentation": "2014J",
  "id_assessment": 37415,
  "target_assessment_type": "TMA",
  "scheduled_date": 61,
  "target_assessment_weight": 0.0,
  "num_previous_scheduled_assessments": 0,
  "registration_timing_days": -45.0,
  "vle_total_clicks_to_date": 12.0,
  "vle_active_days_to_date": 2.0,
  "vle_core_clicks_to_date": 8.0,
  "vle_forum_clicks_to_date": 0.0,
  "vle_quiz_clicks_to_date": 0.0,
  "vle_days_since_last_activity": 22.0,
  "vle_clicks_last_7d": 0.0,
  "vle_clicks_last_14d": 0.0,
  "vle_activity_trend_ratio": 0.0,
  "num_prior_submissions": 0,
  "prior_submission_lateness_mean": null,
  "num_prior_scored_assessments": 0,
  "mean_prior_score": null,
  "std_prior_score": null,
  "prior_at_risk_rate": null,
  "recent_prior_score": null
}
```

### 3.2 Governed Response (`HTTP 200 OK`):
```json
{
  "prediction_id": "7f8b9e10-1a2b-4c3d-8e4f-5a6b7c8d9e0f",
  "timestamp_utc": "2026-11-15T07:25:00Z",
  "pseudonymized_student_id": "e2d7b1a03f8c5b6a7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
  "code_module": "GGG",
  "code_presentation": "2014J",
  "id_assessment": 37415,
  "assessment_type": "TMA",
  "scheduled_date": 61,
  "raw_risk_probability": 0.1245,
  "risk_tier": "HIGH_RISK",
  "operational_threshold_applied": 0.10,
  "predicted_at_risk": true,
  "governance_action": "SHADOW_MODE_ADVISORY_REVIEW",
  "suppress_automated_outreach": true,
  "student_outreach_enabled": false,
  "advising_recommendation": "Module GGG Tutor-Marked Assignment in Module GGG. Risk score (0.1245) meets or exceeds operational threshold (0.10). [SHADOW MODE ACTIVE: External student outreach is strictly prohibited. Passive advisor review only.]",
  "model_name": "HistGradientBoostingClassifier",
  "model_version": "1.0.0",
  "feature_pipeline_version": "1.0.0",
  "evaluation_status": "OFFLINE_RESEARCH_SHADOW_MODE"
}
```
