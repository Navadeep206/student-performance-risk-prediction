# Phase 10: Final Shadow-Mode Safety & Artifact Acceptance Audit
## Student Performance & At-Risk Prediction System (OULAD)

- **Audit Version:** 1.0 (Final Acceptance)  
- **Audit Date:** 2026-11-15T07:25:00Z  
- **System Scope:** Model Persistence, Inference API, Governance Router, Shadow-Mode Safety Gates  
- **Repository Regression Suite:** **106/106 Tests Passed (100% Pass Rate)**  
- **Production Deployment Status:** **WITHHELD (Mandatory Passive Shadow Mode Enforced)**  

---

## Executive Summary

This audit establishes the final code-level safety, cryptographic integrity, and governance compliance of the **Student Performance & At-Risk Prediction System** following Phase 9 finalization and Phase 10 implementation.

Prior to this audit, a governance contradiction was identified: while `ShadowModeManager` declared `ALLOW_STUDENT_OUTREACH = False`, the governance router returned active outreach actions (`SUPPORTIVE_CHECK_IN_ONLY`, `AUTOMATED_OUTREACH_ENABLED`) with `suppress_automated_outreach = False`. This contradiction has been eliminated by strictly separating the **statistical risk classification decision** from **operational communication permissions**.

Under the remediated architecture, **all predictions operate in passive, read-only shadow mode**. No module, assessment type, or risk probability can trigger student communication. The API exposes zero communication or intervention endpoints, silent fallback secrets have been excised, audit logging is provably thread-safe, and both the Primary `HistGradientBoostingClassifier` and Challenger `RandomForestClassifier` artifacts match their certified Phase 6 configurations.

---

## 1. Resolution of Shadow-Mode Outreach Contradiction

### 1.1 Architectural Separation of Risk Scoring vs. Communication Permission
Previously, risk scoring and outreach permission were coupled: if a student was classified as at-risk ($\hat{y} = 1$), the router enabled automated outreach (`suppress_automated_outreach = False`). This violated the pre-deployment shadow-mode constraint.

The architecture now decouples these two concepts into distinct fields across `InstitutionalGovernanceRouter`, `DecisionAuditRecord`, and `PredictionResponse`:
- **Risk Decision:** `predicted_at_risk` ($\text{True} \iff \hat{p} \ge \tau$) and `operational_threshold_applied` ($\tau = 0.10$ for Module GGG TMA; $\tau = 0.20$ for Modules AAA–FFF TMA).
- **Communication Permission:** `student_outreach_enabled = False` and `suppress_automated_outreach = True` hard-locked across all predictions.

### 1.2 Governed Routing Verdicts

#### A. Module GGG TMA ($\hat{p} = 0.12$, $\tau = 0.10$)
```json
{
  "code_module": "GGG",
  "target_assessment_type": "TMA",
  "raw_risk_probability": 0.12,
  "operational_threshold_applied": 0.10,
  "predicted_at_risk": true,
  "governance_action": "SHADOW_MODE_ADVISORY_REVIEW",
  "suppress_automated_outreach": true,
  "student_outreach_enabled": false,
  "advising_recommendation": "Module GGG Tutor-Marked Assignment in Module GGG. Risk score (0.1200) meets or exceeds operational threshold (0.10). [SHADOW MODE ACTIVE: External student outreach is strictly prohibited. Passive advisor review only.]"
}
```

#### B. Standard TMA Predictions (Modules AAA–FFF)
- **Above Threshold ($\hat{p} \ge 0.20$):**  
  `predicted_at_risk = True`, `operational_threshold = 0.20`, `governance_action = "SHADOW_MODE_ADVISORY_REVIEW"`, `suppress_automated_outreach = True`, `student_outreach_enabled = False`.
- **Below Threshold ($\hat{p} < 0.20$):**  
  `predicted_at_risk = False`, `operational_threshold = 0.20`, `governance_action = "SHADOW_MODE_MONITORING"`, `suppress_automated_outreach = True`, `student_outreach_enabled = False`.

#### C. Universal CMA Outreach Suppression (Modules AAA–GGG)
Across all seven modules, Computer-Marked Assessments are strictly treated as non-actionable research telemetry:
- `predicted_at_risk = False`
- `operational_threshold = None`
- `governance_action = "AUTOMATED_OUTREACH_SUPPRESSED"`
- `suppress_automated_outreach = True`
- `student_outreach_enabled = False`

#### D. Outbound Communication Security Lock
Calling `attempt_outreach(student_id, message)` on either `InstitutionalGovernanceRouter` or `ShadowModeManager` strictly raises:
```python
SecurityViolationError(
    "SECURITY LOCK ACTIVE: External student outreach is strictly prohibited in Shadow Mode. "
    "No automated messages, emails, or LMS nudges may be sent."
)
```

#### E. API Route Audit
Inspection of the FastAPI application route table confirms that only four endpoints exist:
1. `GET /health`
2. `GET /model-info`
3. `POST /predict`
4. `POST /predict/batch`

Zero communication, messaging, email, or intervention routes exist. Probing `/intervene`, `/outreach`, or `/send-message` returns HTTP 404.

---

## 2. Keyed HMAC-SHA256 Secret Configuration & Key Rotation Policy

### 2.1 Silent Fallback Removal & Safe Failover
All silent default secrets (`default-fallback-dev-secret-32bytes!`) have been excised from `src/api.py`.
- If the environment variable `STUDENT_PSEUDONYMIZATION_SECRET` is unset or empty:
  `/predict` immediately halts and returns HTTP 500:
  `"Security Configuration Error: Missing STUDENT_PSEUDONYMIZATION_SECRET. Student identifiers cannot be processed without keyed pseudonymization."`
- If the secret is shorter than 16 characters:
  The API halts with HTTP 500:
  `"Cryptographic Error: Pseudonymization secret must be a string of at least 16 characters."`
- The secret key is never written to log files, serialized into JSON, or returned in HTTP responses.

### 2.2 Test Configuration Isolation
Automated tests configure `STUDENT_PSEUDONYMIZATION_SECRET` strictly via pytest fixtures (`setup_test_environment`), saving and restoring pre-existing environment variables to ensure zero leakage across test boundaries.

### 2.3 Production Key Provisioning & Rotation Requirements
For eventual production deployment, the following institutional key management protocol is mandatory:
1. **KMS Provisioning:** The secret must be stored in an enterprise Key Management Service (AWS KMS, Azure Key Vault, or HashiCorp Vault) and injected into the container environment at runtime.
2. **Key Length & Entropy:** Minimum 256 bits (32 bytes) of cryptographically secure random entropy (e.g. `openssl rand -hex 32`).
3. **Scheduled Key Rotation:** A mandatory 90-day key rotation cycle.
4. **Historical Audit Re-identification Map:** When rotating secrets, institutional compliance requires either:
   - Archiving an encrypted map of legacy-to-new pseudonyms within the secure SIS enclave, or
   - Retaining versioned secrets in KMS (`STUDENT_PSEUDONYMIZATION_SECRET_V1`, `_V2`) to permit authorized institutional re-identification under FERPA/GDPR Art. 4(5).

---

## 3. Decision Audit Record & Logger Integrity

### 3.1 Immutability
`DecisionAuditRecord` is implemented as an `@dataclass(frozen=True)`. Any attempt to mutate fields post-instantiation raises a `FrozenInstanceError`.

### 3.2 Thread-Safe Concurrency
`DecisionAuditLogger` encapsulates an in-memory audit buffer and an append-only JSONL log guarded by `threading.Lock()`.
- Verified under high concurrency: 50 concurrent worker threads logging simultaneously produced exactly 50 distinct, uncorrupted records.
- 25 concurrent HTTP requests to `/predict` generated 25 unique UUIDs with zero race conditions or payload overwrites.

### 3.3 Recorded Provenance & Privacy Guarantees
Every logged record captures:
- `prediction_id`: Unique UUIDv4.
- `timestamp_utc`: Standard ISO 8601 UTC timestamp.
- `pseudonymized_student_id`: 64-character lowercase hex digest of `HMAC-SHA256(raw_id, secret)`.
- `input_feature_hash`: Cryptographic SHA-256 hash of the complete 23-feature JSON vector.
- `model_name`, `model_version`, `feature_pipeline_version`.
- `raw_risk_probability`, `operational_threshold`, `predicted_at_risk`.
- `governance_routing_action`, `suppress_automated_outreach`, `student_outreach_enabled`.
- **Privacy Check:** Raw student IDs (`100200`) and the secret key are mathematically absent from both `pseudonymized_student_id` and the serialized audit record.

---

## 4. Model Artifact Integrity Verification

Both model binaries saved in `models/` were independently audited against the certified Phase 6 specifications without retraining:

### 4.1 Primary Production Candidate (`HistGradientBoostingClassifier`)
- **Artifact Path:** `models/optimized_hist_gradient_boosting.joblib`
- **Metadata Path:** `models/optimized_hist_gradient_boosting_metadata.json`
- **Cryptographic SHA-256 Hash:** `a5ad045e8af28e318d71d036bdc6888f171934242511637396581ed8d4d80771`
- **Feature Set:** Exactly 23 approved predictive features in identical order.
- **Categoricals (3 features):** `code_module`, `code_presentation`, `target_assessment_type` handled via `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.
- **Numericals (20 features):** Passthrough with native missingness NaN routing for cold-start learners.
- **Prohibited Feature Audit:** Zero instances of `final_result`, `date_unregistration`, or demographic features (`gender`, `imd_band`, `highest_education`, `age_band`, `disability`, `region`).
- **Provenance:** Trained on $N = 120,860$ snapshots (`2013B`, `2013J`, `2014B`). Prospective intake `2014J` strictly quarantined.

### 4.2 Challenger Benchmark Model (`RandomForestClassifier`)
- **Artifact Path:** `models/challenger_random_forest.joblib`
- **Metadata Path:** `models/challenger_random_forest_metadata.json`
- **Cryptographic SHA-256 Hash:** `93262efb5a02412a42303834a7ff0eef61a3d5b87cd4b7696f3ec5b722c87f2c`
- **Feature Set:** Exactly 23 approved predictive features in identical order.
- **Pipeline:** Median imputation + standard scaling for numericals; most frequent imputation + OneHotEncoder for categoricals.
- **Prohibited Feature Audit:** Zero target leakage or demographic features.

---

## 5. API Safety Verification Results (10 Mandated Scenarios)

All 10 safety scenarios mandated in the audit prompt have been implemented as automated tests in `tests/test_phase10_api.py` and passed with 100% compliance:

| # | Mandated Verification Scenario | Test Function | Test Outcome |
| :-: | :--- | :--- | :---: |
| 1 | **GGG TMA at raw probability 0.12** | `test_ggg_tma_at_probability_012_routing_and_outreach_disabled` | **PASSED** (returns `predicted_at_risk=True`, $\tau=0.10$, `SHADOW_MODE_ADVISORY_REVIEW`, `suppress_automated_outreach=True`, `student_outreach_enabled=False`) |
| 2 | **Standard TMA above threshold ($\ge 0.20$)** | `test_predict_single_standard_tma_above_and_below_threshold` | **PASSED** (`predicted_at_risk=True`, $\tau=0.20$, `SHADOW_MODE_ADVISORY_REVIEW`, `student_outreach_enabled=False`) |
| 3 | **Standard TMA below threshold ($< 0.20$)** | `test_predict_single_standard_tma_above_and_below_threshold` | **PASSED** (`predicted_at_risk=False`, $\tau=0.20$, `SHADOW_MODE_MONITORING`, `student_outreach_enabled=False`) |
| 4 | **CMA for all 7 modules (AAA–GGG)** | `test_predict_single_cma_universal_suppression` | **PASSED** (`predicted_at_risk=False`, $\tau=\text{None}$, `AUTOMATED_OUTREACH_SUPPRESSED`, `student_outreach_enabled=False`) |
| 5 | **Unknown assessment metadata** | `test_predict_unknown_metadata_safe_handling` | **PASSED** (`predicted_at_risk=False`, `MANUAL_ADVISOR_REVIEW`, `suppress_automated_outreach=True`, `student_outreach_enabled=False`) |
| 6 | **Missing HMAC secret** | `test_missing_hmac_secret_fails_safely_no_silent_fallback` | **PASSED** (returns HTTP 500, no silent fallback, zero secret exposure) |
| 7 | **Missing model artifact** | `test_model_artifact_manager_missing_and_corrupt_files` | **PASSED** (`FileNotFoundError` raised immediately) |
| 8 | **Corrupted artifact hash** | `test_model_artifact_manager_missing_and_corrupt_files` | **PASSED** (`ValueError` "SHA-256 hash mismatch" raised) |
| 9 | **Concurrent prediction requests** | `test_concurrent_api_prediction_requests` | **PASSED** (25 concurrent threads scored with 200 OK, unique UUIDs, thread-safe logs) |
| 10 | **Attempted student outreach** | `test_attempted_outreach_raises_security_violation` & `test_api_does_not_expose_outreach_endpoints` | **PASSED** (`SecurityViolationError` raised; zero outreach routes exist in API) |

---

## 6. Repository Regression Suite Verification

The complete test suite was executed across all phases and modules:
```
======================= 106 passed, 3 warnings in 5.18s ========================
```
- Total Tests: **106**
- Passed: **106**
- Failed: **0**
- Pass Rate: **100.0%**

---

## 7. Remaining Operational Limitations & Production Status

1. **Mandatory Shadow-Mode Operation:**  
   The system must run strictly in read-only shadow mode alongside human academic advisors. Direct automated email, SMS, or LMS alerts to students are strictly prohibited.
2. **Non-Punitive Advisory Boundary:**  
   Predictions represent empirical estimates of coursework difficulty. They must never be used for automated grade penalties, admissions screening, prerequisite blocking, or enrollment termination.
3. **ETL Latency Constraints:**  
   The 7-day pre-deadline prediction horizon is the sole validated operational candidate. Predictions at the 1-day horizon remain unverified due to institutional nightly ETL latency.

> [!CAUTION]
> ### Final Production Deployment Status: WITHHELD
> 
> Algorithmic governance, model artifact integrity, API implementation, and shadow-mode security gates are **100% complete, verified, and accepted**.
> 
> However, **Production Deployment Approval remains WITHHELD** until real-world institutional dependencies (advising staffing capacity agreement, live SIS webhook integration, KMS key provisioning, and a one-semester passive shadow-mode pilot) are formally completed and signed off.

