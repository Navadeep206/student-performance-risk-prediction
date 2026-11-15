# Phase 9 Task 5: Operational Readiness & Deployment Feasibility Audit Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Audit Date:** 2026-11-15T06:54:11Z  
- **Audit Mandate:** Investigation & Operational Verification Only (Production Deployment Approval is **WITHHELD**)  
- **Evaluated System:** `HistGradientBoostingClassifier` (Primary Candidate) with `ModuleGGGGovernanceRouter`  
- **Benchmark Population:** Prospective Chronological Holdout in `2014J` ($N = 50,014$ snapshots across 8,607 students)  

---

## 1. Executive Summary & Operational Classification Matrix

A comprehensive operational readiness audit was executed to evaluate whether the Student At-Risk Prediction System can be safely deployed into institutional advising workflows.

> [!CAUTION]
> **Production Approval Remains WITHHELD.**
> While predictive discrimination and cold-start robustness are certified, operational readiness requires institutional staffing alignment, real-world LMS ETL latency verification, and live gradebook integration. Automated student outreach is strictly prohibited.

### Operational Requirements Status Matrix

| Operational Dimension / Requirement | Status Classification | Key Audit Finding / Prerequisite |
| :--- | :---: | :--- |
| **1. Calibration Ranking Invariance** | **VERIFIED** | Monotonic sigmoid preserves AP to 10 decimal places; Phase 6 drop was caused by `CalibratedClassifierCV` refitting/ensembling. |
| **2. Counselor Workload Viability** | **REQUIRES INSTITUTIONAL INPUT** | 7,130 TMA alerts across 3,299 unique students (38.3% of cohort) requires institutional staffing validation. |
| **3. Module GGG Governance Routing** | **VERIFIED** | Router correctly suppresses decoupled GGG CMAs and applies calibrated threshold tau = 0.10 to GGG TMAs. |
| **4. Universal CMA Outreach Suppression** | **PARTIALLY VERIFIED** | GGG CMAs are suppressed, but router currently defaults to outreach-enabled for AAA–FFF CMAs. Must be globally gated. |
| **5. Static Registration & Course Telemetry** | **VERIFIED** | Registration dates and syllabus metadata are fully available in SIS prior to semester start. |
| **6. VLE Dynamic Telemetry (7-Day Cutoff)** | **VERIFIED** | 7-day pre-deadline window provides sufficient operational buffer for standard 24–48h LMS batch ingestion. |
| **7. VLE Real-Time Telemetry (1-Day Cutoff)** | **NOT VERIFIED** | High operational fragility; cannot guarantee real-time sub-24h event streaming in production LMS environments. |
| **8. Grade Publication Timing (14-Day Simulation)** | **REQUIRES INSTITUTIONAL INPUT** | OULAD lacks actual grade publication timestamps; production requires linking to LMS gradebook release events. |
| **9. Shadow-Mode Protocol & Monitoring** | **VERIFIED** | Non-intrusive logging, drift monitoring, and false positive/negative investigation procedures fully specified. |

---

## 2. Calibration Methodology Consistency Audit

### 2.1 The Mathematical Question of Ranking Invariance
Phase 6 reported that applying post-hoc calibration degraded Average Precision (PR-AUC dropped from 0.4669 to 0.3538 for isotonic and 0.2915 for sigmoid). We investigated whether a monotonic sigmoid mapping on frozen predictions could mathematically reduce PR-AUC:

- **Mathematical Proof & Empirical Verification:**
  - Given identical continuous frozen predictions S and true labels Y, any strictly monotonic function f(s) = 1 / (1 + exp(-(A*s + B))) with A > 0 preserves the exact pairwise ordering of all predictions:
    s_i > s_j <=> f(s_i) > f(s_j)
  - Precision-Recall curves evaluate precision across distinct ranking thresholds. Because ties are neither created nor broken, **Average Precision is mathematically identical**:
    - Synthetic Base AP: **0.956685**
    - Sigmoid Mapped AP: **0.956685**
    - Absolute AP Difference: **0.0**
    - Absolute ROC-AUC Difference: **0.0**
  - **Verdict:** A strictly monotonic sigmoid transformation applied to identical frozen scores **cannot** alter PR-AUC.

### 2.2 Root Cause of Reported Phase 6 PR-AUC Reduction
The PR-AUC drop in Phase 6 occurred due to the internal architecture of scikit-learn's `CalibratedClassifierCV`:
1. **Estimator Refitting and Subsampling:** `CalibratedClassifierCV(cv=K)` does not apply calibration to a frozen base estimator trained on 100% of the training set. Instead, it partitions the data into $K$ folds, fits an independent clone estimator on $(K-1)/K$ of the data per fold, and fits calibrators on the held-out portions.
2. **Multi-Model Ensembling:** At test time, `CalibratedClassifierCV` averages the probabilities of $K$ different refitted models. This ensemble has different tree partition splits, shallower interaction depths, and altered rankings compared to the full base model.
3. **Isotonic Step Collapse:** For isotonic calibration, the pool-adjacent-violators algorithm aggregates continuous scores into flat step bins, introducing thousands of tied predictions that collapse ranking resolution under class imbalance.

---

## 3. Counselor Workload & Alert Capacity Audit

Audit conducted on the prospective `2014J` chronological holdout ($N = 50,014$ snapshots, 8,607 unique enrolled students):

### 3.1 Raw Alerting vs. Governed Actionable Alerting

| Alerting Policy Configuration | Total Actionable Alerts | Snapshot Alert Rate | CMA Alerts | TMA Alerts | Unique Students Alerted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Global Threshold (tau = 0.20 Everywhere)** | 9,227 | 18.45% | 2,243 | 6,984 | 4,112 (47.78%) |
| **Governed Operational Policy (TMA Only, tau=0.10 for GGG)** | **7,130** | **14.26%** | **0 (Suppressed)** | **7,130** | **3,299 (38.33%)** |

- **Operational Impact of CMA Outreach Suppression:** Suppressing automated outreach on computer quizzes eliminates **2,243 unnecessary alerts** (a **22.73% reduction** in counselor workload), focusing human intervention exclusively on major coursework milestones.

### 3.2 Student Alert Frequency Distribution
Under the governed TMA-only policy, **3,299 unique students** receive at least one alert:
- **1 Alert Only:** 1,430 students (43.35% of alerted students)
- **2 Alerts:** 709 students (21.49%)
- **3 Alerts:** 441 students (13.37%)
- **4 Alerts:** 663 students (20.10%)
- **5+ Alerts:** 56 students (1.70%)
- **Mean Alerts per Alerted Student:** **2.16 alerts** across the semester.

### 3.3 Module & Milestone Breakdown (Governed TMA Alerts)
- **Course Module Allocation:**
  - **BBB (Social Science):** 2,501 alerts (35.1%) — High volume course
  - **DDD (STEM / Tech):** 1,752 alerts (24.6%)
  - **CCC (STEM / Comp):** 1,532 alerts (21.5%)
  - **FFF (STEM / Tech):** 772 alerts (10.8%)
  - **GGG (Social / Open):** 239 alerts (3.4%) — Governed under calibrated tau = 0.10
  - **AAA (Humanities):** 187 alerts (2.6%)
  - **EEE (Science):** 147 alerts (2.1%)
- **Milestone Allocation:**
  - **TMA 1 (Onboarding, Day <= 70):** 2,280 alerts (32.0%)
  - **TMA 2 (Midterm, Day 71–140):** 1,801 alerts (25.3%)
  - **TMA 3+ (Capstone, Day > 140):** 3,049 alerts (42.8%)

### 3.4 Tiered Advisor Intervention Model
Assuming an academic semester of ~34 weeks, 7,130 total alerts averages to ~210 alerts/week, peaking at 450–650 alerts/week during major submission windows.
- **Recommended Tiered Cadence:**
  - **Tier 1 (Single / Onboarding Alert):** Automated supportive LMS check-in nudges (0 human advisor hours). Handles 1,430 students.
  - **Tier 2 (Persistent Academic Risk):** Mandatory 1-on-1 advisor outreach for students flagged >= 2 times (1,869 unique students across the semester, or ~55 consultations/week across all faculties).
- **Prerequisite:** Institutional advising management must confirm FTE counselor availability before enabling Tier 2 outreach.

---

## 4. Governance Router Integration Audit

Audited `ModuleGGGGovernanceRouter` in `src/module_ggg_remediation.py`:

| Verification Item | Implementation Check | Status | Evidence / Code Trace |
| :--- | :--- | :---: | :--- |
| **Decoupled GGG CMAs Suppressed** | GGG CMAs routed to `AUTOMATED_OUTREACH_SUPPRESSED` with `predicted_at_risk=False` | **VERIFIED** | Lines 103–118 in `module_ggg_remediation.py` |
| **GGG TMA Calibrated Threshold** | GGG TMAs evaluated against `tma_calibrated_threshold = 0.10` | **VERIFIED** | Lines 120–153 in `module_ggg_remediation.py` |
| **Standard Module Default (tau = 0.20)** | Modules AAA–FFF evaluated against `global_threshold = 0.20` | **VERIFIED** | Lines 88–100 in `module_ggg_remediation.py` |
| **Invalid Metadata Safe Handling** | Unknown assessment types routed to `MANUAL_ADVISOR_REVIEW` with `suppress_automated_outreach=True` | **VERIFIED** | Lines 154–165 in `module_ggg_remediation.py` |
| **Universal CMA Suppression Integration** | Suppress automated outreach on CMAs across all modules AAA–FFF | **PARTIALLY VERIFIED** | **Integration Gap:** Router suppresses CMAs for GGG, but AAA–FFF CMAs currently default to `suppress=False`. Must be globally gated. |
| **Decision Audit Trail Logging** | Log `model_name`, `model_version`, and `feature_vector_hash` | **PARTIALLY VERIFIED** | **Integration Gap:** Router logs metadata and thresholds, but lacks model version and feature hash fields. |

---

## 5. Data Telemetry & Feature Availability Audit

Audit of temporal boundaries, grade publication delays, and feature extraction feasibility:

| Feature Domain | Telemetry Source | Feasibility Classification | Operational Verification Findings |
| :--- | :--- | :---: | :--- |
| **Student Registration** | SIS Enrollment Database | **VERIFIED** | Registration dates (`date_registration`) are recorded pre-semester. 100% available at prediction cutoffs. |
| **Course Syllabus & Schedule** | Course Registry | **VERIFIED** | Assessment IDs, scheduled dates, and assessment types are static course metadata available at Day 0. |
| **VLE Clicks (7-Day Horizon)** | LMS Clickstream Event Logs | **VERIFIED** | Cutoff at $T - 7$ days provides ample buffer for nightly batch ETL (24–48h ingestion lag). |
| **VLE Clicks (1-Day Horizon)** | Real-Time LMS Streaming | **NOT VERIFIED** | Operationally fragile. Real-time sub-24h event ingestion cannot be guaranteed across legacy LMS infrastructure. |
| **Historical Assessment Grades** | Gradebook / Tutor Marking | **REQUIRES INSTITUTIONAL INPUT** | **Crucial OULAD Constraint:** OULAD does not record actual grade publication timestamps. The 14-day delay is an offline benchmark. Production requires direct API hooks to SIS grade release events. |

---

## 6. Shadow-Mode Deployment Design

A non-intrusive shadow-mode operational evaluation plan has been formulated:

### 6.1 Shadow Architecture & Data Ingestion
1. **Passive Inference Pipeline:** On scheduled prediction dates ($T - 7$ days before each TMA), the pipeline runs in batch mode, generating risk predictions without sending messages or alerting students.
2. **Audit Event Logging Schema:** Every inference record logs:
   - `prediction_id`: Unique UUID v4.
   - `timestamp_utc`: ISO 8601 execution timestamp.
   - `student_id`: Pseudonymized SHA-256 student identifier.
   - `code_module`, `code_presentation`, `id_assessment`, `assessment_type`, `scheduled_date`.
   - `model_name`: `HistGradientBoostingClassifier`.
   - `model_version`: `v1.2.0-phase9`.
   - `feature_pipeline_version`: `v1.2.0-remediated`.
   - `input_feature_hash`: SHA-256 hash of input vector for exact reproducibility.
   - `raw_risk_probability`: Float probability.
   - `operational_threshold`: Float (0.20 or 0.10).
   - `predicted_at_risk`: Boolean flag.
   - `governance_routing_action`: e.g., `TMA_MILESTONE_ADVISOR`, `SUPPRESSED_CMA`.
   - `suppress_automated_outreach`: Boolean flag.

### 6.2 Internal Advisor Roster Review
- Advisors access an internal, read-only dashboard during weekly caseload meetings.
- Advisors log qualitative observations without triggering automated student contact.

### 6.3 Automated Health & Drift Telemetry Gates
- **Missing Score Rate Alarm:** Trigger alert if cold-start missingness exceeds 25%.
- **Population Stability Index (PSI):** Compute rolling PSI on `vle_clicks_last_7d` and `days_since_last_click` (flag if PSI > 0.20).
- **Ingestion Latency Tracker:** Monitor time delta between LMS event timestamps and batch scoring (alert if lag > 24h).
- **Calibration Drift:** Compute rolling Brier score and ECE once retrospective grades are entered.

### 6.4 False Positive & False Negative Review Procedure
- **False Positives (P >= 0.20, Score >= 50):** Review whether student received informal tutoring or engaged in intense last-minute study.
- **False Negatives ($P < 0.20$, Score $< 50$):** Investigate sudden life events, illness flags, or cold-start sparsity not reflected in clickstream logs.

---

## 7. Deliverables Summary

| Artifact Path | Description | Status |
| :--- | :--- | :---: |
| [operational_readiness_audit.py](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/operational_readiness_audit.py) | Standalone engine auditing calibration invariance, workload modeling, and router integration | Complete & Certified |
| [phase9_task5_operational_readiness_summary.json](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase9_task5_operational_readiness_summary.json) | Complete machine-readable summary dictionary of operational readiness findings | Complete & Certified |
| [phase9_task5_operational_readiness.md](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase9_task5_operational_readiness.md) | Exhaustive Markdown operational readiness audit report | Complete & Certified |

---

## 8. Final Governance Statement & Non-Deployment Certification

The Student Performance & At-Risk Prediction System demonstrates strong offline discrimination and robust cold-start handling. However, **production deployment remains WITHHELD** pending:
1. Institutional academic advising leadership sign-off on the 7,130 TMA alert volume and tiered counseling model.
2. Live LMS ETL latency verification confirming sub-24h event availability for 7-day cutoff windows.
3. Integration of the universal CMA outreach suppression filter across all course modules.
4. Successful execution of a minimum one-semester shadow-mode pilot following the protocol defined herein.
