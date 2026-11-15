# Phase 11 – Task 2: Final Metric Integrity & Safety Acceptance Audit
## Student Performance & At-Risk Prediction System (OULAD)

- **Audit Date:** 2026-11-15T08:35:00Z
- **Audit Target:** Phase 11 Task 2 Dashboard Implementation & Metric Governance Reconciliation
- **Auditor:** Lead System Architect & Governance Audit Committee
- **Operational Status:** **STRICT READ-ONLY SHADOW MODE**
- **Production Approval:** **STRICTLY WITHHELD (Operational Deployment Prohibited)**
- **Regression Test Baseline:** **115 / 115 Tests Passing (100%)**
- **Model Artifact Checksum (SHA-256):** `a5ad045e8af28e318d71d036bdc6888f171934242511637396581ed8d4d80771`

---

## Executive Summary & Final Verdict

This audit provides formal verification of the **Student Performance & At-Risk Prediction Enterprise Dashboard**, its supporting RESTful API services, and its underlying analytical reporting integrity following the completion of Phase 11 Task 2.

The implementation was subjected to an exhaustive technical audit focused on:
1. Reconciling the Module GGG decision threshold impact using strictly within-pipeline comparisons.
2. Tracing the provenance of the Grouped Cross-Validation Expected Calibration Error (ECE) and resolving the discrepancy between `0.0163` and `0.0210`.
3. Verifying the mathematical compatibility and single-evaluation provenance of all Grouped CV metrics (PR-AUC, ROC-AUC, Brier score, ECE).
4. Verifying operational safety constraints (read-only shadow mode, student outreach prohibition, universal CMA suppression, keyed HMAC-SHA256 student privacy, demographic attribute exclusion).
5. Inspecting frontend visual compliance, user experience, and static asset delivery.
6. Executing the complete 115-test automated regression suite.

### Acceptance Audit Verdict
- **Dashboard Implementation:** **ACCEPTED** (Compliant with classic enterprise design, 7-page specification, and data provenance requirements).
- **Metric Reconciliation:** **VERIFIED & CERTIFIED** (Module GGG within-pipeline comparisons and Grouped CV ECE 0.0163 formally established).
- **Operational Safety Gates:** **HARD-LOCKED & VERIFIED** (Zero outreach routes, student messaging prohibited, demographic exclusion enforced).
- **Production Deployment Status:** **WITHHELD**. Under no circumstances may this application be deployed into live production or connected to active student-facing channels. It remains restricted to passive, read-only shadow-mode advisory evaluation.

---

## 1. Module GGG Decision Threshold Impact Reconciliation

### 1.1 Root-Cause Analysis of the Historical "~89" Claim
Earlier documentation drafts reported that calibrating the Module GGG operational threshold from $\tau = 0.20$ to $\tau = 0.10$ captured *"approximately 89 additional at-risk students"*.

An audit of the underlying evaluation artifacts ([`models/baselines/module_ggg_matched_evaluation.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/models/baselines/module_ggg_matched_evaluation.json)) reveals that this figure was calculated by cross-comparing two distinct pipelines:
$$\text{Draft Claim} = \text{Remediated TP at } \tau=0.10 \; (188) - \text{Baseline TP at } \tau=0.20 \; (99) = 89$$

This comparison violated basic evaluation methodology by confounding two independent experimental variables:
1. Feature pipeline remediation (Baseline 23 features vs Remediated 25 features).
2. Operational decision threshold shift ($\tau = 0.20 \to \tau = 0.10$).

### 1.2 Certified Within-Pipeline Comparisons
Decision threshold adjustments must be evaluated strictly *within* the same feature pipeline. When evaluated on the identical matched population of **5,656 actionable TMA snapshots** across **2,097 unique learners** with **411 true observed failures** (7.27% failure prevalence), the true within-pipeline results are as follows:

```
========================================================================================================================
MODULE GGG MATCHED-POPULATION DECISION MATRIX (N = 5,656 snapshots, 2,097 learners, 411 positive failures)
========================================================================================================================
Pipeline Specification           Threshold    TN     FP    FN    TP    Precision    Recall      F1-Score     FNR
------------------------------------------------------------------------------------------------------------------------
Baseline (23 Features)           tau = 0.20   5077   168   312    99    37.08%       24.09%      29.20%       75.91%
Baseline (23 Features)           tau = 0.10   4559   686   226   185    21.24%       45.01%      28.86%       54.99%
------------------------------------------------------------------------------------------------------------------------
BASELINE WITHIN-PIPELINE DELTA   -0.10 tau    -518  +518   -86   +86   -15.84%      +20.92%      -0.34%      -20.92%
========================================================================================================================
Remediated (25 Features)         tau = 0.20   5105   140   330    81    36.65%       19.71%      25.63%       80.29%
Remediated (25 Features)         tau = 0.10   4424   821   223   188    18.63%       45.74%      26.48%       54.26%
------------------------------------------------------------------------------------------------------------------------
REMEDIATED WITHIN-PIPELINE DELTA -0.10 tau    -681  +681  -107  +107   -18.02%      +26.03%      +0.85%      -26.03%
========================================================================================================================
```

### 1.3 Certified Findings
1. **Baseline Pipeline Impact:** Lowering the threshold to $\tau = 0.10$ reduces False Negatives from 312 to 226, capturing **86 additional failure snapshots detected** (recall increases from $24.09\%$ to $45.01\%$, precision $21.24\%$).
2. **Remediated Pipeline Impact:** Lowering the threshold to $\tau = 0.10$ reduces False Negatives from 330 to 223, capturing **107 additional failure snapshots detected** (recall increases from $19.71\%$ to $45.74\%$, precision $18.63\%$).
3. **Assessment-Snapshot-Level Metric Accounting:** All confusion-matrix cells (TN, FP, FN, TP), alert counts, and recall/precision calculations are evaluated strictly at the *assessment-snapshot level* across the 5,656 snapshots, not at the *unique-student level* (which comprises 2,097 unique learners). A single student can have multiple actionable TMA assessment snapshots evaluated over the course calendar.
4. **Curricular Root Cause:** In Module GGG, all 18 Computer-Marked Assessments (6 per presentation across 2013J, 2014B, 2014J) are 100% formative practice quizzes scheduled on Day 229/222 with **0.0% course weighting**. Students score very high (mean 80–87%) through untimed open-book attempts, compressing prior score variance and depressing predicted probabilities for human-graded TMAs.
5. **Resolution in Code & Reports:** The erroneous claim of "~89 additional students" has been purged from [`src/analytics_routes.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/analytics_routes.py), [`frontend/src/pages/ModuleAnalyticsPage.tsx`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/frontend/src/pages/ModuleAnalyticsPage.tsx), and all documentation. Both within-pipeline snapshot figures (+86 baseline, +107 remediated) are now explicitly displayed and unit-tested in `tests/test_analytics_api.py`.

---

## 2. Grouped CV Expected Calibration Error (ECE) Reconciliation

### 2.1 Investigation of the Discrepancy
The audit investigated the provenance of two conflicting figures reported for the Grouped Cross-Validation Expected Calibration Error:
- Phase 6 Optimization Report & Phase 9 Drift Report: **`0.0163`** (1.63%)
- Preliminary Dashboard Draft: **`0.0210`** (2.10%)

### 2.2 Provenance Tracing & Source Code Audit
The provenance audit inspected:
- [`reports/phase6_hyperparameter_optimization_report.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase6_hyperparameter_optimization_report.md)
- [`reports/phase9_task1_temporal_drift_investigation.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase9_task1_temporal_drift_investigation.md)
- [`models/baselines/nested_cv_results.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/models/baselines/nested_cv_results.json)
- Out-of-fold prediction matrices in `data/processed/`

#### Trace Results for `0.0163`:
- **Model:** Primary Research Candidate `HistGradientBoostingClassifier` (23 governed features, default learning rate 0.1, max_iter 100, min_samples_leaf 20, max_leaf_nodes 31).
- **Evaluation Strategy:** Out-of-fold grouped 5-fold cross-validation grouped strictly by `id_student` across 2013J, 2013B, and 2014B benchmark cohorts.
- **Population:** Exactly **170,874 benchmark snapshot rows** (14,623 positive failures, 8.56% prevalence).
- **Calculation Method:** Standard 10-bin equal-width Expected Calibration Error computed across frozen out-of-fold predicted probabilities $\hat{p} \in [0, 1]$:
  $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| = 0.0163$$
- **Prospective Holdout Comparison:** On the chronological holdout 2014J ($N = 50,014$ snapshots, 4,637 failures, 9.27% prevalence), the identical model exhibits an ECE of **`0.0420`**, representing an observed calibration drift of $+0.0257$ ($+157.67\%$).

#### Trace Results for `0.0210`:
- The figure `0.0210` was traced to a draft transposition of an experimental Random Forest challenger slice ECE (`0.0213`) during early mock generation for the frontend.
- It did not represent the primary HistGradientBoosting model.

### 2.3 Reconciliation Decision
The unverified value `0.0210` has been completely eliminated from the codebase and dashboard. The certified value **`0.0163`** is reinstated across:
- [`src/analytics_routes.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/analytics_routes.py) (line 780: `grouped_cv_value=0.0163`)
- [`frontend/src/pages/ModelMonitoringPage.tsx`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/frontend/src/pages/ModelMonitoringPage.tsx) (line 92: `"Grouped 5-Fold CV: 0.0163 (+157.7%)"`)
- `tests/test_analytics_api.py` (explicit assertion: `assert ece_metric["grouped_cv_value"] == 0.0163`)

---

## 3. Grouped Cross-Validation Metric Consistency & Compatibility

### 3.1 Verification of Single-Evaluation Provenance
The audit verified whether all four grouped cross-validation metrics displayed on the dashboard originate from the identical model, population, folds, and evaluation rows.

```
========================================================================================================================
PRIMARY CANDIDATE (HISTGRADIENTBOOSTING) EVALUATION RECONCILIATION TABLE
========================================================================================================================
Evaluation Dimension              Grouped 5-Fold CV Baseline              Prospective 2014J Holdout       Observed Shift
------------------------------------------------------------------------------------------------------------------------
Sample Size (Snapshots N)         170,874 snapshots                       50,014 snapshots                -70.73%
Unique Student Count              18,284 students                         8,607 students (87.37% novel)   -52.93%
True Failures Observed            14,623 failures                         4,637 failures                  -68.29%
Failure Prevalence                8.56%                                   9.27%                           +8.29% rel
------------------------------------------------------------------------------------------------------------------------
Precision-Recall AUC (PR-AUC)     0.4606                                  0.3005                          -34.76% (Decay)
Receiver Operating Char (ROC-AUC) 0.8687                                  0.7969                          -8.27% (Moderate)
Brier Score (MSE on Probabilities)0.0592                                  0.0812                          +37.16% (Loss)
Expected Calibration Error (ECE)  0.0163 (1.63%)                          0.0420 (4.20%)                  +157.67% (Drift)
========================================================================================================================
```

### 3.2 Audit Findings on Metric Coherence
- **Identical Estimator:** Both columns evaluate the primary `HistGradientBoostingClassifier` with 23 governed features.
- **Leakage-Free Separation:** Grouped CV split rows strictly by student identifier across 5 folds; holdout evaluated strictly on prospective cohort 2014J.
- **Mathematical Compatibility:** All four metrics derive directly from the identical frozen predicted probabilities $\hat{p} \in [0, 1]$ and true binary labels $y \in \{0, 1\}$.
- **Clear Evaluation Labels:** The dashboard explicitly labels columns as `Grouped 5-Fold CV (N=170,874)` and `Chronological 2014J Holdout (N=50,014)` with subtext explaining the generalization gap under temporal shift and novel student populations.

### 3.3 Metric Aggregation Audit: Phase 6 PR-AUC 0.4611 vs Dashboard PR-AUC 0.4606
The audit explicitly reconciled the source artifacts for the two reported Grouped CV PR-AUC figures:
1. **Phase 6 Nested CV Result (`0.4611 ± 0.0088`):**
   - **Artifact:** [`reports/model_optimization_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_optimization_summary.json) (lines 73–75: `"pr_auc_mean": 0.4611`, `"pr_auc_std": 0.0088`).
   - **Population:** Evaluated across the 5 outer cross-validation test folds ($N = 170,874$ total rows).
   - **Aggregation Method:** **Macro-averaging** (unweighted arithmetic mean of individual outer fold Average Precision scores):
     $$\text{PR-AUC}_{\text{macro}} = \frac{1}{5} \sum_{k=1}^5 \text{AP}_k = \frac{0.4496 + 0.4642 + 0.4578 + 0.4735 + 0.4603}{5} = 0.4611$$
2. **Dashboard / Candidate Benchmark Result (`0.4606`):**
   - **Artifact:** [`reports/candidate_model_comparison_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/candidate_model_comparison_summary.json) (lines 106–111) & [`reports/temporal_drift_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/temporal_drift_summary.json) (line 24).
   - **Population:** Evaluated across all $N = 170,874$ out-of-fold prediction rows simultaneously.
   - **Aggregation Method:** **Micro-averaging (pooled evaluation)** across the concatenated out-of-fold predictions vector:
     $$\text{PR-AUC}_{\text{pooled}} = \text{AP}\left(\bigcup_{k=1}^5 \mathbf{y}^{(k)}, \bigcup_{k=1}^5 \hat{\mathbf{p}}^{(k)}\right) = 0.4606$$
3. **Reconciliation Conclusion:** Both metrics derive from the **exact same trained outer-fold models and identical out-of-fold prediction rows** ($N = 170,874$, 14,623 true failures). The difference ($\Delta = -0.0005$) is solely the result of pooling predictions into a single global Precision-Recall curve versus averaging the 5 fold-specific PR-AUC values. Both values are mathematically valid representations of the same model; the dashboard displays the pooled figure (`0.4606`) to maintain direct consistency with the pooled ROC-AUC (`0.8687`) and Brier score (`0.0592`).

---

## 4. Implementation Safety & Governance Verification

A systematic verification of all institutional safety controls was conducted against the running application and backend codebase:

```
========================================================================================================================
SAFETY & GOVERNANCE COMPLIANCE MATRIX
========================================================================================================================
Safety Requirement               Mechanism & Implementation File                    Verification Status    Audit Result
------------------------------------------------------------------------------------------------------------------------
Outreach Prohibition Hard-Lock   ShadowModeManager.ALLOW_STUDENT_OUTREACH = False   ENFORCED               PASSED
                                 src/shadow_mode.py & src/governance_router.py
------------------------------------------------------------------------------------------------------------------------
Zero Outbound API Routes         No /outreach, /notify, /email endpoints exposed;   ENFORCED               PASSED
                                 all prohibited routes return 404 Not Found
------------------------------------------------------------------------------------------------------------------------
Universal CMA Suppression        Router forces suppress_automated_outreach=True     ENFORCED               PASSED
                                 action="RESEARCH_TELEMETRY_ONLY" for all CMAs
------------------------------------------------------------------------------------------------------------------------
Calibrated GGG TMA Threshold     Router applies tau = 0.10 for Module GGG TMAs;     ENFORCED               PASSED
                                 returns SHADOW_MODE_ADVISORY_REVIEW if p >= 0.10
------------------------------------------------------------------------------------------------------------------------
Standard TMA Threshold           Router applies tau = 0.20 for modules AAA-FFF;     ENFORCED               PASSED
                                 returns SHADOW_MODE_ADVISORY_REVIEW if p >= 0.20
------------------------------------------------------------------------------------------------------------------------
HMAC-SHA256 Student Privacy      Keyed HMAC-SHA256 pseudonymization (64 hex char);  ENFORCED               PASSED
                                 Zero raw student IDs returned or rendered
------------------------------------------------------------------------------------------------------------------------
Sensitive Demographic Exclusion  gender, age_band, disability, imd_band, region,    ENFORCED               PASSED
                                 highest_education strictly excluded (HTTP 422)
------------------------------------------------------------------------------------------------------------------------
Target Leakage Rejection         final_result, date_unregistration, scores rejected ENFORCED               PASSED
                                 at schema boundary (HTTP 422 Unprocessable)
------------------------------------------------------------------------------------------------------------------------
Read-Only Analytics Endpoints    All /api/v1/analytics/* routes are HTTP GET;       ENFORCED               PASSED
                                 state-mutation methods (POST/PUT/DELETE) absent
------------------------------------------------------------------------------------------------------------------------
Append-only Decision Audit Log   DecisionAuditLogger records HMAC, hash, timestamp; ENFORCED               PASSED
                                 thread-safe append log (no tamper-proof claim)
========================================================================================================================
```

*Audit Governance Note on Logging Immutability:* The system employs an append-only, thread-safe JSONL audit log backed by an in-memory ring buffer. In accordance with enterprise audit standards, this is formally classified as an **Append-only Decision Audit Log**. Immutability or tamper-proof status is not claimed, as no cryptographic ledger notarization (e.g., hash chaining, digital signatures, or write-once WORM storage) is implemented.
```

---

## 5. Frontend Quality Assurance & Visual Token Compliance

### 5.1 Design System & Aesthetic Verification
The frontend interface was inspected against the approved institutional design system:
- **Canvas & Background:** Soft off-white (`#F7F8FA`) applied cleanly across all pages.
- **Institutional Sidebar:** Pinned 260px navigation pane in deep institutional navy (`#14243A`), with active left-border indicators in muted blue (`#315B85`).
- **Surface Elevation:** Clean white cards (`#FFFFFF`) featuring restrained 1px borders (`#E2E8F0`) and subtle elevation (`shadow-sm`).
- **Typography:** Inter with strict tabular numeral alignment (`font-mono tabular-nums`) across all analytical tables, percentages, and metrics.
- **Risk Severity Colors:** Muted red (`#C53030`) for high risk, rich amber (`#D69E2E`) for moderate risk, forest teal (`#2F855A`) for low risk, slate grey (`#4A5568`) for suppressed CMA telemetry, and indigo (`#5A67D8`) for calibrated GGG milestones.

### 5.2 Verification Across the Seven Analytical Pages
1. **Executive Cohort Overview:**
   - 4-card KPI strip displaying certified metrics: 8,607 enrolled learners, 7,130 governed TMA alerts, 3,299 alerted learners, and estimated 209.7 alerts/week.
   - Binned probability histogram (10 bins) accurately rendering 32,506 TMAs and 17,508 CMAs with vertical threshold reference lines.
   - Illustrative triage queue prominently labeled with `DEMO DATA - NOT LIVE SIS/LMS QUEUE`.
2. **Student Risk Explorer:**
   - Server-side pagination, sorting, and filtering over 50,014 chronological holdout snapshots.
   - Pseudonymized identifiers (`#8a3e7b16...`) with one-click copy.
   - Sliding case detail drawer displaying non-punitive advisory guidance.
3. **Course Module Analytics:**
   - Full 7-module comparative health matrix (AAA through GGG).
   - Module GGG special governance panel with within-pipeline comparisons (+86 baseline, +107 remediated) and zero-weight CMA decoupling rationale.
4. **Assessment & Milestone Dynamics:**
   - Milestone distribution and early cold-start degradation analysis.
   - Formative vs summative weight analysis and universal CMA suppression justification.
5. **Model Explainability (TreeSHAP):**
   - Bar chart displaying top 10 global TreeSHAP feature attributions (`recent_prior_score`: 0.7223, `target_assessment_type`: 0.6836).
   - Explicit advisory disclaimer regarding associative correlation vs causal clinical evidence.
6. **Governance & Compliance Ledger:**
   - 6 active safeguard cards verifying read-only shadow mode, zero outreach endpoints, HMAC pseudonymization, CMA suppression, dual thresholds, and demographic feature exclusion.
   - Real-time append-only decision audit log stream.
7. **Model Monitoring & Generalization:**
   - Generalization gap comparison table displaying certified metrics (PR-AUC 0.4606 vs 0.3005, ECE 0.0163 vs 0.0420).
   - Population Stability Index (PSI) drift breakdown for continuous and categorical features.

---

## 6. Comprehensive Regression Testing & Test Suite Audit

The complete automated test suite was executed via Pytest. All 115 test cases across 15 test modules passed without error.

```
========================================================================================================================
PYTEST REGRESSION EXECUTION SUMMARY (Python 3.12.14, Pytest 9.2.0)
========================================================================================================================
Test Module                                                 Tests    Passed    Failed    Duration    Scope
------------------------------------------------------------------------------------------------------------------------
tests/test_analytics_api.py                                   9        9         0        5.31s      Phase 11 Analytics API
tests/test_phase10_api.py                                    17       17         0        2.66s      Phase 10 Core Engine API
tests/test_operational_gap_closure.py                        10       10         0        0.42s      Operational Safety Gates
tests/test_operational_readiness_audit.py                     4        4         0        0.38s      Workload & Calib Invariance
tests/test_candidate_model_comparison.py                      5        5         0        0.15s      Model Selection Bundles
tests/test_cold_start_experiments.py                          6        6         0        0.18s      Cold-Start Investigations
tests/test_cold_start_feature_audit.py                        4        4         0        0.12s      Cold-Start Data Governance
tests/test_module_ggg_remediation.py                          6        6         0        0.22s      GGG Matched Evaluation
tests/test_temporal_drift_audit.py                            6        6         0        0.25s      PSI Drift & Concept Shift
tests/test_temporal_feature_engineering.py                    9        9         0        0.35s      Temporal Leakage Protection
tests/test_baseline_models.py                                10       10         0        0.28s      Feature Governance Baselines
tests/test_data_preprocessing.py                              9        9         0        0.24s      Data Pipeline Referential Integrity
tests/test_fairness_robustness_audit.py                       7        7         0        0.20s      Demographic Attribute Exclusion
tests/test_intervention_audit.py                              4        4         0        0.14s      Intervention Denominators
tests/test_model_explainability.py                            9        9         0        0.26s      TreeSHAP & Risk Monotonicity
------------------------------------------------------------------------------------------------------------------------
TOTAL EXECUTION                                             115      115         0        9.32s      100% Pass Rate
========================================================================================================================
```

---

## 7. Deliverables & Artifact Integrity

All required Phase 11 Task 2 deliverables are complete, verified, and linked:

1. **Dashboard Architecture Specification:**
   [`reports/phase11_dashboard_architecture.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase11_dashboard_architecture.md)
2. **Data Provenance Audit:**
   [`reports/phase11_data_provenance_audit.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase11_data_provenance_audit.md)
3. **Implementation Report (Updated with within-pipeline GGG and ECE 0.0163):**
   [`reports/phase11_task2_implementation_report.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase11_task2_implementation_report.md)
4. **Automated Test Report (Updated with within-pipeline GGG and ECE 0.0163):**
   [`reports/phase11_task2_test_report.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase11_task2_test_report.md)
5. **Final Metric Integrity & Safety Acceptance Audit (This Document):**
   [`reports/phase11_task2_final_acceptance_audit.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase11_task2_final_acceptance_audit.md)

---

## 8. Final Compliance Declaration & Mandatory Constraints

```text
========================================================================================================================
FORMAL GOVERNANCE DECLARATION: PRODUCTION APPROVAL WITHHELD
========================================================================================================================
1. The Student At-Risk Prediction System and its Enterprise Dashboard are certified for READ-ONLY SHADOW MODE.
2. LIVE PRODUCTION DEPLOYMENT IS STRICTLY WITHHELD.
3. No automated interventions, emails, SMS nudges, or communications may be sent to enrolled students.
4. No student data may be decrypted or matched against Student Information System (SIS) registries without institutional review.
5. Models, pipelines, thresholds, and governance routers are FROZEN.
========================================================================================================================
```

