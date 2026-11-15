# Phase 11: Data Provenance & Metric Verification Audit
## Student Performance & At-Risk Prediction System (OULAD)

- **Audit Document Version:** 1.0  
- **Audit Date:** 2026-11-15T07:40:00Z  
- **Scope:** Complete Metric-to-Source Verification for Phase 11 Enterprise Analytics Dashboard  
- **Audit Objective:** Validate the empirical origin, population denominators, calculation methodology, and dashboard display suitability of every metric, chart, and table.  
- **Operational Status:** **STRICT READ-ONLY SHADOW MODE**  

---

## Executive Summary

This data provenance audit establishes a rigorous chain of custody for all statistical indicators, model performance benchmarks, and curricular metadata planned for the **Phase 11 Enterprise Analytics Dashboard**.

Every dashboard visual component is systematically classified into:
1. **Empirical Ground-Truth Artifacts (100% Real OULAD Data):** Directly computed from certified historical and prospective OULAD datasets and Phase 1–10 audit reports.
2. **Derived Operational Metrics:** Mathematically derived from verified empirical counts (e.g. weekly alert load derived from total alerts divided by 34 semester weeks).
3. **Explicitly Labeled Demonstrative Data:** Non-existent live systems (e.g. simulated live SIS webhook ingestion timestamps and demonstrative advisor triage queues) strictly tagged with `[DEMO DATA]`.

Unverified claims (such as arbitrary institutional capacity ceilings, unverified VLE dependence thresholds, and conflicting calibration errors) have been excised or reconciled against certified repository artifacts.

---

## 1. Master Metric-to-Source Provenance Matrix

| Metric / Visualization | Source Artifact (File Path) | Model & Evaluation Population | Sample Size & Denominator ($N$) | Evaluation Period | Nature of Data | Dashboard Display Suitability |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **2014J Enrolled Learners (8,607)** | `data/processed/enrollment_base.csv`<br>`reports/temporal_drift_summary.json` (L17) | Unique student enrollments in prospective intake 2014J across modules AAA–GGG. | $N = 8,607$ unique enrolled students. | Presentation 2014J | **Empirical** | **SUITABLE**<br>*(Labeled "2014J Enrolled Learners", not active retention)* |
| **Total Prediction Snapshots (50,014)** | `data/processed/oof_predictions_chronological_2014J.csv` | Chronological prospective evaluation snapshots. | $N = 50,014$ snapshots. | Presentation 2014J | **Empirical** | **SUITABLE** |
| **Actionable TMA Alerts (7,130)** | `reports/phase9_task5_operational_readiness_summary.json` (L52) | Primary HistGradientBoosting model under governed TMA-only policy ($\tau=0.10$ GGG, $\tau=0.20$ standard). | $N = 7,130$ alerts generated from $32,506$ actionable TMA snapshots (21.93% TMA alert rate; 14.26% across all 50,014 cohort snapshots). | Presentation 2014J | **Derived Empirical** | **SUITABLE** |
| **Unique Learners Alerted (3,299)** | `reports/phase9_task5_operational_readiness_summary.json` (L60) | Unique learners receiving at least one actionable TMA alert. | $3,299 / 8,607 = 38.33\%$ of enrolled learners. | Presentation 2014J | **Derived Empirical** | **SUITABLE** |
| **Mean Alerts per Alerted Learner (2.16)** | `reports/phase9_task5_operational_readiness_summary.json` (L62) | Ratio of total actionable TMA alerts to alerted learners. | $7,130 / 3,299 = 2.1613 \approx 2.16$ alerts/learner (max: 9). | Presentation 2014J | **Derived Empirical** | **SUITABLE** |
| **Estimated Weekly Alert Load (~209.7)** | `reports/phase9_task5_operational_readiness.md` (L63) | Actionable TMA alerts divided by semester duration. | $7,130 / 34\text{ weeks} = 209.70588 \approx 209.7\text{ alerts/week}$. | Presentation 2014J | **Derived Historical** | **SUITABLE WITH MANDATORY LABEL**<br>*(Derived historical estimate; not an approved capacity quota)* |
| **Historical 2014J Raw CMA Alerts Suppressed (2,243)** | `reports/phase9_task5_operational_readiness_summary.json` (L48, L53) | Raw-policy CMA alerts on 2014J holdout suppressed by governed policy. | $2,243$ CMA alerts out of $9,227$ raw alerts ($-22.73\%$ workload reduction). | Presentation 2014J | **Derived Empirical** | **SUITABLE WITH MANDATORY LABEL**<br>*(Historical 2014J raw-policy count; not universal CMA record count)* |
| **Cohort Risk Distribution Histogram** | `data/processed/oof_predictions_chronological_2014J.csv` | Primary HGB continuous risk probabilities $\hat{p} \in [0, 1]$. | $N = 50,014$ snapshots. | Presentation 2014J | **Empirical** | **SUITABLE** |
| **Module Breakdown & Alert Volume (AAA–GGG)** | `reports/phase9_task5_operational_readiness_summary.json` (L80-88) | Governed TMA alerts by course module: BBB: 2501, DDD: 1752, CCC: 1532, FFF: 772, GGG: 239, AAA: 187, EEE: 147. | $N = 7,130$ total alerts across 7 modules (denominated against 32,506 actionable TMA snapshots). | Presentation 2014J | **Derived Empirical** | **SUITABLE** |
| **Module Historical Failure Prevalence** | `reports/module_ggg_failure_investigation.json` (L45-80) | Benchmark failure prevalence ($\text{score} < 50$): DDD (15.84%), CCC (15.22%), AAA (7.53%), BBB (7.37%), GGG (5.99%), FFF (4.81%), EEE (4.46%). | $N = 170,874$ total benchmark snapshots (14,623 failures, 8.56% overall prevalence). | 2013B–2014J (All intakes) | **Empirical** | **SUITABLE** |
| **Module GGG Matched Evaluation (PR-AUC 0.2508 vs 0.2353)** | `reports/module_ggg_matched_evaluation.json` | Baseline vs Remediated pipeline across identical 5,656 actionable TMA snapshots (411 failures, 7.27% prevalence). Baseline PR-AUC = 0.2508, Remediated PR-AUC = 0.2353. Calibrated $\tau=0.10$ increases recall to 45.01%–45.74% (vs 19.71%–24.09% at $\tau=0.20$). | Exactly $5,656$ matched TMA snapshots across $2,097$ unique learners (411 positive failures). | Presentations 2013J, 2014B, 2014J | **Empirical Matched** | **SUITABLE**<br>*(Draft figure 0.1189 excised as unverified draft error)* |
| **Module GGG Zero-Weight Coursework CMAs** | `data/raw/assessments.csv`<br>`reports/module_ggg_failure_investigation.json` | Curricular audit: 18 total CMAs across 3 presentations (6 per intake: Day 229/222) with $\text{weight} = 0.0$. | $N = 18$ distinct GGG CMA assessments. | Presentations 2013J, 2014B, 2014J | **Empirical Curricular** | **SUITABLE**<br>*(Restricted strictly to Module GGG; other modules have positive-weight CMAs)* |
| **Submission Lateness vs Score Correlation** | `data/processed/assessment_submissions_clean.csv` | Student submission timestamp difference ($\text{date\_submitted} - \text{date\_due}$). | $N = 173,912$ raw submission records. | 2013B–2014J | **Empirical** | **SUITABLE** |
| **TreeSHAP Global Feature Importance** | `reports/model_explainability_summary.json` (L2-26) | Mean absolute SHAP values across 23 predictive features. | $N = 170,874$ benchmark snapshots. | Full Benchmark | **Empirical TreeSHAP** | **SUITABLE** |
| **TreeSHAP Feature Dependence Claims** | None (Unverified Assertion) | "Sharp risk decline after 15 active VLE days". | No verified artifact in repository. | N/A | **Unverified Claim** | **PROHIBITED**<br>*(Excised; replaced with verified TreeSHAP importance & local case studies)* |
| **PSI Feature Drift Matrix (23 Features)** | `reports/temporal_drift_summary.json` (L46-450) | Population Stability Index between Training ($N=120,860$) and 2014J ($N=50,014$). | $N_{\text{total}} = 170,874$. | 2013B/J/2014B vs 2014J | **Empirical Drift** | **SUITABLE** |
| **Generalization Gap (PR-AUC 0.4606 vs 0.3005)** | `reports/temporal_drift_summary.json` (L24-34) | Primary HGB: Grouped 5-Fold Cross-Validation vs Chronological 2014J Holdout. | $N_{\text{CV}} = 170,874$<br>$N_{2014\text{J}} = 50,014$. | Cross-Validation vs 2014J | **Empirical Benchmark** | **SUITABLE** |
| **2014J Brier Score (0.0812)** | `reports/temporal_drift_summary.json` (L29, L466)<br>`reports/cold_start_model_comparison.json` (L39) | Mean squared probability error on prospective 2014J holdout. | $N = 50,014$ snapshots. | Presentation 2014J | **Empirical Benchmark** | **SUITABLE** |
| **2014J Expected Calibration Error (0.0420)** | `reports/temporal_drift_summary.json` (L464)<br>`reports/cold_start_model_comparison.json` (L40) | 10-bin Expected Calibration Error on 2014J holdout ($N=50,014$, prevalence $9.27\%$). | $N = 50,014$ snapshots. | Presentation 2014J | **Empirical Benchmark** | **SUITABLE**<br>*(Reconciled from conflicting 0.0345)* |
| **Illustrative Demonstration Triage Queue** | Synthetic UI Generator | Prototype demonstration showing how high-risk TMA rows render in UI. | 5 demonstrative records. | Illustrative Prototype | **Synthetic Demonstration** | **RESTRICTED WITH MANDATORY LABEL**<br>*(Must display badge `[DEMO DATA - NOT LIVE SIS/LMS QUEUE]`)* |
| **Simulated Live SIS Webhook & Latency Feed** | Synthetic UI Generator | Illustrative feed of student grade releases and advisor review timestamps. | Demonstrative event stream. | Illustrative Prototype | **Synthetic Demonstration** | **RESTRICTED WITH MANDATORY LABEL**<br>*(Must display badge `[DEMO TELEMETRY - NOT LIVE INTEGRATION]`)* |

---

## 2. In-Depth Reconciliation of Critical Discrepancies

### 2.1 Investigation & Reconciliation of PR-AUC Values 0.3203 and 0.4882

#### Identified Conflict
The values **0.3203** and **0.4882** were cited in earlier documentation as cold-start versus non-cold-start PR-AUC, but Phase 9 Task 2 cold-start experiments reported cold-start PR-AUC on 2014J as **0.1687** and non-cold-start as **0.3478**.

#### Empirical Investigation & Resolution
Tracing through repository artifacts reveals that these two sets of numbers represent **two completely different evaluation designs and populations**:

1. **The Phase 7 Robustness Audit Benchmark (`reports/fairness_robustness_summary.json` lines 1750–1810):**
   - **Model:** Phase 6 Optimized `HistGradientBoostingClassifier`.
   - **Evaluation Population:** The entire 5-fold grouped cross-validation benchmark across all four cohorts (`2013B`, `2013J`, `2014B`, `2014J`), comprising $N = 170,874$ snapshots.
   - **Subgroup A: Missing Historical Score (`recent_prior_score.isna()`):**
     - Sample Size: $N = 29,478$ snapshots (share of cohort: $17.25\%$).
     - At-Risk Failures: $3,092$ (empirical prevalence: $10.49\%$).
     - **PR-AUC:** **0.3203**
     - Brier Score: $0.0810$
     - Precision at 0.20: $33.35\%$ (95% CI: $[32.01, 34.71]$)
     - Recall at 0.20: $50.49\%$ (95% CI: $[48.72, 52.25]$)
     - False Positive Rate: $11.82\%$
   - **Subgroup B: Observed Historical Score (`recent_prior_score.notna()`):**
     - Sample Size: $N = 141,396$ snapshots (share of cohort: $82.75\%$).
     - At-Risk Failures: $11,531$ (empirical prevalence: $8.16\%$).
     - **PR-AUC:** **0.4882**
     - Brier Score: $0.0547$
     - Precision at 0.20: $41.95\%$ (95% CI: $[41.20, 42.71]$)
     - Recall at 0.20: $59.22\%$ (95% CI: $[58.32, 60.12]$)
     - False Positive Rate: $7.28\%$

2. **The Phase 9 Task 2 Prospective Chronological Holdout (`reports/cold_start_model_comparison.json` lines 1–42):**
   - **Model:** Primary `HistGradientBoostingClassifier` trained on `2013B`, `2013J`, `2014B` ($N = 120,860$) and evaluated strictly on `2014J` ($N = 50,014$).
   - **Subgroup A: 2014J Cold-Start ($N = 10,407$, prevalence $13.60\%$):**
     - **PR-AUC:** **0.1687** (reflecting severe compounded chronological drift + cold-start penalty).
     - ROC-AUC: $0.5698$, Brier: $0.1263$, ECE: $0.0910$.
   - **Subgroup B: 2014J Non-Cold-Start ($N = 39,607$, prevalence $8.13\%$):**
     - **PR-AUC:** **0.3478**
     - ROC-AUC: $0.8477$, Brier: $0.0693$, ECE: $0.0291$.
   - **Overall 2014J Holdout ($N = 50,014$, prevalence $9.27\%$):**
     - **PR-AUC:** **0.3005**
     - ROC-AUC: $0.7969$, Brier: $0.0812$, ECE: **0.0420**.

#### Audit Decision
Both benchmarks are mathematically valid within their respective experimental frames. To eliminate ambiguity, the dashboard must **never** display 0.3203 vs 0.4882 without explicit labeling:
- When displaying Phase 7 historical cross-validation robustness, cite $N = 170,874$ (0.3203 vs 0.4882).
- When displaying Phase 9 prospective holdout performance, cite $N = 50,014$ (0.1687 vs 0.3478, overall 0.3005).

---

### 2.2 Investigation & Reconciliation of 2014J ECE: 0.0420 vs 0.0345

#### Identified Conflict
Earlier drafts cited an Expected Calibration Error of **0.0345**, whereas Phase 9 Task 1 and Task 2 reported **0.0420**.

#### Empirical Investigation & Resolution
Inspection of `reports/temporal_drift_summary.json` (lines 460–470) and `reports/cold_start_model_comparison.json` (line 40) establishes:
```json
{
  "predicted_probability_psi": 0.0669,
  "train_mean_predicted_prob": 0.0818,
  "test_2014j_mean_predicted_prob": 0.1130,
  "actual_2014j_prevalence": 0.0927,
  "expected_calibration_error_ece": 0.0420,
  "maximum_calibration_error_mce": 0.3295,
  "brier_score_loss": 0.0812
}
```
- The 0.0345 figure originated from an isolated cross-validation fold during preliminary training.
- The certified empirical Expected Calibration Error for the primary candidate on the 2014J prospective holdout ($N = 50,014$) is **0.0420**.

#### Audit Decision
The unverified 0.0345 value is **excised**. The certified holdout value of **0.0420** (with Brier score **0.0812**) is designated as the sole authorized calibration metric for dashboard display.

---

### 2.3 Curricular Audit: Module GGG Zero-Weight Coursework CMAs

#### Identified Claim
Earlier documentation stated that "CMA predictions are decoupled formative quizzes with 0% weight."

#### Curricular Evidence & Correction
Analysis of `data/raw/assessments.csv` across all 7 course modules demonstrates:
1. **Module GGG Exclusivity:** Module GGG is the **only** module where all coursework assessments (3 TMAs and 6 CMAs per presentation across 2013J, 2014B, 2014J; 18 total CMAs) carry `weight = 0.0`, with 100% of course grade determined by the final exam. Furthermore, all 6 CMAs in GGG are scheduled on the exact same date (Day 229 in 2013J/2014J, Day 222 in 2014B) and were submitted by students on average 54.06 days before the scheduled date.
2. **Other Modules Carry Positive CMA Weights:** Modules BBB, DDD, and FFF include Computer-Marked Assessments that contribute positive percentages (e.g. 1%, 2%, or 5%) to final course outcomes.
3. **Universal CMA Outreach Suppression Rationale:** Across modules AAA–FFF, CMAs are suppressed from automated student outreach **not** because they carry zero weight, but because of institutional governance: CMAs are frequent, low-stakes formative quizzes with high student variance that create excessive false-positive alert fatigue.

#### Audit Decision
The zero-weight decoupled quiz narrative is **restricted strictly to Module GGG**. For all other modules, the dashboard documents that CMA outreach suppression is an operational governance policy designed to prevent alert fatigue and preserve CMAs as research telemetry.

---

### 2.4 Removal of Unverified Institutional Capacity & Triage Queue Claims

1. **Institutional Capacity Ceiling:** The 400-alert weekly ceiling cited in initial drafts was an arbitrary illustrative construct with zero university backing. It has been removed. The historical metric (~209.7 alerts/week) is derived strictly from $7,130\text{ alerts} / 34\text{ weeks}$ and is presented strictly as a derived historical estimate.
2. **Pending Advisor Case Queue:** Live pending case queues cannot exist in an offline research sandbox without active SIS/LMS database integrations. The demonstration queue on Page 1 is retained solely as an illustrative layout component and is prominently labeled on every element as `[DEMO DATA - NOT LIVE SIS/LMS QUEUE]`.

---

### 2.5 Curricular Audit: Module-Level Prevalence Denominators & Alert Separation

#### Identified Need
Previous documentation reported approximate, unverified module failure rates (e.g. "AAA (8.4%), BBB (9.2%)") and did not explicitly separate full cohort snapshot denominators from actionable TMA alert denominators.

#### Empirical Verification & Audit
The empirical data in `data/processed/prediction_dataset.csv`, `reports/module_ggg_failure_investigation.json`, and `reports/phase9_task5_operational_readiness_summary.json` provides exact, verified figures across both the full benchmark and the 2014J prospective holdout:

| Module Code | Faculty Domain | Full Benchmark Snapshots ($N=170,874$) | Benchmark Failures | Benchmark Prevalence (%) | 2014J Cohort Snapshots ($N=50,014$) | 2014J Failures | 2014J Prevalence (%) | 2014J Actionable TMA Snapshots ($N=32,506$) | 2014J Governed TMA Alerts ($N=7,130$) | TMA Alert Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAA** | Social Sciences | 3,146 | 237 | **7.53%** | 1,515 | 129 | **8.51%** | 1,515 | 187 | **12.34%** |
| **BBB** | Social Sciences | 42,979 | 3,166 | **7.37%** | 7,398 | 1,236 | **16.71%** | 7,398 | 2,501 | **33.81%** |
| **CCC** | STEM | 17,014 | 2,590 | **15.22%** | 10,277 | 1,386 | **13.49%** | 4,431 *(5,846 CMAs)* | 1,532 | **34.57%** |
| **DDD** | STEM | 29,865 | 4,730 | **15.84%** | 7,051 | 829 | **11.76%** | 7,051 | 1,752 | **24.85%** |
| **EEE** | STEM | 7,886 | 352 | **4.46%** | 3,227 | 126 | **3.90%** | 3,227 | 147 | **4.56%** |
| **FFF** | STEM | 54,769 | 2,636 | **4.81%** | 16,175 | 646 | **3.99%** | 7,260 *(8,915 CMAs)* | 772 | **10.63%** |
| **GGG** | STEM | 15,215 | 912 | **5.99%** | 4,371 | 285 | **6.52%** | 1,624 *(2,747 CMAs)* | 239 | **14.72%** |
| **Total**| — | **170,874** | **14,623** | **8.56%** | **50,014** | **4,637** | **9.27%** | **32,506** | **7,130** | **21.93%** |

#### Audit Findings
1. **Denominator Separation:** The 7,130 actionable alerts are generated strictly from the **32,506 actionable TMA snapshots** in 2014J (representing an alert rate of 21.93% on actionable TMAs, and 14.26% across the entire 50,014 cohort snapshot space).
2. **Prevalence Realities:** High-failure courses are STEM modules CCC (15.22% benchmark, 13.49% holdout) and DDD (15.84% benchmark, 11.76% holdout), alongside Social Science module BBB (7.37% benchmark, surging to 16.71% in 2014J). Low-failure courses are EEE (4.46% benchmark, 3.90% holdout) and FFF (4.81% benchmark, 3.99% holdout).
3. **Threshold Behavior:** The governed policy ensures that alert volumes align with curricular risk: BBB, CCC, and DDD receive 5,785 of the 7,130 alerts (81.1%), whereas low-prevalence courses receive proportionate alert loads without over-alerting.

---

### 2.6 Reconciliation of Module GGG Matched Evaluation Metrics (0.2508 / 0.2353 vs Erroneous 0.1189)

#### Identified Conflict
Initial Phase 11 dashboard architecture drafts referenced a Module GGG matched study PR-AUC of `0.1189` and `415 failures (7.34%)`. In contrast, Phase 9 Task 3 matched evaluation artifacts (`reports/module_ggg_matched_evaluation.json` and `.md`) reported baseline PR-AUC of `0.2508` and remediated PR-AUC of `0.2353` on `5,656` matched snapshots with `411` positive failures (`7.27%`).

#### Source Tracing & Root Cause Analysis
1. **Ground Truth Evidence:** Inspection of `reports/module_ggg_matched_evaluation.json` confirms:
   - Population: Exactly 5,656 actionable TMA prediction opportunities across 2,097 unique learners in presentations 2013J, 2014B, and 2014J.
   - Ground Truth Positives: Exactly 411 observed at-risk failures (7.27% prevalence).
   - Baseline Model: PR-AUC = **0.2508**, ROC-AUC = **0.7510**.
   - Remediated Model: PR-AUC = **0.2353**, ROC-AUC = **0.7451** ($\Delta = -0.0155$).
   - Operational Decision Metrics:
     - Global $\tau = 0.20$: Baseline Recall = **24.09%** (99 TP / 312 FN); Remediated Recall = **19.71%** (81 TP / 330 FN).
     - Calibrated $\tau = 0.10$: Baseline Recall = **45.01%** (185 TP / 226 FN); Remediated Recall = **45.74%** (188 TP / 223 FN).
2. **Origin of Erroneous Figure (0.1189):** The value `0.1189` was an unverified draft artifact. Extensive search across the entire repository confirms that no experiment, script, or JSON artifact produced PR-AUC 0.1189. It appears to have been an accidental misread or typographical corruption (potentially conflating with Brier score slices around 0.1183).

#### Audit Decision
The unverified figure `0.1189` and the inaccurate count `415 failures` are **completely excised**. The certified empirical figures from `reports/module_ggg_matched_evaluation.json` (Baseline PR-AUC **0.2508**, Remediated PR-AUC **0.2353**, Matched Population **5,656**, Failures **411**, and Calibrated Recall **45.01%–45.74%**) are established as the sole authorized ground truth for the dashboard.

---

## 3. Existing Phase 10 API Endpoints vs. Proposed Phase 11 Analytics Routes

| Endpoint Path | HTTP Method | Implementation State | Verification Method | Safe For Production? |
| :--- | :---: | :---: | :--- | :---: |
| `/health` | `GET` | **Existing & Certified** | `tests/test_phase10_api.py::test_health_endpoint` | **Shadow Mode Only** |
| `/model-info` | `GET` | **Existing & Certified** | `tests/test_phase10_api.py::test_model_info_endpoint` | **Shadow Mode Only** |
| `/predict` | `POST` | **Existing & Certified** | `tests/test_phase10_api.py::test_predict_single_standard_tma` | **Shadow Mode Only** |
| `/predict/batch` | `POST` | **Existing & Certified** | `tests/test_phase10_api.py::test_predict_batch_isolated_fault_handling` | **Shadow Mode Only** |
| `/api/v1/analytics/cohort-summary` | `GET` | **Proposed (Task 2)** | To be implemented; pre-computed from `oof_predictions_chronological_2014J.csv` | **Shadow Mode Only** |
| `/api/v1/analytics/student-explorer` | `GET` | **Proposed (Task 2)** | To be implemented; paginated view of pseudonymized 2014J records | **Shadow Mode Only** |
| `/api/v1/analytics/modules-comparison` | `GET` | **Proposed (Task 2)** | To be implemented; aggregates certified Phase 9 module distributions | **Shadow Mode Only** |
| `/api/v1/explainability/global-importance` | `GET` | **Proposed (Task 2)** | To be implemented; reads `reports/model_explainability_summary.json` | **Shadow Mode Only** |
| `/api/v1/governance/audit-records` | `GET` | **Proposed (Task 2)** | To be implemented; reads thread-safe `DecisionAuditLogger` stream | **Shadow Mode Only** |
| `/api/v1/monitoring/psi-feature-drift` | `GET` | **Proposed (Task 2)** | To be implemented; reads `reports/temporal_drift_summary.json` | **Shadow Mode Only** |

---

## 4. Final Governance & Compliance Certification

All dashboard specifications comply 100% with mandatory project governance rules:
- **Read-Only Shadow Mode:** Outbound student communications are strictly locked (`ALLOW_STUDENT_OUTREACH = False`). Direct attempts raise `SecurityViolationError`.
- **Zero Messaging Routes:** The API exposes no endpoints for sending emails, WhatsApp, SMS, or intervention actions.
- **Privacy Safe:** 100% keyed HMAC-SHA256 pseudonymization. Raw student IDs and cryptographic secret keys are never returned in responses or displayed in the UI.
- **Demographic Exclusion:** Sensitive demographic features (`gender`, `imd_band`, `disability`, `age_band`, `highest_education`, `region`) remain strictly excluded from predictive inputs.
- **Threshold Calibration:** GGG TMAs evaluate at calibrated $\tau = 0.10$; standard TMAs evaluate at $\tau = 0.20$.
- **Universal CMA Suppression:** Computer-marked assessments across all 7 modules remain research-only telemetry.

---
*Data provenance audit certified against ground-truth repository evidence. No implementation code executed.*

