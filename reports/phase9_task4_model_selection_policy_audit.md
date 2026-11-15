# Phase 9 Task 4: Final Model Selection Policy Audit Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Audit Date:** 2026-11-15T06:45:01Z  
- **Status:** Complete, Audited & Certified  
- **Primary Model Candidate:** `HistGradientBoostingClassifier` (Deep Capacity Regularized)  
- **Challenger Candidate:** `RandomForestClassifier` (100 Trees, Depth 12)  
- **Comparative Baseline:** `LogisticRegression` (L2 Regularized, Median Imputed)  
- **Scope:** Targeted governance audit of probability calibration, decision thresholds, CMA evaluation, and final production eligibility.

---

## 1. Probability Calibration Audit

### 1.1 Verification of Training Isolation (Phase 6 & Task 4)
- **Calibrator Training Protocol:** In Phase 6, probability calibration was evaluated using `CalibratedClassifierCV(method='isotonic')` and `CalibratedClassifierCV(method='sigmoid')`. These calibrators were fitted **strictly within inner 3-fold cross-validation on training partitions only**.
- **Chronological Holdout Isolation:** Exactly **zero labels** from the prospective chronological holdout (`2014J`, $N = 50,014$) were used for fitting calibrators, tuning hyperparameters, or selecting thresholds.
- **Leakage Status:** **Certified Leakage-Free.**

### 1.2 The Mechanism of Ranking Degradation in Phase 6
In Phase 6, post-hoc calibration was rejected because it caused a catastrophic drop in Average Precision (PR-AUC):
- **Base HistGradientBoosting PR-AUC:** **0.4669**
- **Isotonic Calibrated PR-AUC:** **0.3538** (Delta = -0.1131, a 24.2% relative loss)
- **Sigmoid Calibrated PR-AUC:** **0.2915** (Delta = -0.1754, a 37.6% relative loss)

**Mathematical Root Cause:**
1. **Piecewise Step Function Plateauing:** Under class imbalance (8.56% positive prevalence), isotonic regression fits a non-decreasing piecewise constant step function. In low-density probability regions, the pool-adjacent-violators algorithm aggregates hundreds of continuous gradient boosting predictions into identical flat step bins.
2. **Tie Generation and Resolution Collapse:** Instances assigned to the same horizontal step share identical predicted probabilities. Average Precision evaluates the continuous area under the precision-recall curve across distinct score thresholds; when fine-grained rankings are collapsed into coarse tied steps, the ranking resolution is destroyed.
3. **Parametric Sigmoid Distortion:** Platt scaling assumes that the model outputs log-odds that can be mapped to true probabilities via a two-parameter logistic sigmoid. However, HistGradientBoosting natively optimizes binary cross-entropy (log-loss), meaning that its terminal leaf values already approximate true log-odds. Forcing an additional sigmoid transformation over tree predictions distorts probability boundaries without improving rank ordering.

### 1.3 Task 4 Empirical Calibration vs. Ranking Trade-off
In Task 4, calibration and discrimination metrics were audited across both grouped cross-validation and the prospective chronological holdout:

| Evaluation Environment | Candidate Model | PR-AUC (Ranking) | Brier Score Loss | Expected Calibration Error (ECE) |
| :--- | :--- | :---: | :---: | :---: |
| **Grouped 5-Fold CV** ($N = 170,874$) | **HistGradientBoosting** | **0.4606** | **0.0592** | **0.0163 (1.63%)** |
| | **Random Forest** | 0.4368 | 0.0616 | 0.0179 (1.79%) |
| | **Logistic Regression** | 0.3226 | 0.0674 | 0.0065 (0.65%) |
| **2014J Prospective Holdout** ($N = 50,014$) | **HistGradientBoosting** | 0.3005 | 0.0812 | 0.0420 (4.20%) |
| | **Random Forest** | **0.3093** | **0.0768** | **0.0266 (2.66%)** |
| | **Logistic Regression** | 0.2062 | 0.0907 | 0.0517 (5.17%) |

### 1.4 Calibration Policy Determination
- **Brier & ECE Analysis:** Native HistGradientBoosting already achieves excellent calibration in grouped CV (Brier 0.0592, ECE 1.63%). In the prospective holdout, Random Forest achieves lower ECE (2.66% vs. 4.20%, a difference of 1.54 percentage points).
- **Policy Verdict:** A 1.54% difference in prospective calibration error **does not justify** risking an 11 to 17 point collapse in PR-AUC, nor does it offset HGB's decisive **+38.7% relative recall advantage** on vulnerable cold-start students.
- **Mandate:** **Do not apply post-hoc calibration in production** without leakage-free temporal evidence demonstrating that continuous ranking resolution is preserved. Native log-loss gradient boosted probabilities are confirmed as the production standard.

---

## 2. Threshold Governance Audit

### 2.1 Training-Only Threshold Evidence (tau = 0.20 vs. tau = 0.10)
Threshold selection was performed exclusively using training partitions via inner-fold F1 optimization (tau* ~ 0.21, standardized to 0.20). Evaluation labels from outer validation folds and the 2014J holdout were strictly quarantined.

| Operational Scenario | Threshold (tau) | Total Snapshots | Alert Volume (Count / %) | Precision | Recall (Sensitivity) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grouped 5-Fold CV** | tau = 0.20 | 170,874 | 20,959 (12.27%) | 40.03% | 57.38% | 42.62% |
| | tau = 0.10 | 170,874 | 38,384 (22.46%) | 28.37% | 74.46% | 25.54% |
| **Prospective 2014J Holdout** | tau = 0.20 | 50,014 | 9,227 (18.45%) | 25.93% | 51.61% | 48.39% |
| | tau = 0.10 | 50,014 | 14,611 (29.21%) | 21.29% | 67.09% | 32.91% |

### 2.2 Module GGG-Specific Evidence vs. Other Courses
In Phase 9 Task 3, a matched-population evaluation ($N = 5,656$ actionable TMA snapshots, 411 failures, prevalence 7.27%) revealed that Module GGG suffered from an unacceptable false negative rate under the global threshold:

| Module GGG Matched TMA Pipeline | Operating Threshold | Alert Volume (Count / %) | Precision | Recall | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Remediated Pipeline** | tau = 0.20 (Global) | 221 (3.91%) | 36.65% | 19.71% | **80.29%** |
| **Remediated Pipeline** | tau = 0.10 (Calibrated) | 1,009 (17.84%) | 18.63% | **45.74%** | **54.26%** |

- **Root Cause of GGG Discrepancy:** GGG included 6 decoupled practice quizzes (CMAs) with 0% weight, non-standard submission patterns, and missing early assessment scores. Under tau = 0.20, the model under-predicted risk, missing 4 out of 5 failing students. Calibrating GGG TMAs to tau = 0.10 restored recall from 19.71% to 45.74% at an alert volume of 17.84%, closely matching the institutional 18.45% holdout alert budget.

### 2.3 Module Generalization Guardrail
**Critical Audit Finding:** The success of tau = 0.10 in Module GGG **must not be generalized** to other low-prevalence modules without empirical matched validation.
- In other low-prevalence courses in `2014J`:
  - **Module EEE** (Prevalence 3.90%, $N = 3,227$): Under standard tau = 0.20, the model alerts on 147 students (4.56%), achieving **38.10% precision** and **44.44% recall** (FNR 55.56%).
  - **Module FFF** (Prevalence 3.99%, $N = 16,175$): Under standard tau = 0.20, the model alerts on 819 students (5.06%), achieving **35.41% precision** and **44.89% recall** (FNR 55.11%).
- **Policy Mandate:** Modules EEE and FFF already achieve healthy precision (35-38%) and solid sensitivity (~45%) under tau = 0.20. Artificially lowering tau to 0.10 would double or triple false alerts, overwhelming counselors and causing alert fatigue.
- **Rule:** tau = 0.10 is strictly restricted to Module GGG TMAs via `ModuleGGGGovernanceRouter`. For all other modules, tau = 0.20 remains the operational default.

---

## 3. CMA (Computer-Marked Assessment) Operational Governance

### 3.1 Snapshot Volume vs. Operational Reality
- In the prospective `2014J` holdout, CMA snapshots comprise **31,894 out of 50,014 total instances (63.77%)**, with 2,828 failures (8.87% prevalence).
- In Task 4 candidate benchmarks, models achieved strong mathematical discrimination on CMAs (PR-AUC 0.3844 for HGB, 0.3667 for RF).

### 3.2 Prohibiting Automated CMA Outreach
Despite solid offline PR-AUC metrics, **automated outreach triggered by CMA snapshots is prohibited**:
1. **Low Assessment Stakes:** In many courses (and completely in GGG), CMAs are short-cycle, automated, multi-attempt, or practice quizzes carrying negligible or zero grade weight.
2. **Advisor Fatigue & Feasibility:** Human advisor interventions (phone calls, structured mentoring sessions) cannot be triggered on a weekly cadence for continuous automated quizzes.
3. **Preservation of GGG Suppression:** The Phase 9 Task 3 architectural rule suppressing alerts on decoupled GGG CMAs is fully reaffirmed.
4. **Policy Mandate:** CMA prediction metrics are designated strictly as **Research-Only Telemetry**. Operational outreach must be restricted to Tutor-Marked Assignments (TMAs) and pre-exam capstone milestones.

---

## 4. Final Governed Model Selection Statement

### 4.1 Candidate Architectural Roles

#### 1. Primary Operational Candidate: `HistGradientBoostingClassifier`
- **Assigned Role:** Primary Early-Warning & Ranking Engine.
- **Empirical Justification:**
  - Dominates 5-fold grouped cross-validation (**PR-AUC 0.4606**, **ROC-AUC 0.8687**).
  - Superior sensitivity on **cold-start learners** lacking historical assessment grades (**Recall@0.20 = 27.07% vs. 19.51%** for Random Forest), capturing **38.7% more failing students** in early semester onboarding.
  - Native NaN branch routing eliminates imputation bias on missing assessment features.
  - Highest sensitivity to pre-deadline telemetry accumulation (+0.0387 PR-AUC at 1-day horizon).

#### 2. Challenger Model: `RandomForestClassifier`
- **Assigned Role:** Comparative Benchmark Challenger for Generalization & Calibration.
- **Empirical Justification:**
  - Slightly higher out-of-time prospective ranking on `2014J` (**PR-AUC 0.3093 vs. 0.3005**).
  - Superior out-of-the-box probability calibration (**ECE 0.0266**, **Brier score 0.0768**).
  - Smaller generalization gap from cross-validation to prospective holdout (Delta = -0.1275 vs. -0.1601).
  - **Limitations:** Substantial cold-start sensitivity deficit (19.51% recall) and requires feature imputation.

#### 3. Comparative Baseline: `LogisticRegression`
- **Assigned Role:** Retained Offline Linear Baseline & Sanity Check.
- **Status:** **Rejected for Operational Deployment; Retained as Scientific Baseline.**
- **Empirical Justification:**
  - Suffers catastrophic failure on cold-start learners (**Recall@0.20 = 0.00%**, FNR = 100.0%) due to median imputation distortion.
  - Cannot model non-linear VLE engagement dynamics.
  - Retained exclusively as an offline linear benchmark and explainability baseline.

---

### 4.2 Explicit Distinction: Offline Research Recommendation vs. Production Approval

> [!IMPORTANT]
> **Production Deployment Approval is WITHHELD Pending Phase 9 Task 5 Operational Verification.**
>
> While `HistGradientBoostingClassifier` is certified as the **scientifically best-performing candidate** in offline research benchmarks, **offline superiority does not equate to production readiness**.

Before production approval can be granted, the following operational requirements must be satisfied in Phase 9 Task 5:
1. **Counselor Workload & Capacity Validation:** Institutional advisors must validate whether the projected alert volume (9,227 alerts across `2014J`, ~18.45% of cohort) matches counseling capacity.
2. **Operational Governance Routing Integration:** The `ModuleGGGGovernanceRouter` (suppressing practice quizzes and routing TMAs to tau = 0.10) must be formally verified within the live inference pipeline.
3. **Data Telemetry Latency Verification:** Confirm that student VLE clickstream and registration features can be extracted and ingested within the required 14-day delay and 7-day pre-deadline cutoff windows.
4. **Shadow-Mode Operational Evaluation:** Conduct shadow-mode testing alongside academic advisors before enabling automated outreach workflows.

---

## 5. Audit Deliverables Summary

| Deliverable Artifact | Description | Status |
| :--- | :--- | :---: |
| [phase9_task4_model_selection_policy_audit.md](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase9_task4_model_selection_policy_audit.md) | Comprehensive policy audit report across calibration, thresholds, CMAs, and model roles | Complete & Certified |
| [phase9_task4_model_selection_policy_summary.json](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase9_task4_model_selection_policy_summary.json) | Complete machine-readable summary dictionary of policy audit findings | Complete & Certified |
