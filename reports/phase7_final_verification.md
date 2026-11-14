# Phase 7 Final Verification Audit: Model Explainability & Risk Assessment
## Student Performance & At-Risk Prediction System (OULAD)

- **Audit Date:** 2026-11-14  
- **Auditor:** Automated System Verification Framework  
- **Audit Target:** Phase 7 Implementation, Reports, Figures, and Test Suites  
- **Primary Source Code:** [`src/model_explainability.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/model_explainability.py)  
- **Primary Report:** [`reports/model_explainability_report.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_explainability_report.md)  
- **Summary Metrics Artifact:** [`reports/model_explainability_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_explainability_summary.json)  
- **Test Suite:** [`tests/test_model_explainability.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/tests/test_model_explainability.py)  
- **Overall Project Test Status:** **41 / 41 passing (100%)**  

---

## 1. Audit Scorecard & Summary of Findings

| Category | Verification Item | Verdict | Key Finding / Evidence |
| :--- | :--- | :---: | :--- |
| **1. SHAP Attribution** | Reported values are Mean Absolute SHAP values | **PASS** | Evaluates $I_j = \frac{1}{N} \sum_{i=1}^{N} \|\phi_j(x^{(i)})\|$ via `np.abs(shap_explanation.values).mean(axis=0)`. |
| | Directionality interpretation (positive vs. negative) | **PASS** | Formally mapped: $\phi_j > 0$ increases failure risk log-odds; $\phi_j < 0$ decreases risk (protective factor). |
| | Feature governance compliance (approved features only) | **PASS** | Exactly 23 allowlisted features passed via `FeatureGovernor.get_feature_matrix`. Zero prohibited columns used. |
| | Explainer uses intended optimized HistGradientBoosting | **PASS** | Uses `HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_depth=8, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=5.0)`. |
| **2. Risk Thresholds** | Selected using training data only ($\tau_{\text{low}}=0.05, \tau_{\text{high}}=0.20$) | **PASS** | $\tau_{\text{high}}=0.20$ matches Phase 6 train-tuned F1 threshold (mean 0.206); $\tau_{\text{low}}=0.05$ derived from train calibration curves ($>98\%$ NPV). |
| | Evaluation labels not used for threshold selection | **PASS** | Fixed decision boundaries applied ex-ante to out-of-fold validation probabilities; zero label leakage. |
| | Threshold-selection procedure reproducibility | **PASS** | Fully reproducible via deterministic grid scan (`optimize_threshold_on_train`). |
| **3. Risk-Tier Denominators** | 62.15%, 25.59%, 12.27% cohort share verification | **PASS** | Verified on 170,874 snapshots: Low = 106,196 (62.15%), Mod = 43,719 (25.59%), High = 20,959 (12.27%). |
| | Snapshot-level vs. student-level clarity | **PASS** | Headline 62.15% / 25.59% / 12.27% shares confirmed as **snapshot-level**; student-level shares reported separately for prospective horizons. |
| | Isolation of Assessment 1 and Day 30 benchmarks | **PASS** | Benchmark A (Assessment 1, $N=23,351$) and Benchmark B (Day 30, $N=21,719$) evaluated independently; zero pooling. |
| | Elimination of retrospective peak-risk pooling | **PASS** | Retrospective peak-risk metric ($P_s = \max_a \hat{p}_{s, a}$) is completely rejected and excluded from operational results. |
| **4. Additional Checks** | Local explanation examples reproducibility | **PASS** | Cases Alpha, Beta, Gamma, and Delta reproduce exact probabilities and top SHAP attributions to 4 decimal places. |
| | Sensitive attributes excluded from predictive features | **PASS** | Demographic features (`gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`) verified absent. |
| | Phase 7 and full test suite passing | **PASS** | `tests/test_model_explainability.py` passes 4/4; all 41 test cases across 6 suites pass cleanly in 3.64s. |

---

## 2. In-Depth Verification Details

### 2.1 Area 1: SHAP Attribution & Model Architecture

#### A. Mean Absolute SHAP Verification
- **Code Audit:** In [`src/model_explainability.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/model_explainability.py) (lines 120–124):
  ```python
  mean_abs = np.abs(shap_explanation.values).mean(axis=0)
  importance_dict = {
      feature_names[i]: round(float(mean_abs[i]), 4)
      for i in np.argsort(-mean_abs)
  }
  ```
- **Finding:** The calculated values represent the exact empirical expectation of absolute Shapley values across the evaluation sample:
  $$I_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_j(x^{(i)})|$$
- **Verdict:** **PASS**

#### B. Directionality of Positive and Negative SHAP Values
- **Mathematical Specification:** Binary logistic regression in `HistGradientBoostingClassifier` maps raw tree logit score $f(x)$ through sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  - A positive Shapley value ($\phi_j > 0$) increases $f(x)$, thereby increasing the predicted risk of assessment failure ($\hat{p} = P(y=1|x)$).
  - A negative Shapley value ($\phi_j < 0$) decreases $f(x)$, thereby decreasing the predicted risk of failure (protective factor).
- **Implementation Audit:** In [`src/model_explainability.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/model_explainability.py) (lines 346–356 and 486–495):
  - Drivers: `[f for f in top_pos if vals[i] > 0]` (rendered in red `#e74c3c`).
  - Protective: `[f for f in top_neg if vals[i] < 0]` (rendered in green `#2ecc71`).
  - Axis label: `"SHAP Impact (+ Increases Risk, - Decreases Risk)"`.
- **Verdict:** **PASS**

#### C. Feature Governance Compliance
- **Allowlist Enforcement:** Evaluated against `src.baseline_models.FeatureGovernor`.
- **Audited Feature Set:** Exactly 23 features:
  - 3 Categorical: `code_module`, `code_presentation`, `target_assessment_type`.
  - 20 Numerical: `target_assessment_weight`, `scheduled_date`, `num_previous_scheduled_assessments`, `registration_timing_days`, `vle_total_clicks_to_date`, `vle_active_days_to_date`, `vle_core_clicks_to_date`, `vle_forum_clicks_to_date`, `vle_quiz_clicks_to_date`, `vle_days_since_last_activity`, `vle_clicks_last_7d`, `vle_clicks_last_14d`, `vle_activity_trend_ratio`, `num_prior_submissions`, `prior_submission_lateness_mean`, `num_prior_scored_assessments`, `mean_prior_score`, `std_prior_score`, `prior_at_risk_rate`, `recent_prior_score`.
- **Prohibited Attributes Check:**
  - `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`: **0 present**.
  - `id_student`, `id_assessment`, `final_result`, `date_unregistration`: **0 present**.
- **Verdict:** **PASS**

#### D. Explainer Model Architecture
- **Model Parameters Audit:** In [`src/model_explainability.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/model_explainability.py) (lines 81–90):
  ```python
  clf = HistGradientBoostingClassifier(
      categorical_features=cat_indices,
      learning_rate=0.05,
      max_iter=150,
      max_depth=8,
      max_leaf_nodes=63,
      min_samples_leaf=50,
      l2_regularization=5.0,
      random_state=random_state
  )
  ```
- **Comparison with Phase 6 Selection:** In [`reports/model_optimization_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_optimization_summary.json), the winning architecture across all 5 folds was `deep_capacity_regularized` with identical hyperparameters.
- **Explainer Type:** `shap.TreeExplainer(clf)` directly ingests this trained tree ensemble.
- **Verdict:** **PASS**

---

### 2.2 Area 2: Risk Thresholds & Calibration Governance

#### A. Threshold Derivation on Training Data Only
- **Audit Findings:**
  1. **High Risk Threshold ($\tau_{\text{high}} = 0.20$):**
     - In Phase 6 (`src/model_optimization.py`, line 399), `optimize_threshold_on_train` scanned candidate thresholds on outer training folds to maximize F1-score.
     - The fold-specific optimal thresholds were: Fold 0 = 0.21, Fold 1 = 0.20, Fold 2 = 0.21, Fold 3 = 0.21, Fold 4 = 0.20 (mean = 0.206).
     - $\tau_{\text{high}} = 0.20$ directly reflects this training-derived optimal decision boundary.
  2. **Low Risk Threshold ($\tau_{\text{low}} = 0.05$):**
     - Derived from training fold probability calibrations (evaluated in `reports/figures/optimization/calibration_curves.png`).
     - Predictions below $\hat{p} = 0.05$ on training folds corresponded to an empirical failure rate of $< 2\%$, delivering a Negative Predictive Value (NPV) exceeding $98\%$.
- **Verdict:** **PASS**

#### B. Absence of Evaluation Label Leakage
- **Audit Findings:**
  - Neither $\tau_{\text{low}}$ nor $\tau_{\text{high}}$ was tuned against validation fold ground-truth labels.
  - The thresholds were fixed ex-ante. The evaluation function `evaluate_advisory_risk_tiers(probs, y_true, tau_high=0.20, tau_low=0.05)` simply bins out-of-fold predictions into the pre-specified tiers to evaluate prospective performance.
- **Verdict:** **PASS**

#### C. Reproducibility of Threshold Optimization
- **Audit Findings:**
  - `optimize_threshold_on_train` implements a deterministic grid scan over `np.linspace(0.05, 0.95, 91)`. Given fixed training inputs, it deterministically outputs the identical threshold.
- **Verdict:** **PASS**

---

### 2.3 Area 3: Risk-Tier Denominators & Prospective Horizon Isolation

#### A. Mathematical Verification of Cohort Shares
In [`reports/model_explainability_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_explainability_summary.json) and [`reports/model_explainability_report.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_explainability_report.md):

- **Dataset:** `data/processed/oof_predictions_optimized_14day_delay.csv` (`cutoff_scenario == '7_day_before'`).
- **Total Validated Records ($N$):** **170,874**
- **Low Risk ($\hat{p} < 0.05$):**
  - Count: **106,196**
  - Share: $106,196 / 170,874 = 0.621487 \rightarrow$ **62.15%**
- **Moderate Risk ($0.05 \le \hat{p} < 0.20$):**
  - Count: **43,719**
  - Share: $43,719 / 170,874 = 0.255855 \rightarrow$ **25.59%**
- **High Risk ($\hat{p} \ge 0.20$):**
  - Count: **20,959**
  - Share: $20,959 / 170,874 = 0.122658 \rightarrow$ **12.27%**
- **Sum of Shares:** $62.15\% + 25.59\% + 12.27\% = 100.01\%$ (exact 100.0% unrounded).
- **Verdict:** **PASS**

#### B. Clarification of Denominators: Snapshot-Level vs. Student-Level
- **Verification Question:** Are 62.15%, 25.59%, and 12.27% snapshot-level or student-level statistics?
- **Finding:**
  - **They are strictly SNAPSHOT-LEVEL statistics.** The denominator is $N = 170,874$ prediction events across all assessment opportunities in the course.
  - At the unique student level, each student has multiple assessment opportunities across time.
  - When evaluated at true prospective single-decision student horizons, the distributions are:
    - **Prospective Benchmark A (Assessment 1 Milestone, $N = 23,351$ unique students):**
      - Low Risk: 9,976 (**42.72%**)
      - Moderate Risk: 9,360 (**40.08%**)
      - High Risk: 4,015 (**17.19%**)
    - **Prospective Benchmark B (Day 30 Calendar Cutoff, $N = 21,719$ active unique students):**
      - Low Risk: 11,806 (**54.36%**)
      - Moderate Risk: 8,259 (**38.03%**)
      - High Risk: 1,654 (**7.62%**)
- **Verdict:** **PASS** (Confirmed that reports explicitly document this distinction in Section 4.1).

#### C. Separation of Prospective Milestones (Assessment 1 vs. Day 30)
- **Finding:**
  - Assessment 1 milestone (`risk_tier_evaluation_alternative_A_first_assessment`) and Day 30 campaign (`risk_tier_evaluation_alternative_B_day30_cutoff`) are evaluated independently in code, JSON, report tables, and figures.
  - Zero pooling between the two benchmarks was detected.
- **Verdict:** **PASS**

#### D. Rejection of Retrospective Peak-Risk Pooling
- **Finding:**
  - Retrospective peak-risk pooling ($P_s = \max_a \hat{p}_{s, a}$) was formally audited in Phase 6 and confirmed to suffer from multi-opportunity label inflation (33.49% base rate).
  - In Phase 7, $P_s = \max_a \hat{p}_{s, a}$ is not used for any advisory risk tier calculations or prospective operational recommendations.
- **Verdict:** **PASS**

---

### 2.4 Area 4: Local Explainability & Test Suite Verification

#### A. Reproducibility of Local Case Studies
The 4 local cases were independently recomputed from the prediction dataset:
1. **Case Alpha (True Positive):** Module `AAA`, Assessment `TMA` at Day 117. Predicted $\hat{p} = 0.7009$, actual $y = 1$. Top risk driver: `recent_prior_score` = 34.0 (SHAP $+1.7888$).
2. **Case Beta (False Positive):** Module `AAA`, Assessment `TMA` at Day 54. Predicted $\hat{p} = 0.5344$, actual $y = 0$. Top risk driver: `recent_prior_score` = 48.0 (SHAP $+1.3625$).
3. **Case Gamma (True Negative):** Module `AAA`, Assessment `TMA` at Day 19. Predicted $\hat{p} = 0.0144$, actual $y = 0$. Top protective factor: `registration_timing_days` = -53.0 (SHAP $-0.4515$).
4. **Case Delta (False Negative):** Module `AAA`, Assessment `TMA` at Day 19. Predicted $\hat{p} = 0.0393$, actual $y = 1$. Top protective factor: `registration_timing_days` = -17.0 (SHAP $-0.3580$).
- **Result:** Exact numerical match with [`reports/model_explainability_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_explainability_summary.json) across all probabilities and SHAP values.
- **Privacy Check:** Raw `id_student` identifiers are masked in all reported narratives.
- **Verdict:** **PASS**

#### B. Sensitive Demographics Exclusion
- Explicit programmatic check verified that `gender`, `region`, `highest_education`, `imd_band`, `age_band`, and `disability` are completely absent from model training, SHAP feature rankings, and summary artifacts.
- **Verdict:** **PASS**

#### C. Automated Test Suite Execution
- **Phase 7 Test Suite:**
  ```bash
  .venv/bin/pytest tests/test_model_explainability.py -v
  tests/test_model_explainability.py::test_risk_tier_partition_and_monotonicity PASSED
  tests/test_model_explainability.py::test_explainability_summary_json_structure PASSED
  tests/test_model_explainability.py::test_local_case_study_anonymity_and_structure PASSED
  tests/test_model_explainability.py::test_explainability_figures_exist PASSED
  4 passed in 1.82s
  ```
- **Full Project Test Suite:**
  ```bash
  .venv/bin/pytest tests/
  ============================== 41 passed in 3.64s ==============================
  ```
- **Verdict:** **PASS**

---

## 3. Final Conclusion & Recommendation

All 13 specific verification criteria across the three audit areas have achieved a rating of **PASS**.

Phase 7 is technically rigorous, leakage-free, fully reproducible, and aligned with educational data ethics and governance standards.

**Phase 7 is certified for final approval.**

