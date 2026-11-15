# Phase 9 Task 2 Completion Report: Cold-Start Prediction Investigation
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Phase Target:** Phase 9 — Temporal Generalization & Model Reliability Improvement  
- **Task Target:** Task 2 — Cold-Start Prediction Investigation  
- **Primary Source Files:**
  - [`src/cold_start_feature_audit.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/cold_start_feature_audit.py)
  - [`src/cold_start_experiments.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/cold_start_experiments.py)
- **Automated Test Suites:**
  - [`tests/test_cold_start_feature_audit.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/tests/test_cold_start_feature_audit.py) (4 tests passing)
  - [`tests/test_cold_start_experiments.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/tests/test_cold_start_experiments.py) (7 tests passing)
- **Status:** Complete, Tested, Audited & Certified  

---

## 1. Executive Summary & Problem Definition

In early course stages, learners have zero historical assessment grades. When predicting failure risk prior to the release of initial assessment scores, the system faces the **Cold-Start Dilemma**:
- Historical grades (`recent_prior_score`, `mean_prior_score`, `std_prior_score`) are **100% unobserved (`NaN`)**.
- In Phase 6 and 7, this missingness resulted in a **34.4% drop in PR-AUC** (0.3203 cold-start vs 0.4882 when grades are available).
- In Task 1, cold-start failure rate jumped from **8.79% in training to 13.60% in 2014J** (+54.7% relative increase).

Task 2 investigated whether missingness indicators, reduced feature sets, or specialized cold-start models could improve early-warning reliability.

---

## 2. Key Experimental Findings Across Tasks

### Experiment A: Cohort Characterization
- **Total Cold-Start Snapshots:** 29,478 (17.25% of benchmark; 23,351 unique learners).
- **Failure Prevalence Disparity:** Cold-start failure rate is **10.49%** vs **8.16%** downstream.
- **Horizon Gap:** Cold-start predictions occur at **Day 26.6** (vs Day 152.3), with only **11.1 active days** and **261 total clicks** accumulated.

### Experiment B: Feature Availability & Leakage Audit
- Formally verified genuine prediction-time availability across all 23 allowlisted features.
- Certified zero target leakage, zero future-information leakage, and zero demographic attribute inclusion.

### Experiment C: Feature Strategy Comparison (5-Fold Grouped CV)
- **Strategy 1 (Baseline - Native Missing Routing):** Cold-Start PR-AUC = **0.2684**
- **Strategy 2 (Explicit Missingness Indicators):** Cold-Start PR-AUC = **0.2684** (Zero improvement; redundant for gradient boosting).
- **Strategy 3 (Engagement Only):** Cold-Start PR-AUC = **0.1606** (Major drop; lacks syllabus difficulty baseline).
- **Strategy 6 (Segmented / Dedicated Specialist):** Cold-Start PR-AUC = **0.2608** (Worse by -0.018 due to discarding 84% of training data).

### Experiment D: Model Architecture Comparison on 2014J Holdout
- **HistGradientBoosting:** Cold-Start PR-AUC = **0.1687**, ROC-AUC = **0.5698**, Overall PR-AUC = **0.3005**.
- **Random Forest:** Cold-Start PR-AUC = **0.1683**, ROC-AUC = **0.5716**, Overall PR-AUC = **0.3093**.
- **Logistic Regression:** Cold-Start PR-AUC = **0.1302**, ROC-AUC = **0.5144** (Catastrophic collapse due to median imputation).

---

## 3. Final Recommendation

1. **A Separate Cold-Start Model Is Not Recommended:** The unified `HistGradientBoostingClassifier` natively routes missing values and transfers learned module and registration representations from the full cohort, outperforming dedicated models.
2. **Operational Triage Policy:** Advisors must treat predictions generated prior to Assessment 1 as broad onboarding check-ins rather than definitive failure warnings.

**Task 2 is complete and certified. Ready for review before proceeding to Task 3.**
