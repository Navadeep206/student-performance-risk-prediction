# Phase 6: Model Optimization, Generalization & Robustness Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Dataset:** Open University Learning Analytics Dataset (OULAD)  
- **Pipeline Implementation:** [`src/model_optimization.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/model_optimization.py)  
- **Automated Test Suite:** [`tests/test_model_optimization.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/tests/test_model_optimization.py)  
- **Summary Metrics:** [`reports/model_optimization_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/model_optimization_summary.json)  
- **Out-of-Fold Optimized Predictions:** `data/processed/oof_predictions_optimized_14day_delay.csv` (341,748 rows, 52 MB)  
- **Chronological Evaluation Predictions:** `data/processed/oof_predictions_chronological_2014J.csv` (50,014 rows, 7.8 MB)  
- **Evaluation Diagnostic Plots:** [`reports/figures/optimization/`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/figures/optimization/)  
- **Status:** Complete, Audited & Verified (All 33 automated tests passing across Phases 3, 4, 5, and 6)  

---

## 1. Executive Summary

Phase 6 performed systematic hyperparameter optimization, probability calibration analysis, student-level intervention modeling, and temporal generalization audits for the prospective at-risk prediction system:
$$\mathbf{Target:}\quad \text{at\_risk} = \begin{cases} 1 & \text{if } \text{next\_assessment\_score} < 50 \\ 0 & \text{if } \text{next\_assessment\_score} \ge 50 \end{cases}$$

### Key Engineering & Empirical Findings:
1. **Nested Cross-Validation Optimization:** Implemented strict nested cross-validation (outer 5-fold `StratifiedGroupKFold` grouped by student, inner 3-fold grouped cross-validation on training partitions only). Selected a regularized, deep-tree gradient boosting architecture (`learning_rate=0.05, max_iter=150, max_depth=8, max_leaf_nodes=63, min_samples_leaf=50, l2_reg=5.0`), improving PR-AUC to **0.4611** (ROC-AUC **0.8687**, Brier Score **0.0592**).
2. **Snapshot-Level vs Unique Student-Level Intervention Capture:**
   - **Snapshot-Level:** Within a 20% outreach budget, the model captures **71.26% of all failing assessment instances**.
   - **Student-Level (Unique Learners):** Aggregating peak predicted risk across upcoming assessments for the 23,351 unique learners, a 20% advisor outreach budget captures **48.38% of all students who experience academic failure** ($2.42\times$ more effective than random outreach).
3. **Temporal Generalization Gap (Chronological Split):**
   - Evaluated strict temporal forecasting: Training on earlier intakes (`2013B`, `2013J`, `2014B`, $N=120,860$) and projecting forward onto the completely unseen future intake (`2014J`, $N=50,014$).
   - Across the full `2014J` intake, the model maintains strong discriminative capacity (**ROC-AUC 0.7969**, Brier score **0.0812**, capturing **54.24% of failing students in the top 20% budget**), though semester drift reduces PR-AUC from 0.4611 (grouped CV) to **0.3005** (chronological test).
4. **Probability Calibration:** Native log-loss probabilities from HistGradientBoosting achieve exceptional calibration (**Brier score 0.0592**, Expected Calibration Error **1.63%**). Empirical audits showed that post-hoc step calibration (`CalibratedClassifierCV`) damages PR-AUC by quantizing predictions into coarse step bins.
5. **Zero Data Leakage:** All 33 automated tests passing in 4.65 seconds. Outer validation partitions were strictly isolated from hyperparameter tuning and threshold selection.

---

## 2. Phase 5 Inspection & Verification Audit

Prior to Phase 6 optimization, the Phase 5 implementation and outputs were independently audited:

| Audit Item | Phase 5 Implementation | Verification Result | Governance Status |
| :--- | :--- | :---: | :---: |
| **Threshold Selection** | Selected via `optimize_threshold_on_train` using only training fold labels ($\tau^* \approx 0.21$). | Verified | Leak-free |
| **Outer Validation Labels** | Never used for model fitting or parameter tuning. | Verified | Leak-free |
| **Preprocessing Fitting** | All ColumnTransformers fit exclusively on training slices. | Verified | Leak-free |
| **Cutoff Scenario Isolation** | 7-day and 1-day horizons evaluated separately without pooling. | Verified | Methodologically sound |
| **Recall@k Unit Formulation** | In Phase 5, Recall@k was evaluated on individual snapshot records ($N=170,874$). | Clarified | Snapshot-level; student-level introduced in Phase 6 |

---

## 3. Nested Hyperparameter Optimization

To avoid optimistic bias, hyperparameters were selected using nested cross-validation:

```mermaid
flowchart TD
    subgraph Nested Cross-Validation Workflow
        O_Train["Outer Training Fold (80% Students)"] --> Inner_CV["Inner 3-Fold Grouped CV (StratifiedGroupKFold on Students)"]
        Inner_CV --> Eval_Grid["Evaluate 5 Principled Regularization Candidates"]
        Eval_Grid --> Best_Param["Select Winning Candidate (Max Inner PR-AUC)"]
        Best_Param --> Train_Full["Train Best Model on Full Outer Training Fold"]
        Train_Full --> O_Val["Evaluate on Held-Out Outer Validation Fold (20% Students)"]
    end
```

### 3.1 Candidate Hyperparameter Search Space

| Candidate Name | `learning_rate` | `max_iter` | `max_depth` | `max_leaf_nodes` | `min_samples_leaf` | `l2_regularization` | Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `baseline_params` | 0.10 | 100 | 6 | 31 | 20 | 0.0 | Phase 5 default baseline. |
| `conservative_regularized` | 0.05 | 120 | 6 | 31 | 50 | 2.0 | Slower shrinkage with leaf constraints. |
| **`deep_capacity_regularized`** | **0.05** | **150** | **8** | **63** | **50** | **5.0** | **Deep interaction capacity restrained by L2 penalty.** |
| `shallow_high_reg` | 0.05 | 100 | 4 | 15 | 100 | 10.0 | High regularization for minimal variance. |
| `balanced_momentum` | 0.08 | 130 | 7 | 45 | 35 | 3.0 | Intermediate balance between speed and depth. |

### 3.2 Optimization Results Across Scenarios (5-Fold Grouped CV)

*All models evaluated using out-of-fold outer validation.*

| Evaluation Scenario | Model Architecture | PR-AUC (AP) | ROC-AUC | Brier Score | ECE | Snapshot Recall@10% | Snapshot Recall@20% | Optimal Train Threshold ($\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **14d Delay / 7d Cutoff** *(Primary)* | Baseline HistGradientBoosting | 0.4564 ± 0.0081 | 0.8668 | 0.0594 | 0.0175 | 0.5107 | 0.7077 | 0.21 ± 0.01 |
| | **Optimized HistGradientBoosting** | **0.4611 ± 0.0084** | **0.8687** | **0.0592** | **0.0163** | **0.5139** | **0.7126** | **0.21 ± 0.01** |
| | Logistic Regression (L2) | 0.3234 ± 0.0068 | 0.8009 | 0.0712 | 0.0410 | 0.4128 | 0.5957 | 0.16 ± 0.01 |
| | Random Forest (max_depth=8) | 0.4378 ± 0.0073 | 0.8541 | 0.0628 | 0.0245 | 0.4922 | 0.6907 | 0.14 ± 0.01 |
| | Previous-Score Heuristic | 0.2058 ± 0.0047 | 0.6659 | 0.1124 | 0.0682 | 0.3405 | 0.4976 | 0.50 (fixed) |
| **14d Delay / 1d Cutoff** | Baseline HistGradientBoosting | 0.4632 ± 0.0068 | 0.8697 | 0.0591 | 0.0172 | 0.5144 | 0.7150 | 0.21 ± 0.01 |
| | **Optimized HistGradientBoosting** | **0.4669 ± 0.0072** | **0.8720** | **0.0588** | **0.0159** | **0.5172** | **0.7187** | **0.21 ± 0.01** |
| **7d Delay / 7d Cutoff** | Baseline HistGradientBoosting | 0.4578 ± 0.0060 | 0.8673 | 0.0593 | 0.0174 | 0.5120 | 0.7115 | 0.21 ± 0.01 |
| | **Optimized HistGradientBoosting** | **0.4624 ± 0.0065** | **0.8691** | **0.0591** | **0.0161** | **0.5161** | **0.7127** | **0.21 ± 0.01** |
| **7d Delay / 1d Cutoff** | Baseline HistGradientBoosting | 0.4664 ± 0.0062 | 0.8719 | 0.0589 | 0.0169 | 0.5184 | 0.7213 | 0.21 ± 0.01 |
| | **Optimized HistGradientBoosting** | **0.4694 ± 0.0064** | **0.8736** | **0.0586** | **0.0157** | **0.5215** | **0.7209** | **0.21 ± 0.01** |

*Winning Architecture: The `deep_capacity_regularized` configuration was selected by inner CV across 19 out of 20 folds, confirming that allowing deeper interaction trees ($depth=8$) while applying stronger L2 shrinkage ($l2\_reg=5.0$) provides the best generalization balance.*

---

## 4. Probability Calibration Analysis

In educational decision support, predicted probabilities must represent true empirical failure likelihoods so that counselors can prioritize outreach based on risk tiers (e.g., $P > 0.50$ = acute danger; $0.20 < P \le 0.50$ = emerging concern).

### 4.1 Calibration Metrics

- **Brier Score Loss:** **0.0592** (Superior to Random Forest at 0.0628, Logistic Regression at 0.0712, and Heuristic at 0.1124).
- **Expected Calibration Error (ECE):** **0.0163** (1.63% mean calibration error across 10 probability bins).
- **Reliability Diagram Inspection (`calibration_curves.png`):** The calibration curve for Optimized HistGradientBoosting closely hugs the diagonal line across all confidence bins from 0.0 to 0.85, confirming reliable posterior estimates directly from log-loss.

### 4.2 Why Post-Hoc Step Calibration Was Rejected
- `CalibratedClassifierCV(method='isotonic')` and `sigmoid` were tested strictly on training partitions.
- Due to the 8.56% class imbalance, isotonic regression collapses probabilities into coarse step bins with heavy ties.
- This collapsed ranking resolution and **severely degraded PR-AUC from 0.4669 down to 0.3538 (isotonic) and 0.2915 (sigmoid)**.
- **Architectural Policy:** Native log-loss gradient boosting probabilities are preserved as the production standard.

---

## 5. Intervention-Level Evaluation: Snapshot vs. Unique Student Recall

In operational deployments, advisors contact **individual learners**, not abstract snapshot records. Phase 6 formalizes this distinction.

### 5.1 Definitions
- **Snapshot-Level:** Unit is an individual assessment submission snapshot ($N=170,874$). Ground truth is whether that specific assessment score was $< 50$.
- **Student-Level (Unique Learner):** Unit is a unique enrolled student ($N=23,351$).
  - Ground truth: $Y_s = 1$ if the student fails *at least one* upcoming assessment during the window ($N=7,821$ students, 33.49% base rate).
  - Risk aggregation: Evaluated under **Max-Risk Rule** ($P_s = \max_a \hat{p}_{s, a}$) and **Mean-Risk Rule** ($P_s = \frac{1}{n_s} \sum_a \hat{p}_{s, a}$).

### 5.2 Comparative Intervention Capture Table (14-Day Delay, 7-Day Horizon)

| Intervention Budget Capacity | Snapshot-Level Recall@k | Student-Level Recall@k (Max-Risk Rule) | Student-Level Precision@k (Max-Risk Rule) | Student-Level Recall@k (Mean-Risk Rule) | Random Outreach Student Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 5% Budget** (1,168 students) | 32.65% | **14.88%** (1,164 failing students caught) | 99.66% | 13.23% | 5.00% |
| **Top 10% Budget** (2,335 students) | 51.39% | **28.74%** (2,248 failing students caught) | 96.27% | 25.92% | 10.00% |
| **Top 15% Budget** (3,503 students) | 63.04% | **39.55%** (3,093 failing students caught) | 88.30% | 36.91% | 15.00% |
| **Top 20% Budget** (4,670 students) | **71.26%** | **48.38%** (3,784 failing students caught) | 80.99% | 46.57% | 20.00% |

#### Practical Institutional Takeaways:
1. **Precision at Realistic Capacity:** In a top 10% intervention campaign (2,335 students contacted), **96.27% of contacted students actually experience academic failure**. Advisor time is virtually never wasted on non-struggling students.
2. **Capture Lift:** Reaching out to the top 20% of high-risk students captures **almost half (48.38%) of all failing students across the entire institution**, outperforming unguided outreach by **$2.42\times$**.
3. **Aggregation Superiority:** Peak-risk (`Max`) aggregation consistently outperforms `Mean` aggregation by +2.0% to +3.0% recall, because student disengagement often manifests initially on a single critical assessment.

---

## 6. Temporal Generalization: Chronological Evaluation on Presentation 2014J

To evaluate real-world forward-looking deployment, a strict temporal split was audited:
- **Historical Training Intakes:** `2013B`, `2013J`, `2014B` (120,860 snapshots; 14,744 unique students; 8.26% at-risk rate).
- **Target Deployment Intake:** `2014J` (50,014 snapshots; 8,607 unique students; 9.27% at-risk rate).

```text
Chronological Evaluation Architecture:
├── Historical Training Set (2013B, 2013J, 2014B):  120,860 records
└── Unobserved Target Cohort (2014J):                 50,014 records
    ├── Continuing Students (Previously Enrolled):     1,211 students (6,316 records)
    └── Strictly Novel Students (Zero Overlap):        7,396 students (43,698 records)
```

### 6.1 Performance Comparison: Grouped CV vs. Future Chronological Cohort

| Evaluation Design | Test Cohort | Evaluation Records ($N$) | PR-AUC (AP) | ROC-AUC | Brier Score | Recall@10% | Recall@20% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5-Fold Grouped CV** | In-distribution (all presentations) | 170,874 | **0.4611** | **0.8687** | **0.0592** | **0.5139** | **0.7126** |
| **Chronological Split** | **All 2014J Students** | 50,014 | **0.3005** | **0.7969** | **0.0812** | **0.3412** | **0.5424** |
| **Chronological Split** | **Strictly Novel 2014J Students** (0 overlap) | 43,698 | **0.2793** | **0.7926** | **0.0776** | **0.3371** | **0.5317** |

### 6.2 Analysis of the Generalization Gap
- **Why Does PR-AUC Drop from 0.4611 to 0.3005?**
  In grouped cross-validation, folds contain training instances from all presentations, allowing models to learn presentation-specific norms. When projecting forward onto `2014J`, the model encounters **inter-annual concept drift**: updated module curricula, modified assessment weights, and changing course platform structures.
- **Operational Viability Maintained:**
  Despite the distribution shift, discrimination remains solid (**ROC-AUC 0.7969**). In a 20% outreach budget, the model still captures **54.24% of all failing instances in the novel presentation** ($2.7\times$ better than random).
- **Governance Mandate:** Cross-validation metrics alone must never be used to claim production readiness. Annual model recalibration (fine-tuning on the most recent completed intake) is mandatory before deploying to a new academic year.

---

## 7. Subgroup Robustness Audits

### 7.1 Performance by Course Module (14-Day Delay, 7-Day Horizon)

| Module Code | Total Snapshots ($N$) | Observed At-Risk ($N$) | At-Risk Rate (%) | Optimized HistGradientBoosting PR-AUC | Recall@10% | Recall@20% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAA** | 3,146 | 237 | 7.53% | 0.3112 | 0.4852 | 0.6287 |
| **BBB** | 42,979 | 3,166 | 7.37% | 0.3854 | 0.5164 | 0.6800 |
| **CCC** | 17,014 | 2,590 | 15.22% | **0.5182** | 0.3703 | 0.5849 |
| **DDD** | 29,865 | 4,730 | 15.84% | **0.5641** | 0.4042 | 0.6190 |
| **EEE** | 7,886 | 352 | 4.46% | 0.3521 | 0.5909 | **0.7898** |
| **FFF** | 54,769 | 2,636 | 4.81% | 0.3842 | 0.5964 | **0.7610** |
| **GGG** | 15,215 | 912 | 5.99% | 0.2014 | 0.3629 | 0.5471 |

- **STEM Modules with Clear Progressive Gradients (CCC & DDD):** High PR-AUCs (0.5182 and 0.5641). Formative programming exercises and mathematics homework provide strong, early warning indicators.
- **Low-Failure Modules with High Intervention Efficiency (EEE & FFF):** Reaching the top 20% of flagged students captures **over 76%–79% of all failing students**.

### 7.2 Performance by Assessment Type

| Assessment Type | Total Snapshots ($N$) | Observed At-Risk ($N$) | At-Risk Rate (%) | Optimized HistGradientBoosting PR-AUC | Recall@10% | Recall@20% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CMA** (Computer-Marked) | 70,527 | 3,998 | 5.67% | 0.3768 | **0.5583** | **0.7534** |
| **TMA** (Tutor-Marked) | 98,253 | 10,102 | 10.28% | **0.4791** | 0.4811 | 0.6842 |
| **Exam** (Final Summative) | 2,094 | 523 | 24.98% | **0.6452** | 0.3117 | 0.5124 |

---

## 8. Diagnostic Visualizations Catalog

Generated plots in [`reports/figures/optimization/`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/figures/optimization/):

1. **Probability Calibration Reliability Diagram (`calibration_curves.png`):**  
   Shows that native log-loss HistGradientBoosting aligns with the diagonal line across all confidence tiers (Brier score 0.0592), outperforming Logistic Regression (0.0712) and Random Forest (0.0628).
2. **Precision-Recall Curves (`pr_curves_optimized.png`):**  
   Demonstrates the performance lift of the nested-tuned model (AP = 0.4611) over baseline HGB (0.4564), Random Forest (0.4378), Logistic Regression (0.3234), and Previous-Score Heuristic (0.2058).
3. **Generalization Gap (`chronological_vs_cv.png`):**  
   Contrasts 5-fold grouped CV against true prospective deployment on `2014J`, highlighting the drop from 0.4611 to 0.3005 due to inter-semester concept drift.
4. **Intervention Efficiency (`intervention_recall_comparison.png`):**  
   Visualizes the difference between snapshot-level recall (71.26% at 20%) and unique student-level recall (48.38% at 20%).

---

## 9. Final Model Selection & Operational Specifications

| Selection Criterion | Winner: `HistGradientBoostingClassifier` (`deep_capacity_regularized`) | Justification |
| :--- | :--- | :--- |
| **Predictive Ranking (PR-AUC)** | **0.4611** (14d delay, 7d cutoff) | Outperforms Logistic Regression (0.3234) and Random Forest (0.4378) by wide margins. |
| **Intervention Efficiency** | **71.26% Snapshot Recall@20%** / **48.38% Student Recall@20%** | Captures nearly half of all failing students in a practical 20% counselor workload budget. |
| **Probability Calibration** | **Brier Score 0.0592**, **ECE 1.63%** | Reliable empirical risk probabilities directly out of log-loss optimization. |
| **Missing Score Robustness** | Native tree routing for `NaN` | Seamlessly predicts risk on Assessment 1 where no historical scores exist. |
| **Decision Threshold Policy** | $\tau^* = 0.21$ (Train-tuned F1 threshold) | Yields balanced precision (38.5%) and recall (55.9%) under 8.56% positive prevalence. |

---

## 10. Automated Testing Verification (33/33 Passing)

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.14, pytest-9.2.0, pluggy-1.7.0 -- .venv/bin/python
rootdir: /Users/navadeepguduru/Student At Risk Prediction, configfile: pytest.ini
collected 33 items

tests/test_baseline_models.py::test_feature_governance_and_allowlist PASSED         [  3%]
tests/test_baseline_models.py::test_student_level_group_separation PASSED          [  6%]
tests/test_baseline_models.py::test_train_only_preprocessing_isolation PASSED     [  9%]
tests/test_baseline_models.py::test_deterministic_random_baseline PASSED          [ 12%]
tests/test_baseline_models.py::test_previous_score_heuristic_and_coverage PASSED  [ 15%]
tests/test_baseline_models.py::test_recall_at_k_exact_and_tie_apportionment PASSED [ 18%]
tests/test_baseline_models.py::test_train_tuned_threshold_optimization PASSED     [ 21%]
tests/test_baseline_models.py::test_standard_metrics_dictionary_validity PASSED   [ 24%]
tests/test_baseline_models.py::test_oof_predictions_exist_and_complete PASSED     [ 27%]
tests/test_baseline_models.py::test_scenario_isolation_in_summary_json PASSED     [ 30%]
tests/test_data_preprocessing.py::test_raw_file_integrity PASSED                  [ 33%]
tests/test_data_preprocessing.py::test_processed_files_exist PASSED               [ 36%]
tests/test_data_preprocessing.py::test_enrollment_composite_key_uniqueness PASSED  [ 39%]
tests/test_data_preprocessing.py::test_enrollment_count_preservation PASSED        [ 42%]
tests/test_data_preprocessing.py::test_preservation_of_zero_activity_enrollments PASSED [ 45%]
tests/test_data_preprocessing.py::test_foreign_key_referential_integrity PASSED  [ 48%]
tests/test_data_preprocessing.py::test_score_and_click_validity PASSED            [ 51%]
tests/test_data_preprocessing.py::test_missing_value_handling PASSED              [ 54%]
tests/test_data_preprocessing.py::test_reproducibility_of_aggregation PASSED      [ 57%]
tests/test_model_optimization.py::test_ece_calculation_exactness PASSED           [ 60%]
tests/test_model_optimization.py::test_student_level_aggregation_metrics PASSED   [ 63%]
tests/test_model_optimization.py::test_candidate_search_space_validity PASSED     [ 66%]
tests/test_model_optimization.py::test_chronological_split_integrity PASSED       [ 69%]
tests/test_model_optimization.py::test_phase6_artifacts_exist_after_run PASSED     [ 72%]
tests/test_temporal_feature_engineering.py::test_datasets_exist_and_non_empty PASSED [ 75%]
tests/test_temporal_feature_engineering.py::test_no_prohibited_features PASSED    [ 78%]
tests/test_temporal_feature_engineering.py::test_target_label_accuracy PASSED     [ 81%]
tests/test_temporal_feature_engineering.py::test_cutoff_calculation PASSED        [ 84%]
tests/test_temporal_feature_engineering.py::test_no_target_leakage_on_first_assessment PASSED [ 87%]
tests/test_temporal_feature_engineering.py::test_temporal_delay_sensitivity PASSED [ 90%]
tests/test_temporal_feature_engineering.py::test_vle_temporal_monotonicity PASSED  [ 93%]
tests/test_temporal_feature_engineering.py::test_valid_referential_integrity PASSED [ 96%]
tests/test_temporal_feature_engineering.py::test_class_imbalance_bounds PASSED   [100%]

============================== 33 passed in 4.65s ==============================
```

---

## 11. Unresolved Limitations & Governance Recommendations

1. **Inter-Annual Concept Drift:** The transition from grouped cross-validation (PR-AUC 0.4611) to chronological deployment (PR-AUC 0.3005) demonstrates that course curricula and assessment difficulty change across years. Institutional systems must implement **annual model retraining and drift monitoring**.
2. **First-Assessment Cold Start:** While VLE clickstream momentum successfully predicts risk on Assessment 1 where no previous grades exist, predictive confidence increases significantly once the first graded assignment is returned. Advisors should be informed when a prediction is based solely on clickstream data vs historical assessment grades.
3. **Non-Causal Interpretation:** Features such as `vle_core_clicks_to_date` and `registration_timing_days` are observational predictors, not causal levers. Interventions must address pedagogical understanding rather than merely encouraging students to click arbitrarily.

