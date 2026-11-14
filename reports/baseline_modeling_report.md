# Phase 5: Baseline Model Development & Cross-Validation Strategy Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Dataset:** Open University Learning Analytics Dataset (OULAD)  
- **Pipeline Implementation:** [`src/baseline_models.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/baseline_models.py)  
- **Automated Test Suite:** [`tests/test_baseline_models.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/tests/test_baseline_models.py)  
- **Summary Metrics:** [`reports/baseline_model_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/baseline_model_summary.json)  
- **Out-of-Fold Predictions:**  
  - Baseline (14-day delay): `data/processed/oof_predictions_14day_delay.csv` (341,748 rows, 52 MB)  
  - Alternative (7-day delay): `data/processed/oof_predictions_7day_delay.csv` (341,748 rows, 52 MB)  
- **Evaluation Plots:** [`reports/figures/baseline/`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/figures/baseline/)  
- **Status:** Complete, Audited & Verified (All 28 automated tests passing across Phases 3, 4, and 5)  

---

## 1. Executive Summary & Verification of Prior Execution

Phase 5 established the machine learning benchmarking framework for the prospective task:
$$\mathbf{Target:}\quad \text{at\_risk} = \begin{cases} 1 & \text{if } \text{next\_assessment\_score} < 50 \\ 0 & \text{if } \text{next\_assessment\_score} \ge 50 \end{cases}$$

### Key Execution Verifications:
1. **Execution Status:** The cross-validation pipeline was fully executed via [`src/baseline_models.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/baseline_models.py) in **168.68 seconds**.
2. **Scenario Completion:** All **4 evaluation scenarios** (14-day delay @ 7-day cutoff, 14-day delay @ 1-day cutoff, 7-day delay @ 7-day cutoff, and 7-day delay @ 1-day cutoff) completed **5 validation folds each** (20 cross-validation folds total).
3. **Data Completeness:** Both Out-of-Fold (OOF) prediction CSV files were fully written with **341,748 rows each**, containing zero missing predictions across all six candidate models.
4. **Student Isolation:** Programmatically verified that across all 20 folds, **0 students overlap** between training and validation splits.
5. **Testing Status:** The entire automated test suite passed with **28/28 tests passing** in 4.09 seconds.

```text
Cross-Validation Execution Summary:
├── Total Scenarios Evaluated:         4 scenarios (isolated)
├── Folds per Scenario:                5 folds (StratifiedGroupKFold on id_student)
├── Total Models Trained & Evaluated:  6 models (2 heuristics, 1 linear, 2 tree, 1 dummy)
├── Total Out-of-Fold Predictions:     683,496 model prediction records
├── Student-Level Overlap:             0 across all folds (100% leak-free)
└── Total Automated Tests Passing:     28 / 28 (100%)
```

---

## 2. Feature Governance & Allowlist Audit

To ensure compliance with educational data privacy, fairness, and strict leakage prevention, an explicit feature allowlist was enforced prior to training. Any feature not on the allowlist was excluded.

### 2.1 Predictive Feature Allowlist (23 Features)

| Feature Name | Category | Type | Rationale & Temporal Availability |
| :--- | :--- | :---: | :--- |
| `code_module` | Course Context | Categorical | Course discipline code (`AAA`–`GGG`); fixed at enrollment. |
| `code_presentation` | Course Context | Categorical | Intake semester (`2013B`–`2014J`); fixed at enrollment. |
| `target_assessment_type` | Assessment Context | Categorical | Mode of upcoming evaluation (`TMA`, `CMA`, `Exam`); syllabus-defined. |
| `target_assessment_weight` | Assessment Context | Numerical | Percentage syllabus weight; published in module syllabus. |
| `scheduled_date` | Assessment Context | Numerical | Official due date (days from Day 0); published in syllabus. |
| `num_previous_scheduled_assessments` | Course Progress | Numerical | Ordinal count of preceding syllabus assessments. |
| `registration_timing_days` | Student Behavior | Numerical | Days registered relative to Day 0 (negative = early); determined prior to launch. |
| `vle_total_clicks_to_date` | VLE Cumulative | Numerical | Cumulative clicks on learning platform strictly $\le T$. |
| `vle_active_days_to_date` | VLE Cumulative | Numerical | Distinct calendar days with recorded VLE activity $\le T$. |
| `vle_core_clicks_to_date` | VLE Content | Numerical | Clicks on primary learning materials (`oucontent`, `resource`, `subpage`) $\le T$. |
| `vle_forum_clicks_to_date` | VLE Social | Numerical | Clicks in discussion forums (`forumng`) $\le T$. |
| `vle_quiz_clicks_to_date` | VLE Formative | Numerical | Clicks on formative self-assessment quizzes $\le T$. |
| `vle_days_since_last_activity` | VLE Recency | Numerical | Days elapsed between last platform interaction and cutoff $T$. |
| `vle_clicks_last_7d` | VLE Velocity | Numerical | Total clicks in the 7 days immediately preceding $T$. |
| `vle_clicks_last_14d` | VLE Velocity | Numerical | Total clicks in the 14 days immediately preceding $T$. |
| `vle_activity_trend_ratio` | VLE Momentum | Numerical | Ratio of recent (7d) clicks to prior (8-14d) clicks: $\frac{\text{clicks}_{7\text{d}}}{\text{prior}_{7\text{d}} + 1}$. |
| `num_prior_submissions` | Academic History | Numerical | Count of preceding assessments completed and submitted on or before $T$. |
| `prior_submission_lateness_mean` | Academic Timeliness | Numerical | Mean days submitted after scheduled due date for prior assessments. |
| `num_prior_scored_assessments` | Graded History | Numerical | Count of prior assessments graded on or before $T$ under grading delay. |
| `mean_prior_score` | Graded History | Numerical | Mean score on historically graded assessments. |
| `std_prior_score` | Graded History | Numerical | Score standard deviation across historically graded assessments. |
| `prior_at_risk_rate` | Graded History | Numerical | Historical failure rate: proportion of graded prior assessments with score $< 50$. |
| `recent_prior_score` | Graded History | Numerical | Numerical score on the most recently returned prior assessment. |

### 2.2 Prohibited & Excluded Columns

| Column Name | Category | Exclusion Reason |
| :--- | :--- | :--- |
| `id_student` | Identifier | Memorization shortcut; used strictly for group partitioning in CV. |
| `id_assessment` | Identifier | Raw database primary key; syllabus features used instead. |
| `target_assessment_score` | Ground Truth | Future outcome; strictly masked at cutoff $T$. |
| `at_risk` | Target Variable | Supervised learning label; prohibited as a predictor. |
| `final_result` | End Outcome | Severe post-hoc target leakage (`Distinction`, `Pass`, `Fail`, `Withdrawn`). |
| `date_unregistration` | End Outcome | Post-hoc event occurring after course drop; leakage. |
| `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability` | Sensitive Demographics | Fairness and governance policy; models must predict based on demonstrated academic engagement. |

---

## 3. Cross-Validation Strategy: StratifiedGroupKFold

Because individual learners enroll in multiple modules or submit multiple assessments across time, random splitting causes severe optimistic data leakage (an algorithm could memorize student-specific traits across snapshots).

### 3.1 Design Specifications
- **Partition Algorithm:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- **Grouping Variable:** `id_student` (all snapshots for a student are assigned to the exact same fold).
- **Stratification Target:** `at_risk` (ensures identical class balance across folds).
- **Programmatic Leakage Verification:** Enforced in code:
  $$\text{Train}_{\text{students}} \cap \text{Val}_{\text{students}} = \emptyset \quad \forall \text{ folds}$$

### 3.2 Empirical Fold Distribution Audit (14-Day Delay Baseline, 7-Day Horizon)

| Fold Index | Training Snapshots ($N$) | Validation Snapshots ($N$) | Validation Students | Overlapping Students | Validation At-Risk Count | Validation At-Risk Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 136,699 | 34,175 | 4,692 | **0** | 2,925 | 8.56% |
| **Fold 1** | 136,700 | 34,174 | 4,646 | **0** | 2,924 | 8.56% |
| **Fold 2** | 136,697 | 34,177 | 4,689 | **0** | 2,926 | 8.56% |
| **Fold 3** | 136,699 | 34,175 | 4,638 | **0** | 2,924 | 8.56% |
| **Fold 4** | 136,701 | 34,173 | 4,686 | **0** | 2,924 | 8.56% |
| **Total** | — | **170,874** | **23,351** | **0** | **14,623** | **8.56%** |

*Note: The exact same student partition was preserved across both cutoff horizons (7-day and 1-day) and both delay scenarios (14-day and 7-day).*

---

## 4. Evaluation Scenarios & Protocol

In accordance with strict operational separation rules:
1. **Grading Delay Assumptions Evaluated Separately:**
   - **Baseline (14-Day Grading Delay):** Realistic human grading turnaround ($\text{date\_submitted} + 14 \le T$).
   - **Alternative (7-Day Grading Delay):** Accelerated marking turnaround ($\text{date\_submitted} + 7 \le T$).
2. **Cutoff Horizons Evaluated Separately:**
   - **7-Day Early Warning ($T = D_k - 7$):** Allows academic advisors 1 full week for proactive intervention.
   - **1-Day Immediate Checkpoint ($T = D_k - 1$):** Captures eleventh-hour revision surges and non-engagement.
3. **No Metric Pooling:** Headline metrics are never averaged or pooled across horizons or delays.

---

## 5. Baseline Models & Benchmark Performance

Six distinct model architectures were evaluated across all folds. Because the dataset has an **8.56% at-risk positive class prevalence**, raw accuracy is misleading. Evaluation relies on **PR-AUC (Average Precision)**, **ROC-AUC**, and **Recall@k** (proportion of at-risk students captured within the highest-risk 10% and 20% prediction budgets).

### 5.1 Scenario 1: Baseline 14-Day Delay — 7-Day Early Warning Horizon ($T = D_k - 7$)

*Primary operational benchmark for student intervention.*

| Model Benchmark | PR-AUC (AP) | ROC-AUC | Recall@10% | Recall@20% | Precision (0.5) | Recall (0.5) | F1-Score (0.5) | Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Classifier** | 0.0856 ± 0.0000 | 0.5000 ± 0.0000 | 0.1000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% |
| **Uniform Random** | 0.0864 ± 0.0008 | 0.5020 ± 0.0031 | 0.1021 ± 0.0039 | 0.2041 ± 0.0041 | 0.0860 | 0.5020 | 0.1471 | 100.0% |
| **Previous-Score Heuristic** | 0.2058 ± 0.0047 | 0.6659 ± 0.0037 | 0.3405 ± 0.0049 | 0.4976 ± 0.0062 | 0.4072 | 0.2335 | 0.2964 | **82.75%** |
| **Logistic Regression** | 0.3234 ± 0.0068 | 0.8009 ± 0.0045 | 0.4128 ± 0.0092 | 0.5957 ± 0.0088 | 0.5076 | 0.1035 | 0.1717 | 100.0% |
| **Random Forest (max_depth=8)** | 0.4378 ± 0.0073 | 0.8541 ± 0.0047 | 0.4922 ± 0.0095 | 0.6907 ± 0.0078 | 0.6409 | 0.1404 | 0.2302 | 100.0% |
| **HistGradientBoosting (Best)** | **0.4565 ± 0.0081** | **0.8668 ± 0.0042** | **0.5107 ± 0.0125** | **0.7077 ± 0.0097** | **0.5480** | **0.2319** | **0.3258** | **100.0%** |

### 5.2 Scenario 2: Baseline 14-Day Delay — 1-Day Immediate Checkpoint ($T = D_k - 1$)

*Immediate pre-deadline prediction incorporating final-week study velocity.*

| Model Benchmark | PR-AUC (AP) | ROC-AUC | Recall@10% | Recall@20% | Precision (0.5) | Recall (0.5) | F1-Score (0.5) | Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Classifier** | 0.0856 ± 0.0000 | 0.5000 ± 0.0000 | 0.1000 ± 0.0000 | 0.2000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% |
| **Uniform Random** | 0.0864 ± 0.0008 | 0.5020 ± 0.0031 | 0.1021 ± 0.0039 | 0.2041 ± 0.0041 | 0.0860 | 0.5020 | 0.1471 | 100.0% |
| **Previous-Score Heuristic** | 0.2081 ± 0.0050 | 0.6693 ± 0.0037 | 0.3436 ± 0.0051 | 0.5013 ± 0.0062 | 0.4074 | 0.2372 | 0.2995 | **82.90%** |
| **Logistic Regression** | 0.3278 ± 0.0065 | 0.8023 ± 0.0046 | 0.4173 ± 0.0091 | 0.6011 ± 0.0079 | 0.5109 | 0.1098 | 0.1795 | 100.0% |
| **Random Forest (max_depth=8)** | 0.4409 ± 0.0075 | 0.8562 ± 0.0047 | 0.4944 ± 0.0101 | 0.6938 ± 0.0084 | 0.6385 | 0.1425 | 0.2328 | 100.0% |
| **HistGradientBoosting (Best)** | **0.4632 ± 0.0068** | **0.8697 ± 0.0042** | **0.5144 ± 0.0122** | **0.7150 ± 0.0085** | **0.5517** | **0.2393** | **0.3331** | **100.0%** |

### 5.3 Scenario 3: Alternative 7-Day Delay Sensitivity ($T = D_k - 7$)

*Evaluates sensitivity to faster grading turnaround (grades available 7 days after submission).*

| Model Benchmark | PR-AUC (AP) | ROC-AUC | Recall@10% | Recall@20% | F1-Score (0.5) | Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Previous-Score Heuristic** | 0.2083 ± 0.0044 | 0.6696 ± 0.0035 | 0.3438 ± 0.0050 | 0.5020 ± 0.0060 | 0.2996 | **82.93%** |
| **Logistic Regression** | 0.3256 ± 0.0064 | 0.8022 ± 0.0046 | 0.4158 ± 0.0090 | 0.5990 ± 0.0082 | 0.1748 | 100.0% |
| **HistGradientBoosting (Best)** | **0.4578 ± 0.0060** | **0.8673 ± 0.0041** | **0.5120 ± 0.0121** | **0.7115 ± 0.0089** | **0.3279** | **100.0%** |

### 5.4 Scenario 4: Alternative 7-Day Delay Sensitivity ($T = D_k - 1$)

| Model Benchmark | PR-AUC (AP) | ROC-AUC | Recall@10% | Recall@20% | F1-Score (0.5) | Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Previous-Score Heuristic** | 0.2132 ± 0.0024 | 0.6777 ± 0.0035 | 0.3487 ± 0.0049 | 0.5098 ± 0.0058 | 0.3048 | **84.26%** |
| **Logistic Regression** | 0.3288 ± 0.0039 | 0.8023 ± 0.0045 | 0.4214 ± 0.0092 | 0.6019 ± 0.0081 | 0.1772 | 100.0% |
| **HistGradientBoosting (Best)** | **0.4664 ± 0.0062** | **0.8719 ± 0.0041** | **0.5184 ± 0.0120** | **0.7213 ± 0.0084** | **0.3350** | **100.0%** |

### 5.5 Train-Tuned Threshold Evaluation (Zero Validation Leakage)

Because positive prevalence is 8.56%, a fixed classification threshold of 0.50 favors precision over recall. To evaluate optimal operating points without leakage, decision thresholds were optimized strictly on training folds to maximize F1, then applied to out-of-fold validation sets:

| Model | Optimal Train Threshold ($\tau^*$) | Validation Precision | Validation Recall | Validation F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.16 \pm 0.01$ | 0.2782 ± 0.0041 | 0.5694 ± 0.0118 | **0.3739 ± 0.0071** |
| **Random Forest** | $0.14 \pm 0.01$ | 0.3312 ± 0.0052 | 0.6045 ± 0.0124 | **0.4278 ± 0.0069** |
| **HistGradientBoosting** | $0.21 \pm 0.01$ | 0.3845 ± 0.0060 | 0.5592 ± 0.0115 | **0.4554 ± 0.0074** |

---

## 6. Key Empirical Findings & Heuristic Comparison

```mermaid
flowchart TD
    subgraph Benchmark Performance Comparison (PR-AUC)
        M["Majority Baseline: 0.0856"] --> R["Random Baseline: 0.0864"]
        R --> PS["Previous-Score Heuristic: 0.2058<br/>(Coverage: 82.75%)"]
        PS --> LR["Logistic Regression: 0.3234<br/>(+57.1% over Heuristic)"]
        LR --> RF["Random Forest: 0.4378<br/>(+112.7% over Heuristic)"]
        RF --> HGB["HistGradientBoosting: 0.4565<br/>(+121.8% over Heuristic)"]
    end
```

### 1. Massive Superiority Over Heuristics
- The **Previous-Score Heuristic** achieves a PR-AUC of 0.2058. It suffers from a fatal blind spot: **17.25% of snapshots have zero historical score coverage** (Assessment 1 in every module, or early submissions where tutor grades have not yet been published).
- In contrast, ML models achieve **100% coverage** by leveraging early VLE interaction patterns (`vle_total_clicks_to_date`, `vle_active_days_to_date`, `registration_timing_days`).
- **HistGradientBoosting improves PR-AUC by +121.8% over the heuristic** (0.4565 vs 0.2058) and achieves an ROC-AUC of **0.8668**.

### 2. Practical Operational Utility: Recall@k
- In real-world educational support, student advisors have finite time and can only contact a targeted subset of learners.
- At a **10% intervention capacity** (`Recall@10%`), HistGradientBoosting successfully captures **51.07% of all at-risk students** across the university.
- At a **20% intervention capacity** (`Recall@20%`), HistGradientBoosting captures **70.77% of all at-risk students**.
- This is **over $3.5\times$ more effective** than random advisor outreach (which captures only 10% and 20% respectively) and substantially outperforms the previous-score heuristic (34.05% and 49.76%).

### 3. Horizon Dynamics: 7-Day vs 1-Day Checkpoints
- Transitioning from the 7-day early warning horizon to the 1-day immediate checkpoint yields a modest performance boost (PR-AUC improves from 0.4565 to 0.4632; Recall@20% from 70.77% to 71.50%).
- This demonstrates that **early warning 7 days in advance captures over 98% of the predictive signal** present at 1 day before the deadline, validating the 7-day window as an effective, actionable timeframe for advisor interventions.

### 4. Grading Delay Sensitivity: 14-Day vs 7-Day Turnaround
- Reducing tutor marking delay from 14 days to 7 days increases previous-score coverage only marginally (from 82.75% to 82.93% at 7 days; 82.90% to 84.26% at 1 day).
- Model PR-AUC shifts from 0.4565 to 0.4578 at the 7-day horizon.
- This proves that **system utility is robust and does not degrade significantly under realistic 14-day institutional grading turnaround**.

---

## 7. Subgroup Robustness & Diagnostic Breakdown

To ensure the model is fair, stable, and effective across diverse academic environments, out-of-fold performance was audited across modules and assessment types.

### 7.1 Performance by Course Module (14-Day Delay, 7-Day Horizon)

| Module Code | Total Snapshots ($N$) | Observed At-Risk ($N$) | At-Risk Rate (%) | Logistic Regression PR-AUC | HistGradientBoosting PR-AUC | HGB Recall@10% | HGB Recall@20% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAA** | 3,146 | 237 | 7.53% | 0.3117 | 0.3061 | 0.4768 | 0.6203 |
| **BBB** | 42,979 | 3,166 | 7.37% | 0.1535 | 0.3791 | 0.5117 | 0.6766 |
| **CCC** | 17,014 | 2,590 | 15.22% | 0.4317 | **0.5109** | 0.3653 | 0.5799 |
| **DDD** | 29,865 | 4,730 | 15.84% | 0.4787 | **0.5591** | 0.4008 | 0.6150 |
| **EEE** | 7,886 | 352 | 4.46% | 0.2690 | 0.3474 | 0.5852 | **0.7812** |
| **FFF** | 54,769 | 2,636 | 4.81% | 0.2794 | 0.3780 | 0.5918 | **0.7568** |
| **GGG** | 15,215 | 912 | 5.99% | 0.1413 | 0.1998 | 0.3586 | 0.5428 |

#### Subgroup Observations:
- **STEM Modules with Higher Base Risk (CCC & DDD):** Modules CCC (15.22% failure) and DDD (15.84% failure) exhibit the highest PR-AUCs (0.5109 and 0.5591). The higher prevalence and strong reliance on sequential technical homework assignments make early risk highly distinguishable.
- **Low-Failure Modules with Strong Capture (EEE & FFF):** In modules EEE (4.46% failure) and FFF (4.81% failure), base prevalence is low, but the model achieves exceptional recall: **Recall@20% exceeds 75%–78%**, catching over three-quarters of all failing students within the top quintile.
- **Diagnostic Flag (Module GGG):** Module GGG has lower PR-AUC (0.1998). As established in Phase 2 EDA, Module GGG has no Tutor-Marked Assignments (TMAs)—it uses only non-credit Computer-Marked practice quizzes and a final Exam, resulting in fewer graded formative touchpoints.

### 7.2 Performance by Assessment Type

| Assessment Type | Total Snapshots ($N$) | Observed At-Risk ($N$) | At-Risk Rate (%) | Logistic Regression PR-AUC | HistGradientBoosting PR-AUC | HGB Recall@10% | HGB Recall@20% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CMA** (Computer-Marked) | 70,527 | 3,998 | 5.67% | 0.2757 | 0.3728 | **0.5540** | **0.7491** |
| **TMA** (Tutor-Marked) | 98,253 | 10,102 | 10.28% | 0.3269 | **0.4754** | **0.4771** | **0.6796** |
| **Exam** (Final Summative) | 2,094 | 523 | 24.98% | **0.5756** | **0.6396** | 0.3059 | 0.5067 |

- **TMAs represent the core formative backbone:** With 98,253 snapshots, TMAs yield a strong PR-AUC of 0.4754 and capture 67.96% of failing students in the top 20%.
- **CMAs exhibit high capture efficiency:** Despite a lower failure rate (5.67%), the top 20% budget captures **74.91%** of at-risk students.
- **Exams show high predictability:** With a 24.98% base failure rate, the model reaches a PR-AUC of **0.6396**.

---

## 8. Evaluation Figures Catalog

Diagnostic plots generated by [`src/baseline_models.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/baseline_models.py) are archived in [`reports/figures/baseline/`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/figures/baseline/):

1. **Precision-Recall Curves (`pr_curves.png`):**  
   Side-by-side comparison of 7-day vs 1-day horizons. Highlights the lift of HistGradientBoosting (AP = 0.457 / 0.463) over Logistic Regression (AP = 0.323 / 0.328), the Previous-Score Heuristic (AP = 0.206), and the no-skill prevalence line (0.086).
2. **ROC Curves (`roc_curves.png`):**  
   Shows discrimination across all models against the random diagonal, with HistGradientBoosting achieving AUC = 0.867 (7-day) and 0.870 (1-day).
3. **Recall@k Intervention Capture Efficiency (`recall_at_k_comparison.png`):**  
   Bar chart illustrating capture rates across 5%, 10%, 15%, and 20% outreach budgets. Demonstrates that 70.8% of failing students are caught within a 20% advisor workload budget.
4. **Cross-Validation Fold Stability (`cv_fold_stability.png`):**  
   Boxplot and jittered points across the 5 validation folds showing tight bounds for HistGradientBoosting ($0.4565 \pm 0.0081$), proving that model stability is unaffected by student grouping.

---

## 9. Automated Testing & Verification Audit

The test suite was executed via `pytest tests/ -v`, verifying 28 test cases across all pipeline stages:

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.14, pytest-9.2.0, pluggy-1.7.0 -- .venv/bin/python
rootdir: /Users/navadeepguduru/Student At Risk Prediction, configfile: pytest.ini
collected 28 items

tests/test_baseline_models.py::test_feature_governance_and_allowlist PASSED         [  3%]
tests/test_baseline_models.py::test_student_level_group_separation PASSED          [  7%]
tests/test_baseline_models.py::test_train_only_preprocessing_isolation PASSED     [ 10%]
tests/test_baseline_models.py::test_deterministic_random_baseline PASSED          [ 14%]
tests/test_baseline_models.py::test_previous_score_heuristic_and_coverage PASSED  [ 17%]
tests/test_baseline_models.py::test_recall_at_k_exact_and_tie_apportionment PASSED [ 21%]
tests/test_baseline_models.py::test_train_tuned_threshold_optimization PASSED     [ 25%]
tests/test_baseline_models.py::test_standard_metrics_dictionary_validity PASSED   [ 28%]
tests/test_baseline_models.py::test_oof_predictions_exist_and_complete PASSED     [ 32%]
tests/test_baseline_models.py::test_scenario_isolation_in_summary_json PASSED     [ 35%]
tests/test_data_preprocessing.py::test_raw_file_integrity PASSED                  [ 39%]
tests/test_data_preprocessing.py::test_processed_files_exist PASSED               [ 42%]
tests/test_data_preprocessing.py::test_enrollment_composite_key_uniqueness PASSED  [ 46%]
tests/test_data_preprocessing.py::test_enrollment_count_preservation PASSED        [ 50%]
tests/test_data_preprocessing.py::test_preservation_of_zero_activity_enrollments PASSED [ 53%]
tests/test_data_preprocessing.py::test_foreign_key_referential_integrity PASSED  [ 57%]
tests/test_data_preprocessing.py::test_score_and_click_validity PASSED            [ 60%]
tests/test_data_preprocessing.py::test_missing_value_handling PASSED              [ 64%]
tests/test_data_preprocessing.py::test_reproducibility_of_aggregation PASSED      [ 67%]
tests/test_temporal_feature_engineering.py::test_datasets_exist_and_non_empty PASSED [ 71%]
tests/test_temporal_feature_engineering.py::test_no_prohibited_features PASSED    [ 75%]
tests/test_temporal_feature_engineering.py::test_target_label_accuracy PASSED     [ 78%]
tests/test_temporal_feature_engineering.py::test_cutoff_calculation PASSED        [ 82%]
tests/test_temporal_feature_engineering.py::test_no_target_leakage_on_first_assessment PASSED [ 85%]
tests/test_temporal_feature_engineering.py::test_temporal_delay_sensitivity PASSED [ 89%]
tests/test_temporal_feature_engineering.py::test_vle_temporal_monotonicity PASSED  [ 92%]
tests/test_temporal_feature_engineering.py::test_valid_referential_integrity PASSED [ 96%]
tests/test_temporal_feature_engineering.py::test_class_imbalance_bounds PASSED   [100%]

============================== 28 passed in 4.09s ==============================
```

---

## 10. Guidance & Technical Transition to Phase 6

Phase 5 has established a verified, leakage-free benchmark. Key takeaways inform the upcoming **Phase 6 (Model Exploration & Iterative Optimization)**:

1. **Strong Non-Linear Signal:** The jump from Logistic Regression (PR-AUC 0.3234) to HistGradientBoosting (PR-AUC 0.4565) confirms that student risk is governed by non-linear thresholds (e.g. abrupt drops in weekly study momentum combined with assessment weight). Tree-based ensembles and gradient boosting methods (XGBoost, LightGBM, CatBoost) are strong candidates for Phase 6.
2. **Missing Historical Score Handling:** Because early assessments lack prior grades, tree models that handle `NaN` natively without artificial zero-imputation possess a distinct architectural advantage.
3. **Threshold Calibration:** Under severe class imbalance (8.56%), post-hoc probability calibration (Isotonic Regression / Platt Scaling) should be explored in Phase 6 to provide calibrated posterior probabilities for decision makers.
4. **Primary Optimization Metric:** Optimization should continue to prioritize **PR-AUC (Average Precision)** and **Recall@20%** rather than ROC-AUC or unweighted accuracy.

