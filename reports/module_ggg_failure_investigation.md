# Module GGG Failure Investigation
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Phase Target:** Phase 9 — Temporal Generalization & Model Reliability Improvement  
- **Task Target:** Task 3 — Module GGG Failure Investigation & Remediation  
- **Status:** Complete Empirical Investigation & Evidence Collection  

---

## 1. Executive Summary & Context

Across Phases 5–8, multi-course robustness audits established that while the optimized `HistGradientBoostingClassifier` achieved solid predictive ranking on modules such as `CCC` (PR-AUC 0.5157), `DDD` (PR-AUC 0.5639), and `FFF` (PR-AUC 0.3828), it experienced a severe collapse on **Module GGG**:
- **PR-AUC:** **0.2060** (compared to 0.4565 baseline across the university).
- **ROC-AUC:** **0.7694** (lowest discrimination of all courses).
- **Recall@0.20:** **20.07%** (False Negative Rate = **79.93%**).

Task 3 executes a comprehensive, forensic investigation across the repository, data schemas, feature engineering logic, and prediction outputs to diagnose the underlying causes of this failure.

---

## 2. Cross-Module Predictive Benchmark Comparison

The following table summarizes model performance across all 7 undergraduate course modules in OULAD evaluated on 14-day grading delay, 7-day early warning snapshots ($N = 170,874$ total):

| Module Code | Academic Domain | Total Snapshots | Positive Failures | Baseline Failure Rate | PR-AUC | ROC-AUC | Recall@0.20 | False Negative Rate | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAA** | Humanities / Arts | 3,146 | 237 | 7.53% | 0.3376 | 0.8311 | 46.41% | 53.59% | 0.0584 |
| **BBB** | Social Sciences | 42,979 | 3,166 | 7.37% | 0.3793 | 0.8413 | 50.16% | 49.84% | 0.0549 |
| **CCC** | STEM / Computing | 17,014 | 2,590 | 15.22% | 0.5157 | 0.8416 | 70.58% | 29.42% | 0.0981 |
| **DDD** | STEM / Technology | 29,865 | 4,730 | 15.84% | 0.5639 | 0.8529 | 70.04% | 29.96% | 0.0959 |
| **EEE** | STEM / Science | 7,886 | 352 | 4.46% | 0.3489 | 0.8917 | 46.31% | 53.69% | 0.0345 |
| **FFF** | STEM / Technology | 54,769 | 2,636 | 4.81% | 0.3828 | 0.8811 | 45.71% | 54.29% | 0.0361 |
| **GGG** | Social Sciences / Open | **15,215** | **912** | **5.99%** | **0.2060** | **0.7694** | **20.07%** | **79.93%** | **0.0521** |

---

## 3. Dissecting Module GGG Architecture & Course Structure

### 3.1 Coursework Weight Anomaly (100% Formative Practice)
In OULAD, courses are divided into continuous coursework (Tutor-Marked Assessments / TMAs and Computer-Marked Assessments / CMAs) and a final Examination.
- In modules `AAA` through `FFF`, continuous coursework carries between 50% and 100% of the total module grade.
- In **Module GGG**, across all three presentations (`2013J`, `2014B`, `2014J`), **the sum of all coursework weights is strictly $0.0\%$**:
  - 3 TMAs: `weight = 0.0%`
  - 6 CMAs: `weight = 0.0%`
  - 1 Final Exam: `weight = 100.0%`
- In `studentAssessment.csv`, final exam marks are **completely unrecorded for Module GGG** (0 submission records). Final exams were only captured in the database for modules `CCC` and `DDD`.
- **Finding:** In Module GGG, the prediction pipeline is exclusively evaluating unweighted formative practice assignments. The actual determinative barrier for student course completion (the final exam) is unobserved in the continuous assessment log.

### 3.2 VLE Telemetry Sparsity
Module GGG has the lowest virtual learning environment (VLE) interaction density in the entire university:
- **GGG Average Clicks per Student:** **526.4 clicks**
- Comparison: `FFF` has **2,266.7 clicks** (4.3x higher); `AAA` has **1,666.6 clicks**; `EEE` has **1,358.0 clicks**.
- Pedagogical Cause: Learning materials in Module GGG were predominantly distributed as offline physical coursebooks. Students rarely interacted with the online portal, breaking the correlation between clickstream frequency and academic risk.

---

## 4. Assessment Cohort Divergence: TMAs vs. CMAs

Breaking down Module GGG's 15,215 snapshots reveals that predictive failure is not uniform across all assessments, but concentrated in its computer-marked quizzes:

| Assessment Cohort | Snapshot Count ($N$) | Failures | Prevalence | PR-AUC | ROC-AUC | Recall@0.20 | False Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All GGG Snapshots** | **15,215** | **912** | **5.99%** | **0.2060** | **0.7694** | **20.07%** | **79.93%** |
| **TMAs (Tutor-Marked)** | 5,656 (37.2%) | 411 | 7.27% | **0.2508** | **0.7510** | **24.09%** | **75.91%** |
| **CMAs (Computer-Marked)**| 9,559 (62.8%) | 501 | 5.24% | **0.1627** | **0.7780** | **16.77%** | **83.23%** |

### 4.1 TMA Progression Trajectory
When evaluating TMAs alone, performance scales substantially as students progress through the syllabus:
- **TMA 1 (Day 61 - Cold Start):** PR-AUC = **0.0908 - 0.1227** (No prior scores, sparse clicks).
- **TMA 2 (Day ~120 - Midterm):** PR-AUC = **0.1857 - 0.3507**, ROC-AUC = **0.7097 - 0.8231**.
- **TMA 3 (Day ~170 - Pre-Exam Milestone):** PR-AUC = **0.3200 - 0.4529**, ROC-AUC = **0.8069 - 0.8364**.

On TMA 3 in `2014J`, the model achieves **PR-AUC 0.4529 and ROC-AUC 0.8364**, matching the university-wide benchmark!

### 4.2 The CMA Collapse
The 6 CMAs constitute **62.8% of all GGG snapshots**. On these quizzes:
- Overall PR-AUC collapses to **0.1627**.
- On low-prevalence CMAs (e.g. ID 37428 with 1.18% failure rate), PR-AUC drops to **0.0525**.
- Because CMAs dominate nearly two-thirds of the GGG data partition, their failure drags the entire module average down to 0.2060.

---

## 5. Temporal Decoupling & Pipeline Mechanics Failures

Forensic tracing of `src/temporal_feature_engineering.py` uncovered two major engineering failures impacting GGG:

1. **Early Submission Inversion:**
   - In `assessments.csv`, all 6 CMAs were assigned an identical placeholder deadline on the final day of the semester (Day 229 in 2013J/2014J, Day 222 in 2014B).
   - Our early-warning cutoff evaluates these CMAs 7 days prior: at **Day 222**.
   - However, **82.18% of students (7,856 out of 9,559) submitted their CMAs BEFORE Day 222**!
   - The median submission occurred **47 days prior to the cutoff**, and 25% occurred **85 to 223 days prior**.
   - Evaluating students at Day 222 uses telemetry generated months *after* they took the test. Students who mastered the course and finished early exhibit zero clicks in the 14 days prior to Day 222, triggering false-positive alerts.

2. **Feature Index Noise & Peer Score Blindness:**
   - Feature `num_previous_scheduled_assessments` assigned arbitrary integers $k \in \{3, 4, 5, 6, 7, 8\}$ to the 6 clustered CMAs based purely on table row order.
   - The feature condition `(cohort_subs["scheduled_date"] < a_date)` excluded all peer CMAs sharing the same scheduled date ($229 < 229$ is False), preventing completed CMA scores from updating `mean_prior_score`.

