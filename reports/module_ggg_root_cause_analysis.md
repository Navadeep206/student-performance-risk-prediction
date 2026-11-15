# Module GGG Root Cause Analysis
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Phase Target:** Phase 9 — Temporal Generalization & Model Reliability Improvement  
- **Task Target:** Task 3 — Root Cause Analysis of Module GGG Predictive Collapse  
- **Status:** Complete & Formally Categorized  

---

## 1. Executive Summary & Failure Classification

Through repository tracing, schema audits, temporal feature verification, and empirical subgroup benchmarking, five distinct failures were confirmed. Each finding is classified using the five-tier severity taxonomy:

1. **Critical:** Direct cause of operational prediction breakdown or invalid evaluation.
2. **High:** Substantial threat to temporal generalization and model reliability.
3. **Medium:** Engineering or logic inconsistency injecting noise or omitting valid signal.
4. **Low:** Minor cosmetic or reporting artifact.
5. **Informational:** Background curricular or environmental characteristic.

---

## 2. Forensic Failure Profiles

### Finding FAIL-GGG-01 (Severity: CRITICAL)
- **Failure Identifier:** `FAIL-GGG-01`
- **Affected Component:** Curricular Assessment Architecture (`assessments.csv`, `studentAssessment.csv`)
- **Reproduction Steps:**
  ```python
  import pandas as pd
  ass = pd.read_csv("data/raw/assessments.csv")
  sa = pd.read_csv("data/raw/studentAssessment.csv")
  ggg = ass[ass["code_module"] == "GGG"]
  print("Coursework weights sum:", ggg[ggg["assessment_type"] != "Exam"]["weight"].sum())
  print("Exam submissions count:", len(sa[sa["id_assessment"].isin(ggg[ggg["assessment_type"] == "Exam"]["id_assessment"])]))
  ```
- **Expected Behavior:** Continuous coursework carries substantive course credit weight (e.g., 50–100%) that aligns early performance with student graduation and course completion.
- **Actual Behavior:** Continuous coursework has `weight = 0.0%` for all 9 continuous assessments. The final exam carries `weight = 100.0%`, but has zero submission records in `studentAssessment.csv`.
- **Root Cause:** Module GGG was designed with 100% formative continuous coursework and a single high-stakes summative exam. Because OULAD did not capture final exam scores for GGG, the prediction model trains exclusively on practice quizzes with zero academic credit impact.
- **Impact:** Decouples assessment failure risk from actual course completion and module passing.
- **Proposed Remediation:** Institutional governance rule: Flag GGG continuous coursework as formative practice. Exclude unweighted continuous assessments from high-stakes academic probation triggers.
- **Verification Strategy:** Verify in dataset metadata and audit report that zero coursework weight is explicitly logged and handled by policy.

---

### Finding FAIL-GGG-02 (Severity: HIGH)
- **Failure Identifier:** `FAIL-GGG-02`
- **Affected Component:** Temporal Alignment Pipeline (`src/temporal_feature_engineering.py`)
- **Reproduction Steps:**
  ```python
  import pandas as pd
  pred = pd.read_csv("data/processed/prediction_dataset.csv")
  sa = pd.read_csv("data/raw/studentAssessment.csv")
  cma = pred[(pred["code_module"] == "GGG") & (pred["target_assessment_type"] == "CMA") & (pred["cutoff_scenario"] == "7_day_before")]
  m = pd.merge(cma, sa[["id_assessment", "id_student", "date_submitted"]], on=["id_assessment", "id_student"])
  print("Early submission pct:", (m["date_submitted"] <= m["cutoff_date"]).mean() * 100)
  ```
- **Expected Behavior:** Cutoff date ($t = \text{scheduled\_date} - 7$) occurs before students complete their assessment, capturing active preparation telemetry.
- **Actual Behavior:** **82.18% (7,856 of 9,559) of CMA submissions occurred before the cutoff date**, with a median submission 47 days prior to Day 222 (and up to 223 days prior).
- **Root Cause:** All 6 CMAs were assigned a nominal end-of-semester deadline (Day 229 / Day 222) in `assessments.csv`. In practice, they were open-access self-paced quizzes. Setting the prediction horizon to Day 222 inverts the chronological order of learning and evaluation.
- **Impact:** Feature telemetry measured in the 7 and 14 days prior to Day 222 reflects post-completion behavior rather than pre-test study effort. Students who completed the course early and logged out exhibit zero clicks, generating false-positive high-risk predictions.
- **Proposed Remediation:** Document early submission artifact in governance guidelines; restrict automated risk scoring on self-paced quizzes with terminal placeholder deadlines.
- **Verification Strategy:** Measure distribution of `cutoff_date - date_submitted` across all modules to verify GGG is uniquely affected.

---

### Finding FAIL-GGG-03 (Severity: MEDIUM)
- **Failure Identifier:** `FAIL-GGG-03`
- **Affected Component:** Feature Engineering Logic (`src/temporal_feature_engineering.py`, line 100 & 199)
- **Reproduction Steps:**
  ```python
  import pandas as pd
  pred = pd.read_csv("data/processed/prediction_dataset.csv")
  cma = pred[(pred["code_module"] == "GGG") & (pred["target_assessment_type"] == "CMA") & (pred["cutoff_scenario"] == "7_day_before")]
  print(cma["num_previous_scheduled_assessments"].value_counts())
  ```
- **Expected Behavior:** All 6 CMAs scheduled on the exact same date (Day 229) should share the identical value for `num_previous_scheduled_assessments` (exactly 3, corresponding to the 3 preceding TMAs).
- **Actual Behavior:** `num_previous_scheduled_assessments` takes values $3, 4, 5, 6, 7, 8$ distributed across identical-date assessments.
- **Root Cause:** The code used loop index `k` from `enumerate(cohort_sched.iterrows())` rather than computing the number of assessments strictly preceding the current date.
- **Impact:** Injects arbitrary sequential integer noise into an approved numerical feature, distorting tree splits for clustered assessments.
- **Proposed Remediation:** Compute `num_previous_scheduled_assessments` as `(cohort_sched['scheduled_date'] < current_date).sum()`.
- **Verification Strategy:** Unit test asserting that assessments with identical scheduled dates receive identical `num_previous_scheduled_assessments` values.

---

### Finding FAIL-GGG-04 (Severity: MEDIUM)
- **Failure Identifier:** `FAIL-GGG-04`
- **Affected Component:** Feature Engineering Logic (`src/temporal_feature_engineering.py`, line 161)
- **Reproduction Steps:**
  ```python
  # Trace condition: (cohort_subs["scheduled_date"] < a_date)
  # When evaluating CMA with scheduled_date=229, any peer CMA has scheduled_date=229.
  # 229 < 229 is strictly False.
  ```
- **Expected Behavior:** A student who completed peer CMAs earlier in the term should have those graded scores included in `mean_prior_score` and `num_prior_scored_assessments`.
- **Actual Behavior:** Prior CMA scores are completely omitted; `num_prior_scored_assessments` never exceeds 3 (the 3 TMAs).
- **Root Cause:** The SQL-like filtering condition strictly enforced `scheduled_date < a_date`. For assessments sharing the same nominal deadline, peer assessments are excluded regardless of their completion date.
- **Impact:** Information loss: The model is blind to student quiz performance on peer CMAs completed earlier in the presentation.
- **Proposed Remediation:** Filter historical assessments by submission and grading timestamp (`date_submitted + delay <= cutoff_t`) rather than nominal scheduled date.
- **Verification Strategy:** Test verifying that completed prior assessments sharing the same nominal deadline are included in prior score metrics.

---

### Finding FAIL-GGG-05 (Severity: CRITICAL)
- **Failure Identifier:** `FAIL-GGG-05`
- **Affected Component:** Model Calibration & Global Risk Thresholding
- **Reproduction Steps:**
  ```python
  import pandas as pd
  from sklearn.metrics import recall_score
  oof = pd.read_csv("data/processed/oof_predictions_optimized_14day_delay.csv")
  ggg = oof[(oof["code_module"] == "GGG") & (oof["cutoff_scenario"] == "7_day_before")]
  y_true = ggg["at_risk"]
  y_pred = (ggg["prob_optimized_hgb"] >= 0.20).astype(int)
  print("GGG Recall@0.20:", recall_score(y_true, y_pred) * 100)
  ```
- **Expected Behavior:** The operational threshold $\tau = 0.20$ provides equitable, reliable risk detection across all course disciplines.
- **Actual Behavior:** Recall on Module GGG collapses to **20.07%**, failing to detect **79.93% of at-risk students (FNR = 79.93%)**.
- **Root Cause:**
  1. Low baseline failure prevalence on late CMAs (1.18% to 2.45%).
  2. Sparse VLE telemetry (526 clicks per student).
  3. Formative 0% course weights.
  The global tree ensemble suppresses predicted risk probabilities on GGG learners, keeping them below the 0.20 threshold.
- **Impact:** Applying standard automated interventions on GGG results in widespread under-intervention on vulnerable students.
- **Proposed Remediation:** Implement an explicit Model Governance Protocol:
  1. Designate Module GGG as **UNRELIABLE / LOW-CONFIDENCE** for automated threshold triage.
  2. Route GGG advising exclusively to structured TMA milestones (TMA 2 and TMA 3, where PR-AUC reaches 0.35–0.45 and ROC-AUC reaches 0.81–0.84).
  3. Require manual advisor review and offline academic check-ins for GGG learners.
- **Verification Strategy:** Integrate governance routing tests and verification assertions into the test suite.

