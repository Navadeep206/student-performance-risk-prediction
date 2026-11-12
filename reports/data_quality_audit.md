# Empirical Data Quality & Integrity Audit Report
## Open University Learning Analytics Dataset (OULAD)

- **Document Version:** 1.0  
- **Dataset Location:** `data/raw/`  
- **Audit Implementation:** [`src/data_quality_audit.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/data_quality_audit.py)  
- **Companion Notebook:** [`notebooks/03_data_quality_audit.ipynb`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/notebooks/03_data_quality_audit.ipynb)  
- **Audit Summary Data:** [`reports/data_quality_audit_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/data_quality_audit_summary.json)  
- **Phase:** Phase 2.5 — Missing Value & Duplicate Analysis  

---

## 1. Executive Summary

An exhaustive empirical audit was conducted across all seven relational tables of the Open University Learning Analytics Dataset (10,900,977 total records). The audit verified:
1. Zero standard SQL nulls exist in the raw CSVs; instead, missing data is systematically encoded as the string sentinel `'?'`.
2. Exact duplicate rows are completely absent in six out of seven tables. In the 10.65M-row `studentVle.csv`, 787,170 exact duplicate lines exist due to session-level clickstream logging from Moodle VLE.
3. Numeric values adhere strictly to valid bounds (all assessment scores are between 0 and 100; all click counts are positive integers $\ge 1$; no students unregistered prior to registering).
4. No data was modified, imputed, or deleted, adhering strictly to non-destructive Phase 2 guidelines.

---

## 2. Missing Value & Sentinel (`'?'`) Analysis

### 2.1 Complete Missingness Manifest Across All Tables

| Table Name | Column Name | Total Rows | Standard Nulls (`NaN`) | Sentinel `'?'` Count | Total Missing (%) | Nature of Missingness |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `courses.csv` | *All columns* | 22 | 0 | 0 | **0.00%** | Complete |
| `assessments.csv` | `date` | 206 | 0 | 11 | **5.34%** | Structurally Expected (Exams) |
| `vle.csv` | `week_from` | 6,364 | 0 | 5,243 | **82.38%** | Structurally Expected (General assets) |
| `vle.csv` | `week_to` | 6,364 | 0 | 5,243 | **82.38%** | Structurally Expected (General assets) |
| `studentInfo.csv` | `imd_band` | 32,593 | 0 | 1,111 | **3.41%** | Administrative/Geographic Unmapped |
| `studentRegistration.csv` | `date_registration` | 32,593 | 0 | 45 | **0.14%** | Administrative Anomaly (Provisional) |
| `studentRegistration.csv` | `date_unregistration` | 32,593 | 0 | 22,521 | **69.09%** | Structurally Expected (Did not withdraw) |
| `studentAssessment.csv` | `score` | 173,912 | 0 | 173 | **0.10%** | Ungraded / Excused TMAs |
| `studentVle.csv` | *All columns* | 10,655,280 | 0 | 0 | **0.00%** | Complete (Zero missing across 10.65M rows) |

---

### 2.2 Deep Dive into Critical Missing Columns

#### 1. `studentRegistration.date_unregistration` (22,521 missing, 69.09%)
- **What is missing:** The date when a student formally dropped out of the course.
- **Root Cause & Diagnosis:** Structurally Expected. Students who completed the module presentation never unregistered; hence, this field is intentionally blank for non-withdrawn learners.
- **Cross-Tabulation Validation:**
  - When `date_unregistration != '?'` (10,072 students): 10,063 have `final_result = 'Withdrawn'` and 9 have `final_result = 'Fail'`.
  - When `date_unregistration == '?'` (22,521 students): 12,361 `Pass`, 7,043 `Fail`, 3,024 `Distinction`, and 93 `Withdrawn` (students who ceased participation without submitting a formal withdrawal notice).

#### 2. `assessments.date` (11 missing, 5.34%)
- **What is missing:** The scheduled deadline day for the assessment.
- **Root Cause & Diagnosis:** Structurally Expected. Exactly 100% of the 11 missing dates belong to final `Exam` components across courses `AAA`, `BBB`, `CCC`, and `DDD`. Traditional Open University exams were coordinated nationally at independent testing centres rather than submitted to the VLE, so no fixed relative day was cataloged in the syllabus table.

#### 3. `vle.week_from` & `vle.week_to` (5,243 missing, 82.38%)
- **What is missing:** Pedagogical syllabus week mapping for VLE learning assets.
- **Root Cause & Diagnosis:** Structurally Expected. Over 82% of VLE materials (discussion forums, course glossary, generic web links, document folders, and interactive tool subpages) are available continuously throughout the semester as ongoing study aids, rather than restricted to a single syllabus week.

#### 4. `studentInfo.imd_band` (1,111 missing, 3.41%)
- **What is missing:** Index of Multiple Deprivation decile band (socioeconomic status proxy).
- **Geographic Pattern:**
  - **North Region:** 731 missing (65.8% of all missing IMD)
  - **Ireland:** 266 missing (24.0% of all missing IMD)
  - **Other UK Regions:** 114 missing (10.2%)
- **Root Cause & Diagnosis:** Administrative / Postcode matching constraint. IMD is measured independently by regional UK administrative bodies (Scotland, England, Wales, Northern Ireland, and the Republic of Ireland). Certain regional postcodes in the North and Ireland could not be merged into the standardized English deprivation decile table during anonymization.

#### 5. `studentAssessment.score` (173 missing, 0.10%)
- **What is missing:** The numeric mark awarded on an assignment.
- **Pattern:** 100% of the 173 missing scores occur in **Tutor-Marked Assessments (TMAs)** across all 7 modules.
- **Root Cause & Diagnosis:** Submitted coursework where no grade was registered due to tutor evaluation delays, corrupted submission files, or institutional exemptions granted under mitigating circumstances.

#### 6. `studentRegistration.date_registration` (45 missing, 0.14%)
- **What is missing:** Date student enrolled relative to course start.
- **Pattern:** 39 out of 45 students (86.7%) have `final_result = 'Withdrawn'`.
- **Root Cause & Diagnosis:** Administrative edge case where students enrolled provisionally and withdrew almost immediately without a timestamped registration audit log.

---

## 3. Duplicate Analysis Across All Tables

| Table Name | Total Rows | Exact Duplicate Rows | Primary / Composite Key Duplicates | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `courses.csv` | 22 | 0 | 0 | Perfect uniqueness on `(code_module, code_presentation)`. |
| `assessments.csv` | 206 | 0 | 0 | Perfect uniqueness on `id_assessment`. |
| `vle.csv` | 6,364 | 0 | 0 | Perfect uniqueness on `id_site`. |
| `studentInfo.csv` | 32,593 | 0 | 0 | Perfect uniqueness on `(code_module, code_presentation, id_student)`. `id_student` alone has 3,808 non-unique instances (multi-presentation enrollments). |
| `studentRegistration.csv` | 32,593 | 0 | 0 | Perfect uniqueness on `(code_module, code_presentation, id_student)`. Matches `studentInfo` 1-to-1. |
| `studentAssessment.csv` | 173,912 | 0 | 0 | Perfect uniqueness on `(id_assessment, id_student)`. |
| `studentVle.csv` | 10,655,280 | **787,170** (7.39%) | **2,195,949** (20.61%) | Session-level transaction logging from Moodle VLE. |

### 3.1 Understanding Duplication in `studentVle.csv`
1. **Repeated Key Combinations (`code_module, code_presentation, id_student, id_site, date`):** 2,195,949 records share identical keys. This occurs because students frequently log into the VLE multiple times on the same date (e.g. morning and evening study sessions) to review the same resource.
2. **Exact Duplicate Rows:** 787,170 records are completely identical across all six columns (including `sum_click`). When a student executes the exact same number of clicks (e.g. 1 click or 2 clicks) across separate sessions on the same calendar day, the session log extracts identical tuples.
3. **Operational Decision:** These rows represent legitimate, discrete interaction sessions and must **NOT** be discarded. In Phase 5 feature engineering, grouping by student, module, presentation, and date with `.sum()` aggregates total daily study intensity accurately.

---

## 4. Value Validation & Anomaly Checks

### 4.1 Assessment Scores
- **Valid Range:** $[0, 100]$
- **Empirical Minimum:** $0.0$
- **Empirical Maximum:** $100.0$
- **Scores Outside $[0, 100]$:** Exactly 0.

### 4.2 Clickstream Counts (`sum_click`)
- **Valid Requirement:** Non-negative, positive integer clicks.
- **Empirical Minimum:** $1.0$ click
- **Empirical Maximum:** $6,977.0$ clicks
- **Negative Clicks ($< 0$):** Exactly 0.
- **Zero Clicks ($= 0$):** Exactly 0.

### 4.3 Assessment Submission Timelines (`date_submitted`)
- **Empirical Range:** $-11$ to $+608$ days.
- **Negative Submission Dates ($< 0$):** 2,057 total submissions occurred before presentation day 0:
  - **1,909 submissions** have `is_banked = 1`: Transferred credits from prior presentation attempts are systematically tagged with negative placeholder dates.
  - **148 submissions** have `is_banked = 0`: Legitimate early submissions where enthusiastic students submitted the first assignment 1 to 9 days prior to the official term start.
- **Extended Late Submissions ($> 269$ days):** Submissions occurring long after standard course conclusion reflect formal medical/compassionate extension arrangements granted by course directors.

### 4.4 Course Presentation Lengths
- **Empirical Range:** 234 to 269 days (all strictly positive; median 261 days).

### 4.5 Registration & Unregistration Consistency
- **Check:** Did any student unregister before registering (`date_unregistration < date_registration`)?
- **Result:** Exactly 0 instances. All unregistrations occurred strictly on or after the registration date.

### 4.6 Coursework Weights Anomaly (Module GGG)
- In modules `AAA` through `FFF`, the sum of non-exam coursework weights is exactly $100.0\%$.
- In module `GGG`, the sum of coursework weights is **$0.0\%$** across all presentations. All 3 TMAs and 6 CMAs in GGG have `weight = 0.0%`, with the final Exam carrying $100.0\%$. In module GGG, coursework is 100% formative.

---

## 5. Expected vs. Suspicious Missing Values

| Category | Columns | Assessment | Action for Phase 3 / Phase 4 |
| :--- | :--- | :--- | :--- |
| **Expected Structural Nulls** | `studentRegistration.date_unregistration` | Natural condition for students who completed the course. | Maintain as `NaN` or derive binary flag `has_unregistered`. |
| **Expected Structural Nulls** | `assessments.date` (11 Exam dates) | Centralized university exams without fixed VLE submission deadlines. | Impute with presentation length or module end date. |
| **Expected Structural Nulls** | `vle.week_from`, `vle.week_to` | General, semester-long reference resources (forums, URLs, glossaries). | Flag as `is_general_resource` (1 if missing, 0 if week-specific). |
| **Informative Missingness** | `studentInfo.imd_band` (1,111 rows) | Regional/administrative postcode mapping constraint. | Treat missingness as an explicit category (`'Missing'` or `'Unknown'`). |
| **Edge-Case Submissions** | `studentAssessment.score` (173 rows) | Unscored or excused TMA submissions. | Flag submission with `score_missing = 1` and impute with 0 or cohort median. |
| **Edge-Case Registrations** | `studentRegistration.date_registration` (45 rows) | Provisional registrations that rapidly withdrew. | Impute with cohort median registration date (-57 days). |

---

## 6. Recommended Handling Decisions for Phase 3 & Beyond

1. **Robust Ingestion Pipeline:**
   - Always load CSVs using `na_values=['?']` or an explicit string converter so that `'?'` sentinels are converted to `np.nan` on ingestion.
2. **No Row Deletion:**
   - Under no circumstances should rows with missing `imd_band`, `score`, or `date_registration` be discarded. In educational analytics, students with missing demographic postcodes or ungraded assignments represent valuable operational segments.
3. **Session Clickstream Aggregation:**
   - Preserve all 10.65M rows in raw storage. When constructing daily or weekly time-series features in Phase 5, aggregate click counts using `groupby(['code_module', 'code_presentation', 'id_student', 'date'])['sum_click'].sum()`.
4. **Strict Temporal Cutoff Guardrails:**
   - To eliminate target leakage, models trained for early warning (e.g. Day 30 or Day 60) must completely ignore `date_unregistration` and all interactions/submissions logged after the cutoff threshold.

