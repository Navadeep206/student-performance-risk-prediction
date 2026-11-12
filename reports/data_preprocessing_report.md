# Phase 3: Data Cleaning & Preprocessing Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Dataset:** Open University Learning Analytics Dataset (OULAD)  
- **Pipeline Implementation:** [`src/data_preprocessing.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/data_preprocessing.py)  
- **Automated Test Suite:** [`tests/test_data_preprocessing.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/tests/test_data_preprocessing.py)  
- **Pipeline Execution Summary:** [`reports/preprocessing_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/preprocessing_summary.json)  
- **Status:** Complete, Tested & Verified (All 9 tests passing)  

---

## 1. Executive Summary

Phase 3 established a reproducible, non-destructive data cleaning and relational integration pipeline that converts the seven raw OULAD CSV tables into validated, typed processed datasets.

### Core Achievements:
1. **Raw File Immutability:** All seven raw CSV files remain 100% byte-identical to their initial state, verified via SHA-256 cryptographic hashing.
2. **Context-Aware Sentinel Ingestion:** The string sentinel `'?'` was converted to proper typed nulls (`NaN` / `<NA>`) without data loss or premature zero-substitution.
3. **Zero Enrollment Loss:** The master enrollment table maintains the exact 32,593 enrollment rows, preserving all 6,750 zero-submission students and 3,365 zero-click students.
4. **Automated Test Suite:** Pytest test suite verified byte integrity, referential integrity, numerical validity, and macro-aggregation reproducibility.

---

## 2. Input and Output Manifest

| Dataset Role | File Path | Input Rows | Processed Rows | Output Schema & Grain |
| :--- | :--- | :---: | :---: | :--- |
| **Clean Enrollments** | [`data/processed/enrollments_clean.csv`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/data/processed/enrollments_clean.csv) | 32,593 | **32,593** | Grain: `(module, pres, student)`. Enriched with timeline dates and course duration. |
| **Clean Assessments** | [`data/processed/assessments_clean.csv`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/data/processed/assessments_clean.csv) | 206 | **206** | Grain: `id_assessment`. Types, weights, and submission presence flag. |
| **Clean Submissions** | [`data/processed/assessment_submissions_clean.csv`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/data/processed/assessment_submissions_clean.csv) | 173,912 | **173,912** | Grain: `(id_assessment, id_student)`. Enriched with scheduled dates and lateness days. |
| **Clean VLE Events** | [`data/processed/vle_interactions_clean.csv`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/data/processed/vle_interactions_clean.csv) | 10,655,280 | **10,655,280** | Grain: Transactional click log. Preserves all raw event records; enriched with `activity_type`. |
| **Master Enrollment Base** | [`data/processed/enrollment_base.csv`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/data/processed/enrollment_base.csv) | 32,593 | **32,593** | Grain: `(module, pres, student)`. 39 columns: static traits + aggregated descriptive metrics. |

---

## 3. Context-Aware Missing-Value Treatment

| Table | Column | Raw Sentinel `'?'` Count | Processed Representation | Rationale & Handling Decision |
| :--- | :--- | :---: | :---: | :--- |
| `studentRegistration` | `date_unregistration` | 22,521 | `<NA>` (Nullable `Int64`) | **Structurally Expected:** Student never unregistered. Must **not** be replaced with 0 (which would imply withdrawal on Day 0). A dedicated flag `has_unregistered` was derived. |
| `assessments` | `date` | 11 | `<NA>` (Nullable `Int64`) | **Structurally Expected:** Exam dates managed centrally without fixed VLE deadlines. Left as null; not arbitrarily imputed. |
| `studentInfo` | `imd_band` | 1,111 | `'Missing'` (String category) | **Informative Administrative Constraint:** Regional postcode unmapped in Northern Ireland and North Region. Treated as an explicit category. |
| `studentAssessment` | `score` | 173 | `<NA>` (Nullable `Float64`) | **Ungraded / Excused:** Distinct from actual failing 0 scores. Flagged via `score_is_missing = 1` and excluded from initial score averages. |
| `studentRegistration` | `date_registration` | 45 | `<NA>` (Nullable `Int64`) | **Administrative Edge-Case:** Provisional enrollments that withdrew immediately. Maintained as nullable integer. |
| `vle` | `week_from`<br>`week_to` | 5,243<br>5,243 | `<NA>` (Nullable `Int64`) | **Semester-Long Reference Assets:** Assets available across the entire course. |

---

## 4. Relational Integration & Join Validation

### 4.1 Join Topologies Executed in Pipeline
1. **`studentInfo ⟕ studentRegistration` (Symmetric 1:1 Join):**
   - Keys: `(code_module, code_presentation, id_student)`
   - Input: 32,593 rows ⟕ 32,593 rows $\rightarrow$ Output: 32,593 rows.
   - Result: 0 duplicate keys, 0 dropped rows, 0 expansion.
2. **`enrollments ⟕ courses` (Lookup N:1 Join):**
   - Keys: `(code_module, code_presentation)`
   - Attached: `module_presentation_length` (234–269 days).
   - Result: 32,593 rows preserved.
3. **`studentAssessment ⟕ assessments` (Enrichment N:1 Join):**
   - Keys: `id_assessment`
   - Attached: `scheduled_date`, `weight`, `assessment_type`.
   - Derived: `submission_lateness_days = date_submitted - scheduled_date`.
   - Result: Exactly 173,912 rows.
4. **`studentVle ⟕ vle` (Chunked Stream N:1 Join):**
   - Keys: `id_site`
   - Streamed in chunks of 1,000,000 rows.
   - Attached: `activity_type` (`oucontent`, `resource`, `forumng`, `quiz`, etc.).
   - Result: Exactly 10,655,280 rows written to `vle_interactions_clean.csv`.

### 4.2 Aggregated Base Integration (`enrollment_base.csv`)
To prevent Cartesian multiplication and preserve the student grain:
- **Assessment Submissions Aggregation:** Pre-aggregated to `(code_module, code_presentation, id_student)`.
  - Merged via `LEFT JOIN` to enrollments.
  - 6,750 students with zero submissions preserved with `has_zero_submissions = 1`, `total_assessments_submitted = 0`.
- **VLE Engagement Aggregation:** Pre-aggregated across chunks to `(code_module, code_presentation, id_student)`.
  - Merged via `LEFT JOIN` to enrollments.
  - 3,365 students with zero clicks preserved with `is_vle_dormant = 1`, `total_vle_clicks = 0`.
- **Final Grain Verification:** Exactly 32,593 rows and 39 columns. Uniqueness verified.

---

## 5. Duplicate Handling Decisions

1. **`studentVle.csv` Event Duplication:**
   - Raw `studentVle.csv` contains 787,170 exact duplicate rows across all six columns.
   - **Decision:** All 10,655,280 records were preserved intact in `vle_interactions_clean.csv`. These records stem from distinct Moodle VLE sessions that coincidentally generated identical click counts on the same day. Deleting them would artificially deflate genuine student engagement volume.
2. **Entity Tables:**
   - Zero duplicates exist or were found in `courses`, `assessments`, `vle`, `studentInfo`, `studentRegistration`, or `studentAssessment`.

---

## 6. Automated Test Suite Execution Results

Automated regression and integrity tests were executed via `pytest`:
```bash
$ .venv/bin/pytest tests/test_data_preprocessing.py -v
============================= test session starts ==============================
platform darwin -- Python 3.12.14, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/navadeepguduru/Student At Risk Prediction
collected 9 items

tests/test_data_preprocessing.py::test_raw_file_integrity PASSED         [ 11%]
tests/test_data_preprocessing.py::test_processed_files_exist PASSED      [ 22%]
tests/test_data_preprocessing.py::test_enrollment_composite_key_uniqueness PASSED [ 33%]
tests/test_data_preprocessing.py::test_enrollment_count_preservation PASSED [ 44%]
tests/test_data_preprocessing.py::test_preservation_of_zero_activity_enrollments PASSED [ 55%]
tests/test_data_preprocessing.py::test_foreign_key_referential_integrity PASSED [ 66%]
tests/test_data_preprocessing.py::test_score_and_click_validity PASSED   [ 77%]
tests/test_data_preprocessing.py::test_missing_value_handling PASSED     [ 88%]
tests/test_data_preprocessing.py::test_reproducibility_of_aggregation PASSED [100%]

============================== 9 passed in 1.31s ===============================
```

### Verified Test Properties:
- **Raw File Immutability:** SHA-256 hashes of all 7 raw CSVs verified byte-for-byte.
- **Cardinality Preservation:** Exactly 32,593 enrollments confirmed.
- **Zero Inactivity Loss:** 6,750 zero-submission students and 3,365 zero-click students confirmed present.
- **Score Validity:** All valid scores verified in $[0, 100]$; exactly 173 missing scores flagged.
- **Aggregation Total:** Exact match on 173,912 submissions and 39,605,099 total clicks.

---

## 7. Strict Restrictions Adherence

1. **Raw Files Unchanged:** Checked and confirmed via SHA-256.
2. **No Target Created:** The predictive target (`at_risk`) was **not** created in this phase.
3. **No Prohibited Features Used:** `final_result` and `date_unregistration` are retained purely as descriptive outcome attributes in the processed storage; they are not encoded as predictive features.
4. **No Premature Feature Selection or Modeling:** Zero models were trained.
5. **No Data Deletion:** Zero rows were dropped to make data cleaner.

---

## 8. Unresolved Issues & Phase 4 Transition

- **Unresolved Issues:** None. All five processed datasets are created, validated, typed, and indexed.
- **Readiness for Phase 4 (Feature Engineering & Target Formulation):** The pipeline now provides clean, indexed datasets ready for temporal cutoff thresholding (Day 30 / Day 60), cohort standardization, and `at_risk` target construction.

