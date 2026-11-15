# Phase 10: Model Artifact & Integrity Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Date:** 2026-11-15T07:10:20Z  
- **Phase Target:** Phase 10 — Model Saving, API Development & Testing  
- **Status:** Complete, Audited & Certified  

---

## 1. Saved Model Artifacts Overview

Two model artifacts were prepared, fitted on training data, cryptographically hashed, and serialized in `models/`:

### 1.1 Primary Production Candidate
- **Model Architecture:** `HistGradientBoostingClassifier`
- **Configuration:** Deep Capacity Regularized (`learning_rate=0.05, max_iter=150, max_depth=8, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=5.0, random_state=42`)
- **Artifact Path:** `models/optimized_hist_gradient_boosting.joblib`
- **Metadata Path:** `models/optimized_hist_gradient_boosting_metadata.json`
- **Cryptographic Hash (SHA-256):** `a5ad045e8af28e318d71d036bdc6888f171934242511637396581ed8d4d80771`
- **Feature Pipeline:** 
  - Numerical (20 features): `passthrough` (native missingness NaN branch routing)
  - Categorical (3 features): `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`
- **Training Provenance:** Strictly trained on OULAD earlier intakes (`2013B`, `2013J`, `2014B`, $N = 120,860$ snapshots). The `2014J` prospective intake was strictly quarantined.

### 1.2 Challenger Benchmark Model
- **Model Architecture:** `RandomForestClassifier`
- **Configuration:** 100 Trees (`max_depth=12, min_samples_leaf=30, random_state=42`)
- **Artifact Path:** `models/challenger_random_forest.joblib`
- **Metadata Path:** `models/challenger_random_forest_metadata.json`
- **Cryptographic Hash (SHA-256):** `93262efb5a02412a42303834a7ff0eef61a3d5b87cd4b7696f3ec5b722c87f2c`
- **Feature Pipeline:** Median imputation + standard scaling for numericals; most frequent imputation + OneHotEncoder for categoricals.

---

## 2. Feature Schema & Ordering Invariance

The model enforces strict feature order invariance across all 23 allowed predictive features:

| Index | Feature Name | Feature Type | Handling of Missing Values |
| :---: | :--- | :---: | :--- |
| **0** | `target_assessment_weight` | Numerical | Passthrough |
| **1** | `scheduled_date` | Numerical | Passthrough |
| **2** | `num_previous_scheduled_assessments` | Numerical | Passthrough |
| **3** | `registration_timing_days` | Numerical | Passthrough |
| **4** | `vle_total_clicks_to_date` | Numerical | Passthrough |
| **5** | `vle_active_days_to_date` | Numerical | Passthrough |
| **6** | `vle_core_clicks_to_date` | Numerical | Passthrough |
| **7** | `vle_forum_clicks_to_date` | Numerical | Passthrough |
| **8** | `vle_quiz_clicks_to_date` | Numerical | Passthrough |
| **9** | `vle_days_since_last_activity` | Numerical | Passthrough |
| **10** | `vle_clicks_last_7d` | Numerical | Passthrough |
| **11** | `vle_clicks_last_14d` | Numerical | Passthrough |
| **12** | `vle_activity_trend_ratio` | Numerical | Passthrough |
| **13** | `num_prior_submissions` | Numerical | Passthrough |
| **14** | `prior_submission_lateness_mean` | Numerical | Passthrough |
| **15** | `num_prior_scored_assessments` | Numerical | Passthrough |
| **16** | `mean_prior_score` | Numerical | Native NaN routing (Cold Start) |
| **17** | `std_prior_score` | Numerical | Native NaN routing |
| **18** | `prior_at_risk_rate` | Numerical | Native NaN routing |
| **19** | `recent_prior_score` | Numerical | Native NaN routing |
| **20** | `code_module` | Categorical | `OrdinalEncoder` |
| **21** | `code_presentation` | Categorical | `OrdinalEncoder` |
| **22** | `target_assessment_type` | Categorical | `OrdinalEncoder` |

---

## 3. Artifact Integrity Validation Protocol

The `ModelArtifactManager` enforces four mandatory integrity gates upon artifact loading:
1. **File Existence Gate:** Verifies both joblib artifact and metadata JSON exist on disk.
2. **Cryptographic Hash Gate:** Computes SHA-256 hash of the joblib binary and asserts exact match against metadata. Prevents silent corruption or unauthorized alteration.
3. **Feature Count & Ordering Gate:** Verifies exactly 23 features in identical ordering.
4. **Prohibited Feature Gate:** Rejects any prediction request containing target leakage (`final_result`, `date_unregistration`) or sensitive demographic features (`gender`, `imd_band`, `highest_education`, `age_band`, `disability`, `region`).
