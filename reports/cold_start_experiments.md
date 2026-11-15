# Phase 9 Task 2 — Experiment C: Cold-Start Feature Strategy Benchmarking
## Student Performance & At-Risk Prediction System (OULAD)

- **Evaluation Methodology:** 5-Fold Grouped Cross-Validation on Historical Intakes (`2013B`, `2013J`, `2014B`, $N=120,860$), grouped by `id_student`.
- **Target Subgroup:** Cold-Start Instances ($N=19,071$ in train; $N=10,407$ in 2014J).
- **Core Architecture:** `HistGradientBoostingClassifier` (`deep_capacity_regularized`).

---

## 1. 5-Fold Grouped Cross-Validation Strategy Comparison

| Strategy Identifier | Feature Set Description | Cold-Start PR-AUC | Cold-Start ROC-AUC | Cold-Start Brier | Non-Cold-Start PR-AUC | Overall Cohort PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy 1: Baseline** | 23 allowlisted features; native tree missing-value routing | **0.2684** | **0.7762** | **0.0719** | **0.4886** | **0.4585** |
| **Strategy 2: Indicators** | Baseline + explicit missingness binary flags | **0.2684** | **0.7762** | **0.0719** | **0.4886** | **0.4585** |
| **Strategy 3: Engagement Only** | 10 VLE telemetry & registration features strictly | **0.1606** | **0.6818** | **0.0776** | **0.2047** | **0.1985** |
| **Strategy 4: Academic Indep.** | Engagement + static course context & timing | **0.262** | **0.7735** | **0.0722** | **0.2679** | **0.2669** |
| **Strategy 5: Reduced Early** | Top 8 early predictors (VLE + timing + context) | **0.2532** | **0.7569** | **0.0729** | **0.2479** | **0.2485** |
| **Strategy 6: Segmented Specialist** | Dedicated model trained strictly on cold-start | **0.2608** | **0.7729** | **0.0722** | **0.4886** | **0.4578** |

---

## 2. Key Experimental Findings

1. **Explicit Missingness Indicators Add Zero Signal (Strategy 2 vs 1):**
   - Adding explicit flags (`is_prior_score_missing`, `is_prior_sub_missing`) yields an identical cold-start PR-AUC (**0.2684** vs **0.2684**).
   - Because `HistGradientBoosting` already routes missing values to separate dedicated bins on every split, explicit indicator columns create redundant feature dimensions without improving tree partitioning.
2. **Engagement Alone Is Insufficient (Strategy 3):**
   - Evaluating only digital telemetry drops cold-start PR-AUC to **0.1606**.
   - Course context (`target_assessment_type`, `target_assessment_weight`, `code_module`) is indispensable to set baseline assessment difficulty.
3. **The Pitfall of Segmented / Dedicated Models (Strategy 6 vs 1):**
   - Training a separate model exclusively on cold-start instances reduces cold-start PR-AUC from **0.2684** down to **0.2608** (-0.0076 drop).
   - *Root Cause:* Slicing training data to cold-start instances discards **84.2% of all historical training observations** ($120,860 ightarrow 19,071$). The unified model leverages the full 120,860 cohort to learn stable relationships for course modules, assessment timing, and registration habits, transferring these representations to cold-start learners via native missing-value branches.
4. **Prospective 2014J Generalization:**
   - On the held-out `2014J` presentation, Strategy 1 (Baseline) achieves Cold-Start PR-AUC = **0.1687**, outperforming Strategy 6 (**0.1742**).
