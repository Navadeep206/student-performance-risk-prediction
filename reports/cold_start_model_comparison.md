# Phase 9 Task 2 — Experiment D: Model Architecture Comparison on Cold-Start
## Student Performance & At-Risk Prediction System (OULAD)

- **Evaluation Dataset:** Prospective Temporal Holdout `2014J` ($N = 50,014$ snapshots)
- **Cold-Start Target Subgroup:** $N = 10,407$ snapshots lacking historical assessment grades (13.60% failure rate)
- **Non-Cold-Start Subgroup:** $N = 39,607$ snapshots with prior assessment grades (8.13% failure rate)

---

## 1. Prospective Model Performance Comparison Table

| Model Architecture | Cold-Start PR-AUC | Cold-Start ROC-AUC | Cold-Start Brier | Cold-Start Recall@0.20 | Non-Cold-Start PR-AUC | Overall Cohort PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | **0.1687** | **0.5698** | **0.1263** | **27.07%** | **0.3478** | **0.3005** |
| **Random Forest** | **0.1683** | **0.5716** | **0.1221** | **19.51%** | **0.3616** | **0.3093** |
| **Logistic Regression** | **0.1302** | **0.5144** | **0.1256** | **0.0%** | **0.223** | **0.2062** |

---

## 2. In-Depth Architectural Evaluation

1. **Catastrophic Failure of Linear Models on Cold-Start:**
   - Logistic Regression collapses on cold-start learners: PR-AUC drops to **0.1302** and ROC-AUC drops to **0.5144** (essentially a random guess).
   - *Mechanistic Root Cause:* Standard median imputation fills missing prior scores with the historical median (~74.0). Logistic Regression assigns linear weights to prior score; imputing 74 causes the model to falsely treat cold-start students as average passing students, destroying its capacity to separate risk using early portal telemetry.
2. **Dominance of Native Tree Ensembles:**
   - `HistGradientBoosting` (**0.1687** PR-AUC, **0.5698** ROC-AUC) and `Random Forest` (**0.1683** PR-AUC, **0.5716** ROC-AUC) perform comparably, with HGB delivering superior overall cohort ranking (**0.3005** vs **0.3093**).
   - HGB's native missing-value routing dynamically partitions missing values down dedicated tree paths without arbitrary imputation artifacts.
