# Phase 9 Task 2 — Experiment E: Cold-Start Practical Recommendation
## Student Performance & At-Risk Prediction System (OULAD)

- **Audit Question:** Is a separate, specialized cold-start model statistically or operationally justified?
- **Finding:** **NO. A separate cold-start model is NOT justified.**

---

## 1. Empirical Evidence Summary

1. **Data Discard Penalty:** Training a dedicated cold-start model on cold-start snapshots alone reduces training data from 120,860 to 19,071 instances, causing cold-start PR-AUC to drop from **0.2672** to **0.2492**.
2. **Transfer of Macro Context:** The unified model learns stable structural representations of course modules, credit weights, and registration habits from the full student population, which transfers to cold-start learners via native missing-value tree branches.
3. **No Operational Overhead:** A unified model eliminates pipeline branching, routing heuristics, and multi-model synchronization.

---

## 2. Institutional Advisory Guidelines

1. **Early Onboarding vs. Academic Remediations:** Alerts generated prior to Assessment 1 reflect pre-course disengagement and should trigger supportive onboarding checks rather than remedial tutoring.
2. **Avoid Goodhart's Law:** Advisors must never advise students to "click more in the portal" to lower their risk scores.
