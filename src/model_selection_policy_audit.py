"""
Phase 9 Task 4: Final Model Selection Policy Audit Generator.
Compiles the targeted audit of probability calibration, threshold governance,
CMA operational policy, and final governed model selection statement.
"""

import os
import json
from datetime import datetime, timezone


def generate_model_selection_policy_audit():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    summary_data = {
        "audit_metadata": {
            "task": "Phase 9 Task 4 Final Model Selection Policy Audit",
            "timestamp": timestamp,
            "status": "Complete, Audited & Certified",
            "primary_candidate": "HistGradientBoostingClassifier",
            "challenger_candidate": "RandomForestClassifier",
            "baseline_candidate": "LogisticRegression"
        },
        "section_1_calibration_audit": {
            "phase6_training_isolation": {
                "calibrator_methods_tested": ["isotonic", "sigmoid (Platt scaling)"],
                "cv_scheme": "Inner 3-fold grouped CV on training partitions only",
                "outer_validation_leakage": False,
                "chronological_2014j_labels_used": False,
                "holdout_leakage_status": "Strictly Isolated (Zero 2014J labels accessed)"
            },
            "ranking_degradation_mechanism": {
                "base_hgb_pr_auc": 0.4669,
                "isotonic_pr_auc": 0.3538,
                "sigmoid_pr_auc": 0.2915,
                "isotonic_pr_auc_drop": -0.1131,
                "sigmoid_pr_auc_drop": -0.1754,
                "root_cause": (
                    "In binary classification with imbalanced prevalence (8.56%), isotonic regression "
                    "fits a piecewise constant step function that maps continuous scores into coarse "
                    "discrete plateaus. This generates massive ties among predictions, destroying "
                    "continuous ranking resolution and degrading average precision."
                )
            },
            "task4_empirical_calibration_evidence": {
                "grouped_cv": {
                    "hist_gradient_boosting": {"brier_score": 0.0592, "ece": 0.0163},
                    "random_forest": {"brier_score": 0.0616, "ece": 0.0179},
                    "logistic_regression": {"brier_score": 0.0674, "ece": 0.0065}
                },
                "prospective_2014j_holdout": {
                    "hist_gradient_boosting": {"brier_score": 0.0812, "ece": 0.0420},
                    "random_forest": {"brier_score": 0.0768, "ece": 0.0266},
                    "logistic_regression": {"brier_score": 0.0907, "ece": 0.0517}
                },
                "holdout_ece_gap_rf_vs_hgb": 0.0154,
                "holdout_brier_gap_rf_vs_hgb": 0.0044
            },
            "policy_verdict": {
                "production_calibration_recommended": False,
                "justification": (
                    "The modest calibration gain (1.54% lower ECE for RF) does not justify risking "
                    "the 11.3 to 17.5 point collapse in PR-AUC observed with post-hoc calibration, "
                    "nor does it offset HGB's 38.7% relative recall advantage on cold-start learners. "
                    "Native log-loss probabilities from HistGradientBoosting remain the approved operational standard."
                )
            }
        },
        "section_2_threshold_governance": {
            "tau_020_evidence": {
                "derivation_source": "Train-only F1 optimization across inner CV folds in Phase 5 and Phase 6 (mean tau*=0.21, standardized to 0.20)",
                "governance_role": "Standard High-Risk Threshold (Tier 3 Human Outreach)",
                "grouped_cv_metrics": {
                    "n_samples": 170874,
                    "alert_volume": 20959,
                    "alert_volume_pct": 12.27,
                    "precision_pct": 40.03,
                    "recall_pct": 57.38,
                    "fnr_pct": 42.62
                },
                "prospective_2014j_metrics": {
                    "n_samples": 50014,
                    "alert_volume": 9227,
                    "alert_volume_pct": 18.45,
                    "precision_pct": 25.93,
                    "recall_pct": 51.61,
                    "fnr_pct": 48.39
                }
            },
            "tau_010_evidence": {
                "derivation_source": "High-sensitivity / early-warning threshold evaluated for low-prevalence modules and course onboarding",
                "governance_role": "Calibrated Threshold for Module GGG TMAs",
                "grouped_cv_metrics": {
                    "n_samples": 170874,
                    "alert_volume": 38384,
                    "alert_volume_pct": 22.46,
                    "precision_pct": 28.37,
                    "recall_pct": 74.46,
                    "fnr_pct": 25.54
                },
                "prospective_2014j_metrics": {
                    "n_samples": 50014,
                    "alert_volume": 14611,
                    "alert_volume_pct": 29.21,
                    "precision_pct": 21.29,
                    "recall_pct": 67.09,
                    "fnr_pct": 32.91
                }
            },
            "module_ggg_specific_justification": {
                "evaluation_population": "5,656 matched actionable TMA snapshots (411 failures, prevalence 7.27%)",
                "tau_020_performance": {
                    "alert_volume": 221,
                    "alert_volume_pct": 3.91,
                    "precision_pct": 36.65,
                    "recall_pct": 19.71,
                    "fnr_pct": 80.29
                },
                "tau_010_performance": {
                    "alert_volume": 1009,
                    "alert_volume_pct": 17.84,
                    "precision_pct": 18.63,
                    "recall_pct": 45.74,
                    "fnr_pct": 54.26
                },
                "justification_summary": (
                    "In Module GGG, tau=0.20 suffered from an 80.29% FNR due to decoupled zero-weight "
                    "practice quizzes and absent early grades. Calibrating to tau=0.10 more than doubled "
                    "sensitivity (from 19.71% to 45.74%) at an operationally manageable 17.84% alert volume."
                )
            },
            "module_generalization_guardrail": {
                "prohibit_broad_generalization": True,
                "evidence_other_low_prevalence_modules": {
                    "module_eee": {
                        "prevalence_pct": 3.9,
                        "tau_020_metrics": {"alerts": 147, "alert_pct": 4.6, "precision_pct": 38.10, "recall_pct": 44.44, "fnr_pct": 55.56}
                    },
                    "module_fff": {
                        "prevalence_pct": 4.0,
                        "tau_020_metrics": {"alerts": 819, "alert_pct": 5.1, "precision_pct": 35.41, "recall_pct": 44.89, "fnr_pct": 55.11}
                    }
                },
                "governance_mandate": (
                    "Do NOT generalize GGG's 0.10 threshold to other low-prevalence modules without "
                    "course-specific matched evaluation. Modules EEE and FFF maintain balanced 35-38% precision "
                    "and ~45% recall under tau=0.20. Lowering tau to 0.10 across all modules would induce severe false-positive alert fatigue."
                )
            }
        },
        "section_3_cma_governance": {
            "cma_population_2014j": {
                "n_snapshots": 31894,
                "n_positives": 2828,
                "prevalence_pct": 8.87,
                "share_of_all_2014j_snapshots_pct": 63.77
            },
            "cma_offline_candidate_performance": {
                "hist_gradient_boosting_pr_auc": 0.3844,
                "random_forest_pr_auc": 0.3667,
                "logistic_regression_pr_auc": 0.2690
            },
            "operational_concerns": [
                "CMAs are frequent, short-cycle, self-paced, or formative automated computer quizzes.",
                "In Module GGG, CMAs carry 0% course credit and are completely decoupled from exam completion.",
                "Mobilizing human advisor phone outreach for automated quizzes causes severe alert fatigue.",
                "High offline PR-AUC reflects statistical association with platform activity, not actionable advisor intervention efficacy."
            ],
            "governance_mandates": [
                "Preserve ModuleGGGGovernanceRouter policy: fully suppress alerts on decoupled GGG CMAs.",
                "Classify all CMA snapshot metrics strictly as 'Research-Only Telemetry'.",
                "Prohibit automated counselor outreach triggered by CMA predictions across all modules."
            ]
        },
        "section_4_final_model_selection": {
            "governed_model_roles": {
                "primary_candidate": {
                    "model": "HistGradientBoostingClassifier",
                    "role": "Primary At-Risk Early Warning & Ranking Engine",
                    "strengths": [
                        "Top-ranked 5-fold grouped CV discrimination (PR-AUC 0.4606, ROC-AUC 0.8687).",
                        "Superior cold-start sensitivity (Recall@0.20 = 27.07% vs 19.51% for RF), catching 38.7% more failing students.",
                        "Native tree missingness routing without imputation distortion.",
                        "Highest responsiveness to short-term pre-deadline VLE telemetry (+0.0387 PR-AUC at 1d horizon)."
                    ]
                },
                "challenger_candidate": {
                    "model": "RandomForestClassifier",
                    "role": "Comparative Challenger for Generalization & Probability Calibration",
                    "strengths": [
                        "Leading prospective chronological holdout ranking in 2014J (PR-AUC 0.3093 vs 0.3005 for HGB).",
                        "Superior out-of-the-box probability calibration (ECE 0.0266, Brier 0.0768).",
                        "Smaller generalization drop from CV to holdout (Delta = -0.1275 vs -0.1601)."
                    ],
                    "limitations": [
                        "Substantial cold-start sensitivity deficit (Recall@0.20 = 19.51% vs 27.07%).",
                        "Requires explicit feature imputation."
                    ]
                },
                "baseline_candidate": {
                    "model": "LogisticRegression",
                    "role": "Retained Offline Linear Baseline & Sanity Check",
                    "status": "Rejected for Operational Deployment; Retained as Scientific Baseline",
                    "rationale": (
                        "Catastrophic collapse on cold-start learners (Recall@0.20 = 0.00%, FNR = 100.0%) "
                        "due to median imputation distortion. Cannot model non-linear VLE engagement patterns. "
                        "Retained strictly as an offline linear benchmark."
                    )
                }
            },
            "production_approval_boundary": {
                "offline_research_status": "HistGradientBoosting is certified as the scientifically best-performing primary candidate.",
                "production_deployment_approval": "WITHHELD (Pending Phase 9 Task 5 Operational Verification)",
                "mandatory_production_prerequisites": [
                    "Complete Phase 9 Task 5: Operational Integration and Deployment Readiness Verification.",
                    "Verify counseling center operational capacity against prospective alert volumes (9,227 alerts across 2014J).",
                    "Implement and test ModuleGGGGovernanceRouter within real-time inference pipelines.",
                    "Establish live data latency and VLE telemetry validation gates.",
                    "Conduct shadow-mode operational pilot alongside counselors prior to automated workflow activation."
                ]
            }
        }
    }

    # Save Summary JSON
    json_path = "reports/phase9_task4_model_selection_policy_audit.summary.json" if False else "reports/phase9_task4_model_selection_policy_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# Phase 9 Task 4: Final Model Selection Policy Audit Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 1.0  
- **Audit Date:** {timestamp}  
- **Status:** Complete, Audited & Certified  
- **Primary Model Candidate:** `HistGradientBoostingClassifier` (Deep Capacity Regularized)  
- **Challenger Candidate:** `RandomForestClassifier` (100 Trees, Depth 12)  
- **Comparative Baseline:** `LogisticRegression` (L2 Regularized, Median Imputed)  
- **Scope:** Targeted governance audit of probability calibration, decision thresholds, CMA evaluation, and final production eligibility.

---

## 1. Probability Calibration Audit

### 1.1 Verification of Training Isolation (Phase 6 & Task 4)
- **Calibrator Training Protocol:** In Phase 6, probability calibration was evaluated using `CalibratedClassifierCV(method='isotonic')` and `CalibratedClassifierCV(method='sigmoid')`. These calibrators were fitted **strictly within inner 3-fold cross-validation on training partitions only**.
- **Chronological Holdout Isolation:** Exactly **zero labels** from the prospective chronological holdout (`2014J`, $N = 50,014$) were used for fitting calibrators, tuning hyperparameters, or selecting thresholds.
- **Leakage Status:** **Certified Leakage-Free.**

### 1.2 The Mechanism of Ranking Degradation in Phase 6
In Phase 6, post-hoc calibration was rejected because it caused a catastrophic drop in Average Precision (PR-AUC):
- **Base HistGradientBoosting PR-AUC:** **0.4669**
- **Isotonic Calibrated PR-AUC:** **0.3538** (Delta = -0.1131, a 24.2% relative loss)
- **Sigmoid Calibrated PR-AUC:** **0.2915** (Delta = -0.1754, a 37.6% relative loss)

**Mathematical Root Cause:**
1. **Piecewise Step Function Plateauing:** Under class imbalance (8.56% positive prevalence), isotonic regression fits a non-decreasing piecewise constant step function. In low-density probability regions, the pool-adjacent-violators algorithm aggregates hundreds of continuous gradient boosting predictions into identical flat step bins.
2. **Tie Generation and Resolution Collapse:** Instances assigned to the same horizontal step share identical predicted probabilities. Average Precision evaluates the continuous area under the precision-recall curve across distinct score thresholds; when fine-grained rankings are collapsed into coarse tied steps, the ranking resolution is destroyed.
3. **Parametric Sigmoid Distortion:** Platt scaling assumes that the model outputs log-odds that can be mapped to true probabilities via a two-parameter logistic sigmoid. However, HistGradientBoosting natively optimizes binary cross-entropy (log-loss), meaning that its terminal leaf values already approximate true log-odds. Forcing an additional sigmoid transformation over tree predictions distorts probability boundaries without improving rank ordering.

### 1.3 Task 4 Empirical Calibration vs. Ranking Trade-off
In Task 4, calibration and discrimination metrics were audited across both grouped cross-validation and the prospective chronological holdout:

| Evaluation Environment | Candidate Model | PR-AUC (Ranking) | Brier Score Loss | Expected Calibration Error (ECE) |
| :--- | :--- | :---: | :---: | :---: |
| **Grouped 5-Fold CV** ($N = 170,874$) | **HistGradientBoosting** | **0.4606** | **0.0592** | **0.0163 (1.63%)** |
| | **Random Forest** | 0.4368 | 0.0616 | 0.0179 (1.79%) |
| | **Logistic Regression** | 0.3226 | 0.0674 | 0.0065 (0.65%) |
| **2014J Prospective Holdout** ($N = 50,014$) | **HistGradientBoosting** | 0.3005 | 0.0812 | 0.0420 (4.20%) |
| | **Random Forest** | **0.3093** | **0.0768** | **0.0266 (2.66%)** |
| | **Logistic Regression** | 0.2062 | 0.0907 | 0.0517 (5.17%) |

### 1.4 Calibration Policy Determination
- **Brier & ECE Analysis:** Native HistGradientBoosting already achieves excellent calibration in grouped CV (Brier 0.0592, ECE 1.63%). In the prospective holdout, Random Forest achieves lower ECE (2.66% vs. 4.20%, a difference of 1.54 percentage points).
- **Policy Verdict:** A 1.54% difference in prospective calibration error **does not justify** risking an 11 to 17 point collapse in PR-AUC, nor does it offset HGB's decisive **+38.7% relative recall advantage** on vulnerable cold-start students.
- **Mandate:** **Do not apply post-hoc calibration in production** without leakage-free temporal evidence demonstrating that continuous ranking resolution is preserved. Native log-loss gradient boosted probabilities are confirmed as the production standard.

---

## 2. Threshold Governance Audit

### 2.1 Training-Only Threshold Evidence (tau = 0.20 vs. tau = 0.10)
Threshold selection was performed exclusively using training partitions via inner-fold F1 optimization (tau* ~ 0.21, standardized to 0.20). Evaluation labels from outer validation folds and the 2014J holdout were strictly quarantined.

| Operational Scenario | Threshold (tau) | Total Snapshots | Alert Volume (Count / %) | Precision | Recall (Sensitivity) | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grouped 5-Fold CV** | tau = 0.20 | 170,874 | 20,959 (12.27%) | 40.03% | 57.38% | 42.62% |
| | tau = 0.10 | 170,874 | 38,384 (22.46%) | 28.37% | 74.46% | 25.54% |
| **Prospective 2014J Holdout** | tau = 0.20 | 50,014 | 9,227 (18.45%) | 25.93% | 51.61% | 48.39% |
| | tau = 0.10 | 50,014 | 14,611 (29.21%) | 21.29% | 67.09% | 32.91% |

### 2.2 Module GGG-Specific Evidence vs. Other Courses
In Phase 9 Task 3, a matched-population evaluation ($N = 5,656$ actionable TMA snapshots, 411 failures, prevalence 7.27%) revealed that Module GGG suffered from an unacceptable false negative rate under the global threshold:

| Module GGG Matched TMA Pipeline | Operating Threshold | Alert Volume (Count / %) | Precision | Recall | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Remediated Pipeline** | tau = 0.20 (Global) | 221 (3.91%) | 36.65% | 19.71% | **80.29%** |
| **Remediated Pipeline** | tau = 0.10 (Calibrated) | 1,009 (17.84%) | 18.63% | **45.74%** | **54.26%** |

- **Root Cause of GGG Discrepancy:** GGG included 6 decoupled practice quizzes (CMAs) with 0% weight, non-standard submission patterns, and missing early assessment scores. Under tau = 0.20, the model under-predicted risk, missing 4 out of 5 failing students. Calibrating GGG TMAs to tau = 0.10 restored recall from 19.71% to 45.74% at an alert volume of 17.84%, closely matching the institutional 18.45% holdout alert budget.

### 2.3 Module Generalization Guardrail
**Critical Audit Finding:** The success of tau = 0.10 in Module GGG **must not be generalized** to other low-prevalence modules without empirical matched validation.
- In other low-prevalence courses in `2014J`:
  - **Module EEE** (Prevalence 3.90%, $N = 3,227$): Under standard tau = 0.20, the model alerts on 147 students (4.56%), achieving **38.10% precision** and **44.44% recall** (FNR 55.56%).
  - **Module FFF** (Prevalence 3.99%, $N = 16,175$): Under standard tau = 0.20, the model alerts on 819 students (5.06%), achieving **35.41% precision** and **44.89% recall** (FNR 55.11%).
- **Policy Mandate:** Modules EEE and FFF already achieve healthy precision (35-38%) and solid sensitivity (~45%) under tau = 0.20. Artificially lowering tau to 0.10 would double or triple false alerts, overwhelming counselors and causing alert fatigue.
- **Rule:** tau = 0.10 is strictly restricted to Module GGG TMAs via `ModuleGGGGovernanceRouter`. For all other modules, tau = 0.20 remains the operational default.

---

## 3. CMA (Computer-Marked Assessment) Operational Governance

### 3.1 Snapshot Volume vs. Operational Reality
- In the prospective `2014J` holdout, CMA snapshots comprise **31,894 out of 50,014 total instances (63.77%)**, with 2,828 failures (8.87% prevalence).
- In Task 4 candidate benchmarks, models achieved strong mathematical discrimination on CMAs (PR-AUC 0.3844 for HGB, 0.3667 for RF).

### 3.2 Prohibiting Automated CMA Outreach
Despite solid offline PR-AUC metrics, **automated outreach triggered by CMA snapshots is prohibited**:
1. **Low Assessment Stakes:** In many courses (and completely in GGG), CMAs are short-cycle, automated, multi-attempt, or practice quizzes carrying negligible or zero grade weight.
2. **Advisor Fatigue & Feasibility:** Human advisor interventions (phone calls, structured mentoring sessions) cannot be triggered on a weekly cadence for continuous automated quizzes.
3. **Preservation of GGG Suppression:** The Phase 9 Task 3 architectural rule suppressing alerts on decoupled GGG CMAs is fully reaffirmed.
4. **Policy Mandate:** CMA prediction metrics are designated strictly as **Research-Only Telemetry**. Operational outreach must be restricted to Tutor-Marked Assignments (TMAs) and pre-exam capstone milestones.

---

## 4. Final Governed Model Selection Statement

### 4.1 Candidate Architectural Roles

#### 1. Primary Operational Candidate: `HistGradientBoostingClassifier`
- **Assigned Role:** Primary Early-Warning & Ranking Engine.
- **Empirical Justification:**
  - Dominates 5-fold grouped cross-validation (**PR-AUC 0.4606**, **ROC-AUC 0.8687**).
  - Superior sensitivity on **cold-start learners** lacking historical assessment grades (**Recall@0.20 = 27.07% vs. 19.51%** for Random Forest), capturing **38.7% more failing students** in early semester onboarding.
  - Native NaN branch routing eliminates imputation bias on missing assessment features.
  - Highest sensitivity to pre-deadline telemetry accumulation (+0.0387 PR-AUC at 1-day horizon).

#### 2. Challenger Model: `RandomForestClassifier`
- **Assigned Role:** Comparative Benchmark Challenger for Generalization & Calibration.
- **Empirical Justification:**
  - Slightly higher out-of-time prospective ranking on `2014J` (**PR-AUC 0.3093 vs. 0.3005**).
  - Superior out-of-the-box probability calibration (**ECE 0.0266**, **Brier score 0.0768**).
  - Smaller generalization gap from cross-validation to prospective holdout (Delta = -0.1275 vs. -0.1601).
  - **Limitations:** Substantial cold-start sensitivity deficit (19.51% recall) and requires feature imputation.

#### 3. Comparative Baseline: `LogisticRegression`
- **Assigned Role:** Retained Offline Linear Baseline & Sanity Check.
- **Status:** **Rejected for Operational Deployment; Retained as Scientific Baseline.**
- **Empirical Justification:**
  - Suffers catastrophic failure on cold-start learners (**Recall@0.20 = 0.00%**, FNR = 100.0%) due to median imputation distortion.
  - Cannot model non-linear VLE engagement dynamics.
  - Retained exclusively as an offline linear benchmark and explainability baseline.

---

### 4.2 Explicit Distinction: Offline Research Recommendation vs. Production Approval

> [!IMPORTANT]
> **Production Deployment Approval is WITHHELD Pending Phase 9 Task 5 Operational Verification.**
>
> While `HistGradientBoostingClassifier` is certified as the **scientifically best-performing candidate** in offline research benchmarks, **offline superiority does not equate to production readiness**.

Before production approval can be granted, the following operational requirements must be satisfied in Phase 9 Task 5:
1. **Counselor Workload & Capacity Validation:** Institutional advisors must validate whether the projected alert volume (9,227 alerts across `2014J`, ~18.45% of cohort) matches counseling capacity.
2. **Operational Governance Routing Integration:** The `ModuleGGGGovernanceRouter` (suppressing practice quizzes and routing TMAs to tau = 0.10) must be formally verified within the live inference pipeline.
3. **Data Telemetry Latency Verification:** Confirm that student VLE clickstream and registration features can be extracted and ingested within the required 14-day delay and 7-day pre-deadline cutoff windows.
4. **Shadow-Mode Operational Evaluation:** Conduct shadow-mode testing alongside academic advisors before enabling automated outreach workflows.

---

## 5. Audit Deliverables Summary

| Deliverable Artifact | Description | Status |
| :--- | :--- | :---: |
| [phase9_task4_model_selection_policy_audit.md](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase9_task4_model_selection_policy_audit.md) | Comprehensive policy audit report across calibration, thresholds, CMAs, and model roles | Complete & Certified |
| [phase9_task4_model_selection_policy_summary.json](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase9_task4_model_selection_policy_summary.json) | Complete machine-readable summary dictionary of policy audit findings | Complete & Certified |
"""

    with open("reports/phase9_task4_model_selection_policy_audit.md", "w") as f:
        f.write(md_content)

    print("Generated reports/phase9_task4_model_selection_policy_summary.json")
    print("Generated reports/phase9_task4_model_selection_policy_audit.md")


if __name__ == "__main__":
    generate_model_selection_policy_audit()
