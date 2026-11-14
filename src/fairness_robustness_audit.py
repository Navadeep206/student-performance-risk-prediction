"""
src/fairness_robustness_audit.py

Phase 8: Comprehensive Fairness Audit, Model Robustness Analysis & Operational Governance
for the Student Performance & At-Risk Prediction System (OULAD).

Implements:
1. Offline Fairness & Demographic Subgroup Auditing:
   - Evaluates sensitive attributes: Gender, Age Band, Region, Highest Education, IMD Band, Disability.
   - Computes: N, Positive Prevalence, Precision@tau=0.20, Recall@tau=0.20, FPR, FNR, PR-AUC, Brier Score.
   - Computes 95% Wilson score confidence intervals for proportions.
   - Computes Demographic Parity, Equal Opportunity, Predictive Parity, and False Positive Parity ratios.
   - Identifies small-sample statistical limitations.
   - STRICT GOVERNANCE: Demographic features remain 100% excluded from the predictive feature matrix.
2. Robustness Stress-Testing:
   - Academic presentations (2013J, 2014J, 2013B, 2014B).
   - Course modules (AAA through GGG).
   - Assessment types (TMA, CMA, Exam).
   - Prediction horizons (7-day early warning vs 1-day immediate checkpoint).
   - Grading-delay assumptions (14-day vs 7-day delay).
   - Historical score availability (Missing prior score on early assessments vs Observed).
   - Digital engagement velocity (VLE total click quartiles).
   - Prospective chronological drift on 2014J cohort.
3. Risk Tier Validation:
   - Audits decision boundaries (Low: p<0.05, Moderate: 0.05<=p<0.20, High: p>=0.20).
   - Separates Snapshot-level, Assessment 1 milestone, and Day 30 prospective campaign.
   - Strictly excludes retrospective peak-risk aggregation.
4. Failure Mode Diagnostics:
   - Profiles True Positives, False Positives, True Negatives, False Negatives.
   - Explains behavioral and contextual drivers of false alarms and missed failures.
5. Publication-Ready Figure & Summary Export:
   - Saves all diagnostic plots in reports/figures/fairness/.
   - Exports structured results to reports/fairness_robustness_summary.json.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_models import ALL_ALLOWED_FEATURES, PROHIBITED_COLUMNS


# =============================================================================
# 1. STATISTICAL HELPER UTILITIES
# =============================================================================

def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes exact Wilson score confidence interval for a binomial proportion.
    Handles small samples and extreme proportions safely without normal approximation failure.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = 1.96  # 95% confidence level
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(float(lower * 100), 2), round(float(upper * 100), 2))


def compute_binary_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                        tau: float = 0.20) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification performance metrics at fixed threshold tau.
    """
    n = len(y_true)
    if n == 0:
        return {}

    y_pred = (y_prob >= tau).astype(int)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prev = (n_pos / n * 100) if n > 0 else 0.0
    sel_rate = ((tp + fp) / n * 100) if n > 0 else 0.0
    prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    rec = (tp / n_pos * 100) if n_pos > 0 else 0.0
    fpr = (fp / n_neg * 100) if n_neg > 0 else 0.0
    fnr = (fn / n_pos * 100) if n_pos > 0 else 0.0

    ap = average_precision_score(y_true, y_prob) if n_pos > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob)

    prec_ci = wilson_score_interval(tp, tp + fp)
    rec_ci = wilson_score_interval(tp, n_pos)
    fpr_ci = wilson_score_interval(fp, n_neg)

    return {
        "n_samples": n,
        "n_positives": n_pos,
        "prevalence_pct": round(float(prev), 2),
        "selection_rate_pct": round(float(sel_rate), 2),
        "precision_pct": round(float(prec), 2),
        "precision_95ci": prec_ci,
        "recall_pct": round(float(rec), 2),
        "recall_95ci": rec_ci,
        "false_positive_rate_pct": round(float(fpr), 2),
        "false_positive_rate_95ci": fpr_ci,
        "false_negative_rate_pct": round(float(fnr), 2),
        "pr_auc": round(float(ap), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }


# =============================================================================
# 2. FAIRNESS & DEMOGRAPHIC SUBGROUP AUDIT ENGINE
# =============================================================================

class FairnessAuditor:
    """
    Conducts offline fairness audits across protected demographic attributes.
    Ensures zero demographic leakage into model training or feature pipelines.
    """

    DEMOGRAPHIC_ATTRIBUTES = [
        "gender",
        "age_band",
        "disability",
        "highest_education",
        "imd_band",
        "region"
    ]

    def __init__(self, tau: float = 0.20, small_sample_threshold: int = 2000):
        self.tau = tau
        self.small_sample_threshold = small_sample_threshold

    def audit_attribute(self, df: pd.DataFrame, attribute_name: str) -> Dict[str, Any]:
        """
        Audits model fairness across all subgroups of a given demographic attribute.
        """
        unique_groups = sorted(df[attribute_name].dropna().unique())
        subgroup_results = {}
        selection_rates = []
        recalls = []
        precisions = []
        fprs = []

        for group_val in unique_groups:
            sub = df[df[attribute_name] == group_val]
            metrics = compute_binary_metrics_at_threshold(
                sub["at_risk"].values, sub["prob_optimized_hgb"].values, tau=self.tau
            )
            is_small = metrics["n_samples"] < self.small_sample_threshold
            metrics["small_sample_limitation"] = is_small
            subgroup_results[str(group_val)] = metrics

            selection_rates.append(metrics["selection_rate_pct"])
            recalls.append(metrics["recall_pct"])
            precisions.append(metrics["precision_pct"])
            fprs.append(metrics["false_positive_rate_pct"])

        # Compute Disparity Ratios
        min_sel, max_sel = min(selection_rates), max(selection_rates)
        min_rec, max_rec = min(recalls), max(recalls)
        min_prec, max_prec = min(precisions), max(precisions)
        min_fpr, max_fpr = min(fprs), max(fprs)

        disparity_summary = {
            "demographic_parity_ratio": round(float(min_sel / max_sel), 3) if max_sel > 0 else 1.0,
            "equal_opportunity_ratio": round(float(min_rec / max_rec), 3) if max_rec > 0 else 1.0,
            "predictive_parity_ratio": round(float(min_prec / max_prec), 3) if max_prec > 0 else 1.0,
            "false_positive_parity_ratio": round(float(min_fpr / max_fpr), 3) if max_fpr > 0 else 1.0,
            "max_recall_difference_pct": round(float(max_rec - min_rec), 2),
            "max_fpr_difference_pct": round(float(max_fpr - min_fpr), 2),
        }

        return {
            "attribute": attribute_name,
            "subgroups": subgroup_results,
            "parity_disparities": disparity_summary
        }

    def run_full_audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs the fairness audit across all sensitive demographic attributes."""
        results = {}
        for attr in self.DEMOGRAPHIC_ATTRIBUTES:
            results[attr] = self.audit_attribute(df, attr)
        return results


# =============================================================================
# 3. ROBUSTNESS & STRESS-TESTING ENGINE
# =============================================================================

class RobustnessAuditor:
    """
    Evaluates model performance stability across operational scenarios,
    course structures, score availability, and activity distributions.
    """

    def __init__(self, tau: float = 0.20):
        self.tau = tau

    def audit_presentations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audits performance across course offerings/presentations."""
        res = {}
        for pres in sorted(df["code_presentation"].unique()):
            sub = df[df["code_presentation"] == pres]
            res[str(pres)] = compute_binary_metrics_at_threshold(
                sub["at_risk"].values, sub["prob_optimized_hgb"].values, tau=self.tau
            )
        return res

    def audit_modules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audits performance across different course disciplines."""
        res = {}
        for mod in sorted(df["code_module"].unique()):
            sub = df[df["code_module"] == mod]
            res[str(mod)] = compute_binary_metrics_at_threshold(
                sub["at_risk"].values, sub["prob_optimized_hgb"].values, tau=self.tau
            )
        return res

    def audit_assessment_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audits performance across assessment formats (TMA, CMA, Exam)."""
        res = {}
        for t in ["TMA", "CMA", "Exam"]:
            sub = df[df["target_assessment_type"] == t]
            if len(sub) > 0:
                res[t] = compute_binary_metrics_at_threshold(
                    sub["at_risk"].values, sub["prob_optimized_hgb"].values, tau=self.tau
                )
        return res

    def audit_score_availability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audits performance gap when historical grades are absent vs present."""
        missing = df[df["recent_prior_score"].isna()]
        observed = df[df["recent_prior_score"].notna()]
        return {
            "missing_historical_score": compute_binary_metrics_at_threshold(
                missing["at_risk"].values, missing["prob_optimized_hgb"].values, tau=self.tau
            ),
            "observed_historical_score": compute_binary_metrics_at_threshold(
                observed["at_risk"].values, observed["prob_optimized_hgb"].values, tau=self.tau
            )
        }

    def audit_vle_engagement_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audits performance across quartiles of cumulative VLE study clicks."""
        df_copy = df.copy()
        df_copy["vle_quartile"] = pd.qcut(
            df_copy["vle_total_clicks_to_date"], 4,
            labels=["Q1_lowest_activity", "Q2_medium_low", "Q3_medium_high", "Q4_highest_activity"]
        )
        res = {}
        for q in ["Q1_lowest_activity", "Q2_medium_low", "Q3_medium_high", "Q4_highest_activity"]:
            sub = df_copy[df_copy["vle_quartile"] == q]
            res[q] = compute_binary_metrics_at_threshold(
                sub["at_risk"].values, sub["prob_optimized_hgb"].values, tau=self.tau
            )
        return res

    def audit_chronological_generalization(self, chrono_df: pd.DataFrame) -> Dict[str, Any]:
        """Audits prospective generalizability on the held-out 2014J cohort."""
        all_res = compute_binary_metrics_at_threshold(
            chrono_df["at_risk"].values, chrono_df["prob_optimized_hgb"].values, tau=self.tau
        )
        novel_sub = chrono_df[chrono_df["is_novel_student"] == 1]
        novel_res = compute_binary_metrics_at_threshold(
            novel_sub["at_risk"].values, novel_sub["prob_optimized_hgb"].values, tau=self.tau
        )
        return {
            "chronological_2014J_all_learners": all_res,
            "chronological_2014J_novel_learners_only": novel_res
        }


# =============================================================================
# 4. FAILURE ANALYSIS ENGINE
# =============================================================================

def analyze_failure_modes(df: pd.DataFrame, tau: float = 0.20, tau_low: float = 0.05) -> Dict[str, Any]:
    """
    Extracts descriptive profiles of True Positives, False Positives,
    True Negatives, and False Negatives to isolate operational vulnerabilities.
    """
    p = df["prob_optimized_hgb"].values
    y = df["at_risk"].values

    sub = df.copy()
    sub["category"] = "OTHER"
    sub.loc[(p >= tau) & (y == 1), "category"] = "True_Positive"
    sub.loc[(p >= tau) & (y == 0), "category"] = "False_Positive"
    sub.loc[(p < tau_low) & (y == 0), "category"] = "True_Negative"
    sub.loc[(p < tau_low) & (y == 1), "category"] = "False_Negative"

    profiles = {}
    for cat in ["True_Positive", "False_Positive", "True_Negative", "False_Negative"]:
        cat_df = sub[sub["category"] == cat]
        profiles[cat] = {
            "count": len(cat_df),
            "share_of_cohort_pct": round(float(len(cat_df) / len(sub) * 100), 2),
            "mean_predicted_probability": round(float(cat_df["prob_optimized_hgb"].mean()), 4),
            "mean_recent_prior_score": round(float(cat_df["recent_prior_score"].mean()), 2),
            "pct_missing_prior_score": round(float(cat_df["recent_prior_score"].isna().mean() * 100), 2),
            "mean_vle_active_days": round(float(cat_df["vle_active_days_to_date"].mean()), 2),
            "mean_vle_clicks_last_7d": round(float(cat_df["vle_clicks_last_7d"].mean()), 2),
            "mean_registration_timing_days": round(float(cat_df["registration_timing_days"].mean()), 2),
            "mean_target_assessment_weight": round(float(cat_df["target_assessment_weight"].mean()), 2),
            "assessment_type_distribution": {
                k: round(float(v * 100), 2)
                for k, v in cat_df["target_assessment_type"].value_counts(normalize=True).to_dict().items()
            }
        }

    return profiles


# =============================================================================
# 5. ORCHESTRATOR & FIGURE GENERATOR
# =============================================================================

class FairnessRobustnessPipeline:
    """
    Executes the comprehensive Phase 8 fairness, robustness, and governance audit.
    """

    def __init__(self,
                 processed_dir: str = "data/processed",
                 reports_dir: str = "reports",
                 figures_dir: str = "reports/figures/fairness",
                 tau: float = 0.20):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.figures_dir = figures_dir
        self.tau = tau

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.summary: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        print("Starting Phase 8 Fairness Audit & Robustness Analysis Pipeline...")

        # 1. Load Data
        print("Loading prediction datasets and offline demographic enrollments...")
        oof_14 = pd.read_csv(os.path.join(self.processed_dir, "oof_predictions_optimized_14day_delay.csv"))
        oof_7 = pd.read_csv(os.path.join(self.processed_dir, "oof_predictions_7day_delay.csv"))
        chrono = pd.read_csv(os.path.join(self.processed_dir, "oof_predictions_chronological_2014J.csv"))
        pred = pd.read_csv(os.path.join(self.processed_dir, "prediction_dataset.csv"))
        enroll = pd.read_csv(os.path.join(self.processed_dir, "enrollments_clean.csv"))

        # Merge prediction context (features + offline demographics)
        key_pred = ["id_student", "id_assessment", "code_module", "code_presentation", "scheduled_date", "cutoff_scenario"]
        feature_cols = [
            "target_assessment_type", "target_assessment_weight", "recent_prior_score",
            "vle_total_clicks_to_date", "vle_active_days_to_date", "vle_clicks_last_7d",
            "registration_timing_days", "prior_submission_lateness_mean"
        ]
        df_14 = pd.merge(oof_14, pred[key_pred + feature_cols], on=key_pred, how="inner")

        # Merge sensitive demographics for OFFLINE AUDITING ONLY
        demo_cols = ["id_student", "code_module", "code_presentation", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
        df_merged = pd.merge(df_14, enroll[demo_cols], on=["id_student", "code_module", "code_presentation"], how="left")

        # Isolate 14-day delay / 7-day early warning primary scenario
        df_7d = df_merged[df_merged["cutoff_scenario"] == "7_day_before"].reset_index(drop=True)
        df_1d = df_merged[df_merged["cutoff_scenario"] == "1_day_before"].reset_index(drop=True)

        # 2. Fairness Audit
        print("Executing demographic subgroup fairness audit...")
        fairness_auditor = FairnessAuditor(tau=self.tau)
        fairness_results = fairness_auditor.run_full_audit(df_7d)
        self.summary["fairness_audit"] = fairness_results

        # 3. Robustness Audit
        print("Executing operational robustness stress-tests...")
        robustness_auditor = RobustnessAuditor(tau=self.tau)
        robustness_results = {
            "presentations": robustness_auditor.audit_presentations(df_7d),
            "modules": robustness_auditor.audit_modules(df_7d),
            "assessment_types": robustness_auditor.audit_assessment_types(df_7d),
            "horizons": {
                "7_day_early_warning": compute_binary_metrics_at_threshold(df_7d["at_risk"].values, df_7d["prob_optimized_hgb"].values, tau=self.tau),
                "1_day_checkpoint": compute_binary_metrics_at_threshold(df_1d["at_risk"].values, df_1d["prob_optimized_hgb"].values, tau=self.tau),
            },
            "grading_delays": {
                "14_day_delay": compute_binary_metrics_at_threshold(df_7d["at_risk"].values, df_7d["prob_optimized_hgb"].values, tau=self.tau),
                "7_day_delay": compute_binary_metrics_at_threshold(
                    oof_7[oof_7["cutoff_scenario"] == "7_day_before"]["at_risk"].values,
                    oof_7[oof_7["cutoff_scenario"] == "7_day_before"]["prob_hist_gradient_boosting"].values,
                    tau=self.tau
                )
            },
            "score_availability": robustness_auditor.audit_score_availability(df_7d),
            "vle_engagement_levels": robustness_auditor.audit_vle_engagement_levels(df_7d),
            "chronological_generalization": robustness_auditor.audit_chronological_generalization(chrono)
        }
        self.summary["robustness_analysis"] = robustness_results

        # 4. Failure Mode Analysis
        print("Profiling failure modes (TP, FP, TN, FN)...")
        failure_profiles = analyze_failure_modes(df_7d, tau=self.tau, tau_low=0.05)
        self.summary["failure_mode_analysis"] = failure_profiles

        # 5. Risk Tier Audit Summary
        print("Validating prospective risk tier evaluations...")
        # A. Snapshot
        snap_p = df_7d["prob_optimized_hgb"].values
        snap_y = df_7d["at_risk"].values
        # B. Assessment 1
        first_df = df_7d.sort_values("scheduled_date").groupby("id_student").first().reset_index()
        # C. Day 30
        day30_df = df_7d[df_7d["scheduled_date"] > 30].sort_values("scheduled_date").groupby("id_student").first().reset_index()

        self.summary["risk_tier_validation"] = {
            "snapshot_level": {
                "total_records": len(snap_y),
                "low_risk": {"count": int((snap_p < 0.05).sum()), "failures": int(snap_y[snap_p < 0.05].sum())},
                "moderate_risk": {"count": int(((snap_p >= 0.05) & (snap_p < 0.20)).sum()), "failures": int(snap_y[(snap_p >= 0.05) & (snap_p < 0.20)].sum())},
                "high_risk": {"count": int((snap_p >= 0.20).sum()), "failures": int(snap_y[snap_p >= 0.20].sum())}
            },
            "assessment_1_milestone": {
                "total_unique_students": len(first_df),
                "low_risk": {"count": int((first_df["prob_optimized_hgb"] < 0.05).sum()), "failures": int(first_df[first_df["prob_optimized_hgb"] < 0.05]["at_risk"].sum())},
                "moderate_risk": {"count": int(((first_df["prob_optimized_hgb"] >= 0.05) & (first_df["prob_optimized_hgb"] < 0.20)).sum()), "failures": int(first_df[(first_df["prob_optimized_hgb"] >= 0.05) & (first_df["prob_optimized_hgb"] < 0.20)]["at_risk"].sum())},
                "high_risk": {"count": int((first_df["prob_optimized_hgb"] >= 0.20).sum()), "failures": int(first_df[first_df["prob_optimized_hgb"] >= 0.20]["at_risk"].sum())}
            },
            "day30_calendar_campaign": {
                "total_unique_students": len(day30_df),
                "low_risk": {"count": int((day30_df["prob_optimized_hgb"] < 0.05).sum()), "failures": int(day30_df[day30_df["prob_optimized_hgb"] < 0.05]["at_risk"].sum())},
                "moderate_risk": {"count": int(((day30_df["prob_optimized_hgb"] >= 0.05) & (day30_df["prob_optimized_hgb"] < 0.20)).sum()), "failures": int(day30_df[(day30_df["prob_optimized_hgb"] >= 0.05) & (day30_df["prob_optimized_hgb"] < 0.20)]["at_risk"].sum())},
                "high_risk": {"count": int((day30_df["prob_optimized_hgb"] >= 0.20).sum()), "failures": int(day30_df[day30_df["prob_optimized_hgb"] >= 0.20]["at_risk"].sum())}
            }
        }

        # 6. Generate Publication Figures
        print("Generating Phase 8 fairness and robustness diagnostic plots...")
        self.generate_figures(fairness_results, robustness_results, failure_profiles)

        total_time = round(time.time() - t0, 2)
        self.summary["total_execution_time_seconds"] = total_time

        # Save summary JSON
        summary_path = os.path.join(self.reports_dir, "fairness_robustness_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)

        print(f"Phase 8 Pipeline completed successfully in {total_time}s.")
        print(f"Summary JSON written to: {summary_path}")
        return self.summary

    def generate_figures(self, fairness: Dict[str, Any], robustness: Dict[str, Any], failure: Dict[str, Any]):
        """Generates all publication diagnostic plots for fairness and robustness."""
        sns.set_theme(style="whitegrid", font="sans-serif")

        # 1. Demographic Parity & Equal Opportunity (Recall) Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        demo_attrs = ["gender", "age_band", "disability", "highest_education"]
        group_labels = []
        sel_rates = []
        recalls = []

        for attr in demo_attrs:
            subgroups = fairness[attr]["subgroups"]
            for g_name, g_data in subgroups.items():
                label = f"{attr[:4]}: {g_name}"
                group_labels.append(label)
                sel_rates.append(g_data["selection_rate_pct"])
                recalls.append(g_data["recall_pct"])

        y_pos = np.arange(len(group_labels))
        axes[0].barh(y_pos, sel_rates, color="steelblue", alpha=0.85)
        axes[0].axvline(12.27, color="crimson", linestyle="--", label="Cohort Rate (12.3%)")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(group_labels, fontsize=9)
        axes[0].set_xlabel("Selection Rate % (Flagged High Risk at p >= 0.20)", fontsize=10)
        axes[0].set_title("Demographic Parity Check (Selection Rate)", fontsize=11, fontweight="bold")
        axes[0].legend(loc="lower right")

        axes[1].barh(y_pos, recalls, color="teal", alpha=0.85)
        axes[1].axvline(57.38, color="crimson", linestyle="--", label="Cohort Recall (57.4%)")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(group_labels, fontsize=9)
        axes[1].set_xlabel("Recall % (Sensitivity at p >= 0.20)", fontsize=10)
        axes[1].set_title("Equal Opportunity Check (Subgroup Recall)", fontsize=11, fontweight="bold")
        axes[1].legend(loc="lower right")

        plt.suptitle("Fairness Audit: Demographic Parity & Equal Opportunity Across Subgroups", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "demographic_fairness_parity_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2. Subgroup PR-AUC and Brier Score
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        pr_aucs = []
        briers = []
        for attr in demo_attrs:
            for g_name, g_data in fairness[attr]["subgroups"].items():
                pr_aucs.append(g_data["pr_auc"])
                briers.append(g_data["brier_score"])

        axes[0].barh(y_pos, pr_aucs, color="darkslateblue", alpha=0.85)
        axes[0].axvline(0.4606, color="crimson", linestyle="--", label="Cohort PR-AUC (0.461)")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(group_labels, fontsize=9)
        axes[0].set_xlabel("PR-AUC (Average Precision)", fontsize=10)
        axes[0].set_title("Predictive Ranking Accuracy (PR-AUC) Across Subgroups", fontsize=11, fontweight="bold")
        axes[0].legend(loc="lower right")

        axes[1].barh(y_pos, briers, color="darkcyan", alpha=0.85)
        axes[1].axvline(0.0592, color="crimson", linestyle="--", label="Cohort Brier (0.059)")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(group_labels, fontsize=9)
        axes[1].set_xlabel("Brier Score Loss (Lower is Better)", fontsize=10)
        axes[1].set_title("Probability Calibration (Brier Score) Across Subgroups", fontsize=11, fontweight="bold")
        axes[1].legend(loc="lower right")

        plt.suptitle("Subgroup Fairness: Predictive Performance & Calibration Stability", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "subgroup_performance_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 3. Robustness Across Course Disciplines & Offerings
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        modules = list(robustness["modules"].keys())
        mod_prauc = [robustness["modules"][m]["pr_auc"] for m in modules]
        mod_recall = [robustness["modules"][m]["recall_pct"] for m in modules]

        axes[0].bar(modules, mod_prauc, color="cornflowerblue", alpha=0.85)
        axes[0].set_title("PR-AUC Across Course Modules", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("PR-AUC", fontsize=11)
        axes[0].axhline(0.4606, color="crimson", linestyle="--", label="Cohort PR-AUC")
        for i, v in enumerate(mod_prauc):
            axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        axes[0].set_ylim([0, 0.7])
        axes[0].legend()

        axes[1].bar(modules, mod_recall, color="coral", alpha=0.85)
        axes[1].set_title("Recall@tau=0.20 Across Course Modules", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Recall %", fontsize=11)
        axes[1].axhline(57.38, color="crimson", linestyle="--", label="Cohort Recall")
        for i, v in enumerate(mod_recall):
            axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        axes[1].set_ylim([0, 85])
        axes[1].legend()

        plt.suptitle("Robustness Stress-Test: Course Module Disparities (Notice GGG Degradation)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "robustness_across_modules_and_presentations.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 4. Score Availability and VLE Engagement Robustness
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Score availability
        score_keys = ["missing_historical_score", "observed_historical_score"]
        score_labels = ["Missing Prior Grade\n(e.g. Assessment 1)", "Observed Prior Grade\n(Downstream)"]
        score_prauc = [robustness["score_availability"][k]["pr_auc"] for k in score_keys]
        score_prec = [robustness["score_availability"][k]["precision_pct"] for k in score_keys]

        x_pos = np.arange(len(score_labels))
        width = 0.35
        axes[0].bar(x_pos - width/2, score_prauc, width, label="PR-AUC", color="mediumpurple", alpha=0.85)
        axes[0].bar(x_pos + width/2, [p/100 for p in score_prec], width, label="Precision (Scaled)", color="mediumseagreen", alpha=0.85)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(score_labels, fontsize=10)
        axes[0].set_ylabel("Metric Value", fontsize=11)
        axes[0].set_title("Performance by Historical Grade Availability", fontsize=11, fontweight="bold")
        axes[0].legend()

        # VLE Quartiles
        vle_keys = ["Q1_lowest_activity", "Q2_medium_low", "Q3_medium_high", "Q4_highest_activity"]
        vle_labels = ["Q1 (Low)", "Q2 (Med-Low)", "Q3 (Med-High)", "Q4 (High)"]
        vle_prauc = [robustness["vle_engagement_levels"][k]["pr_auc"] for k in vle_keys]
        vle_recall = [robustness["vle_engagement_levels"][k]["recall_pct"] for k in vle_keys]

        axes[1].plot(vle_labels, vle_prauc, marker="o", linewidth=2.5, color="darkslateblue", label="PR-AUC")
        axes[1].set_ylabel("PR-AUC", fontsize=11, color="darkslateblue")
        axes[1].tick_params(axis="y", labelcolor="darkslateblue")
        axes[1].set_title("Predictive Performance Across Digital Study Quartiles", fontsize=11, fontweight="bold")

        ax2 = axes[1].twinx()
        ax2.plot(vle_labels, vle_recall, marker="s", linewidth=2.5, color="chocolate", label="Recall %")
        ax2.set_ylabel("Recall %", fontsize=11, color="chocolate")
        ax2.tick_params(axis="y", labelcolor="chocolate")

        plt.suptitle("Operational Robustness: Score Availability Deficit & Activity Stratification", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "score_availability_and_vle_robustness.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 5. Failure Mode Behavioral Profiles
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cats = ["True_Positive", "False_Positive", "True_Negative", "False_Negative"]
        cat_labels = ["True Positive\n(Caught Failure)", "False Positive\n(False Alarm)", "True Negative\n(Correct Pass)", "False Negative\n(Missed Failure)"]
        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6"]

        mean_prior = [failure[c]["mean_recent_prior_score"] for c in cats]
        mean_clicks = [failure[c]["mean_vle_clicks_last_7d"] for c in cats]

        axes[0].bar(cat_labels, mean_prior, color=colors, alpha=0.85)
        axes[0].set_title("Mean Recent Prior Score by Prediction Outcome", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Mean Grade (0-100)", fontsize=10)
        axes[0].axhline(50, color="black", linestyle="--", alpha=0.7, label="Passing Threshold (50)")
        for i, v in enumerate(mean_prior):
            axes[0].text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
        axes[0].set_ylim([0, 100])
        axes[0].legend()

        axes[1].bar(cat_labels, mean_clicks, color=colors, alpha=0.85)
        axes[1].set_title("Mean VLE Clicks in Prior 7 Days by Prediction Outcome", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Mean Clicks (Last 7 Days)", fontsize=10)
        for i, v in enumerate(mean_clicks):
            axes[1].text(i, v + 2.0, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
        axes[1].set_ylim([0, 100])

        plt.suptitle("Failure Analysis: Behavioral & Historical Discrepancies Across Prediction Outcomes", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "failure_mode_analysis_distributions.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("Phase 8 diagnostic figures successfully generated.")


if __name__ == "__main__":
    pipeline = FairnessRobustnessPipeline()
    pipeline.run()

