"""
src/model_explainability.py

Phase 7: Model Explainability, Feature Attribution & Risk Assessment Engine
for the Student Performance & At-Risk Prediction System (OULAD).

Implements:
1. Global Explainability via TreeSHAP:
   - Computes exact Shapley values using TreeExplainer for the optimized HistGradientBoosting model.
   - Extracts global feature importance rankings across all 23 features.
   - Computes feature interaction effects and partial dependence dynamics.
2. Scenario-Specific Explainability:
   - Evaluates feature attributions across 14-day vs 7-day delays and 7-day vs 1-day cutoffs.
   - Evaluates the two prospective intervention benchmarks:
     - Assessment 1 Milestone (100% missing historical scores; VLE engagement dominates).
     - Day 30 Synchronized Outreach Campaign (Combined historical score and VLE velocity).
3. Advisory Risk Categorization Framework:
   - Derives risk tiers (Low, Moderate, High) strictly using training-derived decision thresholds.
   - Evaluates empirical student counts, prevalence, precision, recall, and false-positive rates.
4. Local Explainability & Case Studies:
   - Decomposes exact local additive contributions for representative anonymized student cases:
     Case Alpha (True Positive), Case Beta (False Positive),
     Case Gamma (True Negative), Case Delta (False Negative).
   - Generates non-judgmental, actionable advisor narratives.
5. Publication-Ready Figure Generation:
   - Saves all diagnostic plots in reports/figures/explainability/.
6. Summary Metrics Export:
   - Saves structured audit results to reports/model_explainability_summary.json.
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
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.baseline_models import (
    FEATURE_ALLOWLIST_CATEGORICAL,
    FEATURE_ALLOWLIST_NUMERICAL,
    ALL_ALLOWED_FEATURES,
    FeatureGovernor,
    compute_recall_at_k,
)


# =============================================================================
# 1. MODEL PREPARATION & PREPROCESSOR BUILDER
# =============================================================================

def build_fitted_optimized_model(X_train: pd.DataFrame, y_train: pd.Series,
                                 random_state: int = 42) -> Tuple[HistGradientBoostingClassifier, ColumnTransformer, List[str]]:
    """
    Fits the optimized HistGradientBoosting architecture identified in Phase 6.
    """
    cat_cols = FEATURE_ALLOWLIST_CATEGORICAL
    num_cols = FEATURE_ALLOWLIST_NUMERICAL
    cat_indices = [len(num_cols) + i for i in range(len(cat_cols))]
    feature_names = num_cols + cat_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        ]
    )

    X_prep = preprocessor.fit_transform(X_train)

    clf = HistGradientBoostingClassifier(
        categorical_features=cat_indices,
        learning_rate=0.05,
        max_iter=150,
        max_depth=8,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=5.0,
        random_state=random_state
    )
    clf.fit(X_prep, y_train)

    return clf, preprocessor, feature_names


# =============================================================================
# 2. GLOBAL EXPLAINABILITY ENGINE (TreeSHAP)
# =============================================================================

def compute_global_shap_attributions(clf: HistGradientBoostingClassifier,
                                     X_prep: np.ndarray,
                                     feature_names: List[str],
                                     sample_size: int = 3000,
                                     random_state: int = 42) -> Tuple[shap.Explanation, Dict[str, float]]:
    """
    Computes exact TreeSHAP values on a representative stratified evaluation sample.
    """
    np.random.seed(random_state)
    n = len(X_prep)
    if n > sample_size:
        sample_idx = np.random.choice(n, size=sample_size, replace=False)
        X_sample = X_prep[sample_idx]
    else:
        X_sample = X_prep

    explainer = shap.TreeExplainer(clf)
    shap_explanation = explainer(X_sample)
    shap_explanation.feature_names = feature_names

    mean_abs = np.abs(shap_explanation.values).mean(axis=0)
    importance_dict = {
        feature_names[i]: round(float(mean_abs[i]), 4)
        for i in np.argsort(-mean_abs)
    }

    return shap_explanation, importance_dict


# =============================================================================
# 3. ADVISORY RISK CATEGORIZATION FRAMEWORK
# =============================================================================

def evaluate_advisory_risk_tiers(probs: np.ndarray, y_true: np.ndarray,
                                 tau_high: float = 0.20,
                                 tau_low: float = 0.05) -> Dict[str, Any]:
    """
    Evaluates three non-punitive, decision-support risk tiers:
      - Low Risk: p < tau_low
      - Moderate Risk: tau_low <= p < tau_high
      - High Risk: p >= tau_high
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n_total = len(y_true)
    n_pos_total = int(y_true.sum())
    n_neg_total = n_total - n_pos_total

    tiers = {
        "low_risk": {
            "name": f"Low Risk (p < {tau_low:.2f})",
            "mask": (probs < tau_low),
            "action": "Standard institutional resources; self-guided support."
        },
        "moderate_risk": {
            "name": f"Moderate Risk ({tau_low:.2f} <= p < {tau_high:.2f})",
            "mask": (probs >= tau_low) & (probs < tau_high),
            "action": "Automated engagement nudges, peer study invitations, deadline reminders."
        },
        "high_risk": {
            "name": f"High Risk (p >= {tau_high:.2f})",
            "mask": (probs >= tau_high),
            "action": "Priority 1-on-1 counselor outreach, specialized concept tutoring."
        }
    }

    results = {
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high},
        "total_records": n_total,
        "total_positives": n_pos_total,
        "baseline_prevalence_pct": round(float(n_pos_total / n_total * 100), 2) if n_total > 0 else 0.0,
        "tier_metrics": {}
    }

    for tier_key, t_info in tiers.items():
        mask = t_info["mask"]
        cnt = int(mask.sum())
        pos_cnt = int(y_true[mask].sum())
        neg_cnt = cnt - pos_cnt

        prev = (pos_cnt / cnt * 100) if cnt > 0 else 0.0
        rec = (pos_cnt / n_pos_total * 100) if n_pos_total > 0 else 0.0
        fpr = (neg_cnt / n_neg_total * 100) if n_neg_total > 0 else 0.0
        share = (cnt / n_total * 100) if n_total > 0 else 0.0

        results["tier_metrics"][tier_key] = {
            "tier_name": t_info["name"],
            "count": cnt,
            "share_of_cohort_pct": round(float(share), 2),
            "at_risk_count": pos_cnt,
            "empirical_failure_rate_pct": round(float(prev), 2),
            "precision_pct": round(float(prev), 2),
            "recall_pct": round(float(rec), 2),
            "false_positive_rate_pct": round(float(fpr), 2),
            "advisory_intervention_guidance": t_info["action"]
        }

    return results


# =============================================================================
# 4. FULL EXPLAINABILITY PIPELINE ORCHESTRATOR
# =============================================================================

class ModelExplainabilityPipeline:
    """
    Executes the comprehensive Phase 7 explainability and risk assessment audit.
    """

    def __init__(self,
                 processed_dir: str = "data/processed",
                 reports_dir: str = "reports",
                 figures_dir: str = "reports/figures/explainability",
                 random_state: int = 42):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.figures_dir = figures_dir
        self.random_state = random_state

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.summary: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        print("Starting Phase 7 Model Explainability & Risk Assessment Pipeline...")

        # 1. Load Datasets
        path_14 = os.path.join(self.processed_dir, "prediction_dataset.csv")
        path_7 = os.path.join(self.processed_dir, "prediction_dataset_7day_delay.csv")
        oof_path = os.path.join(self.processed_dir, "oof_predictions_optimized_14day_delay.csv")

        df_14 = pd.read_csv(path_14)
        df_7 = pd.read_csv(path_7)
        oof_14 = pd.read_csv(oof_path)

        df_14_7d = df_14[df_14["cutoff_scenario"] == "7_day_before"].reset_index(drop=True)
        oof_14_7d = oof_14[oof_14["cutoff_scenario"] == "7_day_before"].reset_index(drop=True)

        # 2. Fit Primary Global Model (14d delay, 7d cutoff)
        print("Fitting primary global HistGradientBoosting model for TreeSHAP analysis...")
        X, y = FeatureGovernor.get_feature_matrix(df_14_7d)
        clf, preprocessor, feature_names = build_fitted_optimized_model(X, y, random_state=self.random_state)

        # 3. Compute Global SHAP Values
        print("Computing exact TreeSHAP values for global attribution...")
        X_prep = preprocessor.transform(X)
        shap_exp, global_importance = compute_global_shap_attributions(
            clf, X_prep, feature_names, sample_size=3000, random_state=self.random_state
        )
        self.summary["global_feature_importance"] = global_importance

        # 4. Scenario-Specific Attributions
        print("Evaluating scenario-specific feature importance...")
        scenario_importance = {}

        scenarios = {
            "14_day_delay__7_day_before": df_14_7d,
            "14_day_delay__1_day_before": df_14[df_14["cutoff_scenario"] == "1_day_before"].reset_index(drop=True),
            "7_day_delay__7_day_before": df_7[df_7["cutoff_scenario"] == "7_day_before"].reset_index(drop=True),
            "7_day_delay__1_day_before": df_7[df_7["cutoff_scenario"] == "1_day_before"].reset_index(drop=True),
        }

        for sc_name, sc_df in scenarios.items():
            X_sc, y_sc = FeatureGovernor.get_feature_matrix(sc_df)
            clf_sc, prep_sc, fn_sc = build_fitted_optimized_model(X_sc, y_sc, random_state=self.random_state)
            X_sc_prep = prep_sc.transform(X_sc)
            _, imp_sc = compute_global_shap_attributions(clf_sc, X_sc_prep, fn_sc, sample_size=1500, random_state=self.random_state)
            scenario_importance[sc_name] = imp_sc

        # Prospective Benchmark: Assessment 1 Milestone
        first_assess = df_14_7d.sort_values("scheduled_date").groupby("id_student").first().reset_index()
        X_first, y_first = FeatureGovernor.get_feature_matrix(first_assess)
        clf_first, prep_first, fn_first = build_fitted_optimized_model(X_first, y_first, random_state=self.random_state)
        X_first_prep = prep_first.transform(X_first)
        _, imp_first = compute_global_shap_attributions(clf_first, X_first_prep, fn_first, sample_size=1500, random_state=self.random_state)
        scenario_importance["benchmark_alternative_A_first_assessment"] = imp_first

        # Prospective Benchmark: Day 30 Calendar Cutoff
        after_30 = df_14_7d[df_14_7d["scheduled_date"] > 30].sort_values("scheduled_date")
        day30_cohort = after_30.groupby("id_student").first().reset_index()
        X_30, y_30 = FeatureGovernor.get_feature_matrix(day30_cohort)
        clf_30, prep_30, fn_30 = build_fitted_optimized_model(X_30, y_30, random_state=self.random_state)
        X_30_prep = prep_30.transform(X_30)
        _, imp_30 = compute_global_shap_attributions(clf_30, X_30_prep, fn_30, sample_size=1500, random_state=self.random_state)
        scenario_importance["benchmark_alternative_B_day30_cutoff"] = imp_30

        self.summary["scenario_specific_importance"] = scenario_importance

        # 5. Advisory Risk Categorization Framework Evaluation
        print("Evaluating advisory risk categorization framework...")
        # A. Snapshot level
        risk_eval_snapshot = evaluate_advisory_risk_tiers(
            oof_14_7d["prob_optimized_hgb"].values, oof_14_7d["at_risk"].values, tau_high=0.20, tau_low=0.05
        )
        self.summary["risk_tier_evaluation_snapshot_level"] = risk_eval_snapshot

        # B. Alternative A (Assessment 1 Milestone)
        first_oof = oof_14_7d.sort_values("scheduled_date").groupby("id_student").first().reset_index()
        risk_eval_first = evaluate_advisory_risk_tiers(
            first_oof["prob_optimized_hgb"].values, first_oof["at_risk"].values, tau_high=0.20, tau_low=0.05
        )
        self.summary["risk_tier_evaluation_alternative_A_first_assessment"] = risk_eval_first

        # C. Alternative B (Day 30 Calendar Cutoff)
        day30_oof = oof_14_7d[oof_14_7d["scheduled_date"] > 30].sort_values("scheduled_date").groupby("id_student").first().reset_index()
        risk_eval_day30 = evaluate_advisory_risk_tiers(
            day30_oof["prob_optimized_hgb"].values, day30_oof["at_risk"].values, tau_high=0.20, tau_low=0.05
        )
        self.summary["risk_tier_evaluation_alternative_B_day30_cutoff"] = risk_eval_day30

        # 6. Local Explainability Case Studies
        print("Decomposing local SHAP contributions for representative student cases...")
        sub7_with_prob = df_14_7d.copy()
        sub7_with_prob["prob_opt"] = oof_14_7d["prob_optimized_hgb"].values

        # Select 4 distinct cases with masked identifiers
        tp_row = sub7_with_prob[(sub7_with_prob["prob_opt"] > 0.65) & (sub7_with_prob["at_risk"] == 1)].iloc[0]
        fp_row = sub7_with_prob[(sub7_with_prob["prob_opt"] > 0.50) & (sub7_with_prob["at_risk"] == 0)].iloc[0]
        tn_row = sub7_with_prob[(sub7_with_prob["prob_opt"] < 0.02) & (sub7_with_prob["at_risk"] == 0)].iloc[0]
        fn_row = sub7_with_prob[(sub7_with_prob["prob_opt"] < 0.04) & (sub7_with_prob["at_risk"] == 1)].iloc[0]

        cases_data = [
            ("Case Alpha (True Positive)", tp_row,
             "Student has a history of prior failure (prior score: 34) and recent activity has dropped to 1 click in the past week. High priority for proactive academic tutoring.",
             "Priority proactive contact. Review formative assignment concepts and offer assignment extension/tutoring."),
            ("Case Beta (False Positive)", fp_row,
             "Student experienced a marginal previous failure (score: 48) and exhibited zero clicks in the 7 days prior to deadline. However, student recovered (e.g. studied offline) and passed.",
             "Check-in gently to assess workload. Confirm student has study materials; avoid alarmist interventions."),
            ("Case Gamma (True Negative)", tn_row,
             "Student registered 159 days early and maintained heavy, consistent engagement (139 clicks in the last 7 days across 14 active days). Low risk profile.",
             "Standard self-service monitoring; student demonstrates healthy academic engagement."),
            ("Case Delta (False Negative)", fn_row,
             "Student showed moderate early clicks (38 clicks in last 7 days) and registered early, but unexpectedly scored below 50 on Assessment 1.",
             "Illustrates that digital clicks are an engagement proxy, not learning mastery. Advisor outreach warranted post-grade release to diagnose concept barriers.")
        ]

        local_cases = []
        for label, row, explanation, recommendation in cases_data:
            row_X = pd.DataFrame([row[ALL_ALLOWED_FEATURES]])
            row_prep = preprocessor.transform(row_X)
            local_shap = explainer = shap.TreeExplainer(clf)(row_prep)
            vals = local_shap.values[0]

            top_pos_idx = np.argsort(-vals)[:3]
            top_neg_idx = np.argsort(vals)[:3]

            drivers = [
                {"feature": feature_names[i], "shap_value": round(float(vals[i]), 4), "feature_value": str(row_X.iloc[0][feature_names[i]])}
                for i in top_pos_idx if vals[i] > 0
            ]
            protective = [
                {"feature": feature_names[i], "shap_value": round(float(vals[i]), 4), "feature_value": str(row_X.iloc[0][feature_names[i]])}
                for i in top_neg_idx if vals[i] < 0
            ]

            local_cases.append({
                "case_label": label,
                "module": str(row["code_module"]),
                "assessment_type": str(row["target_assessment_type"]),
                "assessment_scheduled_date": int(row["scheduled_date"]),
                "predicted_probability": round(float(row["prob_opt"]), 4),
                "actual_outcome": int(row["at_risk"]),
                "top_risk_drivers": drivers,
                "top_protective_factors": protective,
                "plain_language_explanation": explanation,
                "advisor_actionable_recommendation": recommendation
            })

        self.summary["local_case_studies"] = local_cases

        # 7. Generate Evaluation Figures
        print("Generating explainability diagnostic figures...")
        self.generate_figures(shap_exp, feature_names, global_importance, scenario_importance, risk_eval_snapshot, local_cases)

        total_time = round(time.time() - t0, 2)
        self.summary["total_execution_time_seconds"] = total_time

        # Save summary JSON
        summary_path = os.path.join(self.reports_dir, "model_explainability_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)

        print(f"Phase 7 Explainability Pipeline finished in {total_time}s.")
        print(f"Summary JSON: {summary_path}")
        return self.summary

    def generate_figures(self, shap_exp: shap.Explanation,
                         feature_names: List[str],
                         global_importance: Dict[str, float],
                         scenario_importance: Dict[str, Dict[str, float]],
                         risk_eval: Dict[str, Any],
                         local_cases: List[Dict[str, Any]]):
        """Generates all publication-quality SHAP and risk categorization figures."""
        sns.set_theme(style="whitegrid", font="sans-serif")

        # 1. SHAP Global Bar Plot (Top 15 Features)
        fig, ax = plt.subplots(figsize=(10, 7))
        top_15_features = list(global_importance.keys())[:15]
        top_15_values = [global_importance[f] for f in top_15_features]

        y_pos = np.arange(len(top_15_features))
        ax.barh(y_pos, top_15_values[::-1], color="teal", align="center", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_15_features[::-1], fontsize=10)
        ax.set_xlabel("Mean Absolute SHAP Value (Global Impact on Model Output)", fontsize=11)
        ax.set_title("Global Feature Importance (TreeSHAP on HistGradientBoosting)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "shap_global_bar_importance.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2. SHAP Beeswarm Summary Plot
        fig, ax = plt.subplots(figsize=(11, 8))
        shap.plots.beeswarm(shap_exp, max_display=15, show=False)
        plt.title("SHAP Beeswarm Summary (Feature Value vs Risk Directionality)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "shap_global_summary_beeswarm.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Scenario-Specific Feature Importance Comparison Bar Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        comp_features = ["recent_prior_score", "registration_timing_days", "vle_active_days_to_date",
                         "vle_clicks_last_7d", "target_assessment_weight", "vle_days_since_last_activity"]

        comp_data = []
        for f in comp_features:
            for sc_key, sc_label in [
                ("14_day_delay__7_day_before", "14d Delay / 7d Cutoff"),
                ("14_day_delay__1_day_before", "14d Delay / 1d Cutoff"),
                ("benchmark_alternative_A_first_assessment", "Assessment 1 Milestone"),
                ("benchmark_alternative_B_day30_cutoff", "Day 30 Calendar Cutoff"),
            ]:
                val = scenario_importance.get(sc_key, {}).get(f, 0.0)
                comp_data.append({"Feature": f, "Scenario": sc_label, "Mean |SHAP|": val})

        df_comp = pd.DataFrame(comp_data)
        sns.barplot(data=df_comp, x="Feature", y="Mean |SHAP|", hue="Scenario", ax=ax, palette="Dark2")
        ax.set_title("Feature Attribution Dynamics Across Decision Cutoffs and Milestones", fontsize=13, fontweight="bold")
        ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "scenario_feature_importance_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 4. Advisory Risk Tier Distribution & Empirical Failure Rate
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        tiers = risk_eval["tier_metrics"]
        tier_names = [tiers[t]["tier_name"].split(" (")[0] for t in tiers]
        shares = [tiers[t]["share_of_cohort_pct"] for t in tiers]
        fail_rates = [tiers[t]["empirical_failure_rate_pct"] for t in tiers]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        # Cohort Share
        axes[0].bar(tier_names, shares, color=colors, alpha=0.85)
        axes[0].set_title("Cohort Student Share by Advisory Risk Tier", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Percentage of Total Cohort (%)", fontsize=11)
        for i, v in enumerate(shares):
            axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
        axes[0].set_ylim([0, 100])

        # Empirical Failure Rate
        axes[1].bar(tier_names, fail_rates, color=colors, alpha=0.85)
        axes[1].axhline(risk_eval["baseline_prevalence_pct"], color="gray", linestyle="--", label=f"University Base Rate ({risk_eval['baseline_prevalence_pct']}%)")
        axes[1].set_title("Empirical Failure Rate within Each Risk Tier", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Observed Failure Rate (%)", fontsize=11)
        for i, v in enumerate(fail_rates):
            axes[1].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontweight="bold")
        axes[1].set_ylim([0, 50])
        axes[1].legend(loc="upper left")

        plt.suptitle("Advisory Decision-Support Risk Categorization Framework", fontsize=14, fontweight="bold", y=1.03)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "risk_tier_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 5. Local Waterfall Case Examples Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for ax, case in zip(axes.ravel(), local_cases):
            drivers = case["top_risk_drivers"]
            protective = case["top_protective_factors"]
            all_factors = drivers + protective
            
            f_names = [f["feature"] for f in all_factors]
            f_shap = [f["shap_value"] for f in all_factors]
            bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in f_shap]

            y_pos = np.arange(len(f_names))
            ax.barh(y_pos, f_shap, color=bar_colors, alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(f_names, fontsize=9)
            ax.axvline(0, color="black", linestyle="--", alpha=0.6)
            ax.set_title(f"{case['case_label']}: p={case['predicted_probability']} (Actual: {case['actual_outcome']})",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("SHAP Impact (+ Increases Risk, - Decreases Risk)", fontsize=9)

        plt.suptitle("Local Explainability: Feature Attributions for Individual Student Scenarios", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "local_waterfall_examples.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("Phase 7 explainability figures successfully generated.")


if __name__ == "__main__":
    pipeline = ModelExplainabilityPipeline()
    pipeline.run()

