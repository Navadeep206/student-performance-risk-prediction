"""
src/baseline_models.py

Phase 5: Baseline Model Development & Cross-Validation Strategy Pipeline
for the Student Performance & At-Risk Prediction System (OULAD).

Implements:
1. Feature Governance: Strict allowlist and prohibition enforcement to guarantee
   leakage-free modeling.
2. StratifiedGroupKFold Cross-Validation: 5 folds grouped by `id_student` with
   stratification by `at_risk`, ensuring complete student-level separation.
3. Scenario Isolation:
   - Evaluates 14-day grading delay (baseline) vs 7-day grading delay (alternative) separately.
   - Evaluates 7-day early warning horizon vs 1-day immediate checkpoint separately.
4. Baseline Models:
   - A. Majority-Class Classifier
   - B. Uniform Random Classifier (fixed seed)
   - C. Previous-Score Heuristic (with explicit coverage calculation)
   - D. Regularized Logistic Regression (StandardScaler + OneHotEncoder in Scikit-Learn Pipeline)
   - E. HistGradientBoostingClassifier (Tree Baseline with native missing & categorical handling)
   - F. Random Forest Classifier (Controlled depth tree ensemble)
5. Comprehensive Evaluation Metrics:
   - PR-AUC (Average Precision), ROC-AUC, Precision, Recall, F1 (at threshold 0.5)
   - Train-tuned threshold evaluation (optimal F1 threshold chosen strictly on training fold)
   - Recall@10% and Recall@20% with exact tie apportionment
   - Confusion matrix (tn, fp, fn, tp)
   - Fold-wise metrics, mean, and standard deviation
6. Subgroup Robustness Audits:
   - Performance breakdown by Course Module (AAA - GGG)
   - Performance breakdown by Assessment Type (TMA, CMA, Exam)
7. Out-of-Fold (OOF) Prediction Export & Evaluation Plot Generation.
"""

import os
import time
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# =============================================================================
# 1. FEATURE GOVERNANCE & ALLOWLIST
# =============================================================================

FEATURE_ALLOWLIST_CATEGORICAL = [
    "code_module",
    "code_presentation",
    "target_assessment_type",
]

FEATURE_ALLOWLIST_NUMERICAL = [
    "target_assessment_weight",
    "scheduled_date",
    "num_previous_scheduled_assessments",
    "registration_timing_days",
    "vle_total_clicks_to_date",
    "vle_active_days_to_date",
    "vle_core_clicks_to_date",
    "vle_forum_clicks_to_date",
    "vle_quiz_clicks_to_date",
    "vle_days_since_last_activity",
    "vle_clicks_last_7d",
    "vle_clicks_last_14d",
    "vle_activity_trend_ratio",
    "num_prior_submissions",
    "prior_submission_lateness_mean",
    "num_prior_scored_assessments",
    "mean_prior_score",
    "std_prior_score",
    "prior_at_risk_rate",
    "recent_prior_score",
]

ALL_ALLOWED_FEATURES = FEATURE_ALLOWLIST_CATEGORICAL + FEATURE_ALLOWLIST_NUMERICAL

PROHIBITED_COLUMNS = [
    "id_student",
    "id_assessment",
    "target_assessment_score",
    "next_assessment_score",
    "at_risk",
    "final_result",
    "date_unregistration",
    "cutoff_scenario",
    "grading_delay_scenario",
    "cutoff_date",
    "days_until_assessment",
    # Sensitive demographics
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]


class FeatureGovernor:
    """Enforces strict feature governance and auditable allowlists."""

    @staticmethod
    def audit_features(df: pd.DataFrame) -> Dict[str, Any]:
        """Audits columns present in dataframe against allowed and prohibited lists."""
        present_cols = list(df.columns)
        included_features = [col for col in ALL_ALLOWED_FEATURES if col in present_cols]
        missing_allowed = [col for col in ALL_ALLOWED_FEATURES if col not in present_cols]
        prohibited_in_features = [col for col in present_cols if col in PROHIBITED_COLUMNS]

        audit_report = {
            "total_columns_present": len(present_cols),
            "allowed_features_found": len(included_features),
            "allowed_categorical_features": [c for c in FEATURE_ALLOWLIST_CATEGORICAL if c in present_cols],
            "allowed_numerical_features": [c for c in FEATURE_ALLOWLIST_NUMERICAL if c in present_cols],
            "missing_allowed_features": missing_allowed,
            "prohibited_columns_present": prohibited_in_features,
            "is_valid": len(missing_allowed) == 0 and "at_risk" in present_cols and "id_student" in present_cols,
        }
        return audit_report

    @staticmethod
    def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extracts strictly allowed feature matrix X and target y."""
        missing = [c for c in ALL_ALLOWED_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required allowed features: {missing}")
        if "at_risk" not in df.columns:
            raise ValueError("Target column 'at_risk' not found in dataset.")

        X = df[ALL_ALLOWED_FEATURES].copy()
        y = df["at_risk"].astype(int).copy()
        return X, y


# =============================================================================
# 2. METRIC CALCULATION UTILITIES
# =============================================================================

def compute_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """
    Computes Recall@k with exact tie apportionment.
    
    Ranks instances descending by risk probability. For a budget of top m = ceil(k * N)
    instances, computes the exact fraction of total positive cases captured.
    If ties span across the rank m threshold boundary, apportion the tied positive
    cases proportionally to avoid arbitrary ordering bias.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    top_m = int(np.ceil(k * n))
    if top_m >= n:
        return 1.0

    # Stable sort descending
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_y = y_true[order]
    sorted_p = y_prob[order]

    threshold_p = sorted_p[top_m - 1]

    strictly_above = sorted_p > threshold_p
    at_threshold = sorted_p == threshold_p

    n_strictly_above = int(np.sum(strictly_above))
    n_at_threshold = int(np.sum(at_threshold))

    pos_strictly_above = int(np.sum(sorted_y[strictly_above]))
    pos_at_threshold = int(np.sum(sorted_y[at_threshold]))

    if n_at_threshold == 0:
        total_pos = pos_strictly_above
    else:
        needed_from_threshold = top_m - n_strictly_above
        fraction = needed_from_threshold / n_at_threshold
        expected_pos_from_threshold = pos_at_threshold * fraction
        total_pos = pos_strictly_above + expected_pos_from_threshold

    return float(total_pos / n_pos)


def optimize_threshold_on_train(y_train: np.ndarray, y_train_prob: np.ndarray) -> float:
    """
    Identifies optimal F1 decision threshold using training-fold predictions only.
    Guarantees zero threshold leakage onto validation data.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.50
    best_f1 = -1.0

    for thresh in thresholds:
        preds = (y_train_prob >= thresh).astype(int)
        score = f1_score(y_train, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return float(best_thresh)


def calculate_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = 0.50) -> Dict[str, Any]:
    """Computes a standard evaluation dictionary for predicted probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    # Average precision
    ap = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else 0.0

    # ROC-AUC
    try:
        roc = float(roc_auc_score(y_true, y_prob)) if (n_pos > 0 and n_neg > 0) else 0.5
    except ValueError:
        roc = 0.5

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    rec_10 = compute_recall_at_k(y_true, y_prob, k=0.10)
    rec_20 = compute_recall_at_k(y_true, y_prob, k=0.20)

    return {
        "threshold": round(float(threshold), 4),
        "pr_auc": round(ap, 4),
        "roc_auc": round(roc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "recall_at_10": round(rec_10, 4),
        "recall_at_20": round(rec_20, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# =============================================================================
# 3. BASELINE MODEL IMPLEMENTATIONS
# =============================================================================

class MajorityClassifierBaseline(BaseEstimator, ClassifierMixin):
    """Majority-class heuristic baseline: predicts the negative class (at_risk=0)."""

    def __init__(self):
        self.train_prevalence_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityClassifierBaseline":
        self.train_prevalence_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = 1.0 - self.train_prevalence_
        probs[:, 1] = self.train_prevalence_
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


class UniformRandomClassifierBaseline(BaseEstimator, ClassifierMixin):
    """Deterministic uniform random classifier baseline."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UniformRandomClassifierBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        p = rng.uniform(0.0, 1.0, size=len(X))
        probs = np.column_stack([1.0 - p, p])
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class PreviousScoreHeuristicBaseline(BaseEstimator, ClassifierMixin):
    """
    Previous-score domain heuristic.
    
    Predicts at-risk when the most recent historically available assessment score is below 50.
    Under strict OULAD grading delay conditions, when a prior score is missing (e.g. Assessment 1),
    it assigns non-risk (prob 0.0), explicitly avoiding treating missing prior scores as failures.
    Tracks and reports coverage proportion.
    """

    def __init__(self):
        self.coverage_: float = 0.0
        self.train_prevalence_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PreviousScoreHeuristicBaseline":
        self.train_prevalence_ = float(np.mean(y))
        scores = X["recent_prior_score"]
        self.coverage_ = float(scores.notna().mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = X["recent_prior_score"].values
        n = len(scores)
        risk_probs = np.zeros(n, dtype=float)

        valid_mask = ~np.isnan(scores)
        # Higher risk when score is lower; score < 50 => prob > 0.50
        risk_probs[valid_mask] = np.clip((100.0 - scores[valid_mask]) / 100.0, 0.0, 1.0)
        # For missing scores, assign 0.0 (do not treat missing historical score as failure)
        risk_probs[~valid_mask] = 0.0

        probs = np.column_stack([1.0 - risk_probs, risk_probs])
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = X["recent_prior_score"].values
        preds = np.zeros(len(scores), dtype=int)
        valid_mask = ~np.isnan(scores)
        preds[valid_mask] = (scores[valid_mask] < 50.0).astype(int)
        return preds

    def get_coverage(self, X: pd.DataFrame) -> float:
        return float(X["recent_prior_score"].notna().mean())


def build_logistic_regression_pipeline(cat_cols: List[str], num_cols: List[str],
                                       random_state: int = 42) -> Pipeline:
    """Constructs regularized Logistic Regression pipeline with strict train-only ColumnTransformer."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ))
    ])
    return pipeline


def build_hist_gradient_boosting_pipeline(cat_cols: List[str], num_cols: List[str],
                                          random_state: int = 42) -> Pipeline:
    """
    Constructs HistGradientBoostingClassifier pipeline with native support for
    missing values and categorical features.
    """
    # OrdinalEncoder for categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ), cat_cols)
        ]
    )

    cat_indices = [len(num_cols) + i for i in range(len(cat_cols))]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            categorical_features=cat_indices,
            max_depth=6,
            max_iter=100,
            learning_rate=0.1,
            random_state=random_state
        ))
    ])
    return pipeline


def build_random_forest_pipeline(cat_cols: List[str], num_cols: List[str],
                                 random_state: int = 42) -> Pipeline:
    """Constructs controlled-depth Random Forest baseline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline


# =============================================================================
# 4. CROSS-VALIDATION PIPELINE & EVALUATION ENGINE
# =============================================================================

class BaselineEvaluationPipeline:
    """
    Coordinates leak-free 5-fold cross-validation and scenario-isolated
    evaluation of baseline models.
    """

    def __init__(self,
                 processed_dir: str = "data/processed",
                 reports_dir: str = "reports",
                 figures_dir: str = "reports/figures/baseline",
                 n_splits: int = 5,
                 random_state: int = 42):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.figures_dir = figures_dir
        self.n_splits = n_splits
        self.random_state = random_state

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.summary_results: Dict[str, Any] = {}

    def run_cv_for_scenario(self, df_scenario: pd.DataFrame,
                            scenario_name: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Executes leak-free 5-fold StratifiedGroupKFold CV for a single scenario.
        Returns aggregate metrics and out-of-fold prediction dataframe.
        """
        print(f"\n=======================================================")
        print(f"Running Cross-Validation for Scenario: {scenario_name}")
        print(f"Dataset Size: {len(df_scenario):,} rows | Unique Students: {df_scenario['id_student'].nunique():,}")
        print(f"Overall Class Imbalance: {df_scenario['at_risk'].mean()*100:.2f}% positive")
        print(f"=======================================================")

        # Feature extraction
        X, y = FeatureGovernor.get_feature_matrix(df_scenario)
        groups = df_scenario["id_student"].values

        # StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                    random_state=self.random_state)

        # Initialize OOF prediction store
        oof_df = df_scenario[["id_student", "id_assessment", "code_module",
                              "code_presentation", "scheduled_date", "at_risk"]].copy()
        oof_df["fold"] = -1
        oof_df["cutoff_scenario"] = df_scenario["cutoff_scenario"].iloc[0]
        oof_df["grading_delay_scenario"] = df_scenario["grading_delay_scenario"].iloc[0]

        model_names = [
            "majority",
            "random",
            "previous_score",
            "logistic_regression",
            "hist_gradient_boosting",
            "random_forest",
        ]

        for m_name in model_names:
            oof_df[f"prob_{m_name}"] = np.nan

        fold_metrics: Dict[str, List[Dict[str, Any]]] = {m: [] for m in model_names}
        fold_metrics_tuned: Dict[str, List[Dict[str, Any]]] = {
            m: [] for m in ["logistic_regression", "hist_gradient_boosting", "random_forest"]
        }
        fold_audit: List[Dict[str, Any]] = []

        splits = list(sgkf.split(X, y, groups=groups))

        for fold, (train_idx, val_idx) in enumerate(splits):
            t0 = time.time()
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            train_students = set(groups[train_idx])
            val_students = set(groups[val_idx])
            overlap = train_students.intersection(val_students)

            # Programmatic verification
            assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} students shared in fold {fold}!"
            assert len(np.unique(y_val)) == 2, f"Fold {fold} missing classes!"

            oof_df.loc[val_idx, "fold"] = fold

            fold_info = {
                "fold": fold,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_train_students": len(train_students),
                "n_val_students": len(val_students),
                "val_positives": int(y_val.sum()),
                "val_at_risk_pct": round(float(y_val.mean() * 100), 2),
                "student_overlap": len(overlap)
            }
            fold_audit.append(fold_info)

            # Instantiate models
            models = {
                "majority": MajorityClassifierBaseline(),
                "random": UniformRandomClassifierBaseline(random_state=self.random_state + fold),
                "previous_score": PreviousScoreHeuristicBaseline(),
                "logistic_regression": build_logistic_regression_pipeline(
                    FEATURE_ALLOWLIST_CATEGORICAL, FEATURE_ALLOWLIST_NUMERICAL,
                    random_state=self.random_state + fold
                ),
                "hist_gradient_boosting": build_hist_gradient_boosting_pipeline(
                    FEATURE_ALLOWLIST_CATEGORICAL, FEATURE_ALLOWLIST_NUMERICAL,
                    random_state=self.random_state + fold
                ),
                "random_forest": build_random_forest_pipeline(
                    FEATURE_ALLOWLIST_CATEGORICAL, FEATURE_ALLOWLIST_NUMERICAL,
                    random_state=self.random_state + fold
                )
            }

            for m_name, model in models.items():
                model.fit(X_train, y_train)

                # Predictions
                val_probs = model.predict_proba(X_val)[:, 1]
                oof_df.loc[val_idx, f"prob_{m_name}"] = val_probs

                # Metrics at standard 0.5 threshold
                m_dict = calculate_metrics_dict(y_val.values, val_probs, threshold=0.50)
                m_dict["fold"] = fold
                if m_name == "previous_score":
                    m_dict["coverage_pct"] = round(model.get_coverage(X_val) * 100, 2)
                fold_metrics[m_name].append(m_dict)

                # Tuned threshold evaluation using training fold predictions only
                if m_name in fold_metrics_tuned:
                    train_probs = model.predict_proba(X_train)[:, 1]
                    best_thresh = optimize_threshold_on_train(y_train.values, train_probs)
                    m_tuned = calculate_metrics_dict(y_val.values, val_probs, threshold=best_thresh)
                    m_tuned["fold"] = fold
                    m_tuned["train_tuned_threshold"] = best_thresh
                    fold_metrics_tuned[m_name].append(m_tuned)

            elapsed = time.time() - t0
            print(f"  Fold {fold} complete in {elapsed:.1f}s | Val N={len(val_idx):,} | At-Risk={y_val.mean()*100:.2f}%")

        # Aggregate metrics across folds
        summary_models = {}
        for m_name in model_names:
            metrics_list = fold_metrics[m_name]
            keys_to_avg = ["pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_10", "recall_at_20"]
            
            agg = {}
            for k in keys_to_avg:
                vals = [m[k] for m in metrics_list]
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{k}_std"] = round(float(np.std(vals)), 4)

            # Previous score coverage
            if m_name == "previous_score":
                cov_vals = [m["coverage_pct"] for m in metrics_list]
                agg["coverage_pct_mean"] = round(float(np.mean(cov_vals)), 2)
                agg["coverage_pct_std"] = round(float(np.std(cov_vals)), 2)

            agg["fold_metrics"] = metrics_list
            if m_name in fold_metrics_tuned:
                agg["train_tuned_threshold_evaluation"] = fold_metrics_tuned[m_name]

            summary_models[m_name] = agg

        scenario_summary = {
            "scenario_name": scenario_name,
            "n_records": len(df_scenario),
            "unique_students": int(df_scenario["id_student"].nunique()),
            "class_distribution": {
                "at_risk_0": int((df_scenario["at_risk"] == 0).sum()),
                "at_risk_1": int((df_scenario["at_risk"] == 1).sum()),
                "at_risk_pct": round(float(df_scenario["at_risk"].mean() * 100), 2)
            },
            "fold_audit": fold_audit,
            "models": summary_models
        }

        return scenario_summary, oof_df

    def evaluate_subgroups(self, oof_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates subgroup performance by Module and Assessment Type."""
        subgroups: Dict[str, Any] = {"by_module": {}, "by_assessment_type": {}}
        
        # Module evaluation
        for mod, grp in oof_df.groupby("code_module"):
            n = len(grp)
            n_pos = int(grp["at_risk"].sum())
            pct = round(float(grp["at_risk"].mean() * 100), 2)
            if n_pos < 10:
                continue

            subgroups["by_module"][mod] = {
                "n": n,
                "n_at_risk": n_pos,
                "at_risk_pct": pct,
                "lr_pr_auc": round(float(average_precision_score(grp["at_risk"], grp["prob_logistic_regression"])), 4),
                "hgb_pr_auc": round(float(average_precision_score(grp["at_risk"], grp["prob_hist_gradient_boosting"])), 4),
                "hgb_recall_at_10": round(compute_recall_at_k(grp["at_risk"].values, grp["prob_hist_gradient_boosting"].values, 0.10), 4),
                "hgb_recall_at_20": round(compute_recall_at_k(grp["at_risk"].values, grp["prob_hist_gradient_boosting"].values, 0.20), 4),
            }

        # Assessment type evaluation
        assessments_df = pd.read_csv(os.path.join(self.processed_dir, "assessments_clean.csv"))
        a_type_map = assessments_df.set_index("id_assessment")["assessment_type"].to_dict()
        oof_df["assessment_type"] = oof_df["id_assessment"].map(a_type_map)

        for a_type, grp in oof_df.groupby("assessment_type"):
            n = len(grp)
            n_pos = int(grp["at_risk"].sum())
            pct = round(float(grp["at_risk"].mean() * 100), 2)

            subgroups["by_assessment_type"][a_type] = {
                "n": n,
                "n_at_risk": n_pos,
                "at_risk_pct": pct,
                "lr_pr_auc": round(float(average_precision_score(grp["at_risk"], grp["prob_logistic_regression"])), 4),
                "hgb_pr_auc": round(float(average_precision_score(grp["at_risk"], grp["prob_hist_gradient_boosting"])), 4),
                "hgb_recall_at_10": round(compute_recall_at_k(grp["at_risk"].values, grp["prob_hist_gradient_boosting"].values, 0.10), 4),
                "hgb_recall_at_20": round(compute_recall_at_k(grp["at_risk"].values, grp["prob_hist_gradient_boosting"].values, 0.20), 4),
            }

        return subgroups

    def generate_evaluation_figures(self, oof_14_7d: pd.DataFrame, oof_14_1d: pd.DataFrame):
        """Generates publication-quality diagnostic plots in reports/figures/baseline/."""
        print("Generating evaluation figures in reports/figures/baseline/...")
        sns.set_theme(style="whitegrid", font="sans-serif")
        palette = {"majority": "gray", "random": "purple", "previous_score": "orange",
                   "logistic_regression": "blue", "hist_gradient_boosting": "green", "random_forest": "red"}

        # 1. Precision-Recall Curves (7d vs 1d horizon)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        for ax, oof_sub, title in zip(axes, [oof_14_7d, oof_14_1d],
                                     ["7-Day Early Warning Horizon", "1-Day Checkpoint Horizon"]):
            y_true = oof_sub["at_risk"].values
            base_rate = float(y_true.mean())
            ax.axhline(base_rate, color="gray", linestyle="--", label=f"No Skill (Prevalence {base_rate:.1%})")

            for m_key, label, col in [
                ("prob_random", "Random", "purple"),
                ("prob_previous_score", "Previous Score", "orange"),
                ("prob_logistic_regression", "Logistic Regression", "blue"),
                ("prob_hist_gradient_boosting", "HistGradientBoosting", "green"),
                ("prob_random_forest", "Random Forest", "red"),
            ]:
                probs = oof_sub[m_key].values
                prec, rec, _ = precision_recall_curve(y_true, probs)
                ap = average_precision_score(y_true, probs)
                ax.plot(rec, prec, label=f"{label} (AP = {ap:.3f})", color=col, lw=2)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Recall", fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.legend(loc="upper right", fontsize=9)

        plt.suptitle("Out-of-Fold Precision-Recall Curves (14-Day Delay Baseline)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "pr_curves.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2. ROC Curves (7d vs 1d horizon)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        for ax, oof_sub, title in zip(axes, [oof_14_7d, oof_14_1d],
                                     ["7-Day Early Warning Horizon", "1-Day Checkpoint Horizon"]):
            y_true = oof_sub["at_risk"].values
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.500)")

            for m_key, label, col in [
                ("prob_previous_score", "Previous Score", "orange"),
                ("prob_logistic_regression", "Logistic Regression", "blue"),
                ("prob_hist_gradient_boosting", "HistGradientBoosting", "green"),
                ("prob_random_forest", "Random Forest", "red"),
            ]:
                probs = oof_sub[m_key].values
                fpr, tpr, _ = roc_curve(y_true, probs)
                auc_val = roc_auc_score(y_true, probs)
                ax.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.3f})", color=col, lw=2)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.legend(loc="lower right", fontsize=9)

        plt.suptitle("Out-of-Fold ROC Curves (14-Day Delay Baseline)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "roc_curves.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 3. Recall@k Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        k_levels = [0.05, 0.10, 0.15, 0.20]
        y_true = oof_14_7d["at_risk"].values
        
        models_to_plot = [
            ("prob_random", "Random"),
            ("prob_previous_score", "Previous Score"),
            ("prob_logistic_regression", "Logistic Regression"),
            ("prob_hist_gradient_boosting", "HistGradientBoosting"),
        ]

        plot_data = []
        for k in k_levels:
            for col, name in models_to_plot:
                rec_val = compute_recall_at_k(y_true, oof_14_7d[col].values, k=k)
                plot_data.append({"Budget Top-k": f"Top {int(k*100)}%", "Model": name, "Recall@k": rec_val})

        df_bar = pd.DataFrame(plot_data)
        sns.barplot(data=df_bar, x="Budget Top-k", y="Recall@k", hue="Model", ax=ax, palette="Set2")
        ax.set_title("Intervention Capture Efficiency: Recall@k (7-Day Horizon, 14-Day Delay)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Recall (Proportion of At-Risk Students Captured)", fontsize=11)
        ax.set_ylim([0, 1.0])
        ax.axhline(0.10, color="gray", linestyle=":", alpha=0.6)
        ax.axhline(0.20, color="gray", linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, "recall_at_k_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 4. CV Fold Stability Boxplot
        fig, ax = plt.subplots(figsize=(9, 5))
        # Collect fold PR-AUCs
        fold_data = []
        for m_key, name in [
            ("previous_score", "Previous Score"),
            ("logistic_regression", "Logistic Regression"),
            ("hist_gradient_boosting", "HistGradientBoosting"),
            ("random_forest", "Random Forest")
        ]:
            if m_key in self.summary_results.get("14_day_delay__7_day_before", {}).get("models", {}):
                m_info = self.summary_results["14_day_delay__7_day_before"]["models"][m_key]
                for f_m in m_info["fold_metrics"]:
                    fold_data.append({"Model": name, "Fold PR-AUC": f_m["pr_auc"]})

        if fold_data:
            df_fold = pd.DataFrame(fold_data)
            sns.boxplot(data=df_fold, x="Model", y="Fold PR-AUC", ax=ax, palette="Pastel1", width=0.4)
            sns.stripplot(data=df_fold, x="Model", y="Fold PR-AUC", ax=ax, color="black", size=6, jitter=0.1)
            ax.set_title("5-Fold Cross-Validation Stability (PR-AUC, 7-Day Horizon)", fontsize=13, fontweight="bold")
            ax.set_ylim([0.0, 0.6])
            plt.tight_layout()
            fig.savefig(os.path.join(self.figures_dir, "cv_fold_stability.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        print("Diagnostic plots generated successfully.")

    def run(self) -> Dict[str, Any]:
        """Executes the full Phase 5 baseline evaluation across all 4 scenarios."""
        t_start = time.time()
        print("Starting Phase 5 Baseline Model Development & Cross-Validation Pipeline...")

        # 1. Load Datasets
        path_14 = os.path.join(self.processed_dir, "prediction_dataset.csv")
        path_7 = os.path.join(self.processed_dir, "prediction_dataset_7day_delay.csv")

        if not os.path.exists(path_14) or not os.path.exists(path_7):
            raise FileNotFoundError("Phase 4 prediction datasets not found in data/processed/.")

        df_14 = pd.read_csv(path_14)
        df_7 = pd.read_csv(path_7)

        # Feature audit
        audit_14 = FeatureGovernor.audit_features(df_14)
        print("Feature Governance Audit:", audit_14)
        if not audit_14["is_valid"]:
            raise ValueError("Feature governance audit failed!")

        # 2. Run 4 scenarios
        scenarios = [
            ("14_day_delay__7_day_before", df_14[df_14["cutoff_scenario"] == "7_day_before"].reset_index(drop=True)),
            ("14_day_delay__1_day_before", df_14[df_14["cutoff_scenario"] == "1_day_before"].reset_index(drop=True)),
            ("7_day_delay__7_day_before", df_7[df_7["cutoff_scenario"] == "7_day_before"].reset_index(drop=True)),
            ("7_day_delay__1_day_before", df_7[df_7["cutoff_scenario"] == "1_day_before"].reset_index(drop=True)),
        ]

        oof_results = {}
        for sc_name, sc_df in scenarios:
            sc_summary, oof_sc = self.run_cv_for_scenario(sc_df, sc_name)
            self.summary_results[sc_name] = sc_summary
            oof_results[sc_name] = oof_sc

        # Save OOF predictions
        oof_14 = pd.concat([oof_results["14_day_delay__7_day_before"], oof_results["14_day_delay__1_day_before"]], ignore_index=True)
        oof_7 = pd.concat([oof_results["7_day_delay__7_day_before"], oof_results["7_day_delay__1_day_before"]], ignore_index=True)

        oof_14.to_csv(os.path.join(self.processed_dir, "oof_predictions_14day_delay.csv"), index=False)
        oof_7.to_csv(os.path.join(self.processed_dir, "oof_predictions_7day_delay.csv"), index=False)
        print("Saved Out-of-Fold predictions to data/processed/.")

        # Subgroup evaluations
        subgroups_14_7d = self.evaluate_subgroups(oof_results["14_day_delay__7_day_before"])
        self.summary_results["subgroups_14_day_delay__7_day_before"] = subgroups_14_7d

        # Generate figures
        self.generate_evaluation_figures(
            oof_results["14_day_delay__7_day_before"],
            oof_results["14_day_delay__1_day_before"]
        )

        total_time = round(time.time() - t_start, 2)
        self.summary_results["total_execution_time_seconds"] = total_time
        self.summary_results["feature_governance_audit"] = audit_14

        # Save summary JSON
        with open(os.path.join(self.reports_dir, "baseline_model_summary.json"), "w") as f:
            json.dump(self.summary_results, f, indent=2)

        print(f"\nPhase 5 Baseline Pipeline finished in {total_time}s.")
        print("Summary metrics saved to reports/baseline_model_summary.json.")
        return self.summary_results


if __name__ == "__main__":
    pipeline = BaselineEvaluationPipeline()
    pipeline.run()

