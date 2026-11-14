"""
tests/test_baseline_models.py

Automated Test Suite for Phase 5: Baseline Models & Cross-Validation Strategy.
Verifies:
1. Feature governance and strict allowlist enforcement (no target, demographic, or outcome leakage).
2. Student-level group separation in 5-fold StratifiedGroupKFold (zero student overlap).
3. Non-empty positive/negative classes across all folds.
4. Train-only preprocessing fitting (no transformation leakage from validation data).
5. Deterministic random baseline reproducibility.
6. Correct previous-score heuristic behavior and coverage tracking.
7. Exact mathematical precision of Recall@k (including tie apportionment).
8. Standard metric calculation accuracy (PR-AUC, ROC-AUC, Precision, Recall, F1, CM).
9. Scenario isolation: 7-day vs 1-day horizons and 14-day vs 7-day delays evaluated separately.
10. Out-of-fold prediction completeness and validity.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

# Ensure root workspace directory is in python sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

from src.baseline_models import (
    FEATURE_ALLOWLIST_CATEGORICAL,
    FEATURE_ALLOWLIST_NUMERICAL,
    ALL_ALLOWED_FEATURES,
    PROHIBITED_COLUMNS,
    FeatureGovernor,
    compute_recall_at_k,
    optimize_threshold_on_train,
    calculate_metrics_dict,
    MajorityClassifierBaseline,
    UniformRandomClassifierBaseline,
    PreviousScoreHeuristicBaseline,
    build_logistic_regression_pipeline,
    build_hist_gradient_boosting_pipeline,
)


@pytest.fixture(scope="module")
def prediction_dataset_sample():
    """Provides a sample of the 14-day delay prediction dataset."""
    path = "data/processed/prediction_dataset.csv"
    if not os.path.exists(path):
        pytest.skip("prediction_dataset.csv not found.")
    df = pd.read_csv(path, nrows=10_000)
    return df


def test_feature_governance_and_allowlist(prediction_dataset_sample):
    """Verifies that all allowed features exist and no prohibited columns enter feature matrix."""
    df = prediction_dataset_sample
    audit = FeatureGovernor.audit_features(df)
    assert audit["is_valid"], f"Audit failed: {audit}"
    assert len(audit["missing_allowed_features"]) == 0

    X, y = FeatureGovernor.get_feature_matrix(df)
    assert list(X.columns) == ALL_ALLOWED_FEATURES

    # Ensure prohibited columns are strictly excluded from X
    for col in PROHIBITED_COLUMNS:
        assert col not in X.columns, f"Prohibited column {col} found in feature matrix X!"

    # Target variable integrity
    assert set(y.unique()).issubset({0, 1})


def test_student_level_group_separation(prediction_dataset_sample):
    """Verifies that StratifiedGroupKFold strictly isolates students between train and val folds."""
    df = prediction_dataset_sample
    X, y = FeatureGovernor.get_feature_matrix(df)
    groups = df["id_student"].values

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y, groups=groups))

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_students = set(groups[train_idx])
        val_students = set(groups[val_idx])
        overlap = train_students.intersection(val_students)
        assert len(overlap) == 0, f"Fold {fold} contains {len(overlap)} overlapping students!"
        assert len(np.unique(y.iloc[val_idx])) == 2, f"Fold {fold} missing classes!"


def test_train_only_preprocessing_isolation():
    """Verifies that preprocessing transformers fit exclusively on training data."""
    X_train = pd.DataFrame({"num": [10.0, 20.0, 30.0], "cat": ["A", "B", "A"]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({"num": [1000.0, np.nan, 2000.0], "cat": ["B", "C", "A"]})

    pipe = build_logistic_regression_pipeline(cat_cols=["cat"], num_cols=["num"], random_state=42)
    pipe.fit(X_train, y_train)

    # Median imputer must reflect train median (20.0), not val median
    imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 20.0

    # Predictions run without error on unseen category 'C' (handle_unknown='ignore')
    probs = pipe.predict_proba(X_val)
    assert probs.shape == (3, 2)
    assert not np.isnan(probs).any()


def test_deterministic_random_baseline():
    """Verifies that UniformRandomClassifierBaseline is fully reproducible under fixed seed."""
    X = pd.DataFrame({"num": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 0, 1, 0, 1])

    b1 = UniformRandomClassifierBaseline(random_state=42)
    b2 = UniformRandomClassifierBaseline(random_state=42)
    b3 = UniformRandomClassifierBaseline(random_state=999)

    p1 = b1.predict_proba(X)
    p2 = b2.predict_proba(X)
    p3 = b3.predict_proba(X)

    np.testing.assert_array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


def test_previous_score_heuristic_and_coverage():
    """Verifies that PreviousScoreHeuristic assigns risk correctly and tracks coverage."""
    X = pd.DataFrame({
        "recent_prior_score": [80.0, 30.0, np.nan, 50.0, 49.0],
        "other_feature": [1, 2, 3, 4, 5]
    })
    y = pd.Series([0, 1, 0, 0, 1])

    model = PreviousScoreHeuristicBaseline()
    model.fit(X, y)

    # Coverage: 4 out of 5 valid scores = 80.0%
    cov = model.get_coverage(X)
    assert round(cov, 2) == 0.80

    preds = model.predict(X)
    # Expected: 80 -> 0; 30 -> 1; NaN -> 0 (do not treat missing as failure); 50 -> 0; 49 -> 1
    np.testing.assert_array_equal(preds, [0, 1, 0, 0, 1])

    probs = model.predict_proba(X)[:, 1]
    # Continuous risk mapping: (100 - score) / 100
    assert probs[0] == 0.20
    assert probs[1] == 0.70
    assert probs[2] == 0.0   # Missing score assigned non-risk
    assert probs[3] == 0.50
    assert probs[4] == 0.51


def test_recall_at_k_exact_and_tie_apportionment():
    """Verifies that compute_recall_at_k accurately ranks and apportions tied predictions."""
    # Case 1: Perfectly ranked predictions
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # 2 positives out of 10
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.05, 0.1, 0.2, 0.15, 0.1])
    # Top 10% (1 item, prob 0.9, positive): 1 / 2 = 0.50
    assert compute_recall_at_k(y_true, y_prob, k=0.10) == 0.50
    # Top 20% (2 items, probs 0.9 & 0.8, both positive): 2 / 2 = 1.00
    assert compute_recall_at_k(y_true, y_prob, k=0.20) == 1.00

    # Case 2: Complete tie across all instances (e.g. constant majority classifier)
    y_true_flat = np.array([1]*10 + [0]*90)  # 10% prevalence
    y_prob_flat = np.zeros(100)
    # Exact tie apportionment should yield exactly k (expected recall of uniform random selection)
    assert round(compute_recall_at_k(y_true_flat, y_prob_flat, k=0.10), 4) == 0.1000
    assert round(compute_recall_at_k(y_true_flat, y_prob_flat, k=0.20), 4) == 0.2000


def test_train_tuned_threshold_optimization():
    """Verifies that threshold optimization operates on train labels without leakage."""
    y_train = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.45, 0.40, 0.35, 0.32, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01])

    best_thresh = optimize_threshold_on_train(y_train, y_prob)
    assert 0.30 <= best_thresh <= 0.40

    metrics = calculate_metrics_dict(y_train, y_prob, threshold=best_thresh)
    assert metrics["f1"] > 0.0
    assert metrics["threshold"] == round(best_thresh, 4)


def test_standard_metrics_dictionary_validity():
    """Verifies that calculate_metrics_dict outputs all required fields with proper bounds."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.1, 0.4, 0.2, 0.0])

    m = calculate_metrics_dict(y_true, y_prob, threshold=0.5)
    for key in ["threshold", "pr_auc", "roc_auc", "precision", "recall", "f1",
                "recall_at_10", "recall_at_20", "confusion_matrix"]:
        assert key in m

    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["recall_at_10"] <= 1.0
    assert 0.0 <= m["recall_at_20"] <= 1.0

    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)


def test_oof_predictions_exist_and_complete():
    """Verifies that both Out-of-Fold prediction files exist, have 341,748 rows, and no null predictions."""
    for fn in ["oof_predictions_14day_delay.csv", "oof_predictions_7day_delay.csv"]:
        path = os.path.join("data/processed", fn)
        assert os.path.exists(path), f"Missing OOF file: {path}"
        df_oof = pd.read_csv(path)
        assert len(df_oof) == 341748, f"{fn} row count is {len(df_oof)}, expected 341748"
        for col in ["prob_majority", "prob_random", "prob_previous_score",
                    "prob_logistic_regression", "prob_hist_gradient_boosting", "prob_random_forest"]:
            assert col in df_oof.columns, f"Missing prediction column {col} in {fn}"
            assert df_oof[col].isnull().sum() == 0, f"Null predictions found in {fn} column {col}!"


def test_scenario_isolation_in_summary_json():
    """Verifies that all 4 scenarios were evaluated separately without pooling."""
    path = "reports/baseline_model_summary.json"
    assert os.path.exists(path), "baseline_model_summary.json missing!"
    with open(path) as f:
        data = json.load(f)

    expected_scenarios = [
        "14_day_delay__7_day_before",
        "14_day_delay__1_day_before",
        "7_day_delay__7_day_before",
        "7_day_delay__1_day_before",
    ]
    for sc in expected_scenarios:
        assert sc in data, f"Scenario {sc} missing from baseline_model_summary.json!"
        models = data[sc]["models"]
        for m in ["majority", "random", "previous_score", "logistic_regression", "hist_gradient_boosting", "random_forest"]:
            assert m in models, f"Model {m} missing from scenario {sc}!"
            assert "pr_auc_mean" in models[m]
            assert "roc_auc_mean" in models[m]
            assert "recall_at_10_mean" in models[m]
            assert "recall_at_20_mean" in models[m]
            assert len(models[m]["fold_metrics"]) == 5
