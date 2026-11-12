"""
tests/test_temporal_feature_engineering.py

Automated test suite verifying the correctness, temporal validity, and leakage-freedom
of the Phase 4 temporal feature engineering and target construction pipeline.
"""

import os
import pytest
import pandas as pd
import numpy as np


PROCESSED_DIR = "data/processed"


@pytest.fixture(scope="module")
def prediction_df():
    """Loads a slice of the 14-day delay baseline dataset for testing."""
    path = os.path.join(PROCESSED_DIR, "prediction_dataset.csv")
    assert os.path.exists(path), "prediction_dataset.csv missing!"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def prediction_7d_df():
    """Loads a slice of the 7-day delay alternative dataset for testing."""
    path = os.path.join(PROCESSED_DIR, "prediction_dataset_7day_delay.csv")
    assert os.path.exists(path), "prediction_dataset_7day_delay.csv missing!"
    return pd.read_csv(path)


def test_datasets_exist_and_non_empty(prediction_df, prediction_7d_df):
    """Verify that both prediction datasets exist, are non-empty, and share identical row counts."""
    assert len(prediction_df) == 341748, f"Expected 341,748 rows, found {len(prediction_df)}"
    assert len(prediction_7d_df) == 341748, f"Expected 341,748 rows, found {len(prediction_7d_df)}"


def test_no_prohibited_features(prediction_df):
    """Verify that no outcome features or sensitive demographic attributes enter the dataset."""
    prohibited_cols = [
        "final_result",
        "date_unregistration",
        "has_unregistered",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability"
    ]
    for col in prohibited_cols:
        assert col not in prediction_df.columns, f"PROHIBITED FEATURE FOUND: {col} is in prediction dataset!"


def test_target_label_accuracy(prediction_df):
    """Verify that at_risk target is strictly derived as next_score < 50."""
    scores = prediction_df["target_assessment_score"]
    targets = prediction_df["at_risk"]

    assert scores.notna().all(), "Missing values found in target_assessment_score!"
    assert ((scores >= 0) & (scores <= 100)).all(), "Target scores out of [0, 100] bounds!"
    
    expected_targets = (scores < 50).astype(int)
    assert (targets == expected_targets).all(), "Target at_risk mismatch with score < 50 rule!"
    assert set(targets.unique()).issubset({0, 1}), "Target at_risk contains non-binary values!"


def test_cutoff_calculation(prediction_df):
    """Verify that cutoff_date is strictly scheduled_date - offset."""
    # 7-day before scenario
    df_7d = prediction_df[prediction_df["cutoff_scenario"] == "7_day_before"]
    assert (df_7d["cutoff_date"] == df_7d["scheduled_date"] - 7).all(), "Cutoff mismatch in 7_day_before!"
    assert (df_7d["days_until_assessment"] == 7).all()

    # 1-day before scenario
    df_1d = prediction_df[prediction_df["cutoff_scenario"] == "1_day_before"]
    assert (df_1d["cutoff_date"] == df_1d["scheduled_date"] - 1).all(), "Cutoff mismatch in 1_day_before!"
    assert (df_1d["days_until_assessment"] == 1).all()


def test_no_target_leakage_on_first_assessment(prediction_df):
    """Verify that for the very first assessment of a module (k=0), prior score features are empty."""
    first_assessments = prediction_df[prediction_df["num_previous_scheduled_assessments"] == 0]
    assert len(first_assessments) > 0

    assert (first_assessments["num_prior_submissions"] == 0).all(), "First assessment has prior submissions!"
    assert (first_assessments["num_prior_scored_assessments"] == 0).all(), "First assessment has prior scored marks!"
    assert first_assessments["mean_prior_score"].isna().all(), "First assessment has leaked mean_prior_score!"
    assert first_assessments["recent_prior_score"].isna().all(), "First assessment has leaked recent_prior_score!"


def test_temporal_delay_sensitivity(prediction_df, prediction_7d_df):
    """Verify that the 7-day delay scenario makes more historical scores available than 14-day delay."""
    # In the alternative 7-day delay scenario, grades become available 7 days faster than 14-day delay
    total_scored_14d = prediction_df["num_prior_scored_assessments"].sum()
    total_scored_7d = prediction_7d_df["num_prior_scored_assessments"].sum()

    assert total_scored_7d >= total_scored_14d, (
        f"7-day delay total scored ({total_scored_7d}) should be >= 14-day delay ({total_scored_14d})"
    )


def test_vle_temporal_monotonicity(prediction_df):
    """Verify that 1-day before cutoff (T=D-1) has >= cumulative clicks than 7-day before cutoff (T=D-7)."""
    # For the same student and same target assessment, clicks at T=D-1 must be >= clicks at T=D-7
    piv = prediction_df.pivot_table(
        index=["id_student", "id_assessment"],
        columns="cutoff_scenario",
        values="vle_total_clicks_to_date"
    )
    assert (piv["1_day_before"] >= piv["7_day_before"]).all(), "VLE clicks decreased between Day D-7 and Day D-1!"


def test_valid_referential_integrity(prediction_df):
    """Verify that all students and assessments reference valid entities."""
    enrol_path = os.path.join(PROCESSED_DIR, "enrollments_clean.csv")
    assess_path = os.path.join(PROCESSED_DIR, "assessments_clean.csv")

    enrol_df = pd.read_csv(enrol_path)
    assess_df = pd.read_csv(assess_path)

    enrol_keys = set(zip(enrol_df["code_module"], enrol_df["code_presentation"], enrol_df["id_student"]))
    pred_keys = set(zip(prediction_df["code_module"], prediction_df["code_presentation"], prediction_df["id_student"]))
    assert pred_keys.issubset(enrol_keys), "Orphaned enrollment key in prediction dataset!"

    assess_ids = set(assess_df["id_assessment"])
    pred_assess_ids = set(prediction_df["id_assessment"])
    assert pred_assess_ids.issubset(assess_ids), "Orphaned assessment ID in prediction dataset!"


def test_class_imbalance_bounds(prediction_df):
    """Verify empirical class distribution."""
    at_risk_rate = prediction_df["at_risk"].mean()
    assert 0.05 < at_risk_rate < 0.15, f"At-risk rate {at_risk_rate:.4f} outside expected [0.05, 0.15] bounds"
    assert (prediction_df["at_risk"] == 1).sum() == 29246, "Exact at_risk positive count mismatch"

