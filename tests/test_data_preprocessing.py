"""
tests/test_data_preprocessing.py

Automated test suite verifying the integrity, correctness, and reproducibility
of the Phase 3 data cleaning and preprocessing pipeline.
"""

import os
import hashlib
import pytest
import pandas as pd
import numpy as np


# Recorded SHA-256 baselines of raw CSV files
RAW_SHA256_BASELINES = {
    "courses.csv": "4f16eee7454b15e109b0a21a0e43be820e6846ed6f9301bb7feb5ab5ad737a75",
    "assessments.csv": "9ce92381e4a0ac457f8e251b2eb2c179ed51e023ab30d1d671a0268bd62316ba",
    "vle.csv": "d7a146497edd0e47ef536144b58796c07b554002d3753e848b8e4cd8a56f4c38",
    "studentInfo.csv": "815fbc3e2de29f79900bc63f0345a35db62c9146afaa1d86e7107b8b3beffc60",
    "studentRegistration.csv": "bbc87a0de1fe2a9ec6decce1ca6c1a02cf17b54ad3bdcc673fc6413b41c61d47",
    "studentAssessment.csv": "b1839e17618a5de36b6d189a242d3123f7939a56c8c9ccef0c17b9199c614a6b",
    "studentVle.csv": "52668253d876c5becbcb72185977152700cecab2942aca807fecc3dd54b937f0"
}

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def test_raw_file_integrity():
    """Verify that every raw CSV file remains 100% byte-identical and unmodified."""
    for filename, baseline_hash in RAW_SHA256_BASELINES.items():
        filepath = os.path.join(RAW_DIR, filename)
        assert os.path.exists(filepath), f"Raw file {filename} is missing!"
        
        with open(filepath, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
            assert current_hash == baseline_hash, f"RAW FILE MODIFIED: {filename} hash mismatch!"


def test_processed_files_exist():
    """Verify that all five processed CSV files exist and are non-empty."""
    expected_files = [
        "enrollments_clean.csv",
        "assessments_clean.csv",
        "assessment_submissions_clean.csv",
        "vle_interactions_clean.csv",
        "enrollment_base.csv"
    ]
    for filename in expected_files:
        filepath = os.path.join(PROCESSED_DIR, filename)
        assert os.path.exists(filepath), f"Processed file {filename} does not exist!"
        assert os.path.getsize(filepath) > 0, f"Processed file {filename} is empty!"


def test_enrollment_composite_key_uniqueness():
    """Verify that (code_module, code_presentation, id_student) is strictly unique."""
    enrol_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollments_clean.csv"))
    base_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollment_base.csv"))

    key_cols = ["code_module", "code_presentation", "id_student"]
    assert not enrol_df.duplicated(subset=key_cols).any(), "Duplicate key in enrollments_clean.csv"
    assert not base_df.duplicated(subset=key_cols).any(), "Duplicate key in enrollment_base.csv"


def test_enrollment_count_preservation():
    """Verify exact preservation of 32,593 enrollments without row explosion or loss."""
    enrol_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollments_clean.csv"))
    base_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollment_base.csv"))

    assert len(enrol_df) == 32593, f"Expected 32,593 enrollments, got {len(enrol_df)}"
    assert len(base_df) == 32593, f"Expected 32,593 base rows, got {len(base_df)}"


def test_preservation_of_zero_activity_enrollments():
    """Verify that students with 0 assessment submissions and 0 VLE clicks are preserved."""
    base_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollment_base.csv"))

    # Zero submissions check
    zero_sub_count = (base_df["has_zero_submissions"] == 1).sum()
    assert zero_sub_count == 6750, f"Expected 6,750 zero-submission students, got {zero_sub_count}"

    # Zero VLE clicks check
    zero_vle_count = (base_df["is_vle_dormant"] == 1).sum()
    assert zero_vle_count == 3365, f"Expected 3,365 dormant VLE students, got {zero_vle_count}"

    # Verify that total clicks is 0 when is_vle_dormant == 1
    dormant_clicks = base_df[base_df["is_vle_dormant"] == 1]["total_vle_clicks"].sum()
    assert dormant_clicks == 0, f"Dormant students must have 0 clicks, found {dormant_clicks}"


def test_foreign_key_referential_integrity():
    """Verify referential integrity across processed datasets."""
    assess_df = pd.read_csv(os.path.join(PROCESSED_DIR, "assessments_clean.csv"))
    sub_df = pd.read_csv(os.path.join(PROCESSED_DIR, "assessment_submissions_clean.csv"))
    enrol_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollments_clean.csv"))

    # Assessment ID references
    assess_ids = set(assess_df["id_assessment"])
    sub_assess_ids = set(sub_df["id_assessment"])
    assert sub_assess_ids.issubset(assess_ids), "Orphaned id_assessment in submissions!"

    # Student enrollment references
    enrol_keys = set(zip(enrol_df["code_module"], enrol_df["code_presentation"], enrol_df["id_student"]))
    sub_enrol_keys = set(zip(sub_df["code_module"], sub_df["code_presentation"], sub_df["id_student"]))
    assert sub_enrol_keys.issubset(enrol_keys), "Orphaned enrollment key in assessment submissions!"


def test_score_and_click_validity():
    """Verify numeric constraints on scores and clicks."""
    sub_df = pd.read_csv(os.path.join(PROCESSED_DIR, "assessment_submissions_clean.csv"))
    base_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollment_base.csv"))

    # Scores
    valid_scores = sub_df["score"].dropna()
    assert (valid_scores >= 0).all() and (valid_scores <= 100).all(), "Scores out of [0, 100] range!"
    assert sub_df["score"].isna().sum() == 173, "Expected exactly 173 missing scores"

    # Clicks
    assert (base_df["total_vle_clicks"] >= 0).all(), "Negative clicks in enrollment_base!"


def test_missing_value_handling():
    """Verify context-aware missing-value representations."""
    enrol_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollments_clean.csv"))
    assess_df = pd.read_csv(os.path.join(PROCESSED_DIR, "assessments_clean.csv"))

    # Assessments missing date
    assert assess_df["date"].isna().sum() == 11, "Missing dates count in assessments must be 11"

    # Registration & unregistration
    assert enrol_df["date_registration"].isna().sum() == 45, "Missing date_registration must be 45"
    assert enrol_df["date_unregistration"].isna().sum() == 22521, "Missing date_unregistration must be 22,521"
    assert (enrol_df["has_unregistered"] == 1).sum() == 10072, "Recorded unregistrations must be 10,072"


def test_reproducibility_of_aggregation():
    """Verify macro aggregation totals match exactly."""
    base_df = pd.read_csv(os.path.join(PROCESSED_DIR, "enrollment_base.csv"))

    # Total submissions sum
    total_subs = base_df["total_assessments_submitted"].sum()
    assert total_subs == 173912, f"Total submissions aggregated ({total_subs}) != raw count (173,912)"

    # Total clicks sum
    total_clicks = base_df["total_vle_clicks"].sum()
    assert total_clicks == 39605099, f"Total clicks aggregated ({total_clicks}) != raw click sum (39,605,099)"

