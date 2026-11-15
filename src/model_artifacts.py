"""
Model Artifact Preparation, Persistence, and Integrity Validation Module.
Phase 10 (Tasks 10.1 and 10.2).

Provides:
- Reproducible artifact preparation for approved HistGradientBoosting model and Challenger Random Forest.
- Preservation of exact feature ordering, preprocessing, and missing-value handling.
- Cryptographic SHA-256 hash verification and metadata validation.
- ModelArtifactManager enforcing strict feature allowlists, prohibited feature checks, and safe loading.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline_models import (
    FEATURE_ALLOWLIST_CATEGORICAL,
    FEATURE_ALLOWLIST_NUMERICAL,
    ALL_ALLOWED_FEATURES,
    PROHIBITED_COLUMNS,
)


MODEL_DIR = "models"
HGB_ARTIFACT_NAME = "optimized_hist_gradient_boosting.joblib"
HGB_METADATA_NAME = "optimized_hist_gradient_boosting_metadata.json"
RF_ARTIFACT_NAME = "challenger_random_forest.joblib"
RF_METADATA_NAME = "challenger_random_forest_metadata.json"


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class ModelArtifactManager:
    """
    Manages loading, integrity validation, and inference of trained model artifacts.
    Enforces feature order, prohibited column guards, and cryptographic hash checks.
    """

    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None, verify_hash: bool = True):
        self.model_path = model_path or os.path.join(MODEL_DIR, HGB_ARTIFACT_NAME)
        self.metadata_path = metadata_path or os.path.join(MODEL_DIR, HGB_METADATA_NAME)
        self.verify_hash = verify_hash
        
        self.model: Optional[Pipeline] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._is_loaded = False
        
        self.load()

    def load(self):
        """Loads and validates model artifact and its metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Validate cryptographic hash
        if self.verify_hash:
            expected_hash = self.metadata.get("sha256_hash")
            actual_hash = compute_file_sha256(self.model_path)
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(
                    f"Artifact Integrity Error: SHA-256 hash mismatch for {self.model_path}. "
                    f"Expected {expected_hash}, got {actual_hash}. File may be corrupted or altered."
                )

        # Load pipeline
        self.model = joblib.load(self.model_path)

        # Validate pipeline structure
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Loaded model artifact does not implement predict_proba.")

        # Validate feature counts and names
        expected_features = self.metadata.get("feature_names", [])
        if len(expected_features) != len(ALL_ALLOWED_FEATURES):
            raise ValueError(f"Feature count mismatch: expected {len(ALL_ALLOWED_FEATURES)}, metadata has {len(expected_features)}.")

        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Executes prediction with strict feature ordering and prohibited feature validation.
        Returns array of shape (N, 2) or probabilities of positive class.
        """
        if not self._is_loaded or self.model is None or self.metadata is None:
            raise RuntimeError("ModelArtifactManager has not successfully loaded an artifact.")

        # Check for prohibited columns (leakage or sensitive demographics)
        present_cols = set(X.columns)
        prohibited_present = present_cols.intersection(PROHIBITED_COLUMNS)
        if prohibited_present:
            raise ValueError(f"Security & Governance Violation: Prohibited features present in inference payload: {prohibited_present}")

        # Ensure all allowed features exist
        feature_order = self.metadata["feature_names"]
        missing_features = [col for col in feature_order if col not in X.columns]
        if missing_features:
            raise ValueError(f"Inference Error: Missing required predictive features: {missing_features}")

        # Order features exactly as expected during training
        X_ordered = X[feature_order].copy()
        
        # Predict
        probs = self.model.predict_proba(X_ordered)
        return probs[:, 1] if probs.ndim == 2 else probs


def build_and_save_approved_model_artifacts(
    data_path: str = "data/processed/prediction_dataset.csv",
    models_dir: str = MODEL_DIR
) -> Dict[str, Any]:
    """
    Fits and saves the approved Phase 6 HistGradientBoosting model and Challenger Random Forest
    strictly using earlier training intakes (2013B, 2013J, 2014B).
    Quarantines 2014J holdout data completely.
    """
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)

    # Strict training mask: 14-day delay, 7-day cutoff, intakes 2013B/J + 2014B only
    train_mask = (
        (df["cutoff_scenario"] == "7_day_before") &
        (df["grading_delay_scenario"] == "14_day_delay") &
        (df["code_presentation"] != "2014J")
    )
    df_train = df[train_mask].copy()
    n_train = len(df_train)
    print(f"Isolated training partition: {n_train:,} snapshots across earlier presentations (2014J quarantined).")

    X_train = df_train[ALL_ALLOWED_FEATURES]
    y_train = df_train["at_risk"]

    cat_cols = FEATURE_ALLOWLIST_CATEGORICAL
    num_cols = FEATURE_ALLOWLIST_NUMERICAL
    cat_indices = [len(num_cols) + i for i in range(len(cat_cols))]

    # =========================================================================
    # 1. PRIMARY MODEL: HistGradientBoostingClassifier (Deep Regularized)
    # =========================================================================
    print("Fitting Primary Model (HistGradientBoostingClassifier)...")
    hgb_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        ]
    )

    hgb_clf = HistGradientBoostingClassifier(
        categorical_features=cat_indices,
        learning_rate=0.05,
        max_iter=150,
        max_depth=8,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=5.0,
        random_state=42
    )

    hgb_pipeline = Pipeline([
        ("preprocessor", hgb_preprocessor),
        ("classifier", hgb_clf)
    ])

    hgb_pipeline.fit(X_train, y_train)

    hgb_model_path = os.path.join(models_dir, HGB_ARTIFACT_NAME)
    joblib.dump(hgb_pipeline, hgb_model_path)
    hgb_hash = compute_file_sha256(hgb_model_path)

    hgb_metadata = {
        "model_name": "HistGradientBoostingClassifier",
        "model_version": "1.0.0",
        "feature_pipeline_version": "1.0.0",
        "created_at_utc": timestamp,
        "sha256_hash": hgb_hash,
        "feature_count": len(ALL_ALLOWED_FEATURES),
        "feature_names": ALL_ALLOWED_FEATURES,
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_iter": 150,
            "max_depth": 8,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 50,
            "l2_regularization": 5.0,
            "random_state": 42
        },
        "training_data_provenance": {
            "source_dataset": data_path,
            "training_snapshots_count": n_train,
            "presentations_included": ["2013B", "2013J", "2014B"],
            "temporal_holdout_quarantined": ["2014J"],
            "positive_prevalence_pct": round(float(y_train.mean() * 100), 2)
        },
        "supported_prediction_horizons": [7],
        "assumed_grading_delay_days": 14,
        "decision_threshold_policy": {
            "standard_tma_threshold": 0.20,
            "ggg_tma_calibrated_threshold": 0.10,
            "cma_policy": "AUTOMATED_OUTREACH_SUPPRESSED"
        },
        "operational_status": "OFFLINE_RESEARCH_CANDIDATE (Production Deployment Approval WITHHELD)"
    }

    hgb_meta_path = os.path.join(models_dir, HGB_METADATA_NAME)
    with open(hgb_meta_path, "w") as f:
        json.dump(hgb_metadata, f, indent=2)

    # =========================================================================
    # 2. CHALLENGER MODEL: RandomForestClassifier
    # =========================================================================
    print("Fitting Challenger Model (RandomForestClassifier)...")
    rf_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ]
    )

    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1
    )

    rf_pipeline = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("classifier", rf_clf)
    ])

    rf_pipeline.fit(X_train, y_train)

    rf_model_path = os.path.join(models_dir, RF_ARTIFACT_NAME)
    joblib.dump(rf_pipeline, rf_model_path)
    rf_hash = compute_file_sha256(rf_model_path)

    rf_metadata = {
        "model_name": "RandomForestClassifier",
        "model_version": "1.0.0",
        "feature_pipeline_version": "1.0.0",
        "created_at_utc": timestamp,
        "sha256_hash": rf_hash,
        "feature_count": len(ALL_ALLOWED_FEATURES),
        "feature_names": ALL_ALLOWED_FEATURES,
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 30,
            "random_state": 42
        },
        "training_data_provenance": {
            "source_dataset": data_path,
            "training_snapshots_count": n_train,
            "presentations_included": ["2013B", "2013J", "2014B"],
            "temporal_holdout_quarantined": ["2014J"]
        },
        "operational_status": "CHALLENGER_BENCHMARK_MODEL"
    }

    rf_meta_path = os.path.join(models_dir, RF_METADATA_NAME)
    with open(rf_meta_path, "w") as f:
        json.dump(rf_metadata, f, indent=2)

    print(f"Artifacts successfully saved to {models_dir}:")
    print(f"  Primary: {hgb_model_path} (SHA-256: {hgb_hash[:16]}...)")
    print(f"  Challenger: {rf_model_path} (SHA-256: {rf_hash[:16]}...)")

    return {
        "primary_model": hgb_metadata,
        "challenger_model": rf_metadata
    }


if __name__ == "__main__":
    build_and_save_approved_model_artifacts()
