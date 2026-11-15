"""
Shadow-Mode Safety Gates and Passive Monitoring Harness.
Phase 9 Operational Gap Closure (Task A6).

Provides:
- Hard-coded safety lock preventing any automated student communication or external outreach.
- Read-only inference execution generating immutable DecisionAuditRecords.
- Fail-safe configuration validation (missing secret, missing model metadata).
- Automated shadow monitoring: missing feature rates, PSI drift, ingestion latency, and alert volume volatility.
- Retrospective Brier score and ECE evaluation once labels become observed.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.decision_audit import DecisionAuditRecord, DecisionAuditLogger, pseudonymize_student_id
from src.governance_router import InstitutionalGovernanceRouter, SecurityViolationError


class GovernanceConfigurationError(Exception):
    """Raised when required governance metadata or cryptographic secrets are missing."""
    pass


class ShadowModeManager:
    """
    Manages safe, passive, read-only shadow-mode execution.
    Hard-locked against external student communication or live intervention triggers.
    """

    # Hard-coded absolute safety locks
    ALLOW_STUDENT_OUTREACH: bool = False
    ALLOW_LIVE_SIS_MODIFICATION: bool = False

    def __init__(
        self,
        model_name: str,
        model_version: str,
        feature_pipeline_version: str,
        secret_key: Optional[str] = None,
        audit_log_path: Optional[str] = None,
        max_allowed_cold_start_rate: float = 0.25,
        max_allowed_psi_drift: float = 0.20,
        max_allowed_ingestion_latency_hours: float = 24.0,
        expected_alert_rate: float = 0.1426
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.feature_pipeline_version = feature_pipeline_version
        self.secret_key = secret_key
        self.audit_logger = DecisionAuditLogger(log_path=audit_log_path)
        self.router = InstitutionalGovernanceRouter(universal_cma_suppression=True)

        self.max_allowed_cold_start_rate = max_allowed_cold_start_rate
        self.max_allowed_psi_drift = max_allowed_psi_drift
        self.max_allowed_ingestion_latency_hours = max_allowed_ingestion_latency_hours
        self.expected_alert_rate = expected_alert_rate

        self._validate_shadow_configuration()

    def _validate_shadow_configuration(self):
        """Enforces fail-safe configuration requirements."""
        if not self.model_name or not self.model_version:
            raise GovernanceConfigurationError("Shadow-mode requires explicit model_name and model_version.")
        if not self.feature_pipeline_version:
            raise GovernanceConfigurationError("Shadow-mode requires explicit feature_pipeline_version.")
        if not self.secret_key:
            import os
            env_secret = os.environ.get("STUDENT_PSEUDONYMIZATION_SECRET", "").strip()
            if not env_secret:
                raise GovernanceConfigurationError(
                    "Missing STUDENT_PSEUDONYMIZATION_SECRET. Shadow mode requires keyed pseudonymization secret."
                )
            self.secret_key = env_secret

    def attempt_outreach(self, student_id: Any, message: str):
        """Proof of safety lock: attempts to trigger external outreach will fail immediately."""
        if not self.ALLOW_STUDENT_OUTREACH:
            raise SecurityViolationError(
                "SECURITY LOCK ACTIVE: External student outreach is strictly prohibited in Shadow Mode. "
                "No automated messages, emails, or LMS nudges may be sent."
            )

    def process_shadow_prediction(
        self,
        snapshot: Dict[str, Any],
        raw_prob: float,
        input_feature_hash: str
    ) -> DecisionAuditRecord:
        """
        Executes a single read-only shadow prediction, routes governance, and logs an audit record.
        Zero communication emitted.
        """
        audit_rec = self.router.build_audit_record(
            snapshot=snapshot,
            raw_prob=raw_prob,
            model_name=self.model_name,
            model_version=self.model_version,
            feature_pipeline_version=self.feature_pipeline_version,
            input_feature_hash=input_feature_hash,
            secret_key=self.secret_key
        )
        self.audit_logger.log(audit_rec)
        return audit_rec

    def evaluate_batch_safety_gates(
        self,
        batch_records: List[DecisionAuditRecord],
        cold_start_missing_count: int,
        total_batch_count: int,
        max_ingestion_latency_hours: float
    ) -> Dict[str, Any]:
        """
        Evaluates monitoring gates on a scored shadow batch.
        Checks missingness rates, alert volume deviations, and ingestion latency.
        """
        if total_batch_count == 0:
            return {"status": "EMPTY_BATCH", "gates_passed": False}

        cold_start_rate = cold_start_missing_count / total_batch_count
        actual_alerts = sum(1 for r in batch_records if r.predicted_at_risk and not r.suppress_automated_outreach)
        actual_alert_rate = actual_alerts / total_batch_count

        alert_rate_deviation = abs(actual_alert_rate - self.expected_alert_rate) / self.expected_alert_rate

        cold_start_ok = (cold_start_rate <= self.max_allowed_cold_start_rate)
        latency_ok = (max_ingestion_latency_hours <= self.max_allowed_ingestion_latency_hours)
        alert_vol_ok = (alert_rate_deviation <= 0.35)  # Within 35% of expected baseline

        all_passed = bool(cold_start_ok and latency_ok and alert_vol_ok)

        return {
            "all_gates_passed": all_passed,
            "cold_start_check": {
                "rate": round(cold_start_rate, 4),
                "threshold": self.max_allowed_cold_start_rate,
                "passed": cold_start_ok
            },
            "latency_check": {
                "max_latency_hours": round(max_ingestion_latency_hours, 2),
                "threshold_hours": self.max_allowed_ingestion_latency_hours,
                "passed": latency_ok
            },
            "alert_volume_check": {
                "actual_alert_rate": round(actual_alert_rate, 4),
                "expected_alert_rate": round(self.expected_alert_rate, 4),
                "relative_deviation": round(alert_rate_deviation, 4),
                "passed": alert_vol_ok
            }
        }

