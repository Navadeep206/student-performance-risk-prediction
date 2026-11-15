"""
Decision-Level Audit Logging and Privacy-Safe Pseudonymization Module.
Phase 9 Operational Gap Closure (Tasks A2 and A3).

Provides:
- Keyed HMAC-SHA256 pseudonymization with environment secret configuration.
- Structured, immutable DecisionAuditRecord with comprehensive model and feature metadata.
- Validation rejecting invalid probabilities, invalid thresholds, and missing metadata.
- Thread-safe DecisionAuditLogger for audit compliance.
"""

import os
import hmac
import hashlib
import json
import math
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union


# =============================================================================
# 1. KEYED HMAC-SHA256 PSEUDONYMIZATION (TASK A3)
# =============================================================================

def get_pseudonymization_secret() -> str:
    """
    Retrieves the HMAC secret key from the environment variable.
    Raises ValueError if not configured or empty.
    Never hardcoded or stored in Git.
    """
    secret = os.environ.get("STUDENT_PSEUDONYMIZATION_SECRET", "").strip()
    if not secret:
        raise ValueError(
            "Security Configuration Error: STUDENT_PSEUDONYMIZATION_SECRET environment variable "
            "is not set or empty. Raw student IDs cannot be processed without keyed pseudonymization."
        )
    return secret


def pseudonymize_student_id(raw_student_id: Union[int, str], secret_key: Optional[str] = None) -> str:
    """
    Generates a deterministic, keyed HMAC-SHA256 pseudonym for a student identifier.
    
    Security & Privacy Properties:
    - Keyed Cryptographic Hash: Resistant to dictionary and rainbow-table attacks.
    - Deterministic: Identical student ID + identical secret yields identical pseudonym.
    - Distinct with Different Secrets: Re-keying (secret rotation) produces independent pseudonyms.
    - Pseudonymous Personal Data: In compliance with GDPR Art. 4(5) and FERPA, this is pseudonymous,
      NOT anonymous, because the institution holding the secret key can re-identify learners.
    
    Args:
        raw_student_id: Integer or string student identifier from raw data/LMS.
        secret_key: Optional secret key. If None, reads from STUDENT_PSEUDONYMIZATION_SECRET.
    
    Returns:
        64-character lowercase hexadecimal HMAC-SHA256 digest.
    """
    if secret_key is None:
        secret_key = get_pseudonymization_secret()
    
    if not isinstance(secret_key, str) or len(secret_key) < 16:
        raise ValueError("Cryptographic Error: Pseudonymization secret must be a string of at least 16 characters.")
    
    raw_str = str(raw_student_id).strip()
    if not raw_str:
        raise ValueError("Validation Error: raw_student_id cannot be empty or whitespace.")
    
    return hmac.new(
        key=secret_key.encode("utf-8"),
        msg=raw_str.encode("utf-8"),
        digestmod=hashlib.sha256
    ).hexdigest()


# =============================================================================
# 2. STRUCTURED IMMUTABLE DECISION AUDIT RECORD (TASK A2)
# =============================================================================

@dataclass(frozen=True)
class DecisionAuditRecord:
    """
    Immutable decision-level audit record capturing the complete operational provenance
    of every predictive inference decision made by the system.
    """
    prediction_id: str
    timestamp_utc: str
    pseudonymized_student_id: str
    code_module: str
    code_presentation: str
    id_assessment: int
    assessment_type: str
    scheduled_date: int
    model_name: str
    model_version: str
    feature_pipeline_version: str
    input_feature_hash: str
    raw_risk_probability: float
    operational_threshold: Optional[float]
    predicted_at_risk: bool
    governance_routing_action: str
    suppress_automated_outreach: bool
    routing_reason: str
    student_outreach_enabled: bool = False

    def __post_init__(self):
        """Enforces strict validation of all audit fields upon instantiation."""
        # 1. Identification and Versioning
        if not self.prediction_id or not isinstance(self.prediction_id, str):
            raise ValueError("Audit Validation Error: prediction_id must be a non-empty string.")
        
        if not self.timestamp_utc or not isinstance(self.timestamp_utc, str):
            raise ValueError("Audit Validation Error: timestamp_utc must be a non-empty string.")
            
        if not self.pseudonymized_student_id or len(self.pseudonymized_student_id) != 64:
            raise ValueError("Audit Validation Error: pseudonymized_student_id must be a 64-char HMAC-SHA256 hex digest.")
            
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("Audit Validation Error: model_name must be specified.")
            
        if not self.model_version or not isinstance(self.model_version, str):
            raise ValueError("Audit Validation Error: model_version must be specified.")
            
        if not self.feature_pipeline_version or not isinstance(self.feature_pipeline_version, str):
            raise ValueError("Audit Validation Error: feature_pipeline_version must be specified.")
            
        if not self.input_feature_hash or not isinstance(self.input_feature_hash, str):
            raise ValueError("Audit Validation Error: input_feature_hash must be specified.")

        # 2. Probability Validation
        if not isinstance(self.raw_risk_probability, (int, float)) or math.isnan(self.raw_risk_probability):
            raise ValueError("Audit Validation Error: raw_risk_probability must be a valid float.")
        if not (0.0 <= float(self.raw_risk_probability) <= 1.0):
            raise ValueError(f"Audit Validation Error: raw_risk_probability {self.raw_risk_probability} out of bounds [0.0, 1.0].")

        # 3. Threshold Validation
        if self.operational_threshold is not None:
            if not isinstance(self.operational_threshold, (int, float)) or math.isnan(self.operational_threshold):
                raise ValueError("Audit Validation Error: operational_threshold must be a valid float or None.")
            if not (0.0 < float(self.operational_threshold) <= 1.0):
                raise ValueError(f"Audit Validation Error: operational_threshold {self.operational_threshold} out of bounds (0.0, 1.0].")

        # 4. Governance Routing Validation
        if not self.governance_routing_action or not isinstance(self.governance_routing_action, str):
            raise ValueError("Audit Validation Error: governance_routing_action must be specified.")
            
        if not self.routing_reason or not isinstance(self.routing_reason, str):
            raise ValueError("Audit Validation Error: routing_reason must be specified.")

        if not isinstance(self.suppress_automated_outreach, bool):
            raise ValueError("Audit Validation Error: suppress_automated_outreach must be a boolean.")

        if not isinstance(self.student_outreach_enabled, bool):
            raise ValueError("Audit Validation Error: student_outreach_enabled must be a boolean.")

    def to_dict(self) -> Dict[str, Any]:
        """Returns the audit record as a standard dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Returns serialized JSON string of the audit record."""
        return json.dumps(self.to_dict())


# =============================================================================
# 3. DECISION AUDIT LOGGER
# =============================================================================

class DecisionAuditLogger:
    """Thread-safe append-only logger for decision audit records."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self._in_memory_records: List[DecisionAuditRecord] = []
        self._lock = threading.Lock()

    def log(self, record: DecisionAuditRecord):
        """Logs a decision audit record to memory and optional JSONL file in a thread-safe manner."""
        if not isinstance(record, DecisionAuditRecord):
            raise TypeError("Expected DecisionAuditRecord instance.")
            
        with self._lock:
            self._in_memory_records.append(record)

            if self.log_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(record.to_json() + "\n")

    def get_records(self) -> List[DecisionAuditRecord]:
        """Returns all in-memory audit records in a thread-safe manner."""
        with self._lock:
            return list(self._in_memory_records)

    def clear(self):
        """Clears in-memory records (for testing purposes) in a thread-safe manner."""
        with self._lock:
            self._in_memory_records.clear()

