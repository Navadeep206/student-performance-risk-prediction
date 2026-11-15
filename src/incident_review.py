"""
Threshold-Aware Incident Review Module.
Phase 9 Operational Gap Closure (Task A4).

Provides:
- Threshold-aware review of prediction outcomes against ground-truth assessment scores.
- Distinguishes standard TMA predictions (tau = 0.20) from GGG TMA predictions (tau = 0.10).
- Excludes CMA predictions from automated outreach incident review workflows.
- Handles unknown or unverified assessment metadata safely.
- Classifies outcomes into True Positives, True Negatives, False Positives, False Negatives.
- Avoids defect bias: does not assume every false positive is a model defect or every false negative is an avoidable failure.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.decision_audit import DecisionAuditRecord


@dataclass(frozen=True)
class IncidentReviewRecord:
    """Immutable record capturing the post-assessment review of a predictive decision."""
    prediction_id: str
    pseudonymized_student_id: str
    code_module: str
    code_presentation: str
    id_assessment: int
    assessment_type: str
    raw_risk_probability: float
    operational_threshold_applied: Optional[float]
    predicted_at_risk: bool
    actual_score: Optional[float]
    actual_at_risk: Optional[bool]
    is_incident_eligible: bool
    outcome_classification: str
    incident_category: str
    investigation_notes: str
    routing_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ThresholdAwareIncidentReviewEngine:
    """Evaluates retrospective prediction outcomes with threshold-awareness and CMA outreach exclusion."""

    @staticmethod
    def review_decision(
        audit_record: DecisionAuditRecord,
        actual_score: Optional[float]
    ) -> IncidentReviewRecord:
        """
        Reviews a DecisionAuditRecord against the ground-truth assessment score.
        
        Args:
            audit_record: The immutable DecisionAuditRecord generated during inference.
            actual_score: Retrospective score received by student [0.0 - 100.0] or None if missing.
        """
        atype = audit_record.assessment_type
        mod = audit_record.code_module
        prob = audit_record.raw_risk_probability
        threshold = audit_record.operational_threshold

        # ---------------------------------------------------------------------
        # 1. Unverified or Unknown Assessment Metadata
        # ---------------------------------------------------------------------
        if atype not in {"TMA", "CMA", "Exam"} or audit_record.governance_routing_action == "MANUAL_ADVISOR_REVIEW":
            return IncidentReviewRecord(
                prediction_id=audit_record.prediction_id,
                pseudonymized_student_id=audit_record.pseudonymized_student_id,
                code_module=mod,
                code_presentation=audit_record.code_presentation,
                id_assessment=audit_record.id_assessment,
                assessment_type=atype,
                raw_risk_probability=prob,
                operational_threshold_applied=threshold,
                predicted_at_risk=audit_record.predicted_at_risk,
                actual_score=actual_score,
                actual_at_risk=None if actual_score is None else (actual_score < 50.0),
                is_incident_eligible=False,
                outcome_classification="EXCLUDED_UNVERIFIED_METADATA",
                incident_category="METADATA_ANOMALY",
                investigation_notes=(
                    "Prediction had unverified or missing assessment metadata and was suppressed "
                    "from automated outreach. Excluded from automated advisor incident review."
                ),
                routing_reason=audit_record.routing_reason
            )

        # ---------------------------------------------------------------------
        # 2. Universal CMA Outreach Exclusion
        # ---------------------------------------------------------------------
        if atype == "CMA":
            actual_at_risk = None if actual_score is None else bool(actual_score < 50.0)
            return IncidentReviewRecord(
                prediction_id=audit_record.prediction_id,
                pseudonymized_student_id=audit_record.pseudonymized_student_id,
                code_module=mod,
                code_presentation=audit_record.code_presentation,
                id_assessment=audit_record.id_assessment,
                assessment_type="CMA",
                raw_risk_probability=prob,
                operational_threshold_applied=threshold,
                predicted_at_risk=audit_record.predicted_at_risk,
                actual_score=actual_score,
                actual_at_risk=actual_at_risk,
                is_incident_eligible=False,
                outcome_classification="CMA_RESEARCH_TELEMETRY_EXCLUDED",
                incident_category="NOT_INCIDENT_ELIGIBLE",
                investigation_notes=(
                    "CMA predictions are designated strictly as research-only telemetry with outreach "
                    "suppressed. Computer-marked quizzes do not trigger human advisor outreach and are "
                    "excluded from the operational incident review workflow."
                ),
                routing_reason=audit_record.routing_reason
            )

        # ---------------------------------------------------------------------
        # 3. Tutor-Marked Assignment (TMA) Threshold-Aware Review
        # ---------------------------------------------------------------------
        if actual_score is None:
            # Student did not submit or grade is missing
            actual_at_risk = True  # Non-submission is failing / academic risk
        else:
            actual_at_risk = bool(actual_score < 50.0)

        # Confirm the threshold matches module policy
        expected_threshold = 0.10 if mod == "GGG" else 0.20
        effective_threshold = threshold if threshold is not None else expected_threshold
        predicted_risk = audit_record.predicted_at_risk

        if predicted_risk and actual_at_risk:
            outcome = "TRUE_POSITIVE"
            category = "TIMELY_EARLY_WARNING"
            notes = (
                f"Student was correctly flagged (probability {prob:.4f} >= threshold {effective_threshold:.2f}) "
                f"and experienced academic failure (score: {actual_score}). Early warning successfully alerted."
            )
        elif not predicted_risk and not actual_at_risk:
            outcome = "TRUE_NEGATIVE"
            category = "ACCURATE_NON_ALERT"
            notes = (
                f"Student was not flagged (probability {prob:.4f} < threshold {effective_threshold:.2f}) "
                f"and achieved a passing score ({actual_score}). No advisor burden created."
            )
        elif predicted_risk and not actual_at_risk:
            outcome = "FALSE_POSITIVE"
            category = "INVESTIGATE_BENIGN_ALERT"
            notes = (
                f"Student was flagged (probability {prob:.4f} >= threshold {effective_threshold:.2f}) "
                f"but achieved a passing score ({actual_score}). Potential explanations include: "
                "successful student response to advisor/tutor check-in, late cramming activity between "
                "cutoff and submission deadline, or conservative probability calibration. "
                "Does NOT inherently represent a model defect."
            )
        else:  # not predicted_risk and actual_at_risk
            outcome = "FALSE_NEGATIVE"
            category = "INVESTIGATE_UNALERTED_FAILURE"
            notes = (
                f"Student was not flagged (probability {prob:.4f} < threshold {effective_threshold:.2f}) "
                f"but failed the assessment (score: {actual_score}). Potential explanations include: "
                "acute personal illness/disruption not reflected in LMS clickstream, sudden disengagement, "
                "or cold-start feature sparsity. Does NOT inherently represent an avoidable failure."
            )

        return IncidentReviewRecord(
            prediction_id=audit_record.prediction_id,
            pseudonymized_student_id=audit_record.pseudonymized_student_id,
            code_module=mod,
            code_presentation=audit_record.code_presentation,
            id_assessment=audit_record.id_assessment,
            assessment_type=atype,
            raw_risk_probability=prob,
            operational_threshold_applied=effective_threshold,
            predicted_at_risk=predicted_risk,
            actual_score=actual_score,
            actual_at_risk=actual_at_risk,
            is_incident_eligible=True,
            outcome_classification=outcome,
            incident_category=category,
            investigation_notes=notes,
            routing_reason=audit_record.routing_reason
        )

