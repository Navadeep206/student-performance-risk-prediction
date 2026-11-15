"""
Institutional Governance Router Engine.
Phase 9 Operational Gap Closure (Task A1).

Provides:
- Universal CMA outreach suppression across all modules (AAA through GGG).
- Preserves CMA risk scoring as research-only telemetry.
- Enforces calibrated threshold (0.10) for Module GGG TMAs.
- Enforces standard operational threshold (0.20) for Modules AAA-FFF TMAs.
- Enforces safe handling and outreach suppression for unverified/missing assessment metadata.
- Integrates seamlessly with decision-level audit logging.
"""

from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timezone

from src.decision_audit import DecisionAuditRecord, pseudonymize_student_id


class SecurityViolationError(Exception):
    """Raised when an attempt is made to bypass shadow-mode outreach prohibitions."""
    pass


VALID_ASSESSMENT_TYPES = {"TMA", "CMA", "Exam"}
VALID_OULAD_MODULES = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}


class InstitutionalGovernanceRouter:
    """
    Enterprise-grade institutional governance router for Student At-Risk Prediction.
    Implements universal CMA suppression, module-specific threshold calibration,
    strict shadow-mode outreach prohibition, and fail-safe handling of unverified metadata.
    """

    def __init__(
        self,
        tma_calibrated_threshold: float = 0.10,
        global_threshold: float = 0.20,
        universal_cma_suppression: bool = True,
        shadow_mode: bool = True
    ):
        self.tma_calibrated_threshold = float(tma_calibrated_threshold)
        self.global_threshold = float(global_threshold)
        self.universal_cma_suppression = bool(universal_cma_suppression)
        self.shadow_mode = bool(shadow_mode)

    def attempt_outreach(self, student_id: Any, message: str):
        """Proof of safety lock: attempts to trigger external student communication will fail immediately."""
        if self.shadow_mode:
            raise SecurityViolationError(
                "SECURITY LOCK ACTIVE: External student outreach is strictly prohibited in Shadow Mode. "
                "No automated messages, emails, or LMS nudges may be sent."
            )

    def route_prediction(self, snapshot: Dict[str, Any], raw_prob: float) -> Dict[str, Any]:
        """
        Routes a student prediction snapshot through institutional governance rules.
        
        Args:
            snapshot: Dictionary containing assessment and module metadata:
                      - code_module: Course code (AAA - GGG)
                      - target_assessment_type: Assessment type (TMA, CMA, Exam)
                      - scheduled_date: Assessment deadline day
                      - target_assessment_weight: Course percentage weight (optional)
            raw_prob: Continuous risk probability from predictive model [0.0, 1.0].
            
        Returns:
            Dictionary containing operational governance verdict, threshold, confidence tier,
            suppression flag, outreach permission flag, and advisor recommendation.
        """
        mod = str(snapshot.get("code_module", "")).strip().upper()
        a_type = str(snapshot.get("target_assessment_type", "")).strip()
        weight = float(snapshot.get("target_assessment_weight", 0.0))
        sched_date = int(snapshot.get("scheduled_date", 0))
        prob = round(float(raw_prob), 4)

        # ---------------------------------------------------------------------
        # 1. Metadata Validation & Fail-Safe Route (Task A1.6)
        # ---------------------------------------------------------------------
        if not a_type or a_type not in VALID_ASSESSMENT_TYPES:
            return {
                "code_module": mod if mod else "UNKNOWN",
                "target_assessment_type": a_type if a_type else "UNKNOWN",
                "raw_risk_probability": prob,
                "operational_threshold_applied": None,
                "predicted_at_risk": False,
                "confidence_tier": "UNVERIFIED_ASSESSMENT_TYPE",
                "governance_action": "MANUAL_ADVISOR_REVIEW",
                "suppress_automated_outreach": True,
                "student_outreach_enabled": False,
                "advising_recommendation": (
                    "Unverified assessment metadata. Automated student outreach is suppressed. "
                    "Manual academic advisor verification required."
                )
            }

        # ---------------------------------------------------------------------
        # 2. Universal CMA Outreach Suppression Route (Task A1.1 - A1.3)
        # ---------------------------------------------------------------------
        if a_type == "CMA":
            if mod == "GGG":
                # Module GGG practice quizzes: 0% weight, decoupled telemetry
                return {
                    "code_module": "GGG",
                    "target_assessment_type": "CMA",
                    "raw_risk_probability": prob,
                    "operational_threshold_applied": None,
                    "predicted_at_risk": False,
                    "confidence_tier": "UNRELIABLE_DECOUPLED_QUIZ",
                    "governance_action": "AUTOMATED_OUTREACH_SUPPRESSED",
                    "suppress_automated_outreach": True,
                    "student_outreach_enabled": False,
                    "advising_recommendation": (
                        "Formative self-paced quiz with zero module weight and decoupled telemetry. "
                        "Do NOT trigger automated student alerts. Monitor progress through course tutor."
                    )
                }
            else:
                # Modules AAA through FFF: Computer-marked quizzes
                return {
                    "code_module": mod,
                    "target_assessment_type": "CMA",
                    "raw_risk_probability": prob,
                    "operational_threshold_applied": None,
                    "predicted_at_risk": False,
                    "confidence_tier": "RESEARCH_ONLY_TELEMETRY",
                    "governance_action": "AUTOMATED_OUTREACH_SUPPRESSED",
                    "suppress_automated_outreach": True,
                    "student_outreach_enabled": False,
                    "advising_recommendation": (
                        f"Computer-marked quiz (CMA) in Module {mod}. Designated strictly as research-only "
                        "telemetry. Automated student outreach is prohibited across all modules."
                    )
                }

        # ---------------------------------------------------------------------
        # 3. Tutor-Marked Assignment (TMA) Routing (Task A1.4 - A1.5)
        # ---------------------------------------------------------------------
        if a_type == "TMA":
            if mod == "GGG":
                eff_threshold = self.tma_calibrated_threshold
                is_at_risk = bool(prob >= eff_threshold)
                tier = "LOW_CONFIDENCE_ONBOARDING" if sched_date <= 70 else "MODERATE_CONFIDENCE_MILESTONE"
            else:
                eff_threshold = self.global_threshold
                is_at_risk = bool(prob >= eff_threshold)
                tier = "HIGH_CONFIDENCE_STANDARD"

            if self.shadow_mode:
                gov_action = "SHADOW_MODE_ADVISORY_REVIEW" if is_at_risk else "SHADOW_MODE_MONITORING"
                rec = (
                    f"{'Module GGG' if mod == 'GGG' else f'Standard'} Tutor-Marked Assignment in Module {mod}. "
                    f"Risk score ({prob:.4f}) {'meets or exceeds operational threshold' if is_at_risk else 'is below operational threshold'} ({eff_threshold:.2f}). "
                    f"[SHADOW MODE ACTIVE: External student outreach is strictly prohibited. Passive advisor review only.]"
                )
                return {
                    "code_module": mod,
                    "target_assessment_type": "TMA",
                    "raw_risk_probability": prob,
                    "operational_threshold_applied": eff_threshold,
                    "predicted_at_risk": is_at_risk,
                    "confidence_tier": tier,
                    "governance_action": gov_action,
                    "suppress_automated_outreach": True,
                    "student_outreach_enabled": False,
                    "advising_recommendation": rec
                }
            else:
                # Unconstrained simulation mode (non-shadow)
                if mod == "GGG":
                    if sched_date <= 70:
                        gov_act = "SUPPORTIVE_CHECK_IN_ONLY" if is_at_risk else "MONITORING"
                    else:
                        gov_act = "SYNCHRONOUS_ADVISOR_REVIEW" if is_at_risk else "MONITORING"
                else:
                    gov_act = "AUTOMATED_OUTREACH_ENABLED" if is_at_risk else "MONITORING"

                return {
                    "code_module": mod,
                    "target_assessment_type": "TMA",
                    "raw_risk_probability": prob,
                    "operational_threshold_applied": eff_threshold,
                    "predicted_at_risk": is_at_risk,
                    "confidence_tier": tier,
                    "governance_action": gov_act,
                    "suppress_automated_outreach": not is_at_risk,
                    "student_outreach_enabled": is_at_risk,
                    "advising_recommendation": f"TMA evaluation for Module {mod}."
                }

        # ---------------------------------------------------------------------
        # 4. Final Examination Milestone Route
        # ---------------------------------------------------------------------
        if a_type == "Exam":
            is_at_risk = bool(prob >= self.global_threshold)
            return {
                "code_module": mod,
                "target_assessment_type": "Exam",
                "raw_risk_probability": prob,
                "operational_threshold_applied": self.global_threshold,
                "predicted_at_risk": is_at_risk,
                "confidence_tier": "FINAL_EXAMINATION_ROSTER",
                "governance_action": "SHADOW_MODE_ADVISORY_REVIEW" if is_at_risk else "SHADOW_MODE_MONITORING",
                "suppress_automated_outreach": True,
                "student_outreach_enabled": False,
                "advising_recommendation": (
                    "Final examination milestone. Automated student outreach suppressed; "
                    "advisor roster review only."
                )
            }

        # Fallback (should not reach here due to type check above)
        return {
            "code_module": mod,
            "target_assessment_type": a_type,
            "raw_risk_probability": prob,
            "operational_threshold_applied": self.global_threshold,
            "predicted_at_risk": bool(prob >= self.global_threshold),
            "confidence_tier": "UNVERIFIED_ASSESSMENT_TYPE",
            "governance_action": "MANUAL_ADVISOR_REVIEW",
            "suppress_automated_outreach": True,
            "student_outreach_enabled": False,
            "advising_recommendation": "Manual verification required."
        }

    def build_audit_record(
        self,
        snapshot: Dict[str, Any],
        raw_prob: float,
        model_name: str,
        model_version: str,
        feature_pipeline_version: str,
        input_feature_hash: str,
        secret_key: Optional[str] = None
    ) -> DecisionAuditRecord:
        """
        Routes the prediction and constructs a fully validated, immutable DecisionAuditRecord.
        Safely generates keyed HMAC-SHA256 pseudonym for the student identifier.
        """
        route = self.route_prediction(snapshot, raw_prob)
        raw_student_id = snapshot.get("id_student", snapshot.get("student_id", "UNKNOWN"))
        pseudonym = pseudonymize_student_id(raw_student_id, secret_key=secret_key)
        
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pred_id = str(uuid.uuid4())
        
        return DecisionAuditRecord(
            prediction_id=pred_id,
            timestamp_utc=now_utc,
            pseudonymized_student_id=pseudonym,
            code_module=route["code_module"],
            code_presentation=str(snapshot.get("code_presentation", "")),
            id_assessment=int(snapshot.get("id_assessment", 0)),
            assessment_type=route["target_assessment_type"],
            scheduled_date=int(snapshot.get("scheduled_date", 0)),
            model_name=model_name,
            model_version=model_version,
            feature_pipeline_version=feature_pipeline_version,
            input_feature_hash=input_feature_hash,
            raw_risk_probability=route["raw_risk_probability"],
            operational_threshold=route["operational_threshold_applied"],
            predicted_at_risk=route["predicted_at_risk"],
            governance_routing_action=route["governance_action"],
            suppress_automated_outreach=route["suppress_automated_outreach"],
            student_outreach_enabled=route.get("student_outreach_enabled", False),
            routing_reason=route["advising_recommendation"]
        )

