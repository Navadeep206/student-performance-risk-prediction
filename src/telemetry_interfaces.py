"""
Telemetry and Gradebook Integration Readiness Interfaces.
Phase 9 Operational Gap Closure (Task A5).

Provides:
- Ingestion data schemas for Registration, Course Schedule, VLE Events, and Gradebook Releases.
- Explicit distinction between Event Timestamps and Ingestion Timestamps.
- Temporal cutoff enforcement rejecting late-arriving events.
- Distinction between offline simulated 14-day delay and live production grade publication timestamps.
- Operational horizon feasibility classification (7-day candidate vs 1-day unverified).
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum


class HorizonFeasibilityStatus(Enum):
    RESEARCH_TO_OPERATIONAL_CANDIDATE = "7-Day Pre-Deadline Horizon: Candidate for shadow-mode operational evaluation with 24-48h ingestion buffer."
    OPERATIONALLY_UNVERIFIED_FRAGILE = "1-Day Pre-Deadline Horizon: Operationally unverified. Sub-24h event ingestion cannot be guaranteed across legacy LMS infrastructure."


@dataclass(frozen=True)
class RegistrationRecord:
    student_id: int
    code_module: str
    code_presentation: str
    date_registration: int  # Days relative to course start
    event_timestamp_utc: str
    ingestion_timestamp_utc: str

    def __post_init__(self):
        if not self.code_module or not self.code_presentation:
            raise ValueError("Registration validation error: code_module and code_presentation must be non-empty.")


@dataclass(frozen=True)
class CourseScheduleRecord:
    code_module: str
    code_presentation: str
    id_assessment: int
    assessment_type: str
    scheduled_date: int
    weight: float
    event_timestamp_utc: str
    ingestion_timestamp_utc: str

    def __post_init__(self):
        if self.assessment_type not in {"TMA", "CMA", "Exam"}:
            raise ValueError(f"Course schedule validation error: invalid assessment_type '{self.assessment_type}'.")
        if self.weight < 0.0 or self.weight > 100.0:
            raise ValueError("Course schedule validation error: weight must be between 0.0 and 100.0.")


@dataclass(frozen=True)
class VLEEventRecord:
    student_id: int
    code_module: str
    code_presentation: str
    id_site: int
    activity_type: str
    date: int  # Day relative to course start
    sum_click: int
    event_timestamp_utc: str
    ingestion_timestamp_utc: str

    def __post_init__(self):
        if self.sum_click < 1:
            raise ValueError("VLE validation error: sum_click must be >= 1.")


@dataclass(frozen=True)
class GradebookReleaseRecord:
    student_id: int
    id_assessment: int
    date_submitted: int
    score: Optional[float]
    publication_timestamp_utc: Optional[str]  # Real-world instructor release timestamp
    ingestion_timestamp_utc: str

    def __post_init__(self):
        if self.score is not None and not (0.0 <= float(self.score) <= 100.0):
            raise ValueError(f"Gradebook validation error: score {self.score} out of bounds [0.0, 100.0].")


class CutoffEnforcementPolicy:
    """Enforces temporal cutoff rules, distinguishes event from ingestion time, and verifies grade release."""

    @staticmethod
    def evaluate_vle_event(
        event: VLEEventRecord,
        cutoff_date: int,
        batch_execution_timestamp_utc: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validates whether a VLE event is eligible for feature engineering at a given prediction cutoff.
        Rejects events occurring after cutoff date or ingested after batch execution.
        """
        # 1. Event relative day cutoff check
        if event.date > cutoff_date:
            return False, f"REJECTED: VLE event relative day ({event.date}) occurs after cutoff date ({cutoff_date})."

        # 2. Ingestion latency check
        if batch_execution_timestamp_utc is not None:
            event_ingest = datetime.fromisoformat(event.ingestion_timestamp_utc.replace("Z", "+00:00"))
            batch_run = datetime.fromisoformat(batch_execution_timestamp_utc.replace("Z", "+00:00"))
            if event_ingest > batch_run:
                return False, f"REJECTED: Event ingestion timestamp ({event.ingestion_timestamp_utc}) is after batch execution ({batch_execution_timestamp_utc})."

        return True, "ELIGIBLE: Event occurred and was ingested prior to cutoff."

    @staticmethod
    def is_grade_available_offline_simulation(
        grade_record: GradebookReleaseRecord,
        cutoff_date: int,
        assumed_delay_days: int = 14
    ) -> Tuple[bool, str]:
        """
        Offline simulated benchmark policy (assumes uniform 14-day grading delay).
        Used strictly for reproducing certified Phase 5-9 research benchmarks.
        """
        simulated_available_date = grade_record.date_submitted + assumed_delay_days
        if simulated_available_date <= cutoff_date:
            return True, f"AVAILABLE: date_submitted ({grade_record.date_submitted}) + delay ({assumed_delay_days}d) <= cutoff ({cutoff_date})."
        return False, f"UNAVAILABLE: Grading delay simulated availability ({simulated_available_date}) is after cutoff ({cutoff_date})."

    @staticmethod
    def is_grade_available_production(
        grade_record: GradebookReleaseRecord,
        cutoff_timestamp_utc: str
    ) -> Tuple[bool, str]:
        """
        Live institutional gradebook production interface.
        Accepts actual publication_timestamp_utc from LMS/SIS grade release events.
        """
        if grade_record.publication_timestamp_utc is None:
            return False, "UNAVAILABLE: Instructor has not yet published grade in institutional gradebook."

        pub_time = datetime.fromisoformat(grade_record.publication_timestamp_utc.replace("Z", "+00:00"))
        cutoff_time = datetime.fromisoformat(cutoff_timestamp_utc.replace("Z", "+00:00"))

        if pub_time > cutoff_time:
            return False, f"UNAVAILABLE: Grade published at {grade_record.publication_timestamp_utc}, after prediction cutoff ({cutoff_timestamp_utc})."

        ingest_time = datetime.fromisoformat(grade_record.ingestion_timestamp_utc.replace("Z", "+00:00"))
        if ingest_time > cutoff_time:
            return False, f"UNAVAILABLE: Grade published on time but SIS/LMS ingestion lag ({grade_record.ingestion_timestamp_utc}) exceeded cutoff."

        return True, f"AVAILABLE: Grade officially published and ingested before cutoff ({cutoff_timestamp_utc})."

