"""
src/data_preprocessing.py

Production data-cleaning and relational integration pipeline for the Open University
Learning Analytics Dataset (OULAD).

Converts the 7 raw CSV tables into validated, typed processed datasets:
- data/processed/enrollments_clean.csv
- data/processed/assessments_clean.csv
- data/processed/assessment_submissions_clean.csv
- data/processed/vle_interactions_clean.csv
- data/processed/enrollment_base.csv

Adheres strictly to non-destructive principles: raw data is immutable, no premature
targets are created, and left joins ensure zero student enrollment records are lost.
"""

import os
import time
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class OULADPreprocessor:
    """End-to-end data cleaning, typing, and integration pipeline for OULAD."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.summary_metrics: Dict[str, Any] = {}

    def _raw_path(self, filename: str) -> str:
        return os.path.join(self.raw_dir, filename)

    def _processed_path(self, filename: str) -> str:
        return os.path.join(self.processed_dir, filename)

    def clean_courses(self) -> pd.DataFrame:
        """Loads and types courses.csv."""
        raw_path = self._raw_path("courses.csv")
        df = pd.read_csv(raw_path)
        
        # Schema typing
        df["code_module"] = df["code_module"].astype("string")
        df["code_presentation"] = df["code_presentation"].astype("string")
        df["module_presentation_length"] = df["module_presentation_length"].astype("int64")

        # Key validation
        assert not df.duplicated(subset=["code_module", "code_presentation"]).any(), "Duplicate key in courses"
        assert len(df) == 22, f"Expected 22 courses, found {len(df)}"
        return df

    def clean_assessments(self) -> pd.DataFrame:
        """Cleans and types assessments.csv, handling '?' in date."""
        raw_path = self._raw_path("assessments.csv")
        # Load with '?' converted to NaN
        df = pd.read_csv(raw_path, na_values=["?"])

        df["id_assessment"] = df["id_assessment"].astype("int64")
        df["code_module"] = df["code_module"].astype("string")
        df["code_presentation"] = df["code_presentation"].astype("string")
        df["assessment_type"] = df["assessment_type"].astype("string")
        df["date"] = df["date"].astype("Int64")  # Nullable integer
        df["weight"] = df["weight"].astype("float64")

        # Load submissions to identify assessments with active student submissions
        sa_raw = pd.read_csv(self._raw_path("studentAssessment.csv"), usecols=["id_assessment"])
        active_ids = set(sa_raw["id_assessment"].unique())
        df["has_submissions"] = df["id_assessment"].isin(active_ids).astype("int64")

        # Validation
        assert not df.duplicated(subset=["id_assessment"]).any(), "Duplicate id_assessment in assessments"
        assert len(df) == 206, f"Expected 206 assessments, found {len(df)}"
        assert df["date"].isna().sum() == 11, "Expected 11 missing dates in assessments"

        out_path = self._processed_path("assessments_clean.csv")
        df.to_csv(out_path, index=False)
        return df

    def clean_assessment_submissions(self, assessments_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans studentAssessment.csv, types score, enriches with assessment metadata."""
        raw_path = self._raw_path("studentAssessment.csv")
        df = pd.read_csv(raw_path, na_values=["?"])

        # Track missing scores before casting
        df["score_is_missing"] = df["score"].isna().astype("int64")

        df["id_assessment"] = df["id_assessment"].astype("int64")
        df["id_student"] = df["id_student"].astype("int64")
        df["date_submitted"] = df["date_submitted"].astype("int64")
        df["is_banked"] = df["is_banked"].astype("int64")
        df["score"] = df["score"].astype("Float64")  # Nullable float

        # Merge N:1 with assessments metadata
        meta = assessments_df[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]].copy()
        meta = meta.rename(columns={"date": "scheduled_date"})

        merged = df.merge(meta, on="id_assessment", how="left")
        assert len(merged) == len(df), "Row expansion detected during assessment submissions join"

        # Calculate submission lateness: date_submitted - scheduled_date
        # (Positive = late, Negative = early, Null = missing scheduled date e.g. Exams)
        merged["submission_lateness_days"] = (merged["date_submitted"] - merged["scheduled_date"]).astype("Int64")

        # Validation
        assert not merged.duplicated(subset=["id_assessment", "id_student"]).any(), "Duplicate composite key in submissions"
        assert len(merged) == 173912, f"Expected 173,912 submissions, got {len(merged)}"
        assert merged["score"].isna().sum() == 173, "Expected 173 missing scores"

        out_path = self._processed_path("assessment_submissions_clean.csv")
        merged.to_csv(out_path, index=False)
        return merged

    def clean_enrollments(self, courses_df: pd.DataFrame) -> pd.DataFrame:
        """Merges studentInfo and studentRegistration 1:1, joins courses N:1."""
        si_path = self._raw_path("studentInfo.csv")
        sr_path = self._raw_path("studentRegistration.csv")

        si = pd.read_csv(si_path, na_values=["?"])
        sr = pd.read_csv(sr_path, na_values=["?"])

        # Type studentInfo
        si["code_module"] = si["code_module"].astype("string")
        si["code_presentation"] = si["code_presentation"].astype("string")
        si["id_student"] = si["id_student"].astype("int64")
        si["gender"] = si["gender"].astype("string")
        si["region"] = si["region"].astype("string")
        si["highest_education"] = si["highest_education"].astype("string")
        # Treat missing IMD band explicitly as 'Missing' category
        si["imd_band"] = si["imd_band"].fillna("Missing").astype("string")
        si["age_band"] = si["age_band"].astype("string")
        si["num_of_prev_attempts"] = si["num_of_prev_attempts"].astype("int64")
        si["studied_credits"] = si["studied_credits"].astype("int64")
        si["disability"] = si["disability"].astype("string")
        si["final_result"] = si["final_result"].astype("string")

        # Type studentRegistration
        sr["code_module"] = sr["code_module"].astype("string")
        sr["code_presentation"] = sr["code_presentation"].astype("string")
        sr["id_student"] = sr["id_student"].astype("int64")
        sr["date_registration"] = sr["date_registration"].astype("Int64")
        sr["date_unregistration"] = sr["date_unregistration"].astype("Int64")
        sr["has_unregistered"] = sr["date_unregistration"].notna().astype("int64")

        # Symmetric 1:1 Merge
        enrollments = si.merge(sr, on=["code_module", "code_presentation", "id_student"], how="inner")
        assert len(enrollments) == 32593, f"Expected 32,593 enrollments, got {len(enrollments)}"

        # N:1 Merge with courses to add module_presentation_length
        enrollments = enrollments.merge(courses_df[["code_module", "code_presentation", "module_presentation_length"]],
                                        on=["code_module", "code_presentation"], how="left")
        assert len(enrollments) == 32593, "Row expansion during courses join"
        assert not enrollments.duplicated(subset=["code_module", "code_presentation", "id_student"]).any(), "Duplicate enrollment key"

        out_path = self._processed_path("enrollments_clean.csv")
        enrollments.to_csv(out_path, index=False)
        return enrollments

    def clean_vle_interactions(self, chunksize: int = 1_000_000) -> pd.DataFrame:
        """
        Streams studentVle.csv, enriches with vle.activity_type, preserves all raw records,
        and computes per-student aggregated summary metrics.
        """
        vle_path = self._raw_path("vle.csv")
        vle = pd.read_csv(vle_path, na_values=["?"])
        site_type_map = vle.set_index("id_site")["activity_type"].to_dict()

        raw_vle_path = self._raw_path("studentVle.csv")
        out_vle_clean_path = self._processed_path("vle_interactions_clean.csv")

        # If clean VLE already exists and is non-empty, we can stream through it or process directly
        first_chunk = True
        total_rows = 0
        chunk_aggregations = []

        print("Streaming studentVle.csv for typing, enrichment, and aggregation...")
        for chunk in pd.read_csv(raw_vle_path, chunksize=chunksize):
            total_rows += len(chunk)
            
            # Type chunk
            chunk["code_module"] = chunk["code_module"].astype("string")
            chunk["code_presentation"] = chunk["code_presentation"].astype("string")
            chunk["id_student"] = chunk["id_student"].astype("int64")
            chunk["id_site"] = chunk["id_site"].astype("int64")
            chunk["date"] = chunk["date"].astype("int64")
            chunk["sum_click"] = chunk["sum_click"].astype("int64")

            # Enrich with activity_type
            chunk["activity_type"] = chunk["id_site"].map(site_type_map).fillna("other").astype("string")

            # Append to clean output CSV
            chunk.to_csv(out_vle_clean_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False

            # In-chunk aggregations
            chunk["is_precourse"] = chunk["date"] < 0
            chunk["is_core"] = chunk["activity_type"].isin(["oucontent", "resource", "subpage"])
            chunk["is_forum"] = chunk["activity_type"] == "forumng"
            chunk["is_quiz"] = chunk["activity_type"].isin(["quiz", "externalquiz"])

            # Accumulate metrics
            agg = chunk.groupby(["code_module", "code_presentation", "id_student"]).agg(
                chunk_clicks=("sum_click", "sum"),
                min_date=("date", "min"),
                max_date=("date", "max"),
                precourse_clicks=("sum_click", lambda s: s[chunk.loc[s.index, "is_precourse"]].sum()),
                core_clicks=("sum_click", lambda s: s[chunk.loc[s.index, "is_core"]].sum()),
                forum_clicks=("sum_click", lambda s: s[chunk.loc[s.index, "is_forum"]].sum()),
                quiz_clicks=("sum_click", lambda s: s[chunk.loc[s.index, "is_quiz"]].sum()),
                active_days=("date", set),
                distinct_sites=("id_site", set)
            )
            chunk_aggregations.append(agg)

        assert total_rows == 10655280, f"Expected 10,655,280 VLE rows, processed {total_rows}"
        print("Consolidating VLE chunk aggregations across enrollments...")

        # Consolidate chunks
        combined = pd.concat(chunk_aggregations)
        final_vle_agg = combined.groupby(level=["code_module", "code_presentation", "id_student"]).agg({
            "chunk_clicks": "sum",
            "min_date": "min",
            "max_date": "max",
            "precourse_clicks": "sum",
            "core_clicks": "sum",
            "forum_clicks": "sum",
            "quiz_clicks": "sum",
            "active_days": lambda sets: len(set().union(*sets)),
            "distinct_sites": lambda sets: len(set().union(*sets))
        }).reset_index()

        final_vle_agg = final_vle_agg.rename(columns={
            "chunk_clicks": "total_vle_clicks",
            "min_date": "first_vle_interaction_day",
            "max_date": "last_vle_interaction_day",
            "precourse_clicks": "vle_pre_course_clicks",
            "core_clicks": "vle_core_content_clicks",
            "forum_clicks": "vle_forum_clicks",
            "quiz_clicks": "vle_quiz_clicks",
            "active_days": "vle_active_days_count",
            "distinct_sites": "vle_distinct_materials_accessed"
        })
        return final_vle_agg

    def build_enrollment_base(self, enrollments_df: pd.DataFrame, submissions_df: pd.DataFrame, vle_summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assembles enrollment_base.csv by left-joining aggregated assessment and VLE metrics
        to the 32,593 enrollment anchor table.
        """
        print("Aggregating assessment submissions to student enrollment grain...")
        
        # Assessments aggregation
        sub_agg = submissions_df.groupby(["code_module", "code_presentation", "id_student"]).agg(
            total_assessments_submitted=("id_assessment", "count"),
            total_scored_assessments=("score", "count"),
            missing_score_assessments_count=("score_is_missing", "sum"),
            avg_assessment_score=("score", "mean"),
            min_assessment_score=("score", "min"),
            max_assessment_score=("score", "max"),
            std_assessment_score=("score", "std"),
            failing_assessments_count=("score", lambda s: (s < 40).sum()),
            at_risk_assessments_count=("score", lambda s: (s < 50).sum()),
            passed_assessments_count=("score", lambda s: (s >= 40).sum()),
            banked_assessments_count=("is_banked", "sum"),
            avg_submission_lateness_days=("submission_lateness_days", "mean")
        ).reset_index()

        # Step 1: Merge enrollments with assessment aggregation via LEFT JOIN
        base = enrollments_df.merge(sub_agg, on=["code_module", "code_presentation", "id_student"], how="left")
        assert len(base) == 32593, "Enrollment row count corrupted during assessment aggregation join"

        # Fill zero-submission students
        base["has_zero_submissions"] = base["total_assessments_submitted"].isna().astype("int64")
        base["total_assessments_submitted"] = base["total_assessments_submitted"].fillna(0).astype("int64")
        base["total_scored_assessments"] = base["total_scored_assessments"].fillna(0).astype("int64")
        base["missing_score_assessments_count"] = base["missing_score_assessments_count"].fillna(0).astype("int64")
        base["failing_assessments_count"] = base["failing_assessments_count"].fillna(0).astype("int64")
        base["at_risk_assessments_count"] = base["at_risk_assessments_count"].fillna(0).astype("int64")
        base["passed_assessments_count"] = base["passed_assessments_count"].fillna(0).astype("int64")
        base["banked_assessments_count"] = base["banked_assessments_count"].fillna(0).astype("int64")

        # Step 2: Merge with VLE aggregation via LEFT JOIN
        print("Merging VLE aggregation to enrollment base...")
        base = base.merge(vle_summary_df, on=["code_module", "code_presentation", "id_student"], how="left")
        assert len(base) == 32593, "Enrollment row count corrupted during VLE aggregation join"

        # Fill zero-click students
        base["is_vle_dormant"] = base["total_vle_clicks"].isna().astype("int64")
        base["total_vle_clicks"] = base["total_vle_clicks"].fillna(0).astype("int64")
        base["vle_active_days_count"] = base["vle_active_days_count"].fillna(0).astype("int64")
        base["vle_distinct_materials_accessed"] = base["vle_distinct_materials_accessed"].fillna(0).astype("int64")
        base["vle_pre_course_clicks"] = base["vle_pre_course_clicks"].fillna(0).astype("int64")
        base["vle_core_content_clicks"] = base["vle_core_content_clicks"].fillna(0).astype("int64")
        base["vle_forum_clicks"] = base["vle_forum_clicks"].fillna(0).astype("int64")
        base["vle_quiz_clicks"] = base["vle_quiz_clicks"].fillna(0).astype("int64")

        # Cast nullable date bounds
        base["first_vle_interaction_day"] = base["first_vle_interaction_day"].astype("Int64")
        base["last_vle_interaction_day"] = base["last_vle_interaction_day"].astype("Int64")

        # Final key check
        assert not base.duplicated(subset=["code_module", "code_presentation", "id_student"]).any(), "Duplicate enrollment key in base"

        out_path = self._processed_path("enrollment_base.csv")
        base.to_csv(out_path, index=False)
        print(f"Saved {out_path} with {len(base)} rows and {len(base.columns)} columns.")
        return base

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes the complete preprocessing pipeline."""
        t_start = time.time()
        print("Starting OULAD data cleaning and preprocessing pipeline...")

        # 1. Clean Courses
        courses_clean = self.clean_courses()
        print("Courses cleaned.")

        # 2. Clean Assessments
        assessments_clean = self.clean_assessments()
        print("Assessments cleaned.")

        # 3. Clean Submissions
        submissions_clean = self.clean_assessment_submissions(assessments_clean)
        print("Assessment submissions cleaned and enriched.")

        # 4. Clean Enrollments
        enrollments_clean = self.clean_enrollments(courses_clean)
        print("Enrollments cleaned and unified.")

        # 5. Clean VLE Interactions & Aggregate
        vle_summary = self.clean_vle_interactions()
        print("VLE interactions cleaned and aggregated.")

        # 6. Build Master Enrollment Base Table
        enrollment_base = self.build_enrollment_base(enrollments_clean, submissions_clean, vle_summary)
        print("Master enrollment_base table constructed.")

        t_elapsed = round(time.time() - t_start, 2)

        summary = {
            "execution_time_seconds": t_elapsed,
            "processed_files": {
                "enrollments_clean.csv": {
                    "path": self._processed_path("enrollments_clean.csv"),
                    "rows": len(enrollments_clean),
                    "columns": list(enrollments_clean.columns)
                },
                "assessments_clean.csv": {
                    "path": self._processed_path("assessments_clean.csv"),
                    "rows": len(assessments_clean),
                    "columns": list(assessments_clean.columns)
                },
                "assessment_submissions_clean.csv": {
                    "path": self._processed_path("assessment_submissions_clean.csv"),
                    "rows": len(submissions_clean),
                    "columns": list(submissions_clean.columns)
                },
                "vle_interactions_clean.csv": {
                    "path": self._processed_path("vle_interactions_clean.csv"),
                    "rows": 10655280,
                    "columns": ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click", "activity_type"]
                },
                "enrollment_base.csv": {
                    "path": self._processed_path("enrollment_base.csv"),
                    "rows": len(enrollment_base),
                    "columns_count": len(enrollment_base.columns),
                    "columns": list(enrollment_base.columns)
                }
            },
            "validation_results": {
                "enrollment_preservation_count": len(enrollment_base),
                "is_enrollment_count_exact_32593": bool(len(enrollment_base) == 32593),
                "zero_submission_students_count": int(enrollment_base["has_zero_submissions"].sum()),
                "zero_vle_students_count": int(enrollment_base["is_vle_dormant"].sum()),
                "enrollment_key_unique": bool(not enrollment_base.duplicated(subset=["code_module", "code_presentation", "id_student"]).any())
            }
        }

        self.summary_metrics = summary
        with open("reports/preprocessing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Pipeline completed in {t_elapsed}s. Summary written to reports/preprocessing_summary.json")
        return summary


if __name__ == "__main__":
    preprocessor = OULADPreprocessor()
    preprocessor.run_pipeline()

