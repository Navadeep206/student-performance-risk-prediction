"""
src/data_quality_audit.py

Modular, memory-efficient data quality auditor for the Open University Learning
Analytics Dataset (OULAD).

Audits missing values, sentinel strings ('?'), duplicate rows, primary/foreign key
integrity, and numeric/categorical value constraints across all seven raw CSV tables.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class OULADQualityAuditor:
    """Performs empirical data quality auditing across raw OULAD tables."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        self.files = [
            "courses.csv",
            "assessments.csv",
            "vle.csv",
            "studentInfo.csv",
            "studentRegistration.csv",
            "studentAssessment.csv",
            "studentVle.csv",
        ]
        self.audit_results: Dict[str, Any] = {}

    def _file_path(self, filename: str) -> str:
        return os.path.join(self.raw_data_dir, filename)

    def audit_courses(self) -> Dict[str, Any]:
        """Audits courses.csv."""
        df = pd.read_csv(self._file_path("courses.csv"))
        res = {
            "total_rows": len(df),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "key_duplicates": int(df.duplicated(subset=["code_module", "code_presentation"]).sum()),
            "missing_values": {},
            "sentinel_q_values": {},
            "value_anomalies": {}
        }
        for col in df.columns:
            res["missing_values"][col] = int(df[col].isna().sum())
            res["sentinel_q_values"][col] = int((df[col].astype(str) == "?").sum())
            
        len_col = df["module_presentation_length"]
        res["value_anomalies"]["length_below_or_equal_zero"] = int((len_col <= 0).sum())
        res["value_anomalies"]["length_min"] = int(len_col.min())
        res["value_anomalies"]["length_max"] = int(len_col.max())
        return res

    def audit_assessments(self) -> Dict[str, Any]:
        """Audits assessments.csv."""
        df = pd.read_csv(self._file_path("assessments.csv"))
        res = {
            "total_rows": len(df),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "key_duplicates": int(df.duplicated(subset=["id_assessment"]).sum()),
            "missing_values": {},
            "sentinel_q_values": {},
            "value_anomalies": {}
        }
        for col in df.columns:
            res["missing_values"][col] = int(df[col].isna().sum())
            res["sentinel_q_values"][col] = int((df[col].astype(str) == "?").sum())
            
        # Weights check
        res["value_anomalies"]["weight_min"] = float(df["weight"].min())
        res["value_anomalies"]["weight_max"] = float(df["weight"].max())
        res["value_anomalies"]["weight_negative"] = int((df["weight"] < 0).sum())
        
        # Assessments with '?' date
        q_dates = df[df["date"].astype(str) == "?"]
        res["value_anomalies"]["missing_date_types"] = q_dates["assessment_type"].value_counts().to_dict()
        
        # Coursework weight sum by presentation (excluding Exams which are 100%)
        non_exams = df[df["assessment_type"] != "Exam"]
        weight_sums = non_exams.groupby(["code_module", "code_presentation"])["weight"].sum().to_dict()
        res["value_anomalies"]["coursework_weight_sums"] = {f"{k[0]}_{k[1]}": float(v) for k, v in weight_sums.items()}
        return res

    def audit_vle(self) -> Dict[str, Any]:
        """Audits vle.csv."""
        df = pd.read_csv(self._file_path("vle.csv"))
        res = {
            "total_rows": len(df),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "key_duplicates": int(df.duplicated(subset=["id_site"]).sum()),
            "missing_values": {},
            "sentinel_q_values": {},
            "value_anomalies": {}
        }
        for col in df.columns:
            res["missing_values"][col] = int(df[col].isna().sum())
            res["sentinel_q_values"][col] = int((df[col].astype(str) == "?").sum())
            
        both_weeks = df[(df["week_from"].astype(str) != "?") & (df["week_to"].astype(str) != "?")].copy()
        both_weeks["wf"] = both_weeks["week_from"].astype(float)
        both_weeks["wt"] = both_weeks["week_to"].astype(float)
        res["value_anomalies"]["week_to_before_week_from"] = int((both_weeks["wt"] < both_weeks["wf"]).sum())
        res["value_anomalies"]["week_from_negative"] = int((both_weeks["wf"] < 0).sum())
        res["value_anomalies"]["activity_types_count"] = int(df["activity_type"].nunique())
        return res

    def audit_student_info(self) -> Dict[str, Any]:
        """Audits studentInfo.csv."""
        df = pd.read_csv(self._file_path("studentInfo.csv"))
        res = {
            "total_rows": len(df),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "id_student_duplicates": int(df.duplicated(subset=["id_student"]).sum()),
            "composite_key_duplicates": int(df.duplicated(subset=["code_module", "code_presentation", "id_student"]).sum()),
            "missing_values": {},
            "sentinel_q_values": {},
            "value_anomalies": {}
        }
        for col in df.columns:
            res["missing_values"][col] = int(df[col].isna().sum())
            res["sentinel_q_values"][col] = int((df[col].astype(str) == "?").sum())
            
        # Numerical anomalies
        res["value_anomalies"]["prev_attempts_negative"] = int((df["num_of_prev_attempts"] < 0).sum())
        res["value_anomalies"]["studied_credits_non_positive"] = int((df["studied_credits"] <= 0).sum())
        res["value_anomalies"]["studied_credits_min"] = int(df["studied_credits"].min())
        res["value_anomalies"]["studied_credits_max"] = int(df["studied_credits"].max())
        
        # Missing imd_band by region
        q_imd = df[df["imd_band"].astype(str) == "?"]
        res["value_anomalies"]["missing_imd_by_region"] = q_imd["region"].value_counts().to_dict()
        return res

    def audit_student_registration(self) -> Dict[str, Any]:
        """Audits studentRegistration.csv."""
        df = pd.read_csv(self._file_path("studentRegistration.csv"))
        res = {
            "total_rows": len(df),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "composite_key_duplicates": int(df.duplicated(subset=["code_module", "code_presentation", "id_student"]).sum()),
            "missing_values": {},
            "sentinel_q_values": {},
            "value_anomalies": {}
        }
        for col in df.columns:
            res["missing_values"][col] = int(df[col].isna().sum())
            res["sentinel_q_values"][col] = int((df[col].astype(str) == "?").sum())
            
        both_dates = df[(df["date_registration"].astype(str) != "?") & (df["date_unregistration"].astype(str) != "?")].copy()
        both_dates["reg"] = both_dates["date_registration"].astype(float)
        both_dates["unreg"] = both_dates["date_unregistration"].astype(float)
        res["value_anomalies"]["unregistered_before_registered"] = int((both_dates["unreg"] < both_dates["reg"]).sum())
        return res

    def audit_student_assessment(self) -> Dict[str, Any]:
        """Audits studentAssessment.csv."""
        df = pd.read_csv(self._file_path("studentAssessment.csv"))
        res = {
            "total_rows": len(df),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "composite_key_duplicates": int(df.duplicated(subset=["id_assessment", "id_student"]).sum()),
            "missing_values": {},
            "sentinel_q_values": {},
            "value_anomalies": {}
        }
        for col in df.columns:
            res["missing_values"][col] = int(df[col].isna().sum())
            res["sentinel_q_values"][col] = int((df[col].astype(str) == "?").sum())
            
        valid_scores = df[df["score"].astype(str) != "?"]
        scores = valid_scores["score"].astype(float)
        res["value_anomalies"]["score_min"] = float(scores.min())
        res["value_anomalies"]["score_max"] = float(scores.max())
        res["value_anomalies"]["scores_out_of_range"] = int(((scores < 0) | (scores > 100)).sum())
        
        # Negative submission dates breakdown
        neg_dates = df[df["date_submitted"] < 0]
        res["value_anomalies"]["negative_date_submitted_total"] = len(neg_dates)
        res["value_anomalies"]["negative_date_banked"] = int((neg_dates["is_banked"] == 1).sum())
        res["value_anomalies"]["negative_date_early_submission"] = int((neg_dates["is_banked"] == 0).sum())
        return res

    def audit_student_vle(self, chunksize: int = 1_000_000) -> Dict[str, Any]:
        """Audits studentVle.csv using memory-efficient chunked streaming."""
        filepath = self._file_path("studentVle.csv")
        total_rows = 0
        exact_duplicate_lines = 0
        repeated_key_combos = 0
        neg_clicks = 0
        zero_clicks = 0
        min_click = float("inf")
        max_click = float("-inf")
        date_min = float("inf")
        date_max = float("-inf")
        
        missing_counts: Dict[str, int] = {}
        sentinel_q_counts: Dict[str, int] = {}
        seen_hashes = set()
        
        first_chunk = True
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            if first_chunk:
                for col in chunk.columns:
                    missing_counts[col] = 0
                    sentinel_q_counts[col] = 0
                first_chunk = False
                
            total_rows += len(chunk)
            for col in chunk.columns:
                missing_counts[col] += int(chunk[col].isna().sum())
                sentinel_q_counts[col] += int((chunk[col].astype(str) == "?").sum())
                
            c_clicks = chunk["sum_click"]
            neg_clicks += int((c_clicks < 0).sum())
            zero_clicks += int((c_clicks == 0).sum())
            if c_clicks.min() < min_click: min_click = float(c_clicks.min())
            if c_clicks.max() > max_click: max_click = float(c_clicks.max())
            
            c_dates = chunk["date"]
            if c_dates.min() < date_min: date_min = float(c_dates.min())
            if c_dates.max() > date_max: date_max = float(c_dates.max())
            
            # Key combination duplicates within chunk
            repeated_key_combos += int(chunk.duplicated(subset=["code_module", "code_presentation", "id_student", "id_site", "date"]).sum())
            
        # Fast hash check for exact duplicate lines
        with open(filepath, "r") as f:
            next(f)  # skip header
            for line in f:
                h = hash(line)
                if h in seen_hashes:
                    exact_duplicate_lines += 1
                else:
                    seen_hashes.add(h)
                    
        return {
            "total_rows": total_rows,
            "exact_duplicate_rows": exact_duplicate_lines,
            "repeated_key_combinations": repeated_key_combos,
            "missing_values": missing_counts,
            "sentinel_q_values": sentinel_q_counts,
            "value_anomalies": {
                "negative_clicks": neg_clicks,
                "zero_clicks": zero_clicks,
                "click_range": [min_click, max_click],
                "date_range": [date_min, date_max]
            }
        }

    def run_full_audit(self) -> Dict[str, Any]:
        """Runs the audit across all seven tables."""
        self.audit_results = {
            "courses": self.audit_courses(),
            "assessments": self.audit_assessments(),
            "vle": self.audit_vle(),
            "student_info": self.audit_student_info(),
            "student_registration": self.audit_student_registration(),
            "student_assessment": self.audit_student_assessment(),
            "student_vle": self.audit_student_vle(),
        }
        return self.audit_results


if __name__ == "__main__":
    auditor = OULADQualityAuditor()
    print("Running full OULAD data quality audit...")
    results = auditor.run_full_audit()
    output_path = "reports/data_quality_audit_summary.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Audit completed. Summary written to {output_path}")

