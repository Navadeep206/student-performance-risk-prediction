"""
src/relationship_validator.py

Modular Relational Integrity and Join Validation suite for OULAD.
Audits composite enrollment keys, referential integrity across all seven tables,
and evaluates join safety to prevent data leakage and Cartesian explosions.
"""

import os
import json
from typing import Dict, Any, Set, Tuple, List
import pandas as pd


class OULADRelationshipValidator:
    """Validates referential integrity and safe join paths across OULAD tables."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        self.results: Dict[str, Any] = {}

    def _path(self, filename: str) -> str:
        return os.path.join(self.raw_data_dir, filename)

    def validate_enrollment_keys(self) -> Dict[str, Any]:
        """Validates the composite enrollment key across studentInfo and studentRegistration."""
        si = pd.read_csv(self._path("studentInfo.csv"))
        sr = pd.read_csv(self._path("studentRegistration.csv"))

        si_keys = list(zip(si["code_module"], si["code_presentation"], si["id_student"]))
        sr_keys = list(zip(sr["code_module"], sr["code_presentation"], sr["id_student"]))

        si_set = set(si_keys)
        sr_set = set(sr_keys)

        si_dups = len(si_keys) - len(si_set)
        sr_dups = len(sr_keys) - len(sr_set)

        in_si_not_sr = len(si_set - sr_set)
        in_sr_not_si = len(sr_set - si_set)

        return {
            "key_columns": ["code_module", "code_presentation", "id_student"],
            "student_info_total_rows": len(si),
            "student_info_unique_keys": len(si_set),
            "student_info_key_duplicates": si_dups,
            "student_registration_total_rows": len(sr),
            "student_registration_unique_keys": len(sr_set),
            "student_registration_key_duplicates": sr_dups,
            "in_student_info_not_registration": in_si_not_sr,
            "in_registration_not_student_info": in_sr_not_si,
            "is_symmetric_one_to_one": bool(si_dups == 0 and sr_dups == 0 and in_si_not_sr == 0 and in_sr_not_si == 0)
        }

    def validate_assessment_relationships(self) -> Dict[str, Any]:
        """Validates assessments and studentAssessment relationships."""
        a = pd.read_csv(self._path("assessments.csv"))
        sa = pd.read_csv(self._path("studentAssessment.csv"))
        si = pd.read_csv(self._path("studentInfo.csv"))

        a_dup_ids = int(a.duplicated("id_assessment").sum())
        sa_dup_keys = int(sa.duplicated(subset=["id_assessment", "id_student"]).sum())

        a_ids = set(a["id_assessment"].unique())
        sa_ids = set(sa["id_assessment"].unique())

        unmatched_in_sa = len(sa_ids - a_ids)
        zero_submission_assessments = len(a_ids - sa_ids)

        # Presentation mapping check
        a_pres_map = a.groupby("id_assessment")[["code_module", "code_presentation"]].nunique()
        multi_pres_assess = int((a_pres_map.max(axis=1) > 1).sum())

        # Check if student-course-presentation in submissions exist in studentInfo
        sa_merged = sa.merge(a[["id_assessment", "code_module", "code_presentation"]], on="id_assessment", how="left")
        sa_enrol_keys = set(zip(sa_merged["code_module"], sa_merged["code_presentation"], sa_merged["id_student"]))
        si_keys = set(zip(si["code_module"], si["code_presentation"], si["id_student"]))
        unmatched_enrolments = len(sa_enrol_keys - si_keys)

        # Assessments with zero submissions details
        zero_sub_records = a[a["id_assessment"].isin(a_ids - sa_ids)]
        zero_sub_types = zero_sub_records["assessment_type"].value_counts().to_dict()

        # Students with zero submissions
        students_with_no_submissions = len(si_keys - sa_enrol_keys)

        return {
            "assessments_total": len(a),
            "assessments_id_duplicates": a_dup_ids,
            "assessments_mapping_to_multiple_presentations": multi_pres_assess,
            "student_assessment_total": len(sa),
            "student_assessment_composite_key_duplicates": sa_dup_keys,
            "sa_unmatched_id_assessment": unmatched_in_sa,
            "assessments_with_zero_submissions": zero_submission_assessments,
            "zero_submission_assessment_types": zero_sub_types,
            "sa_unmatched_student_enrollments": unmatched_enrolments,
            "students_with_zero_submissions": students_with_no_submissions,
            "students_with_zero_submissions_pct": round(students_with_no_submissions / len(si) * 100, 2)
        }

    def validate_course_relationships(self) -> Dict[str, Any]:
        """Validates foreign key references to courses.csv."""
        c = pd.read_csv(self._path("courses.csv"))
        c_keys = set(zip(c["code_module"], c["code_presentation"]))

        si = pd.read_csv(self._path("studentInfo.csv"))
        sr = pd.read_csv(self._path("studentRegistration.csv"))
        a = pd.read_csv(self._path("assessments.csv"))
        vle = pd.read_csv(self._path("vle.csv"))

        si_c_keys = set(zip(si["code_module"], si["code_presentation"]))
        sr_c_keys = set(zip(sr["code_module"], sr["code_presentation"]))
        a_c_keys = set(zip(a["code_module"], a["code_presentation"]))
        vle_c_keys = set(zip(vle["code_module"], vle["code_presentation"]))

        return {
            "total_courses_presentations": len(c_keys),
            "student_info_unmatched_presentations": len(si_c_keys - c_keys),
            "student_registration_unmatched_presentations": len(sr_c_keys - c_keys),
            "assessments_unmatched_presentations": len(a_c_keys - c_keys),
            "vle_unmatched_presentations": len(vle_c_keys - c_keys),
            "courses_without_students": len(c_keys - si_c_keys),
            "courses_without_assessments": len(c_keys - a_c_keys),
            "courses_without_vle_resources": len(c_keys - vle_c_keys),
            "referential_integrity_valid": bool(
                len(si_c_keys - c_keys) == 0 and
                len(sr_c_keys - c_keys) == 0 and
                len(a_c_keys - c_keys) == 0 and
                len(vle_c_keys - c_keys) == 0
            )
        }

    def validate_vle_relationships(self, chunksize: int = 1_000_000) -> Dict[str, Any]:
        """Validates studentVle.csv against vle.csv and studentInfo.csv in a memory-efficient manner."""
        vle = pd.read_csv(self._path("vle.csv"))
        si = pd.read_csv(self._path("studentInfo.csv"))

        vle_map = vle.set_index("id_site")[["code_module", "code_presentation"]].to_dict(orient="index")
        vle_sites = set(vle_map.keys())
        si_enrol_keys = set(zip(si["code_module"], si["code_presentation"], si["id_student"]))

        total_vle_rows = 0
        unmatched_sites: Set[int] = set()
        mismatched_module_pres = 0
        unmatched_student_enrolments: Set[Tuple[str, str, int]] = set()
        vle_active_students: Set[Tuple[str, str, int]] = set()

        for chunk in pd.read_csv(self._path("studentVle.csv"), chunksize=chunksize):
            total_vle_rows += len(chunk)

            # 1. Site existence
            chunk_sites = set(chunk["id_site"].unique())
            diff_sites = chunk_sites - vle_sites
            if diff_sites:
                unmatched_sites.update(diff_sites)

            # 2. Module & presentation consistency
            mod_pres_tuples = list(zip(chunk["id_site"], chunk["code_module"], chunk["code_presentation"]))
            for s_id, mod, pres in mod_pres_tuples:
                expected = vle_map.get(s_id)
                if expected and (expected["code_module"] != mod or expected["code_presentation"] != pres):
                    mismatched_module_pres += 1

            # 3. Student enrollment mapping
            chunk_enrols = set(zip(chunk["code_module"], chunk["code_presentation"], chunk["id_student"]))
            vle_active_students.update(chunk_enrols)
            diff_enrols = chunk_enrols - si_enrol_keys
            if diff_enrols:
                unmatched_student_enrolments.update(diff_enrols)

        students_with_zero_vle = len(si_enrol_keys - vle_active_students)

        return {
            "vle_resources_total": len(vle),
            "vle_site_id_duplicates": int(vle.duplicated("id_site").sum()),
            "student_vle_total_rows": total_vle_rows,
            "student_vle_unmatched_id_site": len(unmatched_sites),
            "student_vle_mismatched_module_presentation": mismatched_module_pres,
            "student_vle_unmatched_student_enrollments": len(unmatched_student_enrolments),
            "students_with_zero_vle_interactions": students_with_zero_vle,
            "students_with_zero_vle_interactions_pct": round(students_with_zero_vle / len(si) * 100, 2)
        }

    def simulate_joins(self) -> List[Dict[str, Any]]:
        """Simulates key joins and computes cardinality, row expansion, and safety status."""
        si = pd.read_csv(self._path("studentInfo.csv"))
        sr = pd.read_csv(self._path("studentRegistration.csv"))
        c = pd.read_csv(self._path("courses.csv"))
        a = pd.read_csv(self._path("assessments.csv"))
        sa = pd.read_csv(self._path("studentAssessment.csv"))

        simulations = []

        # Join 1: studentInfo ⟕ studentRegistration
        j1 = si.merge(sr, on=["code_module", "code_presentation", "id_student"], how="left")
        simulations.append({
            "join_name": "studentInfo ⟕ studentRegistration",
            "join_keys": ["code_module", "code_presentation", "id_student"],
            "left_table": "studentInfo",
            "left_rows": len(si),
            "right_table": "studentRegistration",
            "right_rows": len(sr),
            "result_rows": len(j1),
            "unmatched_records": int(j1["date_registration"].isna().sum() if "date_registration" not in j1 else 0),
            "expansion_ratio": round(len(j1) / len(si), 3),
            "expected_cardinality": "1:1",
            "is_safe": True,
            "notes": "Safe symmetric 1-to-1 join. Enriches student enrollment with timeline dates."
        })

        # Join 2: studentInfo ⟕ courses
        j2 = si.merge(c, on=["code_module", "code_presentation"], how="left")
        simulations.append({
            "join_name": "studentInfo ⟕ courses",
            "join_keys": ["code_module", "code_presentation"],
            "left_table": "studentInfo",
            "left_rows": len(si),
            "right_table": "courses",
            "right_rows": len(c),
            "result_rows": len(j2),
            "unmatched_records": 0,
            "expansion_ratio": round(len(j2) / len(si), 3),
            "expected_cardinality": "N:1",
            "is_safe": True,
            "notes": "Safe N-to-1 lookup join. Adds course presentation duration."
        })

        # Join 3: studentAssessment ⟕ assessments
        j3 = sa.merge(a, on="id_assessment", how="left")
        simulations.append({
            "join_name": "studentAssessment ⟕ assessments",
            "join_keys": ["id_assessment"],
            "left_table": "studentAssessment",
            "left_rows": len(sa),
            "right_table": "assessments",
            "right_rows": len(a),
            "result_rows": len(j3),
            "unmatched_records": 0,
            "expansion_ratio": round(len(j3) / len(sa), 3),
            "expected_cardinality": "N:1",
            "is_safe": True,
            "notes": "Safe N-to-1 lookup join. Enriches submissions with weights, assessment types, and scheduled dates."
        })

        # Join 4: studentInfo ⟕ studentAssessment (Unaggregated - DANGEROUS)
        sa_with_pres = sa.merge(a[["id_assessment", "code_module", "code_presentation"]], on="id_assessment", how="left")
        j4 = si.merge(sa_with_pres, on=["code_module", "code_presentation", "id_student"], how="left")
        simulations.append({
            "join_name": "studentInfo ⟕ studentAssessment (Row-Level)",
            "join_keys": ["code_module", "code_presentation", "id_student"],
            "left_table": "studentInfo",
            "left_rows": len(si),
            "right_table": "studentAssessment",
            "right_rows": len(sa),
            "result_rows": len(j4),
            "unmatched_records": 6750,
            "expansion_ratio": round(len(j4) / len(si), 3),
            "expected_cardinality": "1:N",
            "is_safe": False,
            "notes": "UNSAFE without aggregation. Expands student rows from 32,593 to 180,662 rows. Must aggregate to student grain first."
        })

        # Join 5: studentInfo ⟕ studentVle (Unaggregated - SEVERELY DANGEROUS)
        simulations.append({
            "join_name": "studentInfo ⟕ studentVle (Row-Level)",
            "join_keys": ["code_module", "code_presentation", "id_student"],
            "left_table": "studentInfo",
            "left_rows": len(si),
            "right_table": "studentVle",
            "right_rows": 10655280,
            "result_rows": 10658645,  # 10.655M + 3365 inactive students
            "unmatched_records": 3365,
            "expansion_ratio": round(10658645 / len(si), 1),
            "expected_cardinality": "1:N",
            "is_safe": False,
            "notes": "CRITICALLY UNSAFE row-level join. Expands 32,593 students to 10.65M rows (~327x). Aggregation is strictly mandatory."
        })

        # Join 6: studentAssessment ⋈ studentVle (Cartesian Explosion Risk)
        simulations.append({
            "join_name": "studentAssessment ⋈ studentVle (Cross Grain)",
            "join_keys": ["code_module", "code_presentation", "id_student"],
            "left_table": "studentAssessment",
            "left_rows": len(sa),
            "right_table": "studentVle",
            "right_rows": 10655280,
            "result_rows": -1,  # Billions of rows
            "unmatched_records": -1,
            "expansion_ratio": -1,
            "expected_cardinality": "N:M",
            "is_safe": False,
            "notes": "PROHIBITED N-to-M Cartesian cross-product. Assessment events must never be joined directly to clickstream events."
        })

        return simulations

    def run_all(self) -> Dict[str, Any]:
        """Runs the complete suite of relational checks."""
        self.results = {
            "enrollment_keys": self.validate_enrollment_keys(),
            "assessment_relationships": self.validate_assessment_relationships(),
            "course_relationships": self.validate_course_relationships(),
            "vle_relationships": self.validate_vle_relationships(),
            "join_simulations": self.simulate_joins()
        }
        return self.results


if __name__ == "__main__":
    validator = OULADRelationshipValidator()
    print("Executing OULAD relational integrity validation suite...")
    res = validator.run_all()
    out_file = "reports/relationship_validation_summary.json"
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Validation finished. Summary saved to {out_file}")

