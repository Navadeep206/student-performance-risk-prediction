"""
src/temporal_feature_engineering.py

Production Temporal Feature Engineering and Next-Assessment Target Formulation
Pipeline for OULAD.

Constructs strictly leakage-free supervised learning datasets for predicting:
  at_risk = 1 if next_assessment_score < 50 else 0

Implements:
- Cutoff scenarios: 7 days before and 1 day before scheduled assessment dates.
- Grading-delay scenarios: 14-day delay (baseline) and 7-day delay (alternative).
- Strict temporal isolation: zero future VLE clicks, zero future assessment scores.
- Complete exclusion of final_result, date_unregistration, and sensitive demographics.

Outputs:
- data/processed/prediction_dataset.csv (14-day delay baseline)
- data/processed/prediction_dataset_7day_delay.csv (7-day delay alternative)
- reports/temporal_feature_summary.json
"""

import os
import time
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


class TemporalFeaturePipeline:
    """Builds leakage-free temporal prediction datasets for next-assessment risk."""

    def __init__(self, processed_dir: str = "data/processed", reports_dir: str = "reports"):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        self.summary_metrics: Dict[str, Any] = {}

    def _path(self, filename: str) -> str:
        return os.path.join(self.processed_dir, filename)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads clean processed inputs."""
        assessments = pd.read_csv(self._path("assessments_clean.csv"))
        submissions = pd.read_csv(self._path("assessment_submissions_clean.csv"))
        enrollments = pd.read_csv(self._path("enrollments_clean.csv"))
        return assessments, submissions, enrollments

    def build_vle_daily_table(self, chunksize: int = 2_000_000) -> pd.DataFrame:
        """Pre-aggregates VLE interactions by student and date for rapid temporal indexing."""
        vle_path = self._path("vle_interactions_clean.csv")
        usecols = ["code_module", "code_presentation", "id_student", "date", "sum_click", "activity_type"]

        vle_chunks = []
        print("Pre-aggregating VLE events into daily student activity records...")
        for chunk in pd.read_csv(vle_path, chunksize=chunksize, usecols=usecols):
            chunk["core_clicks"] = chunk["sum_click"] * chunk["activity_type"].isin(["oucontent", "resource", "subpage"])
            chunk["forum_clicks"] = chunk["sum_click"] * (chunk["activity_type"] == "forumng")
            chunk["quiz_clicks"] = chunk["sum_click"] * chunk["activity_type"].isin(["quiz", "externalquiz"])

            daily = chunk.groupby(["code_module", "code_presentation", "id_student", "date"])[
                ["sum_click", "core_clicks", "forum_clicks", "quiz_clicks"]
            ].sum().reset_index()
            vle_chunks.append(daily)

        vle_daily = pd.concat(vle_chunks, ignore_index=True)
        vle_daily = vle_daily.groupby(["code_module", "code_presentation", "id_student", "date"]).sum().reset_index()
        vle_daily = vle_daily.rename(columns={"sum_click": "total_clicks"})
        print(f"VLE pre-aggregation completed: {len(vle_daily):,} daily student records.")
        return vle_daily

    def generate_snapshots(self, assessments: pd.DataFrame, submissions: pd.DataFrame,
                           enrollments: pd.DataFrame, vle_daily: pd.DataFrame,
                           delay_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates prediction snapshots for a specific grading delay scenario across all cohorts.
        """
        print(f"Constructing prediction snapshots under {delay_days}-day grading delay scenario...")
        sched = assessments[assessments["date"].notna()].copy()
        sched["scheduled_date"] = sched["date"].astype(int)

        cohort_groups = sched.groupby(["code_module", "code_presentation"])
        all_snapshots = []
        
        total_theoretical = 0
        total_excluded_no_score = 0
        total_eligible = 0

        for (mod, pres), cohort_sched in cohort_groups:
            cohort_sched = cohort_sched.sort_values("scheduled_date")
            cohort_enrol = enrollments[(enrollments["code_module"] == mod) & (enrollments["code_presentation"] == pres)]
            cohort_subs = submissions[(submissions["code_module"] == mod) & (submissions["code_presentation"] == pres)]
            cohort_vle = vle_daily[(vle_daily["code_module"] == mod) & (vle_daily["code_presentation"] == pres)]

            reg_map = cohort_enrol.set_index("id_student")["date_registration"].to_dict()
            scored_subs = cohort_subs[cohort_subs["score"].notna()]
            n_enrolled = len(cohort_enrol)

            for k, (_, assess_row) in enumerate(cohort_sched.iterrows()):
                a_id = int(assess_row["id_assessment"])
                a_date = int(assess_row["scheduled_date"])
                a_type = str(assess_row["assessment_type"])
                a_weight = float(assess_row["weight"])

                # Count theoretical opportunities
                total_theoretical += n_enrolled

                # Submissions for target assessment
                target_subs = scored_subs[scored_subs["id_assessment"] == a_id][["id_student", "score"]].copy()
                n_scored = len(target_subs)
                total_eligible += n_scored
                total_excluded_no_score += (n_enrolled - n_scored)

                if n_scored == 0:
                    continue

                target_subs["at_risk"] = (target_subs["score"] < 50).astype(int)
                target_subs = target_subs.rename(columns={"score": "target_assessment_score"})

                for offset in [7, 1]:
                    cutoff_t = a_date - offset
                    cutoff_name = f"{offset}_day_before"

                    # ---------------------------------------------------------
                    # 1. VLE Features strictly for date <= cutoff_t
                    # ---------------------------------------------------------
                    vle_sub = cohort_vle[cohort_vle["date"] <= cutoff_t]
                    if len(vle_sub) > 0:
                        vle_agg = vle_sub.groupby("id_student").agg(
                            vle_total_clicks_to_date=("total_clicks", "sum"),
                            vle_active_days_to_date=("date", "nunique"),
                            last_active_date=("date", "max"),
                            vle_core_clicks_to_date=("core_clicks", "sum"),
                            vle_forum_clicks_to_date=("forum_clicks", "sum"),
                            vle_quiz_clicks_to_date=("quiz_clicks", "sum")
                        ).reset_index()
                        vle_agg["vle_days_since_last_activity"] = cutoff_t - vle_agg["last_active_date"]
                        vle_agg = vle_agg.drop(columns=["last_active_date"])

                        # Window clicks: 7d [cutoff_t - 6, cutoff_t], 14d [cutoff_t - 13, cutoff_t]
                        vle_7d = vle_sub[vle_sub["date"] >= cutoff_t - 6].groupby("id_student")["total_clicks"].sum().rename("vle_clicks_last_7d")
                        vle_14d = vle_sub[vle_sub["date"] >= cutoff_t - 13].groupby("id_student")["total_clicks"].sum().rename("vle_clicks_last_14d")

                        vle_agg = vle_agg.merge(vle_7d, on="id_student", how="left")
                        vle_agg = vle_agg.merge(vle_14d, on="id_student", how="left")
                        vle_agg["vle_clicks_last_7d"] = vle_agg["vle_clicks_last_7d"].fillna(0)
                        vle_agg["vle_clicks_last_14d"] = vle_agg["vle_clicks_last_14d"].fillna(0)

                        prior_7d_clicks = vle_agg["vle_clicks_last_14d"] - vle_agg["vle_clicks_last_7d"]
                        vle_agg["vle_activity_trend_ratio"] = (vle_agg["vle_clicks_last_7d"] / (prior_7d_clicks + 1.0)).round(3)
                    else:
                        vle_agg = pd.DataFrame(columns=["id_student"])

                    # ---------------------------------------------------------
                    # 2. Historical Assessment Submissions & Scores
                    # ---------------------------------------------------------
                    # Must strictly be prior scheduled assessment (scheduled_date < a_date)
                    # and submitted on or before cutoff_t (date_submitted <= cutoff_t)
                    prior_subs = cohort_subs[(cohort_subs["id_assessment"] != a_id) &
                                             (cohort_subs["scheduled_date"] < a_date) &
                                             (cohort_subs["date_submitted"] <= cutoff_t)]

                    if len(prior_subs) > 0:
                        p_sub_agg = prior_subs.groupby("id_student").agg(
                            num_prior_submissions=("id_assessment", "count"),
                            prior_submission_lateness_mean=("submission_lateness_days", "mean")
                        ).reset_index()

                        # Graded prior assessments: must satisfy date_submitted + delay_days <= cutoff_t
                        graded = prior_subs[(prior_subs["date_submitted"] + delay_days <= cutoff_t) & prior_subs["score"].notna()]
                        if len(graded) > 0:
                            p_graded_agg = graded.groupby("id_student").agg(
                                num_prior_scored_assessments=("score", "count"),
                                mean_prior_score=("score", "mean"),
                                std_prior_score=("score", lambda s: s.std() if len(s) > 1 else 0.0),
                                prior_at_risk_rate=("score", lambda s: (s < 50).mean())
                            ).reset_index()
                            # Most recent score available
                            recent = graded.sort_values("date_submitted").groupby("id_student")["score"].last().rename("recent_prior_score")
                            p_graded_agg = p_graded_agg.merge(recent, on="id_student", how="left")
                            p_sub_agg = p_sub_agg.merge(p_graded_agg, on="id_student", how="left")
                    else:
                        p_sub_agg = pd.DataFrame(columns=["id_student"])

                    # ---------------------------------------------------------
                    # 3. Assemble Snapshot
                    # ---------------------------------------------------------
                    snap = target_subs.copy()
                    snap["code_module"] = mod
                    snap["code_presentation"] = pres
                    snap["id_assessment"] = a_id
                    snap["target_assessment_type"] = a_type
                    snap["target_assessment_weight"] = a_weight
                    snap["scheduled_date"] = a_date
                    snap["cutoff_date"] = cutoff_t
                    snap["cutoff_scenario"] = cutoff_name
                    snap["days_until_assessment"] = offset
                    snap["num_previous_scheduled_assessments"] = k
                    snap["grading_delay_scenario"] = f"{delay_days}_day_delay"
                    snap["registration_timing_days"] = snap["id_student"].map(reg_map)

                    # Merge VLE
                    if len(vle_agg) > 0:
                        snap = snap.merge(vle_agg, on="id_student", how="left")
                    else:
                        for col in ["vle_total_clicks_to_date", "vle_active_days_to_date", "vle_days_since_last_activity",
                                    "vle_clicks_last_7d", "vle_clicks_last_14d", "vle_activity_trend_ratio",
                                    "vle_core_clicks_to_date", "vle_forum_clicks_to_date", "vle_quiz_clicks_to_date"]:
                            snap[col] = 0

                    for col in ["vle_total_clicks_to_date", "vle_active_days_to_date", "vle_clicks_last_7d",
                                "vle_clicks_last_14d", "vle_activity_trend_ratio", "vle_core_clicks_to_date",
                                "vle_forum_clicks_to_date", "vle_quiz_clicks_to_date"]:
                        if col in snap:
                            snap[col] = snap[col].fillna(0)
                    if "vle_days_since_last_activity" in snap:
                        snap["vle_days_since_last_activity"] = snap["vle_days_since_last_activity"].fillna(999)

                    # Merge Assessment History
                    if len(p_sub_agg) > 0:
                        snap = snap.merge(p_sub_agg, on="id_student", how="left")
                    else:
                        for col in ["num_prior_submissions", "prior_submission_lateness_mean",
                                    "num_prior_scored_assessments", "mean_prior_score", "std_prior_score",
                                    "prior_at_risk_rate", "recent_prior_score"]:
                            snap[col] = np.nan

                    snap["num_prior_submissions"] = snap["num_prior_submissions"].fillna(0).astype(int)
                    if "num_prior_scored_assessments" in snap:
                        snap["num_prior_scored_assessments"] = snap["num_prior_scored_assessments"].fillna(0).astype(int)
                    if "prior_at_risk_rate" in snap:
                        snap["prior_at_risk_rate"] = snap["prior_at_risk_rate"].fillna(0.0)

                    all_snapshots.append(snap)

        dataset = pd.concat(all_snapshots, ignore_index=True)
        print(f"Generated {len(dataset):,} snapshots under {delay_days}-day delay scenario.")

        metrics = {
            "grading_delay_days": delay_days,
            "total_theoretical_opportunities": total_theoretical,
            "total_eligible_opportunities": total_eligible,
            "total_excluded_no_score": total_excluded_no_score,
            "exclusion_rate_pct": round(total_excluded_no_score / total_theoretical * 100, 2),
            "total_prediction_snapshots": len(dataset),
            "unique_students": int(dataset["id_student"].nunique()),
            "class_distribution": {
                "at_risk_0": int((dataset["at_risk"] == 0).sum()),
                "at_risk_1": int((dataset["at_risk"] == 1).sum()),
                "at_risk_pct": round(float((dataset["at_risk"] == 1).mean() * 100), 2)
            },
            "by_cutoff_scenario": {
                name: {
                    "count": len(grp),
                    "at_risk_count": int((grp["at_risk"] == 1).sum()),
                    "at_risk_pct": round(float((grp["at_risk"] == 1).mean() * 100), 2)
                } for name, grp in dataset.groupby("cutoff_scenario")
            },
            "by_assessment_type": {
                atype: {
                    "count": len(grp),
                    "at_risk_count": int((grp["at_risk"] == 1).sum()),
                    "at_risk_pct": round(float((grp["at_risk"] == 1).mean() * 100), 2)
                } for atype, grp in dataset.groupby("target_assessment_type")
            }
        }
        return dataset, metrics

    def run_all(self) -> Dict[str, Any]:
        """Runs the entire temporal dataset construction pipeline."""
        t_start = time.time()
        assessments, submissions, enrollments = self.load_data()
        vle_daily = self.build_vle_daily_table()

        # 1. Baseline: 14-day delay scenario
        ds_14, metrics_14 = self.generate_snapshots(assessments, submissions, enrollments, vle_daily, delay_days=14)
        out_14 = self._path("prediction_dataset.csv")
        ds_14.to_csv(out_14, index=False)
        print(f"Saved primary prediction dataset to {out_14}")

        # 2. Alternative: 7-day delay scenario
        ds_7, metrics_7 = self.generate_snapshots(assessments, submissions, enrollments, vle_daily, delay_days=7)
        out_7 = self._path("prediction_dataset_7day_delay.csv")
        ds_7.to_csv(out_7, index=False)
        print(f"Saved 7-day delay prediction dataset to {out_7}")

        t_elapsed = round(time.time() - t_start, 2)
        summary = {
            "execution_time_seconds": t_elapsed,
            "baseline_14day_delay": metrics_14,
            "alternative_7day_delay": metrics_7,
            "columns": list(ds_14.columns)
        }

        sum_path = os.path.join(self.reports_dir, "temporal_feature_summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {sum_path} in {t_elapsed}s.")
        return summary


if __name__ == "__main__":
    pipeline = TemporalFeaturePipeline()
    pipeline.run_all()

