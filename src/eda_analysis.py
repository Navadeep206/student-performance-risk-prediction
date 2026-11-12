"""
src/eda_analysis.py

Comprehensive Exploratory Data Analysis (EDA) module for OULAD.
Computes descriptive statistics, distributions, cross-tabulations,
and generates publication-quality figures under reports/figures/eda/.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class OULADExploratoryAnalysis:
    """Executes full-scale exploratory analysis across all seven OULAD tables."""

    def __init__(self, raw_data_dir: str = "data/raw", figures_dir: str = "reports/figures/eda"):
        self.raw_data_dir = raw_data_dir
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Configure plotting style
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight"
        })
        self.stats: Dict[str, Any] = {}

    def _path(self, filename: str) -> str:
        return os.path.join(self.raw_data_dir, filename)

    def analyze_student_distributions(self) -> Dict[str, Any]:
        """Task 1: Student enrollments, demographics, and prior history."""
        si = pd.read_csv(self._path("studentInfo.csv"))
        sr = pd.read_csv(self._path("studentRegistration.csv"))

        # Enrollments by module
        mod_counts = si["code_module"].value_counts().to_dict()
        pres_counts = si["code_presentation"].value_counts().to_dict()
        cohort_counts = si.groupby(["code_module", "code_presentation"]).size().to_dict()
        cohort_dict = {f"{k[0]}_{k[1]}": int(v) for k, v in cohort_counts.items()}

        # Previous attempts
        prev_attempts_dist = si["num_of_prev_attempts"].value_counts().sort_index().to_dict()

        # Studied credits
        credits = si["studied_credits"]
        credits_stats = {
            "mean": round(float(credits.mean()), 2),
            "std": round(float(credits.std()), 2),
            "median": float(credits.median()),
            "iqr": float(credits.quantile(0.75) - credits.quantile(0.25)),
            "min": int(credits.min()),
            "max": int(credits.max()),
            "q25": float(credits.quantile(0.25)),
            "q75": float(credits.quantile(0.75))
        }

        # Registration dates
        reg_valid = sr[sr["date_registration"].astype(str) != "?"]
        reg_dates = reg_valid["date_registration"].astype(float)
        reg_stats = {
            "missing_count": int((sr["date_registration"].astype(str) == "?").sum()),
            "mean": round(float(reg_dates.mean()), 2),
            "std": round(float(reg_dates.std()), 2),
            "median": float(reg_dates.median()),
            "min": float(reg_dates.min()),
            "max": float(reg_dates.max())
        }

        self.stats["student_distributions"] = {
            "total_enrollments": len(si),
            "module_counts": mod_counts,
            "presentation_counts": pres_counts,
            "cohort_counts": cohort_dict,
            "prev_attempts_dist": prev_attempts_dist,
            "studied_credits_stats": credits_stats,
            "registration_stats": reg_stats
        }
        return self.stats["student_distributions"]

    def analyze_assessment_performance(self) -> Dict[str, Any]:
        """Task 2: Scores, assessment types, weights, and missing marks."""
        sa = pd.read_csv(self._path("studentAssessment.csv"))
        a = pd.read_csv(self._path("assessments.csv"))

        # Merge assessments metadata
        m = sa.merge(a, on="id_assessment", how="left")

        # Missing score isolation
        missing_mask = m["score"].astype(str) == "?"
        missing_count = int(missing_mask.sum())
        missing_by_type = m[missing_mask]["assessment_type"].value_counts().to_dict()

        # Valid scores
        valid = m[~missing_mask].copy()
        valid["score"] = valid["score"].astype(float)
        scores = valid["score"]

        overall_stats = {
            "total_submissions": len(m),
            "valid_submissions": len(valid),
            "missing_score_count": missing_count,
            "missing_score_by_type": missing_by_type,
            "mean": round(float(scores.mean()), 2),
            "std": round(float(scores.std()), 2),
            "median": float(scores.median()),
            "iqr": float(scores.quantile(0.75) - scores.quantile(0.25)),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "q25": float(scores.quantile(0.25)),
            "q75": float(scores.quantile(0.75))
        }

        # By assessment type
        type_stats = {}
        for atype, group in valid.groupby("assessment_type"):
            s = group["score"]
            type_stats[atype] = {
                "count": len(s),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "median": float(s.median()),
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75))
            }

        # By module
        mod_stats = {}
        for mod, group in valid.groupby("code_module"):
            s = group["score"]
            mod_stats[mod] = {
                "count": len(s),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "median": float(s.median())
            }

        # By presentation
        pres_stats = {}
        for pres, group in valid.groupby("code_presentation"):
            s = group["score"]
            pres_stats[pres] = {
                "count": len(s),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "median": float(s.median())
            }

        # Weight distribution
        weight_dist = {
            "TMA_mean": round(float(a[a["assessment_type"] == "TMA"]["weight"].mean()), 2),
            "TMA_max": float(a[a["assessment_type"] == "TMA"]["weight"].max()),
            "CMA_mean": round(float(a[a["assessment_type"] == "CMA"]["weight"].mean()), 2),
            "CMA_max": float(a[a["assessment_type"] == "CMA"]["weight"].max()),
            "Exam_all_100": bool((a[a["assessment_type"] == "Exam"]["weight"] == 100.0).all())
        }

        self.stats["assessment_performance"] = {
            "overall": overall_stats,
            "by_type": type_stats,
            "by_module": mod_stats,
            "by_presentation": pres_stats,
            "weights": weight_dist
        }
        return self.stats["assessment_performance"]

    def analyze_academic_outcomes(self) -> Dict[str, Any]:
        """Task 3: Final result distributions overall and module-wise."""
        si = pd.read_csv(self._path("studentInfo.csv"))

        overall = si["final_result"].value_counts().to_dict()
        overall_pct = {k: round(v / len(si) * 100, 2) for k, v in overall.items()}

        # Module-wise breakdown
        ct = pd.crosstab(si["code_module"], si["final_result"])
        ct_pct = pd.crosstab(si["code_module"], si["final_result"], normalize="index") * 100

        module_outcomes = {}
        for mod in ct.index:
            module_outcomes[mod] = {
                "counts": {col: int(ct.loc[mod, col]) for col in ct.columns},
                "percentages": {col: round(float(ct_pct.loc[mod, col]), 2) for col in ct_pct.columns}
            }

        self.stats["academic_outcomes"] = {
            "overall_counts": overall,
            "overall_percentages": overall_pct,
            "module_outcomes": module_outcomes
        }
        return self.stats["academic_outcomes"]

    def analyze_registration_and_withdrawal(self) -> Dict[str, Any]:
        """Task 4: Registration timing, withdrawal curves, and outcome relationships."""
        si = pd.read_csv(self._path("studentInfo.csv"))
        sr = pd.read_csv(self._path("studentRegistration.csv"))
        m = si.merge(sr, on=["code_module", "code_presentation", "id_student"])

        # Registered dates
        reg_valid = m[m["date_registration"].astype(str) != "?"].copy()
        reg_valid["date_reg"] = reg_valid["date_registration"].astype(float)

        # Categorize registration timing
        # Early (< -30 days), Normal (-30 to 0 days), Late (> 0 days)
        def reg_bracket(days: float) -> str:
            if days < -30:
                return "Early (>30d before start)"
            elif days <= 0:
                return "Standard (0-30d before start)"
            else:
                return "Late (After start)"

        reg_valid["reg_cohort"] = reg_valid["date_reg"].apply(reg_bracket)
        cohort_outcomes = pd.crosstab(reg_valid["reg_cohort"], reg_valid["final_result"], normalize="index") * 100

        # Withdrawal dates
        withdrawn = m[m["date_unregistration"].astype(str) != "?"].copy()
        withdrawn["date_unreg"] = withdrawn["date_unregistration"].astype(float)

        # Withdrawal rates by module
        mod_withdrawals = {}
        for mod, group in m.groupby("code_module"):
            total_students = len(group)
            withdrawn_count = int((group["date_unregistration"].astype(str) != "?").sum())
            mod_withdrawals[mod] = {
                "total": total_students,
                "withdrawn_recorded": withdrawn_count,
                "withdrawal_rate_pct": round(withdrawn_count / total_students * 100, 2)
            }

        unreg_stats = {
            "count": len(withdrawn),
            "mean": round(float(withdrawn["date_unreg"].mean()), 2),
            "std": round(float(withdrawn["date_unreg"].std()), 2),
            "median": float(withdrawn["date_unreg"].median()),
            "q25": float(withdrawn["date_unreg"].quantile(0.25)),
            "q75": float(withdrawn["date_unreg"].quantile(0.75)),
            "min": float(withdrawn["date_unreg"].min()),
            "max": float(withdrawn["date_unreg"].max())
        }

        self.stats["registration_withdrawal"] = {
            "withdrawal_date_stats": unreg_stats,
            "module_withdrawal_rates": mod_withdrawals,
            "registration_timing_outcomes": cohort_outcomes.round(2).to_dict(orient="index")
        }
        return self.stats["registration_withdrawal"]

    def analyze_vle_engagement(self, chunksize: int = 1_000_000) -> Dict[str, Any]:
        """Task 5: Chunked clickstream temporal aggregation and distribution."""
        vle_file = self._path("studentVle.csv")
        
        total_clicks = 0
        total_rows = 0
        module_clicks: Dict[str, int] = {}
        presentation_clicks: Dict[str, int] = {}
        daily_clicks: Dict[int, int] = {}

        for chunk in pd.read_csv(vle_file, chunksize=chunksize):
            total_rows += len(chunk)
            chunk_clicks = int(chunk["sum_click"].sum())
            total_clicks += chunk_clicks

            # Group by module
            for mod, sum_c in chunk.groupby("code_module")["sum_click"].sum().items():
                module_clicks[mod] = module_clicks.get(mod, 0) + int(sum_c)

            # Group by presentation
            for pres, sum_c in chunk.groupby("code_presentation")["sum_click"].sum().items():
                presentation_clicks[pres] = presentation_clicks.get(pres, 0) + int(sum_c)

            # Group by date
            for d, sum_c in chunk.groupby("date")["sum_click"].sum().items():
                d_int = int(d)
                daily_clicks[d_int] = daily_clicks.get(d_int, 0) + int(sum_c)

        # Sort daily timeline
        sorted_daily = {k: daily_clicks[k] for k in sorted(daily_clicks.keys())}

        self.stats["vle_engagement"] = {
            "total_log_entries": total_rows,
            "total_clicks": total_clicks,
            "module_clicks": module_clicks,
            "presentation_clicks": presentation_clicks,
            "daily_clicks_summary": {
                "active_days_count": len(sorted_daily),
                "peak_day": int(max(sorted_daily, key=sorted_daily.get)),
                "peak_day_clicks": int(max(sorted_daily.values()))
            }
        }
        # Save daily time series for plotting
        self._daily_clicks_series = sorted_daily
        return self.stats["vle_engagement"]

    def generate_visualizations(self):
        """Task 6: Generate all requested figures."""
        print("Generating visualizations...")
        
        # Load necessary data frames
        si = pd.read_csv(self._path("studentInfo.csv"))
        sr = pd.read_csv(self._path("studentRegistration.csv"))
        sa = pd.read_csv(self._path("studentAssessment.csv"))
        a = pd.read_csv(self._path("assessments.csv"))
        sa_merged = sa.merge(a, on="id_assessment", how="left")
        sa_valid = sa_merged[sa_merged["score"].astype(str) != "?"].copy()
        sa_valid["score"] = sa_valid["score"].astype(float)

        # -------------------------------------------------------------
        # 1. Score distribution and boxplots by assessment type
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.histplot(data=sa_valid, x="score", hue="assessment_type", kde=True, bins=30, ax=axes[0], palette="tab10", alpha=0.5)
        axes[0].set_title("Assessment Score Distribution by Type", fontweight="bold")
        axes[0].set_xlabel("Score (0 - 100)")
        axes[0].set_ylabel("Submission Frequency")
        axes[0].axvline(40, color="crimson", linestyle="--", label="Passing Threshold (40)")
        axes[0].legend(title="Assessment Type")

        sns.boxplot(data=sa_valid, x="assessment_type", y="score", palette="tab10", ax=axes[1], width=0.45)
        axes[1].set_title("Score Spread by Assessment Type", fontweight="bold")
        axes[1].set_xlabel("Assessment Type")
        axes[1].set_ylabel("Score (0 - 100)")
        axes[1].axhline(40, color="crimson", linestyle="--", label="Passing Threshold (40)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "score_distribution_by_type.png"))
        plt.close()

        # -------------------------------------------------------------
        # 2. Module-wise performance comparison
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=sa_valid, x="code_module", y="score", palette="Set2", ax=ax, width=0.5)
        ax.set_title("Coursework Score Spread Across Modules", fontweight="bold")
        ax.set_xlabel("Module Code")
        ax.set_ylabel("Score (0 - 100)")
        ax.axhline(40, color="crimson", linestyle="--", label="Passing Threshold (40)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "module_score_comparison.png"))
        plt.close()

        # -------------------------------------------------------------
        # 3. Academic outcomes distribution overall and by module
        # -------------------------------------------------------------
        ct_pct = pd.crosstab(si["code_module"], si["final_result"], normalize="index") * 100
        order_cols = ["Distinction", "Pass", "Fail", "Withdrawn"]
        ct_pct = ct_pct[order_cols]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
        ct_pct.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.6)
        ax.set_title("Academic Outcome Distribution by Course Module (%)", fontweight="bold")
        ax.set_xlabel("Module Code")
        ax.set_ylabel("Percentage of Enrolled Students (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Final Result", bbox_to_anchor=(1.02, 1), loc="upper left")
        for c in ax.containers:
            labels = [f"{v.get_height():.1f}%" if v.get_height() > 5 else "" for v in c]
            ax.bar_label(c, labels=labels, label_type="center", color="white", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "academic_outcomes_by_module.png"))
        plt.close()

        # -------------------------------------------------------------
        # 4. Registration and withdrawal timelines
        # -------------------------------------------------------------
        m_reg = si.merge(sr, on=["code_module", "code_presentation", "id_student"])
        reg_dates = m_reg[m_reg["date_registration"].astype(str) != "?"]["date_registration"].astype(float)
        unreg_dates = m_reg[m_reg["date_unregistration"].astype(str) != "?"]["date_unregistration"].astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(reg_dates, bins=50, color="royalblue", kde=True, ax=axes[0])
        axes[0].set_title("Registration Timing Distribution", fontweight="bold")
        axes[0].set_xlabel("Days Relative to Course Start (Day 0)")
        axes[0].set_ylabel("Student Count")
        axes[0].axvline(0, color="crimson", linestyle="--", label="Course Start (Day 0)")
        axes[0].legend()

        sns.histplot(unreg_dates, bins=50, color="firebrick", kde=True, ax=axes[1])
        axes[1].set_title("Withdrawal Timing Distribution (Dropouts)", fontweight="bold")
        axes[1].set_xlabel("Days Relative to Course Start (Day 0)")
        axes[1].set_ylabel("Student Count")
        axes[1].axvline(0, color="navy", linestyle="--", label="Course Start (Day 0)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "registration_withdrawal_timelines.png"))
        plt.close()

        # -------------------------------------------------------------
        # 5. Registration timing vs academic outcome
        # -------------------------------------------------------------
        def reg_group(days: float) -> str:
            if days < -30: return "Early (>30d before)"
            elif days <= 0: return "Standard (0-30d before)"
            else: return "Late (After Day 0)"

        m_reg_valid = m_reg[m_reg["date_registration"].astype(str) != "?"].copy()
        m_reg_valid["reg_cohort"] = m_reg_valid["date_registration"].astype(float).apply(reg_group)
        cohort_pct = pd.crosstab(m_reg_valid["reg_cohort"], m_reg_valid["final_result"], normalize="index")[order_cols] * 100

        fig, ax = plt.subplots(figsize=(9, 5))
        cohort_pct.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.5)
        ax.set_title("Academic Outcome by Registration Timing Window (%)", fontweight="bold")
        ax.set_xlabel("Registration Timing Cohort")
        ax.set_ylabel("Percentage of Students (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Final Result", bbox_to_anchor=(1.02, 1), loc="upper left")
        for c in ax.containers:
            labels = [f"{v.get_height():.1f}%" if v.get_height() > 5 else "" for v in c]
            ax.bar_label(c, labels=labels, label_type="center", color="white", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "registration_timing_vs_outcome.png"))
        plt.close()

        # -------------------------------------------------------------
        # 6. VLE clicks by module & presentation
        # -------------------------------------------------------------
        mod_clicks = pd.Series(self.stats["vle_engagement"]["module_clicks"])
        pres_clicks = pd.Series(self.stats["vle_engagement"]["presentation_clicks"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        mod_clicks.sort_values(ascending=False).plot(kind="bar", color="teal", ax=axes[0], width=0.55)
        axes[0].set_title("Total VLE Clickstream Volume by Module", fontweight="bold")
        axes[0].set_xlabel("Module Code")
        axes[0].set_ylabel("Total Recorded Clicks (Millions)")
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1e6:.1f}M"))

        pres_clicks.plot(kind="bar", color="darkslateblue", ax=axes[1], width=0.5)
        axes[1].set_title("Total VLE Clickstream Volume by Presentation", fontweight="bold")
        axes[1].set_xlabel("Presentation Cohort")
        axes[1].set_ylabel("Total Recorded Clicks (Millions)")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1e6:.1f}M"))

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "vle_clicks_by_module_presentation.png"))
        plt.close()

        # -------------------------------------------------------------
        # 7. Time-based engagement trends over course timeline
        # -------------------------------------------------------------
        daily_series = pd.Series(self._daily_clicks_series)
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(daily_series.index, daily_series.values, color="royalblue", linewidth=1.5, label="Daily Total Clicks")
        ax.fill_between(daily_series.index, daily_series.values, color="royalblue", alpha=0.15)
        ax.axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="Course Start (Day 0)")
        ax.axvline(240, color="darkorange", linestyle=":", linewidth=1.5, label="Standard Exam Period (~Day 240)")
        ax.set_title("Temporal VLE Clickstream Engagement Over Course Presentation Cycle", fontweight="bold")
        ax.set_xlabel("Days Relative to Course Start")
        ax.set_ylabel("Total Daily Clicks Across All Presentations")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1e3:.0f}K"))
        ax.set_xlim(-25, 275)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "vle_timeline_trend.png"))
        plt.close()

        print(f"All 7 figures generated and saved to {self.figures_dir}/")

    def run_all(self) -> Dict[str, Any]:
        """Runs the entire EDA suite."""
        print("Analyzing student distributions...")
        self.analyze_student_distributions()
        print("Analyzing assessment performance...")
        self.analyze_assessment_performance()
        print("Analyzing academic outcomes...")
        self.analyze_academic_outcomes()
        print("Analyzing registration & withdrawals...")
        self.analyze_registration_and_withdrawal()
        print("Analyzing VLE engagement (chunked)...")
        self.analyze_vle_engagement()
        self.generate_visualizations()

        out_path = "reports/eda_summary.json"
        with open(out_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Summary written to {out_path}")
        return self.stats


if __name__ == "__main__":
    eda = OULADExploratoryAnalysis()
    eda.run_all()

