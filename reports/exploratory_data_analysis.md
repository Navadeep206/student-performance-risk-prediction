# Comprehensive Exploratory Data Analysis (EDA) Report
## Student Performance & At-Risk Prediction System

- **Document Version:** 1.0  
- **Dataset:** Open University Learning Analytics Dataset (OULAD)  
- **Data Location:** `data/raw/`  
- **Analysis Implementation:** [`src/eda_analysis.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/eda_analysis.py)  
- **Interactive Notebook:** [`notebooks/05_exploratory_data_analysis.ipynb`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/notebooks/05_exploratory_data_analysis.ipynb)  
- **Summary Statistics:** [`reports/eda_summary.json`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/eda_summary.json)  
- **Visual Artifacts:** [`reports/figures/eda/`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/figures/eda/)  
- **Phase:** Phase 2.7 — Exploratory Data Analysis  

---

## 1. Executive Summary

This exploratory data analysis provides empirical foundational insight into the 32,593 student enrollments, 173,912 assessment submissions, and 39.6 million VLE clickstream interactions across the 22 course presentations of the Open University Learning Analytics Dataset.

### Core Discoveries:
1. **Severe Academic Attrition:** The baseline at-risk rate (`Withdrawn` + `Fail`) across the university is **52.80%** (10,156 withdrawals, 7,052 failures). More than half of enrolled students do not achieve a passing certification.
2. **Early Dropout Hazard:** Withdrawals are concentrated heavily at the beginning of the semester. The **median withdrawal occurs at Day 27**, with 25% of dropouts unregistering prior to Day -2 (before course commencement).
3. **Assessment Modality Hierarchy:** Computer-Marked Assessments (CMAs) yield the highest average scores (mean 81.03), followed by Tutor-Marked Assessments (TMAs, mean 72.56), and Exams (mean 65.57).
4. **VLE Engagement Trajectory:** Clickstream engagement peaks at **Day 18** (427,217 daily clicks across cohorts) and decays steadily thereafter. Inactive students (zero VLE clicks) represent 10.32% of all enrollments, of which **99.9% withdraw or fail**.
5. **Registration Timing Penalty:** Students registering after the official start date suffer significantly elevated course failure rates (28.63% Fail vs. 20.22% for Early registrants).

---

## 2. Student Enrollment & Cohort Distributions

### 2.1 Enrollment by Module & Presentation
The dataset encompasses 32,593 enrollments across 7 module disciplines and 4 semester presentations (`2013B`, `2013J`, `2014B`, `2014J`):

| Module Code | Field / Focus | Total Enrollments | Share (%) | Presentations Offered | Average Presentation Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BBB** | Business / STEM | 7,909 | 24.27% | 4 (`2013B`, `2013J`, `2014B`, `2014J`) | 1,977 |
| **FFF** | STEM / Computing | 7,762 | 23.81% | 4 (`2013B`, `2013J`, `2014B`, `2014J`) | 1,941 |
| **DDD** | STEM / Tech | 6,272 | 19.24% | 4 (`2013B`, `2013J`, `2014B`, `2014J`) | 1,568 |
| **CCC** | STEM / Sciences | 4,434 | 13.60% | 2 (`2014B`, `2014J`) | 2,217 |
| **EEE** | Education / Humanities | 2,934 | 9.00% | 3 (`2013J`, `2014B`, `2014J`) | 978 |
| **GGG** | Social Sciences | 2,534 | 7.77% | 3 (`2013J`, `2014B`, `2014J`) | 845 |
| **AAA** | Humanities / Arts | 748 | 2.30% | 2 (`2013J`, `2014J`) | 374 |
| **Total** | — | **32,593** | **100.0%** | **22 Cohorts** | **1,481** |

**Presentation Seasonality:**
- October (`J`) presentations attract significantly higher enrollment volumes (**20,105 students, 61.7%**) than February (`B`) presentations (**12,488 students, 38.3%**).

### 2.2 Repeat Attempts & Academic Workload
- **Previous Attempts (`num_of_prev_attempts`):**
  - `0 attempts`: 28,421 (87.20%)
  - `1 attempt`: 3,299 (10.12%)
  - `2 attempts`: 675 (2.07%)
  - `3 to 6 attempts`: 198 (0.61%)
  - *Pattern*: While 87.2% are first-time entrants, repeaters represent an important subset with distinct historical failure patterns.
- **Studied Credits (`studied_credits`):**
  - Mean: 79.76 credits ($\pm 41.07$), Median: 60.0 credits (IQR: 60.0), Range: 30 to 655 credits.
  - 60 credits (10,948 students) and 120 credits (8,944 students) constitute standard half-time and full-time academic loads. Students carrying $\ge 180$ credits represent extreme workload stress cases.

---

## 3. Assessment Performance & Grading Distributions

### 3.1 Overall Score Distribution
- **Total Submissions:** 173,912 records.
- **Valid Scored Submissions:** 173,739 records.
- **Missing Score Records:** 173 records (0.10%), all occurring strictly on TMAs.
- **Summary Metrics:**
  - Mean: **75.80** | Standard Deviation: **18.80** | Median: **80.00**
  - 25th Percentile ($Q_1$): **65.00** | 75th Percentile ($Q_3$): **90.00** | IQR: **25.00**
  - Minimum: **0.00** | Maximum: **100.00**
  - Submissions failing institutional threshold ($< 40$): **8,416 (4.84%)**

### 3.2 Performance by Assessment Modality

| Modality | Count | Mean Score | Median Score | Std Dev | $Q_1$ (25%) | $Q_3$ (75%) | Weight Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CMA (Computer-Marked)** | 70,527 | **81.03** | 82.00 | 17.84 | 72.00 | 98.00 | Formative / Low Weight (Mean: 1.18%) |
| **TMA (Tutor-Marked)** | 98,253 | **72.56** | 76.00 | 18.43 | 63.00 | 86.00 | Summative Coursework (Mean: 17.08%) |
| **Exam** | 4,959 | **65.57** | 67.00 | 20.58 | 51.00 | 82.00 | Summative Final (Weight: 100.00%) |

![Score Distribution by Type](../reports/figures/eda/score_distribution_by_type.png)

### 3.3 Score Spread by Module
Module-level coursework performance varies substantially across curricula:
- **Highest Scoring Modules:** `EEE` (mean 81.18, median 85.0), `GGG` (mean 79.70, median 80.0), and `FFF` (mean 77.71, median 80.0).
- **Lowest Scoring Modules:** `AAA` (mean 69.03, median 71.0) and `DDD` (mean 70.09, median 74.0).

![Module Score Comparison](../reports/figures/eda/module_score_comparison.png)

---

## 4. Academic Outcomes & Course Level Differences

### 4.1 Global Outcome Distribution
- **Pass:** 12,361 (37.93%)
- **Withdrawn:** 10,156 (31.16%)
- **Fail:** 7,052 (21.64%)
- **Distinction:** 3,024 (9.28%)
- **Aggregate At-Risk Rate (`Withdrawn` + `Fail`):** **52.80%** (17,208 students)

> [!IMPORTANT]
> **Descriptive Outcome vs. Prediction Target:**
> `final_result` is an immutable, post-presentation certification outcome. It is analyzed here solely to understand the historical base rate of student difficulty. In accordance with strict leakage guidelines, `final_result` must **never** be used as an input feature in predictive pipelines.

### 4.2 Module-Level Academic Outcome Breakdown

| Module | Distinction (%) | Pass (%) | Fail (%) | Withdrawn (%) | Total At-Risk (`Withdrawn` + `Fail`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AAA** | 5.88% | 65.11% | 12.17% | 16.84% | **29.01%** (Lowest Risk) |
| **BBB** | 8.56% | 38.91% | 22.34% | 30.19% | **52.53%** |
| **CCC** | 11.23% | 26.61% | 17.61% | 44.54% | **62.15%** (Highest Attrition) |
| **DDD** | 6.11% | 35.51% | 22.51% | 35.87% | **58.38%** |
| **EEE** | 12.13% | 44.10% | 19.15% | 24.61% | **43.76%** |
| **FFF** | 8.63% | 38.37% | 22.04% | 30.96% | **53.00%** |
| **GGG** | 15.63% | 44.12% | 28.73% | 11.52% | **40.25%** (Highest Failure) |

![Academic Outcomes by Module](../reports/figures/eda/academic_outcomes_by_module.png)

**Key Curricular Differences:**
- **Module CCC Attrition Crisis:** 44.54% of students withdraw, the highest in the university. Passing rate is only 26.61%.
- **Module GGG Examination Barrier:** Students rarely unregister formally (only 11.52% withdraw), but 28.73% fail the final examination (where all coursework was 100% formative with 0% weight).

---

## 5. Registration & Withdrawal Patterns

### 5.1 Registration Timing Dynamics
- **Mean Registration Date:** -69.41 days ($\pm 49.26$).
- **Median Registration Date:** -57.00 days.
- **Range:** -322 days to +167 days.
- **Late Registrants (After Day 0):** 468 students registered late after module start.

### 5.2 Withdrawal Survival Hazard
- **Recorded Withdrawal Dates:** 10,072 students (with 93 informal dropouts having missing dates).
- **Mean Withdrawal Date:** Day 49.76 ($\pm 82.46$).
- **Median Withdrawal Date:** **Day 27.00**.
- **First Quartile ($Q_1$):** **Day -2.00** (25% drop out before the course even opens!).
- **Third Quartile ($Q_3$):** Day 109.00 (75% of dropouts occur within the first half of the semester).

![Registration & Withdrawal Timelines](../reports/figures/eda/registration_withdrawal_timelines.png)

### 5.3 Registration Timing vs. Final Result
Analyzing outcomes across registration cohorts reveals clear behavioral divergence:
- **Early Registrants ($>30$ days before start):** 37.53% Pass, 9.01% Distinction, 20.22% Fail, 33.24% Withdrawn.
- **Standard Registrants ($0 - 30$ days before start):** 39.02% Pass, 10.12% Distinction, 25.30% Fail, 25.57% Withdrawn.
- **Late Registrants (After start):** 43.59% Pass, 6.84% Distinction, **28.63% Fail**, 20.94% Withdrawn.
- *Insight*: Students registering late have a **41.6% higher failure rate** compared to early registrants (28.63% vs 20.22%), confirming registration timeliness as a vital static prior for risk assessment.

![Registration Timing vs Outcome](../reports/figures/eda/registration_timing_vs_outcome.png)

---

## 6. VLE Clickstream Engagement Analysis

### 6.1 Volume & Distribution
- **Total Log Records:** 10,655,280 daily student-site interaction records.
- **Total Clicks Logged:** **39,605,099 clicks**.
- **Clicks by Module:**
  - `FFF`: 17,593,922 clicks (**44.42% of all clicks**)
  - `DDD`: 5,530,089 clicks (13.96%)
  - `BBB`: 5,234,176 clicks (13.22%)
  - `CCC`: 4,682,149 clicks (11.82%)
  - `EEE`: 3,984,227 clicks (10.06%)
  - `GGG`: 1,333,884 clicks (3.37%)
  - `AAA`: 1,246,652 clicks (3.15%)
- **Clicks by Presentation:**
  - `2014J`: 13,800,298 clicks (34.8%)
  - `2013J`: 10,945,847 clicks (27.6%)
  - `2014B`: 7,903,521 clicks (20.0%)
  - `2013B`: 6,955,433 clicks (17.6%)

![VLE Clicks by Module & Presentation](../reports/figures/eda/vle_clicks_by_module_presentation.png)

### 6.2 Temporal Lifecycle Engagement Curve
Tracking total daily interactions from Day -25 to Day 269 reveals:
1. **Pre-Course Ramp (Days -25 to 0):** Low-level onboarding activity (average 20K–50K daily clicks).
2. **First Assessment Surge (Days 1 to 30):** Activity surges dramatically, reaching an all-time dataset peak on **Day 18 (427,217 clicks in a single day)** as students tackle early readings and TMA 1.
3. **Mid-Semester Plateau & Attrition Decay (Days 30 to 180):** Sustained lower activity with periodic periodic spikes corresponding to subsequent assessment due dates.
4. **Final Examination Sprint (Days 200 to 240):** Modest revision spikes before term close.

![VLE Timeline Trend](../reports/figures/eda/vle_timeline_trend.png)

---

## 7. Implications for Early-Warning Machine Learning

1. **Temporal Cut-Off Windows (Critical for Day 30 / Day 60 Milestones):**
   - Because the **median withdrawal day is Day 27**, an early-warning intervention system deployed at **Day 30** will capture the critical transition point where 50% of eventual dropouts leave.
   - A model evaluated at **Day 60** will capture initial coursework marks (TMA 1 and CMA 1) alongside one full month of steady-state VLE interaction data.
2. **Handling the Inactive 10.3% Population:**
   - 3,365 students never generate a single VLE click. Feature pipelines must use zero-fill left joins; models that rely solely on active click rates will fail unless zero-activity indicators (`total_clicks == 0`) are explicitly represented.
3. **Curricular Normalization:**
   - Modules vary dramatically in engagement intensity (Module FFF generated 17.6M clicks while Module AAA generated 1.2M). Features must include cohort-normalized metrics (e.g. z-scores or percentile ranks of clicks within a specific module-presentation) to prevent model bias toward heavily click-dependent STEM curricula.
4. **Class Balance Strategy:**
   - With an overall 52.8% at-risk base rate, binary classification (`At-Risk` vs. `Not At-Risk`) enjoys natural class balance, avoiding severe extreme-minority imbalance pathologies.

---

## 8. Limitations & Potential Biases

1. **Clickstream Granularity:** Click counts measure frequency of interaction, not cognitive depth or quality of learning.
2. **Exam Grade Asymmetry:** Only modules CCC and DDD record exam grades in the submissions table; other modules integrate exams directly into final outcomes.
3. **Correlation vs. Causation:** Early inactivity correlates strongly with dropout, but may reflect external life events (employment change, health crises) rather than course dissatisfaction.

