# Phase 11 – Task 2: Classic Enterprise Dashboard Implementation Report
## Student Performance & At-Risk Prediction System (OULAD)

- **Date:** 2026-11-15T08:25:00Z
- **Phase Target:** Phase 11 — Frontend & API Implementation (Task 2)
- **Operational Mode:** **STRICT READ-ONLY SHADOW MODE**
- **Production Status:** **WITHHELD (Mandatory Shadow Mode Enforced)**
- **Regression Test Baseline:** **115 / 115 Tests Passing (100%)**

---

## 1. Executive Summary

Phase 11 Task 2 delivers the complete frontend interface and supporting backend analytics API for the **Student Performance & At-Risk Prediction Enterprise Analytics Dashboard**. 

Constructed in strict accordance with the approved architecture specification ([`reports/phase11_dashboard_architecture.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase11_dashboard_architecture.md)) and data provenance audit ([`reports/phase11_data_provenance_audit.md`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/reports/phase11_data_provenance_audit.md)), the platform delivers a classic, authoritative academic computing environment. It avoids generic consumer-AI tropes (neon gradients, glassmorphism, floating cards, oversized pills) in favor of a restrained, high-density, institutional analytics workspace tailored for university academic leadership, student success directors, and authorized academic advisors.

The application operates in **read-only shadow mode**. All outbound communication pathways, automated student interventions, and punitive administrative actions remain architecturally prohibited.

---

## 2. Technology Stack & Architectural Decision

| Component Layer | Technology Selected | Rationale & Enterprise Standards Compliance |
| :--- | :--- | :--- |
| **Frontend Framework** | **React 19 + TypeScript 5.x** | Strong type safety for complex probability and threshold contracts; modular component lifecycle. |
| **Build & Tooling** | **Vite 6** | Fast HMR development, modern ES module bundling, compiled directly to production static assets in `frontend/dist`. |
| **Styling Engine** | **Tailwind CSS v3.4** | Configured with custom institutional color token architecture, 8px spacing grid, and tabular numerals. |
| **Iconography** | **Lucide React** | Consistent 1.5px stroke, restrained academic iconography. |
| **Backend Framework** | **FastAPI (Python 3.12, Pydantic v2)** | High-throughput asynchronous service operating certified Phase 10 inference endpoints alongside read-only analytics routes. |
| **Static Delivery** | **Starlette StaticFiles & FileResponse** | Production bundle served directly by FastAPI at `/` and `/dashboard`, unifying frontend and backend into a single deployment unit. |

---

## 3. Mandatory Visual Direction & Design Tokens

The interface adheres strictly to the palette and typography tokens approved in Task 1:

- **Canvas Background:** `#F7F8FA` (Soft off-white)
- **Navigation Sidebar:** `#14243A` (Deep institutional navy)
- **Primary Brand Blue:** `#315B85` (Muted professional blue)
- **Card Surface:** `#FFFFFF` (Pure white with restrained `border: 1px solid #E2E8F0` and `shadow-sm`)
- **Typography:** Inter with `tabular-nums` enforced across all numerical metrics and percentages.
- **Risk Severity Palette:**
  - **High Risk ($\hat{p} \ge \tau$):** `#C53030` (Muted academic crimson)
  - **Moderate Risk ($0.05 \le \hat{p} < \tau$):** `#D69E2E` (Rich amber)
  - **Low Risk ($\hat{p} < 0.05$):** `#2F855A` (Forest teal)
  - **Suppressed Telemetry (CMA):** `#4A5568` (Slate grey)
  - **Calibrated Milestone (GGG TMA):** `#5A67D8` (Muted royal indigo)

---

## 4. Application Shell & Persistent Safety Enforcement

### 4.1 Persistent Shadow-Mode Safety Banner
Rendered fixed at the top of the interface across all views:
```text
SHADOW MODE ACTIVE — PASSIVE ACADEMIC ADVISORY VIEW ONLY. STUDENT COMMUNICATION AND AUTOMATED OUTREACH ARE PROHIBITED.
```

### 4.2 Fixed 260px Institutional Sidebar
- **Header:** Academic Analytics Portal branding with neutral institutional iconography (no fabricated crests or fake university seals).
- **Navigation Links:** 7 primary analytical pages with active left-border indicator (`border-l-[3px] border-primary-blue`).
- **Footer Status Panel:**
  - Mode: `Shadow (Read-Only)`
  - Model: `HGB v1.0.0`
  - System Health: `Operational` (Live status pulse)

### 4.3 Top Navigation Bar
- Contextual breadcrumbs (`Institutional Portal / [Page Name]`).
- Cohort selector pinned to `2014J (Holdout)`.
- User session badge: `Role: Academic Advisor [READ-ONLY]`.

---

## 5. Seven Implemented Analytical Pages

### Page 1: Executive Cohort Overview
- **Standardized KPI Strip (Max 4 Cards):**
  1. *2014J Enrolled Learners:* `8,607` (Unique enrolled students; excludes assumption of 100% retention).
  2. *Governed TMA Alerts:* `7,130` (Generated from 32,506 actionable TMA snapshots; 21.93% alert rate).
  3. *Unique Learners Alerted:* `3,299` (38.33% student coverage; mean 2.16 alerts per alerted learner).
  4. *Estimated Weekly Alert Load:* `~209.7` (Labeled strictly as a derived historical estimate over 34 weeks, not an approved institutional capacity quota).
- **Visualizations:**
  - *Binned Risk Probability Distribution:* SVG histogram comparing 32,506 actionable TMAs against 17,508 suppressed CMAs, with vertical reference lines at $\tau=0.10$ and $\tau=0.20$.
  - *TMA Alerts by Course Milestone:* Horizontal progress indicators breaking down 2,280 Onboarding alerts (31.98%), 1,801 Midterm alerts (25.26%), and 3,049 Capstone alerts (42.76%).
  - *Illustrative Demonstration Triage Queue:* Exemplar table showing top high-risk snapshots, prominently stamped with `DEMO DATA - NOT LIVE SIS/LMS QUEUE`.

### Page 2: Student Risk Explorer
- **High-Density Ledger:**
  - Server-side paginated, filterable, and searchable table loading 50,014 chronological 2014J snapshots.
  - Multi-criteria filtering: Module (AAA–GGG), Risk Tier (High, Medium, Low, Suppressed), and Assessment Type (TMA, CMA).
  - Column sorting on Predicted Risk, Scheduled Day, Module, and Assessment ID.
- **Privacy Enforcement:**
  - 100% keyed HMAC-SHA256 pseudonymized identifiers (`#8a3e7b16...`). Zero raw student IDs exposed.
  - One-click copy pseudonymized ID button.
- **Case Detail Drawer:**
  - Sliding drawer displaying predicted probability, threshold applied, milestone context, and curricular trajectory signals.
  - Displays mandatory non-punitive advisory disclaimer: *"Advisory Guidance Only: Risk predictions are generated in read-only shadow mode to assist academic advisors with proactive study support. Automated messaging is disabled."*

### Page 3: Course Module Analytics
- **7-Module Comparative Health Matrix:**
  - Displays exact benchmark ($N=170,874$) and holdout ($N=50,014$) failure prevalence alongside actionable TMA denominators ($N=32,506$) and alert volumes ($N=7,130$).
  - Full disaggregation: AAA (12.34% TMA alert rate), BBB (33.81%), CCC (34.57%), DDD (24.85%), EEE (4.56%), FFF (10.63%), GGG (14.72%).
- **Visual Chart:** Grouped bar chart contrasting raw failure prevalence against governed TMA alert rates.
- **Module GGG Special Governance Audit Panel:**
  - Certified matched-evaluation results on exactly 5,656 actionable TMA snapshots across 2,097 unique learners (411 true failures, 7.27% prevalence).
  - Baseline PR-AUC = `0.2508`, Remediated PR-AUC = `0.2353` ($\Delta = -0.0155$).
  - Decision threshold comparison (strictly within-pipeline, assessment-snapshot-level):
    - **Baseline Pipeline:** Lowering $\tau$ from $0.20$ to $0.10$ reduces FN from 312 to 226, capturing **86 additional failure snapshots detected** (recall rises from 24.09% to 45.01%, precision 21.24%).
    - **Remediated Pipeline:** Lowering $\tau$ from $0.20$ to $0.10$ reduces FN from 330 to 223, capturing **107 additional failure snapshots detected** (recall rises from 19.71% to 45.74%, precision 18.63%).
    - *Methodological Warning:* Confusion-matrix counts and recall calculations are strictly assessment-snapshot-level, not unique-student-level (5,656 snapshots across 2,097 unique learners). The previous draft claim of "~89 additional students" erroneously subtracted Baseline TP (99) from Remediated TP (188), conflating feature pipeline changes with threshold adjustments. Pipelines must never be cross-compared when evaluating threshold impact.
  - Documents that Module GGG CMAs are 18 zero-weight formative quizzes scheduled at Day 229/222.
  - Certifies that the erroneous draft figure `0.1189` has been completely excised.

### Page 4: Assessment & Milestone Dynamics
- **Assessment Class Overview:** TMA Coursework (32,506), CMA Quizzes (17,508), Final Exams (Unobserved In-Course).
- **Cold-Start Milestone 1 Investigation:**
  - Grouped CV PR-AUC (0.3203 cold-start vs 0.4882 non-cold-start) vs Chronological 2014J PR-AUC (0.1687 cold-start vs 0.3478 non-cold-start).
  - Documents that cold-start requires supportive nudges due to unobserved assessment history.
- **Submission Lateness Dynamics:** Correlates submission delay with failure rates (3.4% early vs 38.2% severely overdue).
- **Universal CMA Suppression Rationale:** Documents 22.73% alert workload reduction and clarifies that positive-weight CMAs in BBB/CCC/DDD/FFF are suppressed for alert fatigue prevention, not zero-weight decoupling.

### Page 5: Model Explainability (TreeSHAP)
- **Global Feature Attribution Bar Chart:**
  - Recent prior score: `0.7223`
  - Target assessment type: `0.6836`
  - Registration timing days: `0.4473`
  - VLE active days to date: `0.4226`
  - Target assessment weight: `0.2601`
  - Prior submission lateness mean: `0.1873`
  - Prior score standard deviation: `0.1753`
  - VLE days since last activity: `0.1467`
  - VLE clicks last 14 days: `0.1408`
  - VLE total clicks: `0.1396`
- **Methodological Guidance:** Explains that TreeSHAP values reflect associative impact on log-odds of failure, not real-world clinical causality.

### Page 6: Governance & Compliance Ledger
- **Control Matrix:** 6 active safeguard cards verifying read-only shadow mode, zero outreach endpoints, HMAC pseudonymization, CMA suppression, dual thresholds, and demographic feature exclusion.
- **Formal Threshold Policy Registry:** Documents legal basis and operational thresholds ($\tau=0.10$ for GGG TMA, $\tau=0.20$ for standard TMA, suppressed for CMA).
- **Append-only Decision Audit Log:** Streaming table capturing `DecisionAuditRecord` entries with prediction UUID, timestamp, HMAC student pseudonym, input feature SHA-256 hash, and governance verdict. Thread-safe append log without claims of cryptographic immutability.

### Page 7: Model Monitoring & Generalization
- **Generalization Gap Table & Metric Provenance:**
  - Grouped 5-Fold CV PR-AUC: `0.4606` vs Chronological 2014J Holdout: `0.3005` ($-34.76\%$).
  - *Provenance Note on PR-AUC 0.4611 vs 0.4606:* The figure `0.4611 ± 0.0088` reported in Phase 6 nested CV is the unweighted arithmetic mean (macro-average) of Average Precision scores computed across the 5 outer folds (`reports/model_optimization_summary.json`). The figure `0.4606` reported on the dashboard and candidate comparison benchmark is the pooled (micro-average) Average Precision computed across all concatenated $N = 170,874$ out-of-fold prediction rows (`reports/candidate_model_comparison_summary.json`). Both originate from the identical Phase 6 model predictions on identical benchmark rows.
  - ROC-AUC: `0.8687` vs `0.7969` ($-8.27\%$).
  - Brier Score: `0.0592` vs `0.0812` ($+37.16\%$).
  - Expected Calibration Error (ECE): `0.0163` (Grouped 5-Fold CV) vs `0.0420` (Chronological 2014J Holdout) ($+157.67\%$). Reconciled from unverified draft entry `0.0210`.
- **PSI Feature Drift Breakdown:**
  - Severe: `target_assessment_type` (`0.4077`, curricular addition of CMAs).
  - Moderate: `registration_timing_days` (`0.2191`), `vle_active_days_to_date` (`0.1102`).
  - Stable: `prior_at_risk_rate` (`0.0821`), `mean_prior_score` (`0.0645`).
- **Telemetry Limitations:** Documents absence of actual grade release timestamps (14-day operational delay assumption) and lack of client-side VLE dwell time.

---

## 6. Backend API Integration & Architecture

The FastAPI backend (`src/api.py` and [`src/analytics_routes.py`](file:///Users/navadeepguduru/Student%20At%20Risk%20Prediction/src/analytics_routes.py)) implements 7 read-only RESTful endpoints:

1. `GET /api/v1/analytics/cohort-summary`: Certified 2014J workload and coverage metrics.
2. `GET /api/v1/analytics/risk-distribution`: Binned continuous probability distribution and threshold markers.
3. `GET /api/v1/analytics/student-explorer`: Paginated, filtered, and sorted assessment snapshots with keyed HMAC pseudonymization.
4. `GET /api/v1/analytics/modules-comparison`: 7-module comparative metrics and GGG matched evaluation panel.
5. `GET /api/v1/explainability/global-importance`: Certified TreeSHAP global feature attributions.
6. `GET /api/v1/governance/audit-records`: Decision audit records and compliance policy configuration.
7. `GET /api/v1/monitoring/psi-feature-drift`: Generalization gap metrics, PSI drift, and calibration telemetry.
8. `GET /` & `GET /dashboard`: Serves the compiled production React SPA (`frontend/dist/index.html`).
9. `GET /assets/*`: Serves compiled JS and CSS bundles via `StaticFiles`.

---

## 7. Operational Safety Safeguards & Verification

1. **Outreach Prohibition Hard-Lock:** `ALLOW_STUDENT_OUTREACH = False` remains active. All outreach permission fields return `False`.
2. **Zero Communication Routes:** Prohibited routes (`/intervene`, `/outreach`, `/send-message`, `/email`, `/notify`, `/nudge`) return 404.
3. **Student Privacy Protection:** Zero raw student IDs in UI or responses; all student identifiers are transformed into 64-character keyed HMAC-SHA256 digests.
4. **Demographic Fairness:** All sensitive demographic features (`gender`, `imd_band`, `disability`, `age_band`, `highest_education`, `region`) remain excluded.
5. **No Model Retraining:** Models, thresholds, and pipelines remain untouched.

---

## 8. Compilation & Artifact Status

- **Frontend Build:** `npm run build` completed in **527ms**; generated clean production bundle in `frontend/dist/` (`dist/assets/index-CIU3nB-E.js`, `dist/assets/index-DV91sqDq.css`).
- **API Tests:** 25 integration tests passed across inference, governance, and analytics routes.
- **Full Test Suite:** **115 of 115 tests passed in 9.32s** (106 baseline regression tests + 9 new analytics tests).

Production deployment remains **WITHHELD**. The platform operates exclusively as an offline academic research shadow-mode tool.

