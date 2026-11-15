# Phase 11 – Task 1: Classic Enterprise Dashboard Architecture & UI Design Specification
## Student Performance & At-Risk Prediction System (OULAD)

- **Document Version:** 2.0 (Post-Provenance Audit & Architectural Correction)  
- **Date:** 2026-11-15T07:40:00Z  
- **Phase Target:** Phase 11 — Enterprise Dashboard Architecture & Implementation  
- **Operational Mode:** **STRICT READ-ONLY SHADOW MODE**  
- **Production Status:** **WITHHELD (Mandatory Shadow Mode Enforced)**  

---

## Executive Summary

This specification defines the complete technical architecture, design system, component hierarchy, API data contracts, and governance safeguards for the **Student Performance & At-Risk Prediction Enterprise Analytics Dashboard**.

Designed as a **classic, high-density, academic enterprise product**, the interface rejects generic consumer-AI aesthetics (neon gradients, floating glassmorphic panels, oversized pills, and decorative charts) in favor of a restrained, authoritative, institutional analytics workspace tailored for university academic leadership, student success directors, and authorized academic advisors.

The dashboard functions strictly in **passive shadow mode**: it visualizes historical and prospective student assessment risk, course-level vulnerabilities, TreeSHAP model explainability, feature drift, and governance compliance ledgers. **All outbound student communication, automated interventions, and punitive actions remain architecturally prohibited.**

---

## 1. System & Technical Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    ENTERPRISE APPLICATION SHELL                                    |
|                                                                                                    |
|  +------------------------------+  +------------------------------------------------------------+  |
|  |     FIXED SIDEBAR (260px)    |  |                 TOP HEADER & CONTEXT BAR (64px)            |  |
|  |  - Neutral Academic Portal   |  |  - Breadcrumbs & Page Title                                |  |
|  |  - Primary Navigation (7 pp) |  |  - Global Intake Selector (2013B - 2014J)                  |  |
|  |  - System Health Indicator   |  |  - Shadow-Mode Safety Badge [READ-ONLY ACTIVE]             |  |
|  |  - Shadow-Mode Status Pill   |  |  - Authorized Advisor Session Indicator                    |  |
|  +------------------------------+  +------------------------------------------------------------+  |
|                                 |                                                                  |
|                                 |  +------------------------------------------------------------+  |
|                                 |  |              MAIN CONTENT AREA (Responsive 12-Col)         |  |
|                                 |  |  - Global Sticky Filter Toolbar                            |  |
|                                 |  |  - KPI Metric Strip (Restrained, Tabular Numerals)         |  |
|                                 |  |  - Primary Analytic Visualizations (Recharts)              |  |
|                                 |  |  - High-Density Data Tables (Pseudonymized HMAC Records)   |  |
|                                 |  |  - Explainability & Audit Drawers                          |  |
|                                 |  +------------------------------------------------------------+  |
+---------------------------------+------------------------------------------------------------------+
                                  |
                                  v (Typed REST API over HTTP/JSON)
+----------------------------------------------------------------------------------------------------+
|                             FASTAPI ENTERPRISE BACKEND SERVICE (src/api.py)                        |
|                                                                                                    |
|  +---------------------------------------------+  +---------------------------------------------+  |
|  |    EXISTING VERIFIED PHASE 10 ENDPOINTS     |  |       PROPOSED PHASE 11 ANALYTICS ROUTES    |  |
|  |  - GET /health                              |  |  - GET /api/v1/analytics/cohort-summary     |  |
|  |  - GET /model-info                          |  |  - GET /api/v1/analytics/student-explorer   |  |
|  |  - POST /predict                            |  |  - GET /api/v1/analytics/modules-comparison |  |
|  |  - POST /predict/batch                      |  |  - GET /api/v1/monitoring/psi-feature-drift |  |
|  +---------------------------------------------+  +---------------------------------------------+  |
|                                                                                                    |
|  +----------------------------------------------------------------------------------------------+  |
|  |                        CERTIFIED MODEL ARTIFACTS & DATA STORAGE                              |  |
|  |  - Primary HGB Joblib (23 features, Passthrough Numericals, Ordinal Categoricals, SHA-256)  |  |
|  |  - Challenger Random Forest Joblib (100 Trees, Median/OHE Pipelines)                         |  |
|  |  - OULAD Benchmarks: 170,874 historical snapshots & 50,014 2014J prospective evaluations    |  |
|  |  - Immutable Audit Records: decision_audit.jsonl (thread-safe append log)                   |  |
|  +----------------------------------------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
```

### 1.1 Technology Stack Selection
- **Frontend Framework:** React 19 with TypeScript 5.x. Provides strong type safety for complex risk probability contracts and high component reusability.
- **Build & Development Tooling:** Vite 6. Fast HMR, minimal bundle footprint, and native ES module bundling.
- **Styling Architecture:** Tailwind CSS v3.4 configured with a strict custom color token palette, tabular typography, and an 8px spacing grid.
- **Data Visualization Library:** Recharts v2.x. Composable, responsive SVG charts matching publication standards with custom tooltips and accessible color palettes.
- **Icons:** Lucide React. Consistent 1.5px stroke, restrained academic iconography.
- **State Management & Caching:** TanStack Query (React Query v5) for server-state caching, optimistic updates, and background invalidation, combined with Zustand for global filter state.
- **Backend Service:** FastAPI (Python 3.12, Uvicorn, Pydantic v2). Operates existing certified Phase 10 API endpoints and defines read-only aggregated analytics routes for Phase 11.

---

## 2. Visual Direction & Design System Specification

### 2.1 Design Philosophy
The visual direction follows classic **Institutional Academic Computing**: calm, functional, high-density, and authoritative. It prioritizes legibility, information density, and unambiguous risk communication over decorative flair.
- **Branding Constraints:** Generic institutional styling solely. Fabricated university crests, fictitious institutional endorsements, or fake advisor names (e.g. "Dr. Smith") are strictly avoided. Navigation elements identify the system neutrally as the **Academic Analytics Portal**.

### 2.2 Color Token Architecture

| Semantic Token | Hex Code | Purpose & Usage Guidelines |
| :--- | :--- | :--- |
| `canvas-bg` | `#F7F8FA` | Primary soft off-white canvas background across all pages. |
| `surface-card` | `#FFFFFF` | Pure white background for analytic cards, tables, and modal drawers. |
| `sidebar-bg` | `#14243A` | Deep institutional navy for the fixed left navigation sidebar. |
| `sidebar-hover` | `#1C3250` | Subtle navy elevation for hover states on sidebar navigation items. |
| `sidebar-active` | `#233E63` | Focused navy container with a 3px primary blue left border indicator. |
| `primary-blue` | `#315B85` | Muted professional blue for primary active controls, tabs, and headers. |
| `primary-hover` | `#26496B` | Deepened blue for hovered interactive elements. |
| `primary-light` | `#EBF2F8` | Light blue tint for selected table rows and primary badge backgrounds. |
| `border-subtle` | `#E2E8F0` | Restrained 1px border for card boundaries, table cell dividers, and headers. |
| `border-medium` | `#CBD5E0` | Elevated border for interactive inputs, active dropdowns, and chart grids. |
| `text-primary` | `#1A202C` | Deep charcoal for primary headings, table cell values, and metric values. |
| `text-secondary` | `#4A5568` | Neutral slate for field labels, table headers, and breadcrumb trails. |
| `text-muted` | `#718096` | De-emphasized grey for micro-labels, timestamps, and secondary captions. |

#### Risk Status Palette (Calibrated for Accessibility & Severity)
- **High Risk ($\hat{p} \ge \tau$):**
  - Base: `#C53030` (Muted Academic Crimson)
  - Text: `#9B2C2C`
  - Surface: `#FFF5F5`
  - Border: `#FEB2B2`
  - Semantic Usage: Actionable TMA alerts requiring advisor case review.
- **Moderate Risk ($0.05 \le \hat{p} < \tau$):**
  - Base: `#D69E2E` (Rich Amber)
  - Text: `#B7791F`
  - Surface: `#FFFAF0`
  - Border: `#FBD38D`
  - Semantic Usage: Watchlist learners with borderline activity trajectories.
- **Low Risk ($\hat{p} < 0.05$):**
  - Base: `#2F855A` (Muted Forest / Teal)
  - Text: `#22543D`
  - Surface: `#F0FFF4`
  - Border: `#C6F6D5`
  - Semantic Usage: On-track learners with stable submission and VLE engagement.
- **Outreach Suppressed (CMA Telemetry):**
  - Base: `#4A5568` (Slate Grey)
  - Text: `#2D3748`
  - Surface: `#EDF2F7`
  - Border: `#CBD5E0`
  - Semantic Usage: Computer-marked quiz predictions strictly excluded from automated outreach.
- **Calibrated Milestone (Module GGG TMA):**
  - Base: `#5A67D8` (Muted Royal Indigo)
  - Text: `#434190`
  - Surface: `#EBF4FF`
  - Border: `#C3DAFE`
  - Semantic Usage: GGG coursework milestone routed at calibrated $\tau = 0.10$.

### 2.3 Typography Scale (Inter Font Family)
- **Display / H1:** 24px (`text-2xl`), font-weight 600 (`font-semibold`), line-height 32px, tracking `-0.02em`. Used for page titles.
- **Section / H2:** 18px (`text-lg`), font-weight 600, line-height 26px, tracking `-0.01em`. Used for card headers.
- **Subsection / H3:** 14px (`text-sm`), font-weight 600, line-height 20px. Used for sub-panel labels.
- **Body Regular:** 13px (`text-[13px]`), font-weight 400 (`font-normal`), line-height 18px. Used for table content and prose.
- **Caption / Meta:** 11px (`text-xs`), font-weight 500 (`font-medium`), line-height 16px. Used for status tags, disclaimers, and chart axes.
- **Tabular Figures:** `font-mono tabular-nums` used for all numerical statistics, probabilities, dates, and SHA-256 hashes.

### 2.4 Spacing & Elevation System
- **Strict 8px Grid:** Spacing adheres strictly to multiples of 8px (`8px, 16px, 24px, 32px, 48px`). A 4px half-unit is permitted only for compact badge padding.
- **Corner Radii:** Subtle, restrained rounding (`rounded-lg` = 8px for cards, inputs, and buttons; `rounded-md` = 6px for small tags). Pill buttons (`rounded-full`) are prohibited.
- **Restrained Elevation:**
  - Card Shadow: `0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.03)`
  - Dropdown / Flyout: `0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.04)`
  - Border: 1px solid `#E2E8F0` on all cards and surfaces.

### 2.5 Strict Anti-Patterns (Explicit Exclusions)
- **NO Gradients or Neons:** Bright cyan, neon purple, or multi-stop gradients are prohibited.
- **NO Glassmorphism:** Backdrop blur, translucent overlays, and frosted glass cards are prohibited.
- **NO Excessive Animations:** Bouncing cards, animated pulse rings, and layout-shift animations are prohibited. Chart animations must be disabled or subtle ($<200\text{ms}$ ease-out).
- **NO Decorative KPI Clutter:** Pages must never display more than 4 primary KPI cards.
- **NO Dark-Mode Default:** The interface is engineered exclusively as a high-legibility, daylight-optimized enterprise application.

---

## 3. Application Layout & Component Hierarchy

```
+----------------------------------------------------------------------------------------------------+
|                                      GLOBAL APPLICATION SHELL                                      |
+---------------------+------------------------------------------------------------------------------+
| FIXED SIDEBAR       | TOP HEADER (64px)                                                            |
| (260px Width)       | [Breadcrumb: Home > Overview]     [Intake: 2014J v] [Session: Advisor Auth]  |
|                     | [BANNER: SHADOW MODE ACTIVE -- READ-ONLY ADVISORY VIEW -- NO STUDENT OUTREACH]      |
|                     +------------------------------------------------------------------------------+
|                     | MAIN CONTENT CONTAINER (p-6, max-w-[1440px])                                 |
| - Portal Shield     |                                                                              |
|                     | +--------------------------------------------------------------------------+ |
| NAVIGATION:         | | GLOBAL FILTER BAR: Module [All v] | Milestone [All v] | Horizon [7d v]   | |
| 1. Overview         | +--------------------------------------------------------------------------+ |
| 2. Student Explorer |                                                                              |
| 3. Module Analytics | +--------------------+ +--------------------+ +--------------------+         |
| 4. Assessment Anal. | | KPI Card 1         | | KPI Card 2         | | KPI Card 3         |         |
| 5. Explainability   | +--------------------+ +--------------------+ +--------------------+         |
| 6. Governance/Audit |                                                                              |
| 7. Model Monitoring | +------------------------------------+ +-----------------------------------+ |
|                     | | Primary Analytic Chart (7 cols)    | | Secondary Distribution (5 cols)   | |
| FOOTER:             | +------------------------------------+ +-----------------------------------+ |
| - Model: HGB v1.0.0 |                                                                              |
| - Shadow Mode Tag   | +--------------------------------------------------------------------------+ |
| - Audit Status OK   | | High-Density Tabular Data Grid / Demonstrative Triage Queue              | |
+---------------------+ +--------------------------------------------------------------------------+ |
```

### 3.1 Responsive Breakpoints
1. **Desktop ($> 1024px$):** Fixed 260px sidebar, fluid content container capped at 1440px, full multi-column grid layouts.
2. **Tablet ($768px - 1024px$):** Sidebar collapses to an icon-only 64px rail with flyout tooltips. Analytic grids collapse from 4 columns to 2 columns.
3. **Mobile ($< 768px$):** Sidebar shifts to an off-canvas drawer accessed via a top header hamburger button. Analytic cards stack into a single column. Tables enable horizontal scrolling with fixed first-column pinning.

---

## 4. Planned Dashboard Pages: Comprehensive Specifications

### Page 1: Overview Dashboard
- **Primary Purpose:** Provide university academic leadership and advising directors with a high-level operational pulse of cohort risk posture and historical derived advising workload.
- **Grid Layout:** 
  - Top: 4-metric KPI summary strip (12-column grid, 3 cols each).
  - Middle: Risk Distribution Histogram (7 cols) + Derived Weekly Alert Volume Distribution (5 cols).
  - Bottom: Assessment Milestone Risk Comparison (6 cols) + Illustrative Demonstration Triage Queue (6 cols).
- **Core Components & Visualizations:**
  1. **Metric Strip:**
     - *2014J Enrolled Learners:* 8,607 (Total unique learners enrolled in prospective intake 2014J; does not assume 100% active retention).
     - *Actionable TMA Alerts:* 7,130 alerts across 3,299 unique learners (38.33% coverage on 2014J holdout).
     - *Estimated Weekly Alert Load:* ~209.7 alerts/week (*Historical derived estimate based on 7,130 alerts across 34 academic weeks; not an approved institutional capacity quota*).
     - *Historical 2014J Raw-Policy CMA Alerts Suppressed:* 2,243 alerts (-22.73% alert reduction relative to the 9,227 unconstrained alerts; represents the historical raw 0.20 policy CMA alert count on 2014J, not a universal count of all CMA records across the entire dataset).
  2. **Risk Distribution Histogram:** Binned probability distribution $[0.0, 1.0]$ with vertical dashed reference lines at $\tau=0.10$ (GGG TMA threshold) and $\tau=0.20$ (Standard TMA threshold). Bars color-coded by tier (teal, amber, crimson).
  3. **Historical Weekly Alert Arrival Distribution:** Derived weekly alert arrival volume across academic weeks 1–34 based on scheduled assessment deadlines, showing peak alert volumes during Midterm (Week 8) and Capstone (Week 24). Unverified institutional capacity ceilings are strictly excluded.
  4. **Assessment Milestone Risk Comparison:** Grouped bar chart comparing the distribution of risk tiers and alert volumes across assessment stages (TMA 1 Onboarding, TMA 2 Midterm, TMA 3 Capstone). Avoids sequential conversion funnel metaphors.
  5. **Illustrative Demonstration Triage Queue:** Compact demonstration table showing how top high-risk TMA snapshots would appear to advisors. Every row and metric is explicitly badged `[DEMO DATA - NOT LIVE SIS/LMS QUEUE]`.

---

### Page 2: Student Risk Explorer
- **Primary Purpose:** Enable academic advisors to inspect individual student risk profiles, understand drivers of coursework vulnerability, and prepare supportive check-in notes without contacting students directly.
- **Grid Layout:**
  - Top: Search & Filter Toolbar (Search by HMAC, Module multi-select, Milestone filter, Risk Tier toggle).
  - Main: Full-width paginated high-density data grid (12 cols).
  - Side / Flyout: Sliding Right Drawer (480px width) containing individual TreeSHAP explanation and engagement timeline.
- **Core Components & Data Columns:**
  1. **Triage Data Grid:**
     - `Pseudonymized ID`: Truncated 12-char HMAC (`8a3e7b16...`) with click-to-copy full 64-char hash.
     - `Module & Intake`: e.g. `BBB 2014J`.
     - `Next Assessment`: Assessment ID, type badge (`TMA`), weight (`18%`), and due date (`Day 54`).
     - `Risk Probability`: Exact value (`0.3472`) with mini inline horizontal progress bar.
     - `Risk Category`: Badge (`HIGH_RISK`, `MEDIUM_RISK`, `LOW_RISK`).
     - `Operational Status`: Governance tag (`SHADOW_MODE_ADVISORY_REVIEW`).
     - `Outreach Status`: Strict lock badge (`OUTREACH PROHIBITED`).
     - `Action`: "Inspect Drivers" text button (triggers side drawer).
  2. **Student Detail Slide-Over Drawer:**
     - Header: Pseudonymized Hash, Course Context, and Shadow-Mode Caution Tag.
     - Metric Row: Total VLE Clicks to date, Active Days to date, Days Since Last Click.
     - Local TreeSHAP Waterfall Chart: Top positive risk pushers and negative protective engagement factors computed from verified snapshot features.
     - Advisory Recommendation Box: Non-punitive supportive guidance.

---

### Page 3: Module Analytics
- **Primary Purpose:** Deliver comparative performance, failure vulnerability, and alert burden analytics across all 7 undergraduate course modules (AAA through GGG).
- **Grid Layout:**
  - Top: 7-Module Comparative Health Matrix (Full Width).
  - Middle: Module Alert Volume vs Failure Prevalence Bar Chart (7 cols) + Course Milestone Difficulty Profiles (5 cols).
  - Bottom: Module GGG Special Governance Audit Panel (12 cols).
- **Core Components & Visualizations:**
  1. **Comparative Module Table:**
     - Columns: Module Code, Faculty Domain, Total Benchmark Snapshots ($N=170,874$), Historical Benchmark Failure Prevalence, 2014J Holdout Snapshots ($N=50,014$), 2014J Holdout Failure Prevalence, 2014J Actionable TMA Snapshots ($N=32,506$), Governed TMA Alerts ($N=7,130$), TMA Alert Rate (%).
     - Strictly Separated Denominators & Verified Empirical Rates:
       - **AAA** (Social Sciences): Benchmark $N = 3,146$ (237 failures, 7.53% prevalence); 2014J Cohort $N = 1,515$ (129 failures, 8.51% prevalence); 2014J TMA Snapshots = 1,515; Governed TMA Alerts = 187 (12.34% TMA alert rate).
       - **BBB** (Social Sciences): Benchmark $N = 42,979$ (3,166 failures, 7.37% prevalence); 2014J Cohort $N = 7,398$ (1,236 failures, 16.71% prevalence); 2014J TMA Snapshots = 7,398; Governed TMA Alerts = 2,501 (33.81% TMA alert rate).
       - **CCC** (STEM): Benchmark $N = 17,014$ (2,590 failures, 15.22% prevalence); 2014J Cohort $N = 10,277$ (1,386 failures, 13.49% prevalence); 2014J TMA Snapshots = 4,431 (5,846 CMAs suppressed); Governed TMA Alerts = 1,532 (34.57% TMA alert rate).
       - **DDD** (STEM): Benchmark $N = 29,865$ (4,730 failures, 15.84% prevalence); 2014J Cohort $N = 7,051$ (829 failures, 11.76% prevalence); 2014J TMA Snapshots = 7,051; Governed TMA Alerts = 1,752 (24.85% TMA alert rate).
       - **EEE** (STEM): Benchmark $N = 7,886$ (352 failures, 4.46% prevalence); 2014J Cohort $N = 3,227$ (126 failures, 3.90% prevalence); 2014J TMA Snapshots = 3,227; Governed TMA Alerts = 147 (4.56% TMA alert rate).
       - **FFF** (STEM): Benchmark $N = 54,769$ (2,636 failures, 4.81% prevalence); 2014J Cohort $N = 16,175$ (646 failures, 3.99% prevalence); 2014J TMA Snapshots = 7,260 (8,915 CMAs suppressed); Governed TMA Alerts = 772 (10.63% TMA alert rate).
       - **GGG** (STEM): Benchmark $N = 15,215$ (912 failures, 5.99% prevalence); 2014J Cohort $N = 4,371$ (285 failures, 6.52% prevalence); 2014J TMA Snapshots = 1,624 (2,747 CMAs suppressed); Governed TMA Alerts = 239 (14.72% TMA alert rate under calibrated $\tau=0.10$).
     - Cohort Aggregates:
       - Full Benchmark: $N = 170,874$ snapshots, 14,623 failures, 8.56% overall prevalence.
       - 2014J Prospective Holdout: $N = 50,014$ snapshots, 4,637 failures, 9.27% overall prevalence.
       - 2014J Actionable TMA Snapshots: $N = 32,506$ snapshots, 3,556 failures, 10.94% TMA failure prevalence.
       - Governed Actionable TMA Alerts: $N = 7,130$ alerts (21.93% of actionable TMA snapshots; 14.26% across all 50,014 cohort snapshots).
  2. **Alert Burden vs Prevalence Chart:** Grouped bar chart comparing raw failure prevalence to model alert volume per module. Demonstrates that governed thresholds prevent over-alerting in low-prevalence courses while capturing vulnerability in high-prevalence courses (BBB, CCC, DDD).
  3. **Module GGG Remediation & Governance Panel:**
     - Dedicated card detailing the Phase 9 remediation findings.
     - Visualizes Module GGG's 18 total computer-marked quizzes (6 per presentation across 2013J, 2014B, 2014J, clustered at Day 229/222 with 0% weight) decoupled from assessment outcomes.
     - Clarifies that zero-weight CMA decoupling applies strictly to Module GGG coursework quizzes; CMAs in other modules carry positive weights and are suppressed based on operational research telemetry designation.
     - Displays the verified matched-evaluation results on exactly 5,656 actionable TMA snapshots across 2,097 unique learners (411 positive failures, 7.27% baseline prevalence):
       - Baseline Pipeline (Phase 6): PR-AUC = **0.2508**, ROC-AUC = **0.7510**.
       - Remediated Pipeline: PR-AUC = **0.2353**, ROC-AUC = **0.7451** ($\Delta = -0.0155$).
       - Threshold Calibration Impact ($\tau=0.10$ vs Global $\tau=0.20$):
         - At Global $\tau = 0.20$: Baseline Recall is 24.09% (99/411 detected, 312 missed, FNR 75.91%); Remediated Recall is 19.71% (81/411 detected, 330 missed, FNR 80.29%).
         - At Calibrated $\tau = 0.10$: Baseline Recall rises to 45.01% (185/411 detected, 226 missed, FNR 54.99%); Remediated Recall rises to 45.74% (188/411 detected, 223 missed, FNR 54.26%).
         - Operational Gain: Calibrated thresholding captures ~89 additional at-risk students (+21.65 percentage points in recall), overcoming the low-prevalence signal dilution caused by high practice-quiz scores.
       - Discrepancy Reconciliation: The draft figure "PR-AUC 0.1189" was an unverified draft artifact with zero empirical basis and has been excised.

---

### Page 4: Assessment Analytics
- **Primary Purpose:** Track assessment-specific dynamics, submission lateness behavior, cold-start milestone 1 challenges, and formative quiz telemetry.
- **Grid Layout:**
  - Top: 3-Metric Assessment Overview (TMA Count, CMA Count, Exam Count).
  - Middle: Submission Timeliness vs Risk Scatter Plot (7 cols) + Assessment Score Distributions by Type (5 cols).
  - Bottom: Cold-Start Assessment 1 Performance Investigation Panel (12 cols).
- **Core Components & Visualizations:**
  1. **Submission Timeliness vs Score Plot:** Scatter plot showing student submission lateness ($\text{date\_submitted} - \text{date\_due}$) on x-axis versus final score on y-axis from `data/processed/assessment_submissions_clean.csv`.
  2. **TMA vs CMA Score Distribution Chart:** Box plots contrasting TMA score variance against CMA score distributions.
  3. **Cold-Start Assessment 1 Panel:**
     - Reconciles historical Phase 7 cross-validation benchmarks against Phase 9 2014J holdout benchmarks:
       - *Phase 7 Grouped CV Benchmark ($N = 170,874$ across all 4 intakes):* Missing historical grade subgroup ($N = 29,478$, prevalence 10.49%) achieved PR-AUC **0.3203** (Brier 0.0810); observed grade subgroup ($N = 141,396$, prevalence 8.16%) achieved PR-AUC **0.4882** (Brier 0.0547).
       - *Phase 9 Task 2 2014J Chronological Holdout ($N = 50,014$):* Cold-start subgroup ($N = 10,407$, prevalence 13.60%) achieved PR-AUC **0.1687**; non-cold-start subgroup ($N = 39,607$, prevalence 8.13%) achieved PR-AUC **0.3478**; overall 2014J holdout achieved PR-AUC **0.3005**.
     - Highlights that prior-grade missingness increased from 15.78% in training to 20.81% in 2014J, driving chronological degradation.

---

### Page 5: Model Explainability (TreeSHAP)
- **Primary Purpose:** Provide algorithmic transparency and explainability, enabling advisors and auditors to inspect global and subgroup model decision boundaries based strictly on verified TreeSHAP artifacts.
- **Grid Layout:**
  - Top: Global TreeSHAP Feature Importance Ranking (12 cols).
  - Middle: Feature Dependence Visualizer (7 cols) + Risk Tier Factor Decomposition (5 cols).
  - Bottom: Cold-Start Feature Substitution Analysis (12 cols).
- **Core Components & Visualizations:**
  1. **Global Feature Importance Chart:** Horizontal bar chart displaying mean absolute SHAP values across all 23 features from `reports/model_explainability_summary.json`:
     - Top verified features: `recent_prior_score` (0.7223), `target_assessment_type` (0.6836), `registration_timing_days` (0.4473), `vle_active_days_to_date` (0.4226), `target_assessment_weight` (0.2601), `prior_submission_lateness_mean` (0.1873).
  2. **Feature Dependence Visualizer:** Interactive display presenting verified SHAP feature values and importance distributions without unverified claims.
  3. **Risk Tier Factor Breakdown:** Grouped bar chart comparing the relative contribution of prior academic history versus real-time VLE engagement across High-Risk, Medium-Risk, and Low-Risk tiers.

---

### Page 6: Governance & Audit
- **Primary Purpose:** Serve as the definitive compliance, fairness, and security ledger for institutional regulators, ethics boards, and model validators.
- **Grid Layout:**
  - Top: Governance Status Summary Strip (Outreach Lock: Active, KMS Rotation: Healthy, Audit Logging: 100%).
  - Middle: Threshold Configuration Matrix (6 cols) + Retrospective Incident Review Outcomes (6 cols).
  - Bottom: Real-Time Decision Audit Log Table (12 cols).
- **Core Components & Compliance Records:**
  1. **Threshold Configuration Matrix:** Formal policy table displaying module code, assessment type, operational threshold, outreach permission, and legal basis:
     - Modules AAA–FFF TMA: $\tau = 0.20$ (Outreach Prohibited in Shadow Mode).
     - Module GGG TMA: $\tau = 0.10$ (Outreach Prohibited in Shadow Mode).
     - Universal CMA (AAA–GGG): Suppressed (Research Telemetry Only).
  2. **Retrospective Incident Review Distribution:** Donut / Bar chart displaying the classification of retrospective prediction outcomes without defect bias:
     - True Positives ($41.2\%$), False Positives ($58.8\%$ -- annotated with *Expected Tradeoff / Not a Model Defect*), CMA Excluded ($100\%$ excluded from incident counts).
  3. **Decision Audit Ledger:** Searchable, paginated audit table streaming records from `DecisionAuditLogger`:
     - Columns: Prediction UUID, Timestamp (UTC), Pseudonymized Student ID, Module, Assessment ID, Feature Vector Hash (SHA-256), Risk Probability, Threshold Applied, Governance Action.

---

### Page 7: Model Monitoring
- **Primary Purpose:** Track temporal generalization stability, concept drift, feature drift (PSI), and probability calibration health across academic presentations.
- **Grid Layout:**
  - Top: Generalization Gap Card Strip (Grouped CV PR-AUC: 0.4606 vs 2014J Holdout: 0.3005).
  - Middle: 23-Feature Population Stability Index (PSI) Heat Grid (8 cols) + Reliability & Calibration Curve (4 cols).
  - Bottom: Ingestion Pipeline & Grading Latency Monitor (12 cols).
- **Core Components & Drift Metrics:**
  1. **Generalization Gap Overview:** Comparative metric cards showing PR-AUC, ROC-AUC, and Brier Score between cross-validation training and the chronological 2014J holdout intake ($N = 50,014$).
  2. **Population Stability Index (PSI) Drift Table:** Table displaying all 23 features, Cohen's $d$, Wasserstein distance, and PSI values from `reports/temporal_drift_summary.json`:
     - Flags severe drift ($\text{PSI} \ge 0.25$) on `target_assessment_weight` ($\text{PSI} = 2.6398$), `scheduled_date` ($\text{PSI} = 1.0964$), and `vle_active_days_to_date` ($\text{PSI} = 0.4578$).
  3. **Reliability Calibration Diagram:** 10-bin calibration curve plotting predicted risk probability against observed empirical failure rate. Reconciled and annotated with Brier score ($0.0812$) and certified Expected Calibration Error (**ECE: 0.0420** from `reports/temporal_drift_summary.json` line 464).
  4. **LMS ETL & Gradebook Latency Telemetry:** Visualizes the 7-day operational candidate horizon versus the 1-day unverified horizon.

---

## 5. API Integration Plan & Data Contracts

```
+----------------------------------------------------------------------------------------------------+
|                                      API CONTRACT MAPPING                                          |
+-------------------------------------------------+--------------------------------------------------+
| DASHBOARD PAGE                                  | FASTAPI ENDPOINTS & DATA SOURCES                 |
+-------------------------------------------------+--------------------------------------------------+
| Existing Certified Phase 10 Endpoints           | GET /health (Health & Shadow Mode Status)        |
|                                                 | GET /model-info (Metadata & 23 Features)         |
|                                                 | POST /predict (Single Snapshot Inference)        |
|                                                 | POST /predict/batch (Batch Ingestion Scoring)    |
+-------------------------------------------------+--------------------------------------------------+
| Proposed Phase 11 Analytics Routes              | GET /api/v1/analytics/cohort-summary             |
| (To be implemented in Task 2)                   | GET /api/v1/analytics/risk-distribution          |
|                                                 | GET /api/v1/analytics/student-explorer           |
|                                                 | GET /api/v1/analytics/modules-comparison         |
|                                                 | GET /api/v1/analytics/assessment-timeliness      |
|                                                 | GET /api/v1/explainability/global-importance     |
|                                                 | GET /api/v1/governance/audit-records             |
|                                                 | GET /api/v1/monitoring/psi-feature-drift         |
|                                                 | GET /api/v1/monitoring/calibration-curve         |
+-------------------------------------------------+--------------------------------------------------+
```

### 5.1 Verification Status of API Endpoints

| Endpoint | HTTP Method | Implementation State | Verification & Safety Guarantees |
| :--- | :---: | :---: | :--- |
| `/health` | `GET` | **Existing & Certified (Phase 10)** | Validated by 17 passing tests in `tests/test_phase10_api.py`. Confirms `SHADOW_MODE_READ_ONLY`. |
| `/model-info` | `GET` | **Existing & Certified (Phase 10)** | Validated by `test_phase10_api.py`. Exposes 23 features, thresholds, and offline status. |
| `/predict` | `POST` | **Existing & Certified (Phase 10)** | Validated by `test_phase10_api.py`. Enforces HMAC pseudonymization, CMA suppression, and shadow mode. |
| `/predict/batch` | `POST` | **Existing & Certified (Phase 10)** | Validated by `test_phase10_api.py`. Batch processing with isolated error handling. |
| `/api/v1/analytics/cohort-summary` | `GET` | **Proposed (Phase 11 Task 2)** | Pre-computed from `oof_predictions_chronological_2014J.csv` and summary JSONs. |
| `/api/v1/analytics/student-explorer` | `GET` | **Proposed (Phase 11 Task 2)** | Paginated read-only view of pseudonymized 2014J predictions. |
| `/api/v1/analytics/modules-comparison` | `GET` | **Proposed (Phase 11 Task 2)** | Aggregated from certified Phase 9 Task 5 module distribution records. |
| `/api/v1/explainability/global-importance` | `GET` | **Proposed (Phase 11 Task 2)** | Reads TreeSHAP feature importance from `reports/model_explainability_summary.json`. |
| `/api/v1/governance/audit-records` | `GET` | **Proposed (Phase 11 Task 2)** | Reads stream from `DecisionAuditLogger`. |
| `/api/v1/monitoring/psi-feature-drift` | `GET` | **Proposed (Phase 11 Task 2)** | Reads 23-feature PSI metrics from `reports/temporal_drift_summary.json`. |

---

## 6. Strict Governance, Privacy & Security Checklist

| Governance & Security Requirement | Implementation Verification |
| :--- | :--- |
| **Read-Only Shadow Mode** | Hard-coded `ALLOW_STUDENT_OUTREACH = False`; prominent persistent banner across all views. |
| **Zero Messaging Endpoints** | API exposes no routes for email, WhatsApp, SMS, or intervention triggers (`/intervene` returns 404). |
| **Student Privacy Protection** | 100% keyed HMAC-SHA256 pseudonymization (`#8a3e7b16...`). Zero raw student IDs in UI or responses. |
| **CMA Telemetry Suppression** | CMAs across all 7 modules strictly suppressed from automated outreach. |
| **Module GGG Calibration** | Enforces $\tau = 0.10$ for Module GGG TMAs with decoupled formative quiz warnings. |
| **Demographic Fairness** | Zero sensitive demographic features accepted or displayed. |
| **Audit Log Compliance** | Thread-safe append-only `DecisionAuditRecord` logging with full feature vector hash. |
| **Advisory Disclaimers** | Non-punitive supportive guidance displayed across student triage drawers and reports. |

### Mandatory Operational Disclaimers
1. **Persistent Header Banner:**
   > `[SHADOW MODE ACTIVE: PASSIVE ACADEMIC ADVISORY VIEW ONLY. DIRECT STUDENT OUTREACH IS STRICTLY PROHIBITED.]`
2. **Student Detail Drawer Footer:**
   > *"Notice: Risk assessments represent statistical probabilities derived from historical LMS interaction and assignment submission timelines. Predictions must never be used for automated punitive interventions, prerequisite blocking, or grade adjustments."*
3. **Audit Ledger Footer:**
   > *"FERPA & GDPR Compliance Notice: Student identifiers are pseudonymized via keyed HMAC-SHA256. Raw identification is strictly restricted to authorized institutional key custodians."*

---

## 7. Implementation Risks & Dependency Analysis

| Risk Category | Identified Technical Challenge | Architectural Mitigation Strategy |
| :--- | :--- | :--- |
| **Data Volume & Memory** | Loading 170,874 snapshots into browser memory causes client crashes. | Server-side pagination, pre-aggregated endpoint summaries, and virtualized table windowing (`@tanstack/react-virtual`). |
| **Secret Key Exposure** | Leakage of `STUDENT_PSEUDONYMIZATION_SECRET` via frontend bundles. | The secret exists strictly server-side in the backend Python runtime. The frontend receives only pre-computed HMAC hex digests. |
| **Chart Re-rendering Jitter** | Complex SVG re-renders on every filter change cause UI stuttering. | Memoize chart data transforms via `useMemo()`, debounce filter inputs, and use lightweight SVG elements. |
| **Missing Node Dependencies** | Initial workspace lacks frontend scaffolding (`package.json`, React). | Formulate a standard Vite + React + TypeScript package manifest in Task 2. |
| **FastAPI Route Latency** | Computing real-time SHAP values on every request exceeds latency targets. | Pre-compute and cache global SHAP values from `reports/model_explainability_summary.json`. Local student explanations read from cached snapshot records. |

---

## 8. Next Steps (Task 2 Readiness)

Upon user review and formal approval of this architecture and accompanying data provenance audit:
1. **Scaffold Frontend Application:** Initialize the enterprise React application in `./app` with Vite, TypeScript, and Tailwind CSS.
2. **Implement Design System Tokens:** Configure `tailwind.config.js` with `#14243A` sidebar, `#F7F8FA` canvas, Inter typography, and risk status palettes.
3. **Extend FastAPI Analytics Routes:** Implement the read-only endpoints defined in Section 5 in `src/api.py`.
4. **Build Core Dashboard Shell:** Implement fixed sidebar, responsive header, filter bar, and 7 core page views.
5. **Execute Verification Tests:** Add automated frontend/API integration tests ensuring 100% compliance with shadow-mode and governance locks.

---
*Architecture specification updated and verified against ground-truth repository evidence. No frontend code executed.*
