# Student Performance Risk Prediction

An end-to-end, production-grade machine learning system designed to identify students at risk of academic underperformance, course failure, or early withdrawal within Virtual Learning Environments (VLEs), enabling timely institutional interventions.

---

## 1. Project Objective

The primary objective of this project is to develop an early-warning predictive system that leverages student demographic data, assessment results, and fine-grained Virtual Learning Environment (VLE) interaction clickstream data to:
- Predict student at-risk status early within an academic presentation cycle.
- Uncover key behavioral and academic indicators associated with student disengagement.
- Deliver interpretable risk scores and actionable insights to academic advisors and educators.
- Provide a robust, modular, and maintainable pipeline ready for real-world deployment.

---

## 2. Dataset: Open University Learning Analytics Dataset (OULAD)

This project utilizes the **Open University Learning Analytics Dataset (OULAD)**, one of the most comprehensive public learning analytics datasets. It encompasses data across 7 courses (modules), 22 course presentations, and 32,593 unique students.

The relational dataset consists of seven core tables:
1. `courses.csv`: Module specifications and presentation duration.
2. `assessments.csv`: Assessment structure, weights, and deadlines.
3. `vle.csv`: Virtual Learning Environment resource metadata and resource types.
4. `studentInfo.csv`: Student demographics, study details, and final course outcomes.
5. `studentRegistration.csv`: Registration and course un-registration dates.
6. `studentAssessment.csv`: Student-level assessment submission dates and scores.
7. `studentVle.csv`: High-volume temporal clickstream interactions with course VLE materials.

---

## 3. Technology Stack

- **Backend & Inference API:** Python 3.12, FastAPI 0.163, Pydantic v2, Uvicorn
- **Machine Learning & Modeling:** Scikit-Learn 1.9 (HistGradientBoostingClassifier primary, RandomForestClassifier challenger)
- **Model Explainability:** TreeSHAP (SHAP 0.54) with background reference masking
- **Data Engineering & Temporal Featurization:** Pandas, NumPy, SciPy
- **Frontend Academic Dashboard:** React 19, TypeScript 5, Vite 6, Tailwind CSS v3.4, Lucide React
- **Testing & Quality Assurance:** Pytest 9.2 (115 automated regression tests), AnyIO, Starlette TestClient
- **Privacy & Security:** Keyed HMAC-SHA256 student ID pseudonymization, strict demographic feature exclusion
- **Operational Mode:** Mandatory Read-Only Shadow Mode (`ALLOW_STUDENT_OUTREACH = False`)

---

## 4. Completed Development Phases & Architecture

| Phase | Milestone Name | Key Deliverables & Verified Governance Controls |
| :---: | :--- | :--- |
| **01–03** | **Data Ingestion, Quality Audit & EDA** | Relational schema integrity, referential foreign key validation, 100% preservation of zero-activity learners. |
| **04** | **Temporal Feature Engineering** | Dynamic cutoff computation (7-day advance warning, 14-day grading delay), zero target leakage, VLE clickstream temporal aggregations. |
| **05** | **Baseline Models & Grouped CV** | Student-grouped 5-fold cross-validation, baseline heuristics, regularized logistic regression, random forest benchmarks. |
| **06** | **Nested CV Optimization & Policy Audit** | Deep-tree HistGradientBoosting tuning, strict inner-fold threshold optimization, counterfactual evaluation design. |
| **07** | **TreeSHAP Explainability** | Global TreeSHAP feature attributions (`recent_prior_score`: 0.7223, `target_assessment_type`: 0.6836), local directional attributions. |
| **08** | **Fairness, Robustness & Demographic Exclusion** | Strict exclusion of protected attributes (`gender`, `imd_band`, `disability`, `age_band`, `region`, `highest_education`), Wilson score intervals. |
| **09** | **Temporal Drift, Cold-Start & GGG Remediation** | Population Stability Index (PSI) drift tracking, cold-start early-warning protocol, Module GGG matched study, calibrated threshold ($\tau=0.10$). |
| **10** | **Pipeline Serialization & Production Engine** | Thread-safe `ModelArtifactManager`, SHA-256 artifact verification, dual-threshold governance router, append-only decision audit logging. |
| **11** | **Enterprise Dashboard & RESTful Analytics** | 7-page classic enterprise academic analytics portal (React 19 / Vite) served directly by FastAPI; 115/115 passing tests. |

---

## 5. Repository Structure

```text
student-performance-risk-prediction/
├── .env.example              # Template environment configuration (includes HMAC secret template)
├── .gitignore                 # GitHub-ready ignore patterns (excludes raw datasets, node_modules, temp files)
├── README.md                  # Comprehensive project documentation
├── requirements.txt           # Production Python dependencies
├── data/
│   ├── raw/                   # Raw OULAD tables (excluded from Git; preserved locally)
│   ├── processed/             # Cleaned, feature-engineered datasets (excluded from Git; preserved locally)
│   └── external/              # External supplementary data (.gitkeep)
├── models/
│   ├── optimized_hist_gradient_boosting.joblib          # Primary research candidate binary (1.15 MB)
│   ├── optimized_hist_gradient_boosting_metadata.json   # Primary model metadata, schema, SHA-256 hash
│   ├── challenger_random_forest.joblib                  # Challenger model binary (9.8 MB)
│   └── challenger_random_forest_metadata.json           # Challenger model metadata and provenance
├── frontend/                  # Classic Enterprise Academic Analytics Portal (React 19 + TypeScript)
│   ├── src/                   # React components, 7 analytical pages, Tailwind tokens, API client
│   ├── package.json           # Frontend dependencies (React, Vite, Tailwind, Lucide)
│   ├── vite.config.ts         # Vite build configuration with API proxy
│   └── dist/                  # Compiled production static bundle (served by FastAPI)
├── notebooks/                 # Exploratory data analysis notebooks (Phases 1–3)
├── src/                       # Modular production source code (29 modules)
│   ├── api.py                 # FastAPI application and inference engine
│   ├── analytics_routes.py    # Read-only RESTful analytics routes for dashboard
│   ├── governance_router.py   # Decision-threshold router and safety policy enforcement
│   ├── shadow_mode.py         # Read-only shadow mode manager (ALLOW_STUDENT_OUTREACH = False)
│   ├── decision_audit.py      # Append-only DecisionAuditLogger and HMAC pseudonymization
│   └── ...                    # Preprocessing, featurization, and audit engines
└── reports/                   # 62 certified Markdown research reports and machine-readable JSON summaries
```

---

## 6. Quickstart & Local Execution

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd "Student At Risk Prediction"

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\Activate.ps1  # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Secrets

Copy the example configuration file and set a secure 32-character HMAC secret key:
```bash
cp .env.example .env
# Ensure STUDENT_PSEUDONYMIZATION_SECRET is set to at least 32 characters in .env
```

### 3. Build Frontend Assets (Optional if using pre-compiled dist)

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Launch the FastAPI Application & Dashboard

```bash
# Start backend server with uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation (Swagger):** `http://localhost:8000/docs`
- **Health Check Endpoint:** `http://localhost:8000/health`
- **Academic Enterprise Dashboard:** `http://localhost:8000/` or `http://localhost:8000/dashboard`

---

## 7. Operational Governance Notice

> [!CAUTION]
> **READ-ONLY SHADOW MODE ENFORCED — PRODUCTION DEPLOYMENT WITHHELD**
> 
> This application is currently authorized **strictly for offline academic research and passive shadow-mode advisory evaluation**.
> - Automated student messaging, emails, and direct interventions are architecturally **prohibited** (`ALLOW_STUDENT_OUTREACH = False`).
> - Computer-Marked Assessments (CMAs) are universally suppressed from outreach (`RESEARCH_TELEMETRY_ONLY`).
> - Student identifiers are transformed using keyed HMAC-SHA256 digests; raw student IDs are never exposed in UI or API responses.
> - Protected demographic attributes are strictly excluded from predictive features.


