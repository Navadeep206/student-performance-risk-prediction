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

## 3. Planned Technology Stack

- **Core Runtime & Language:** Python 3.9+
- **Data Engineering & Analysis:** Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-Learn, LightGBM, XGBoost
- **Model Explainability & Interpretability:** SHAP
- **Visualization & Profiling:** Matplotlib, Seaborn
- **Testing & Quality Assurance:** Pytest
- **Serving & Presentation:** FastAPI, Streamlit
- **Configuration & Environment:** python-dotenv

---

## 4. Twelve Development Phases

| Phase | Phase Name | Focus & Key Deliverables |
| :---: | :--- | :--- |
| **01** | **Project Initialization & Environment Setup** | Directory scaffolding, modular structure, virtual environment configuration, and version control readiness. |
| **02** | **Data Acquisition & Storage** | Dataset download automation, integrity hashing, raw data placement, and schema validation. |
| **03** | **Exploratory Data Analysis (EDA)** | Univariate, bivariate, and multivariate analysis of student demographics, assessment patterns, and VLE interaction distributions. |
| **04** | **Data Preprocessing & Integration** | Missing value imputation, outlier handling, categorical encoding, and relational table joining. |
| **05** | **Feature Engineering** | VLE clickstream temporal aggregations, engagement decay rates, submission timeliness metrics, and cumulative assessment performance features. |
| **06** | **Early-Warning Problem Formulation** | Temporal cut-off thresholding (e.g., day 30, day 60, day 90) to prevent data leakage, and rigorous target variable definition (`At-Risk` vs. `Not At-Risk`). |
| **07** | **Baseline Model & Validation Strategy** | Stratified train/test splitting, cross-validation framework, baseline heuristic / logistic regression benchmarks. |
| **08** | **Advanced Model Exploration & Tuning** | Tree-based gradient boosting models (Random Forest, LightGBM, XGBoost), hyperparameter tuning, and class imbalance mitigation. |
| **09** | **Model Evaluation & Explainability** | Discrimination metrics (ROC-AUC, PR-AUC, F1-Score, Recall@k), SHAP global/local feature importance, and fairness/bias checks. |
| **10** | **Model Serialization & Pipeline Packaging** | End-to-end Scikit-Learn / custom inference pipelines, artifact serialization (`joblib`), and input validation. |
| **11** | **Application & Dashboard Development** | RESTful prediction API (FastAPI) and an interactive educator dashboard (Streamlit) for real-time risk scoring and visual explanations. |
| **12** | **Deployment, Monitoring & Delivery** | Containerization (Docker), comprehensive documentation, operational runbooks, and final project sign-off. |

---

## 5. Repository Structure

```text
student-performance-risk-prediction/
├── .env.example              # Template for environment variables
├── .gitignore                 # Excludes environments, artifacts, and large data
├── README.md                  # Project overview and roadmap
├── requirements.txt           # Core project dependencies
├── data/
│   ├── raw/                   # Immutable raw OULAD tables (.csv)
│   ├── processed/             # Cleaned, joined, and feature-engineered datasets
│   └── external/              # External reference or supplementary data
├── notebooks/                 # Exploratory, prototyping, and analysis notebooks
├── src/                       # Modular production source code
├── models/                    # Serialized model artifacts and metrics
├── app/                       # Serving layers (FastAPI / Streamlit)
├── tests/                     # Unit and integration test suite
└── reports/
    └── figures/               # Generated figures and analytical charts
```

---

## 6. Getting Started

### 1. Activate the Virtual Environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install Starter Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
```

