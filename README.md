# 🏦 Credit Scoring System V2

[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://openjdk.org/projects/jdk/21/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.5.0-6DB33F.svg)](https://spring.io/projects/spring-boot)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.x-47A248.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-grade, full-stack Credit Scoring platform** built on a microservice architecture.
The system delivers an end-to-end ML workflow — from data upload and exploratory analysis, through
feature engineering, model training, SHAP explainability, to real-time credit prediction — all
secured by a Role-Based Access Control (RBAC) layer.

> 📌 **Senior Thesis Project** — Hanoi University of Science and Technology (HUST)
> Building a production-ready Credit Scoring system: from raw data to explainable AI-powered decisions.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [System Requirements](#-system-requirements)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [User Guide](#-user-guide)
- [Role-Based Access Control](#-role-based-access-control)
- [Project Structure](#-project-structure)
- [API Overview](#-api-overview)
- [LLM Configuration (Optional)](#-llm-configuration-optional)
- [Sample Data Format](#-sample-data-format)
- [Notable Technical Contributions](#-notable-technical-contributions)
- [Troubleshooting](#-troubleshooting)
- [Author](#-author)

---

## ✨ Key Features

### 🔐 1. Authentication & Role-Based Access Control
- JWT-based login/register via Spring Boot backend.
- Four roles: **Admin**, **Data Scientist**, **Model Approver**, **Analyst**.
- Each role has restricted access to specific pages and ML operations.
- Persistent user management stored in MongoDB.

### 📤 2. Data Upload & Exploratory Data Analysis (EDA)
- Upload CSV datasets containing customer financial attributes.
- Descriptive statistics: mean, median, standard deviation, quartiles.
- Interactive charts: Histogram, Boxplot, Violin Plot.
- Correlation Heatmap for feature relationships.
- Missing value and outlier detection.
- **🤖 AI-powered Analysis**: Automatic EDA insights via Google Gemini.

### ⚙️ 3. Feature Engineering
| Function | Supported Methods |
|---|---|
| **Missing Values** | Mean, Median, Mode, Constant, Forward/Backward Fill |
| **Outlier Handling** | IQR Method, Z-Score, Winsorization |
| **Categorical Encoding** | One-Hot, Label, Target, Ordinal, Frequency Encoding |
| **Scaling** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler |
| **Binning** | Equal-width, Quantile-based, Custom bins |
| **Data Balancing** | SMOTE, ADASYN, Random Under/Over Sampling, SMOTE-ENN, SMOTE-Tomek |
| **Data Splitting** | Stratified Train (70%) / Validation (15%) / Test (15%) |

> ⚠️ **No Data Leakage Guaranteed**: All preprocessing transformers are fitted on the training set only, then applied to validation and test sets.

### 🤖 4. Model Training (7 Algorithms + Stacking)

**Supported Algorithms:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- **🔥 Stacking Ensemble** (configurable base models + meta-model)

**Advanced Training Features:**
- ✅ Hyperparameter Tuning: Grid Search, Random Search, Optuna (Bayesian Optimization)
- ✅ K-Fold Cross-Validation
- ✅ Early Stopping for boosting models (using Validation set — not Test set)
- ✅ Train / Validation / Test metric comparison for overfitting detection
- ✅ OOF (Out-of-Fold) Tuning for Stacking to prevent data leakage

### ✅ 5. Model Approval Workflow
- Data Scientists train models; Model Approvers review and approve/reject.
- Approval decisions: **Approved**, **Conditional**, **Rejected**.
- Approval history with timestamps and reviewer notes.
- Only the approved model is used for downstream predictions.

### 📊 6. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
- **Visualizations**: Confusion Matrix, ROC Curve.
- **Model Comparison**: Side-by-side table of all trained models.
- **Training History**: Full log of every training run in the session.

### 🔍 7. SHAP Explainability
- **Global Explanation**: Overall feature importance across the dataset.
- **Local Explanation**: Per-sample explanation for any specific prediction.
- **Visualizations**: Summary Plot, Beeswarm Plot, Waterfall Plot, Force Plot.
- **🤖 AI Interpretation**: Gemini translates SHAP values into plain-language explanations.
- **💬 Q&A Chat**: Ask the AI questions about the model's behavior.

### 🎯 8. Credit Prediction & Advisory
- Input customer data via form or CSV upload.
- **Credit Score (300–850)**: Calculated using the Basel II/III log-odds scaling formula.
- **5-Tier Risk Classification**: Very Low → Low → Medium → High → Very High.
- **Approval Decision**: Approved / Conditional / Rejected.
- **🤖 AI Recommendations**: Gemini suggests actions to improve credit score.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Browser (React + Vite)                     │
│                        http://localhost:5173                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / REST (JWT Bearer Token)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Spring Boot Backend (Java 21, Maven)               │
│                        http://localhost:8080                    │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  AuthController  │  │  UserController  │  │MlProxyControl │  │
│  │  /api/auth/**    │  │  /api/users/**   │  │/api/ml/**     │  │
│  └─────────────────┘  └──────────────────┘  └───────┬───────┘  │
│                                                      │           │
│  ┌─────────────────────────────────────────────┐    │           │
│  │ JWT Filter → Security Config → Role Checks  │    │           │
│  └─────────────────────────────────────────────┘    │           │
│                                               WebClient proxy   │
└────────────────────────────────────────────────┬────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FastAPI ML Service (Python 3.11)                 │
│                        http://localhost:8000                    │
│                                                                 │
│  /api/ml/data     → Data Processing (upload, EDA, encoding)     │
│  /api/ml/model    → Model Training, Tuning, Approval            │
│  /api/ml/predict  → Prediction + Credit Score                   │
│  /api/ml/shap     → SHAP Explainability                         │
│  /api/ml/llm      → LLM Analysis (Gemini AI)                    │
│                                                                 │
│  Session Store: In-memory per-session state (DataFrames, Models)│
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────┐
│  MongoDB (Port 27017)│
│  Database: creditscoring │
│  Collection: users   │
└─────────────────────┘
```

---

## 🛠 Tech Stack

### Backend (Spring Boot)
| Technology | Version | Purpose |
|---|---|---|
| **Java** | 21 | Core language |
| **Spring Boot** | 3.5.0 | Web framework |
| **Spring Security** | 6.x | Authentication & authorization |
| **Spring Data MongoDB** | 4.x | Database ORM |
| **Spring WebFlux** | 6.x | Non-blocking ML service proxy (WebClient) |
| **JJWT** | 0.12.5 | JWT token generation & validation |
| **Lombok** | Latest | Boilerplate reduction |
| **MongoDB** | 7.x | User data & configuration storage |

### ML Service (FastAPI + Python)
| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.11 | Core language |
| **FastAPI** | 0.115.0 | REST API framework |
| **Uvicorn** | 0.30.1 | ASGI server |
| **Pandas** | 2.1.4 | Data manipulation |
| **NumPy** | 1.26.2 | Numerical computing |
| **Scikit-learn** | 1.5.2 | ML algorithms & preprocessing |
| **XGBoost** | 2.0.3 | Extreme Gradient Boosting |
| **LightGBM** | 4.1.0 | Light Gradient Boosting Machine |
| **CatBoost** | 1.2.2 | Categorical Boosting |
| **Imbalanced-learn** | 0.12.4 | SMOTE, ADASYN, and resampling |
| **Optuna** | 3.5.0 | Bayesian hyperparameter optimization |
| **SHAP** | 0.44.0 | Shapley Additive Explanations |
| **Plotly** | 5.18.0 | Interactive chart generation |
| **Google Generative AI** | 0.7.2 | Gemini LLM integration |

### Frontend (React)
| Technology | Version | Purpose |
|---|---|---|
| **React** | 19.2.4 | UI framework |
| **Vite** | 8.0.1 | Build tool & dev server |
| **React Router DOM** | 7.x | Client-side routing |
| **Recharts** | 3.x | Chart components |
| **Axios** | 1.x | HTTP client |
| **Lucide React** | 1.x | Icon library |

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 | Windows 10/11 |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 5 GB free | 10 GB+ free |
| **Java** | 21 | 21 (LTS) |
| **Python** | 3.10 | 3.11 |
| **Node.js** | 18 | 20+ |
| **MongoDB** | 6.x | 7.x |

---

## 🚀 Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone <repository-url>
cd CreditScoring-V2
```

### Step 2 — Start MongoDB

Make sure MongoDB is running locally on the default port:

```bash
# Windows — if installed as a service:
net start MongoDB

# Or start manually:
mongod --dbpath "C:\data\db"
```

The Spring Boot backend will automatically create the `creditscoring` database and the `users` collection on first startup.

### Step 3 — Set Up the ML Service (Python)

It is strongly recommended to use a Conda environment:

```bash
# Create a new environment with Python 3.11
conda create -n credit-scoring python=3.11

# Activate the environment
conda activate credit-scoring

# Install dependencies
cd ml-service
pip install -r requirements.txt
```

**Configure environment variables (optional — for AI features):**

```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your Google Gemini API key:
# GOOGLE_API_KEY=your_google_api_key_here
# GOOGLE_MODEL=gemini-2.5-flash
```

### Step 4 — Set Up the Backend (Spring Boot)

No additional dependencies needed beyond Java 21. Maven Wrapper (`mvnw.cmd`) is included.

**Configure JWT secret (optional — default value works for development):**

Edit `backend/src/main/resources/application.yml` if needed:

```yaml
jwt:
  secret: YourSuperSecretKeyHere
  expiration-ms: 86400000   # 24 hours

ml-service:
  url: http://localhost:8000

spring:
  data:
    mongodb:
      uri: mongodb://localhost:27017/creditscoring
```

### Step 5 — Set Up the Frontend (Node.js)

```bash
cd frontend
npm install
```

---

## ▶️ Running the Application

### Option A — One-Click Start (Windows PowerShell)

```powershell
# From the project root, run:
powershell -ExecutionPolicy Bypass -File start-all.ps1
```

This opens three separate terminal windows:
1. **Backend** → Spring Boot on port 8080
2. **ML Service** → FastAPI on port 8000
3. **Frontend** → Vite dev server on port 5173

### Option B — Manual Start (3 Terminals)

**Terminal 1 — Backend:**
```bash
cd backend
./mvnw.cmd spring-boot:run
```

**Terminal 2 — ML Service:**
```bash
cd ml-service
conda activate credit-scoring
python main.py
```

**Terminal 3 — Frontend:**
```bash
cd frontend
npm run dev
```

### Access the Application

| Service | URL | Notes |
|---|---|---|
| **Frontend (Web App)** | http://localhost:5173 | Main user interface |
| **Backend API** | http://localhost:8080 | Spring Boot REST API |
| **ML Service API Docs** | http://localhost:8000/docs | FastAPI Swagger UI |
| **ML Service Health** | http://localhost:8000/health | Health check endpoint |

### Default Login

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | ADMIN |

> 💡 **Note**: The admin account is created automatically on the first startup if it does not exist in the database.

---

## 📖 User Guide

### Standard ML Workflow

```
1. Login              → Authenticate with your credentials
       ↓
2. Data Upload & EDA  → Upload CSV, explore data distributions
       ↓
3. Feature Engineering → Handle missing values, encode, scale, split
       ↓
4. Model Training     → Train and compare ML models
       ↓
5. Model Approval     → Approver reviews and approves a model
       ↓
6. SHAP Explanation   → Understand global and local model behavior
       ↓
7. Prediction         → Score new customers, get credit decisions
```

### Step-by-Step Details

#### Step 1: Login
1. Open http://localhost:5173 in your browser.
2. Enter your credentials (default: `admin` / `admin123`).
3. The sidebar will show pages based on your assigned role.

#### Step 2: Upload Data & EDA
1. Navigate to **Data Upload & Analysis**.
2. Upload a CSV file containing customer features and a target column (0/1).
3. Review auto-generated statistics and charts.
4. Optionally click **"Analyze with AI"** to receive Gemini-powered insights.

#### Step 3: Feature Engineering
1. Navigate to **Feature Engineering**.
2. Handle missing values (choose strategy per column).
3. Handle outliers if needed.
4. Encode categorical columns.
5. Scale numerical features.
6. Balance the dataset if classes are imbalanced (e.g., use SMOTE).
7. Split into Train / Validation / Test sets.
8. Select features to use for training.

#### Step 4: Model Training
1. Navigate to **Model Training**.
2. Choose a model type (e.g., XGBoost, Stacking Ensemble).
3. Configure hyperparameters or use the tuning tools (Grid Search / Random Search / Optuna).
4. Enable **Early Stopping** for boosting models (uses the validation set).
5. Click **"Train Model"**.
6. Review Accuracy, AUC-ROC, F1-Score, and the Confusion Matrix.
7. Compare all trained models in the model history table.

#### Step 5: Model Approval
1. Navigate to **Model Approval** (requires Model Approver or Admin role).
2. Review all trained models and their metrics.
3. Select a model and click **"Approve"**, **"Conditional"**, or **"Reject"**.
4. Add review notes if needed.

#### Step 6: SHAP Explainability
1. Navigate to **SHAP Explanation**.
2. Click **"Initialize SHAP"** to compute Shapley values.
3. View **Global Feature Importance** (Summary Plot, Beeswarm Plot).
4. Select a sample record to view **Local Explanation** (Waterfall Plot, Force Plot).
5. Optionally chat with the **AI assistant** for plain-language explanations.

#### Step 7: Prediction
1. Navigate to **Prediction & Advisory**.
2. Fill in the customer information form, or upload a CSV for batch prediction.
3. View:
   - **Credit Score** (300–850 scale)
   - **Risk Level** (Very Low → Very High)
   - **Approval Decision** (Approved / Conditional / Rejected)
   - **AI Recommendations** for improving the score

---

## 🔑 Role-Based Access Control

| Role | Dashboard | Upload & EDA | Feature Eng. | Training | Approval | SHAP | Prediction | Admin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADMIN** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **DATA_SCIENTIST** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **MODEL_APPROVER** | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **ANALYST** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

**Creating users (Admin only):**

Use the **Admin Settings** page to create new users and assign roles.
Or call the API directly:

```bash
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-jwt-token>" \
  -d '{
    "username": "analyst1",
    "password": "password123",
    "displayName": "Alice Analyst",
    "role": "ANALYST"
  }'
```

---

## 📁 Project Structure

```
CreditScoring-V2/
│
├── 📄 start-all.ps1                    # Windows one-click startup script
├── 📄 frontend-upload-test.csv         # Sample CSV for testing upload
│
├── 📂 backend/                         # Spring Boot Service (Port 8080)
│   ├── pom.xml                         # Maven dependencies
│   └── src/main/java/com/creditscoring/
│       ├── CreditScoringBackendApplication.java
│       ├── 📂 controller/
│       │   ├── AuthController.java     # /api/auth/** — login, register, me
│       │   ├── UserController.java     # /api/users/** — user management (Admin)
│       │   └── MlProxyController.java  # /api/ml/** — transparent proxy to FastAPI
│       ├── 📂 model/
│       │   ├── User.java               # MongoDB user document
│       │   └── Role.java               # ADMIN, DATA_SCIENTIST, MODEL_APPROVER, ANALYST
│       ├── 📂 security/
│       │   ├── JwtTokenProvider.java   # JWT generation & validation
│       │   └── JwtAuthenticationFilter.java
│       ├── 📂 repository/
│       │   └── UserRepository.java     # MongoDB CRUD interface
│       ├── 📂 dto/                     # Request/Response DTOs
│       └── 📂 config/                  # Security config, CORS, WebClient bean
│
├── 📂 ml-service/                      # FastAPI ML Service (Port 8000)
│   ├── main.py                         # Application entry point
│   ├── config.py                       # Settings (env vars, CORS)
│   ├── session_store.py                # In-memory session management
│   ├── requirements.txt                # Python dependencies
│   ├── .env.example                    # Environment variable template
│   ├── 📂 routers/
│   │   ├── data_processing.py          # /api/ml/data — upload, EDA, encoding, split
│   │   ├── model_training.py           # /api/ml/model — train, tune, approve
│   │   ├── prediction.py               # /api/ml/predict — credit scoring
│   │   ├── explainability.py           # /api/ml/shap — SHAP values & plots
│   │   └── llm_analysis.py             # /api/ml/llm — Gemini AI integration
│   └── 📂 services/
│       ├── trainer.py                  # 7 ML algorithms + Stacking + OOF Tuning
│       ├── predictor.py                # Prediction + Basel II/III credit score
│       ├── preprocessing_pipeline.py  # No-leakage preprocessing pipeline
│       ├── encoder.py                  # 5 categorical encoding methods
│       ├── balancer.py                 # SMOTE, ADASYN, resampling methods
│       ├── outlier_handler.py          # IQR, Z-Score, Winsorization
│       ├── eda_analyzer.py             # EDA statistics & chart generation
│       ├── shap_explainer.py           # SHAP integration (Tree/Linear/Kernel)
│       ├── shap_analyzer.py            # AI-powered SHAP interpretation
│       ├── feature_importance.py       # Standalone feature importance
│       ├── data_generator.py           # Synthetic data generation
│       └── llm_config.py              # LLM provider configuration
│
├── 📂 frontend/                        # React + Vite App (Port 5173)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                     # Router & route protection
│       ├── main.jsx                    # App entry point
│       ├── index.css                   # Global styles & design system
│       ├── 📂 pages/
│       │   ├── LoginPage.jsx
│       │   ├── DashboardPage.jsx
│       │   ├── DataUploadPage.jsx       # ~58KB — full EDA interface
│       │   ├── FeatureEngineeringPage.jsx # ~105KB — comprehensive FE UI
│       │   ├── ModelTrainingPage.jsx
│       │   ├── ModelApprovalPage.jsx
│       │   ├── ShapExplanationPage.jsx
│       │   ├── PredictionPage.jsx
│       │   └── AdminSettingsPage.jsx
│       ├── 📂 components/
│       │   ├── Sidebar.jsx
│       │   ├── ProtectedRoute.jsx
│       │   └── charts/                 # Reusable chart components
│       ├── 📂 context/
│       │   └── AuthContext.jsx         # JWT storage & auth state
│       ├── 📂 api/                     # Axios API client modules
│       └── 📂 utils/                   # Utility functions
│
└── 📂 _reference/
    └── CreditScoring-EndToEnd/         # V1 reference project (Streamlit, single-user)
```

---

## 🌐 API Overview

### Backend (Spring Boot — Port 8080)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/login` | None | Login and receive JWT token |
| `POST` | `/api/auth/register` | JWT (Admin) | Create a new user account |
| `GET` | `/api/auth/me` | JWT | Get current user info |
| `GET` | `/api/users` | JWT (Admin) | List all users |
| `PUT` | `/api/users/{id}` | JWT (Admin) | Update user |
| `DELETE` | `/api/users/{id}` | JWT (Admin) | Delete user |
| `*` | `/api/ml/**` | JWT | Proxy to FastAPI ML service |

### ML Service (FastAPI — Port 8000)

Full interactive documentation available at: **http://localhost:8000/docs**

| Prefix | Key Endpoints |
|---|---|
| `/api/ml/data` | `POST /upload`, `GET /eda`, `POST /outlier`, `POST /encode`, `POST /split` |
| `/api/ml/model` | `POST /train`, `POST /train-stacking`, `POST /tune`, `POST /tune-stacking`, `POST /approve` |
| `/api/ml/predict` | `POST /` (single), `POST /batch` (CSV), `GET /credit-score` |
| `/api/ml/shap` | `POST /initialize`, `GET /global`, `POST /local`, `GET /plots` |
| `/api/ml/llm` | `POST /eda-analysis`, `POST /shap-analysis`, `POST /chat` |

> **Session Management**: All ML endpoints require an `X-Session-ID` header (automatically injected by the Spring Boot proxy based on the authenticated user's session).

---

## 🔑 LLM Configuration (Optional)

AI features (EDA analysis, SHAP interpretation, Q&A chat) require a Google Gemini API key.

### Getting an API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account.
3. Create a free API key.
4. Add it to the `.env` file in `ml-service/`:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
LLM_MAX_TOKENS=8000
LLM_TEMPERATURE=0.7
```

> 💡 **Note**: The application works fully without an API key — AI features are simply disabled.

---

## 📊 Sample Data Format

The uploaded CSV file must contain:
- **Feature columns**: Numerical and/or categorical customer attributes.
- **Target column**: A binary column (0 = no default, 1 = default). The column name can be configured in the Feature Engineering step.

### Example CSV Format

```csv
customer_id,age,income,employment_years,loan_amount,credit_history_length,num_credit_cards,debt_ratio,loan_purpose,default
1001,35,50000,5,10000,8,2,0.25,personal,0
1002,42,75000,12,15000,15,3,0.18,mortgage,0
1003,28,30000,2,5000,3,1,0.45,personal,1
1004,55,120000,25,30000,20,4,0.12,business,0
```

### Data Requirements

| Aspect | Requirement |
|---|---|
| **Format** | CSV (UTF-8 encoded) |
| **Minimum rows** | 100+ (500+ recommended for reliable training) |
| **Target column** | Binary (0 or 1) |
| **Missing values** | Allowed — handled in Feature Engineering step |
| **Maximum file size** | 100 MB (configurable in `application.yml`) |

---

## 🏆 Notable Technical Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Microservice Architecture** | Separated concerns: Java for auth/gateway, Python for ML |
| 2 | **JWT + RBAC System** | Four-role access control with MongoDB user persistence |
| 3 | **ML Proxy Pattern** | Spring Boot transparently proxies ML calls — frontend uses a single API host |
| 4 | **No-Leakage Pipeline** | `PreprocessingPipeline` fits on train set, transforms all sets |
| 5 | **OOF Tuning for Stacking** | Hyperparameter tuning via Out-of-Fold predictions — prevents meta-model overfitting |
| 6 | **Early Stopping on Validation** | Boosting models stop early using the validation set, never the test set |
| 7 | **Basel II/III Credit Score** | Log-odds scaling formula maps default probability to 300–850 score range |
| 8 | **5-Tier Risk Classification** | Industry-standard risk tiering: Very Low / Low / Medium / High / Very High |
| 9 | **Multi-model SHAP** | TreeExplainer, LinearExplainer, and KernelExplainer for all model types |
| 10 | **Formal Approval Workflow** | Separated training (DS role) from deployment authorization (Approver role) |
| 11 | **LLM-powered Analysis** | Gemini AI interprets EDA, SHAP values, and answers user Q&A |

---

## 🔧 Troubleshooting

### Common Issues

**1. `ModuleNotFoundError` when starting the ML service**
```bash
conda activate credit-scoring
pip install -r requirements.txt
```

**2. MongoDB connection refused**
```bash
# Check if MongoDB is running
net start MongoDB
# Or start manually
mongod --dbpath "C:\data\db"
```

**3. Backend fails to start — JWT secret too short**
> Edit `backend/src/main/resources/application.yml` and ensure the JWT secret is at least 256 bits (32+ characters).

**4. CORS error in browser**
> Ensure `cors.allowed-origins` in `application.yml` includes `http://localhost:5173`.

**5. LightGBM installation error on Windows**
```bash
pip install lightgbm --prefer-binary
```

**6. CatBoost version conflict**
```bash
pip install catboost==1.2.2
```

**7. SHAP runs slowly**
- Reduce the number of background samples (default: 500).
- For tree-based models, `TreeExplainer` is automatically used (fastest).

**8. Frontend shows blank page after login**
> Clear browser localStorage and reload: `localStorage.clear()` in browser console.

---

## 🗒 Version History

| Version | Date | Notes |
|---|---|---|
| **2.0.0** | 2026-05 | Microservice rewrite: Spring Boot + FastAPI + React, RBAC, Model Approval |
| **1.0.0** | 2026-01 | Original monolithic Streamlit app (single-user, no auth) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**[Your Name]**
- 🎓 Hanoi University of Science and Technology (HUST)
- 📧 Email: your.email@hust.edu.vn
- 🐙 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- [Spring Boot](https://spring.io/projects/spring-boot) — Production-grade Java framework
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [SHAP](https://github.com/shap/shap) — Shapley Additive Explanations
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient Boosting framework
- [Google Gemini](https://ai.google.dev/) — AI analysis integration
- [Scikit-learn](https://scikit-learn.org/) — ML algorithm library

---

<p align="center">
  Made with ❤️ for Credit Scoring Research
</p>
