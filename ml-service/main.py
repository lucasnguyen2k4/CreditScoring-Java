"""
Credit Scoring ML Service - FastAPI Application
Provides REST API endpoints for all ML operations:
  - Data upload & EDA
  - Data processing (encoding, outliers, balancing, pipeline)
  - Model training (7 algorithms + stacking)
  - Prediction & credit scoring
  - SHAP explainability
  - LLM-powered analysis (Gemini)ri
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings

from routers import data_processing, model_training, prediction, explainability, llm_analysis

app = FastAPI(
    title="Credit Scoring ML Service",
    description="Python ML microservice for credit scoring data processing, model training, and prediction.",
    version="2.0.0",
)

# CORS middleware — allow Spring Boot backend and React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(data_processing.router, prefix="/api/ml/data", tags=["Data Processing"])
app.include_router(model_training.router, prefix="/api/ml/model", tags=["Model Training"])
app.include_router(prediction.router, prefix="/api/ml/predict", tags=["Prediction"])
app.include_router(explainability.router, prefix="/api/ml/shap", tags=["Explainability"])
app.include_router(llm_analysis.router, prefix="/api/ml/llm", tags=["LLM Analysis"])


@app.get("/")
def root():
    return {
        "service": "Credit Scoring ML Service",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "llm_configured": settings.is_llm_configured()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
