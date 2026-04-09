"""
LLM Analysis Router
Endpoints for AI-powered EDA analysis and SHAP interpretation using Gemini.
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np

from session_store import store
from config import settings

router = APIRouter()


def get_session(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required")
    return store.get_or_create(session_id)


# ==================== EDA ANALYSIS ====================

@router.post("/analyze-eda")
def analyze_eda(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Run AI-powered EDA analysis using Gemini"""
    session = get_session(x_session_id)
    if session.raw_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        from services.eda_analyzer import LLMEDAAnalyzer

        analyzer = LLMEDAAnalyzer(
            api_key=settings.GOOGLE_API_KEY,
            model=settings.GOOGLE_MODEL,
            provider=settings.LLM_PROVIDER,
        )

        analysis = analyzer.analyze(session.raw_data)
        session.ai_analysis = analysis

        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eda-summary")
def get_eda_summary(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get EDA summary (cached or freshly generated)"""
    session = get_session(x_session_id)
    if session.raw_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        from services.eda_analyzer import EDADataCollector

        collector = EDADataCollector(session.raw_data)
        summary = collector.generate_full_summary()

        # Make it JSON safe
        import json

        def make_safe(obj):
            if isinstance(obj, dict):
                return {k: make_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_safe(i) for i in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return {"summary": make_safe(summary)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SHAP ANALYSIS ====================

@router.post("/analyze-shap-global")
def analyze_shap_global(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Run AI-powered global SHAP analysis"""
    session = get_session(x_session_id)
    if session.shap_values is None:
        raise HTTPException(status_code=400, detail="SHAP not initialized")

    try:
        from services.shap_analyzer import SHAPAnalyzer

        analyzer = SHAPAnalyzer()
        analysis = analyzer.analyze_global(
            model_name=session.model_type or "Unknown",
            feature_importance=session.shap_feature_importance,
            shap_values=session.shap_values,
            expected_value=float(session.shap_expected_value),
            features=session.shap_explainer.feature_names,
        )

        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ShapLocalAnalysisRequest(BaseModel):
    sample_idx: int = 0

@router.post("/analyze-shap-local")
def analyze_shap_local(req: ShapLocalAnalysisRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Run AI-powered local SHAP analysis for a sample"""
    session = get_session(x_session_id)
    if session.shap_values is None:
        raise HTTPException(status_code=400, detail="SHAP not initialized")

    try:
        from services.shap_analyzer import SHAPAnalyzer

        analyzer = SHAPAnalyzer()
        analysis = analyzer.analyze_local(
            model_name=session.model_type or "Unknown",
            feature_importance=session.shap_feature_importance,
            shap_values=session.shap_values,
            expected_value=float(session.shap_expected_value),
            features=session.shap_explainer.feature_names,
            sample_data=session.shap_X_explained,
            sample_idx=req.sample_idx,
        )

        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CHAT ====================

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = None

@router.post("/chat")
def chat_with_ai(req: ChatRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Chat with AI about the model and SHAP values"""
    session = get_session(x_session_id)
    if session.shap_values is None:
        raise HTTPException(status_code=400, detail="SHAP not initialized")

    try:
        from services.shap_analyzer import SHAPAnalyzer

        analyzer = SHAPAnalyzer()
        response = analyzer.chat(
            user_question=req.question,
            model_name=session.model_type or "Unknown",
            feature_importance=session.shap_feature_importance,
            shap_values=session.shap_values,
            expected_value=float(session.shap_expected_value),
            features=session.shap_explainer.feature_names,
            sample_data=session.shap_X_explained,
            conversation_history=req.conversation_history,
        )

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def llm_status():
    """Check LLM configuration status"""
    return {
        "configured": settings.is_llm_configured(),
        "provider": settings.LLM_PROVIDER,
        "model": settings.GOOGLE_MODEL,
    }
