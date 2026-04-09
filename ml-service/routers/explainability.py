"""
Explainability Router
Endpoints for SHAP initialization, global/local explanations.
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import json

from session_store import store
from services.shap_explainer import SHAPExplainer, initialize_shap_explainer

router = APIRouter()


def get_session(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required")
    return store.get_or_create(session_id)


# ==================== INIT SHAP ====================

@router.post("/init")
def init_shap(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Initialize SHAP explainer and compute SHAP values"""
    session = get_session(x_session_id)
    if session.model is None:
        raise HTTPException(status_code=400, detail="No model trained yet")
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="No training data available")

    try:
        explainer, shap_values, X_explained = initialize_shap_explainer(
            session.model, session.X_train, session.model_type
        )

        session.shap_explainer = explainer
        session.shap_values = shap_values
        session.shap_X_explained = X_explained
        session.shap_feature_importance = explainer.get_feature_importance()
        session.shap_expected_value = explainer.expected_value

        importance = session.shap_feature_importance.to_dict(orient="records")

        return {
            "message": "SHAP initialized successfully",
            "n_samples_explained": len(X_explained),
            "expected_value": float(session.shap_expected_value),
            "feature_importance": importance,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== GLOBAL EXPLANATION ====================

@router.get("/global")
def get_global_explanation(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get global SHAP feature importance"""
    session = get_session(x_session_id)
    if session.shap_values is None:
        raise HTTPException(status_code=400, detail="SHAP not initialized. Call /init first.")

    importance = session.shap_feature_importance.to_dict(orient="records")

    # Summary data for plots
    features = session.shap_explainer.feature_names
    mean_abs = np.abs(session.shap_values).mean(axis=0).tolist()

    return {
        "feature_importance": importance,
        "features": features,
        "mean_abs_shap": mean_abs,
        "expected_value": float(session.shap_expected_value),
        "n_samples": len(session.shap_values),
    }


# ==================== LOCAL EXPLANATION ====================

class LocalRequest(BaseModel):
    sample_idx: int = 0

@router.post("/local")
def get_local_explanation(req: LocalRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get local SHAP explanation for a specific sample"""
    session = get_session(x_session_id)
    if session.shap_values is None:
        raise HTTPException(status_code=400, detail="SHAP not initialized")

    if req.sample_idx >= len(session.shap_values):
        raise HTTPException(status_code=400, detail=f"Sample index {req.sample_idx} out of range (max: {len(session.shap_values) - 1})")

    try:
        explanation = session.shap_explainer.get_local_explanation(
            req.sample_idx, session.shap_X_explained
        )

        # Convert to JSON-safe
        contributions = explanation["contributions"].to_dict(orient="records")
        for c in contributions:
            for k, v in c.items():
                if isinstance(v, (np.integer, np.floating)):
                    c[k] = float(v)

        return {
            "sample_idx": explanation["sample_idx"],
            "base_value": float(explanation["base_value"]),
            "prediction": float(explanation["prediction"]),
            "shap_values": [float(v) for v in explanation["shap_values"]],
            "feature_values": [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in explanation["feature_values"]],
            "contributions": contributions,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
