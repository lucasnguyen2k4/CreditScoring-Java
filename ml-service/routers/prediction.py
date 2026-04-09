"""
Prediction Router
Endpoints for single and batch prediction, credit scoring, recommendations.
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from session_store import store
from services.predictor import predict_single, predict_batch, get_feature_contributions, generate_recommendations

router = APIRouter()


def get_session(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required")
    return store.get_or_create(session_id)


def ensure_approved_model_for_prediction(session):
    """Enforce that prediction uses an approved model."""
    if session.approved_model_index is None:
        raise HTTPException(
            status_code=400,
            detail="No approved model yet. Please approve a model in Model Approval before prediction."
        )

    idx = session.approved_model_index
    if idx < 0 or idx >= len(session.trained_models):
        raise HTTPException(status_code=400, detail="Approved model index is invalid. Re-approve a valid model.")
    if idx >= len(session.trained_model_objects) or session.trained_model_objects[idx] is None:
        raise HTTPException(status_code=400, detail="Approved model object is unavailable in memory. Retrain and approve again.")

    session.model = session.trained_model_objects[idx]
    approved = session.trained_models[idx]
    session.model_type = approved.get("model_type")
    session.model_metrics = approved.get("metrics", {})


# ==================== SINGLE PREDICTION ====================

class PredictRequest(BaseModel):
    input_data: Dict[str, Any]

@router.post("/single")
def predict_single_endpoint(req: PredictRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Predict credit score for a single customer"""
    session = get_session(x_session_id)
    ensure_approved_model_for_prediction(session)
    if session.model is None:
        raise HTTPException(status_code=400, detail="No model trained yet")

    try:
        result = predict_single(
            session.model,
            req.input_data,
            session.selected_features,
            session.feature_stats or None,
        )

        # Get feature contributions
        contributions = get_feature_contributions(
            session.model, req.input_data, session.selected_features,
            shap_explainer=session.shap_explainer.explainer if session.shap_explainer else None,
        )

        # Generate recommendations
        recommendations = generate_recommendations(result, req.input_data, contributions)

        result["contributions"] = [(f, float(c)) for f, c in contributions[:10]]
        result["recommendations"] = recommendations

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== FEATURE INFO ====================

@router.get("/features")
def get_features(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get the list of features required for prediction and their stats"""
    session = get_session(x_session_id)
    ensure_approved_model_for_prediction(session)
    if session.model is None:
        raise HTTPException(status_code=400, detail="No model trained yet")

    features_info = []
    for feat in session.selected_features:
        info = {"name": feat}
        if session.X_train is not None and feat in session.X_train.columns:
            col = session.X_train[feat]
            info["min"] = float(col.min())
            info["max"] = float(col.max())
            info["mean"] = float(col.mean())
            info["median"] = float(col.median())
        features_info.append(info)

    return {"features": features_info, "n_features": len(features_info)}
