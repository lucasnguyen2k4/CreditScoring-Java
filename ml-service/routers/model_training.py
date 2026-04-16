"""
Model Training Router
Endpoints for training ML models, cross-validation, hyperparameter tuning.
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json

from session_store import store
from services.trainer import (
    train_model,
    train_stacking_model,
    tune_stacking_with_oof,
    cross_validate_model,
    hyperparameter_tuning,
)

router = APIRouter()


def get_session(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required")
    return store.get_or_create(session_id)


def safe_metrics(metrics: dict) -> dict:
    """Make metrics JSON serializable"""
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            safe[k] = safe_metrics(v)
        elif isinstance(v, (list, tuple)):
            safe[k] = v
        elif v is None:
            safe[k] = None
        else:
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
    return safe


def safe_json(value):
    """Recursively convert values to JSON-serializable types."""
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def map_tuned_params_for_training(model_type: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize tuning param names to train_model parameter names."""
    p = dict(params or {})
    if model_type == "CatBoost":
        if "iterations" in p and "n_estimators" not in p:
            p["n_estimators"] = p.pop("iterations")
        if "depth" in p and "max_depth" not in p:
            p["max_depth"] = p.pop("depth")
    return p


def get_selected_training_frames(session):
    """Return train/valid/test frames using selected features if configured."""
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet")

    selected = [c for c in (session.selected_features or []) if c in session.X_train.columns]
    if not selected:
        selected = session.X_train.columns.tolist()
        session.selected_features = selected

    X_train = session.X_train[selected]
    X_valid = session.X_valid[selected] if session.X_valid is not None else None
    X_test = session.X_test[selected] if session.X_test is not None else None
    return X_train, X_valid, X_test


# ==================== TRAIN ====================

class TrainRequest(BaseModel):
    model_type: str  # Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting
    params: Optional[Dict[str, Any]] = None
    early_stopping_rounds: Optional[int] = None

@router.post("/train")
def train(req: TrainRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Train a single ML model"""
    session = get_session(x_session_id)
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet. Call /data/split first.")

    try:
        X_train, X_valid, X_test = get_selected_training_frames(session)
        model, metrics = train_model(
            X_train, session.y_train,
            X_test, session.y_test,
            model_type=req.model_type,
            params=req.params,
            X_valid=X_valid,
            y_valid=session.y_valid,
            early_stopping_rounds=req.early_stopping_rounds,
        )

        session.model = model
        session.model_type = req.model_type
        session.model_metrics = metrics

        # Add to trained models history
        import datetime
        session.trained_models.append({
            "model_type": req.model_type,
            "metrics": safe_metrics(metrics),
            "params": req.params or {},
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })
        session.trained_model_objects.append(model)

        return {
            "message": f"{req.model_type} trained successfully",
            "metrics": safe_metrics(metrics),
            "model_index": len(session.trained_models) - 1,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== STACKING ====================

class StackingRequest(BaseModel):
    base_models: List[str]  # ['LR', 'DT', 'SVM', 'KNN', 'RF', 'GB']
    meta_model: str = "Random Forest"
    params: Optional[Dict[str, Any]] = None

@router.post("/train-stacking")
def train_stacking(req: StackingRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Train a Stacking Classifier"""
    session = get_session(x_session_id)
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet")

    try:
        X_train, _, X_test = get_selected_training_frames(session)
        model, metrics = train_stacking_model(
            X_train, session.y_train,
            X_test, session.y_test,
            base_models=req.base_models,
            meta_model=req.meta_model,
            params=req.params,
        )

        session.model = model
        session.model_type = "Stacking"
        session.model_metrics = metrics

        import datetime
        session.trained_models.append({
            "model_type": f"Stacking ({', '.join(req.base_models)} → {req.meta_model})",
            "metrics": safe_metrics(metrics),
            "params": safe_json(req.params or {}),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })
        session.trained_model_objects.append(model)

        return {
            "message": "Stacking model trained successfully",
            "metrics": safe_metrics(metrics),
            "model_index": len(session.trained_models) - 1,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== CROSS-VALIDATION ====================

class CVRequest(BaseModel):
    model_type: str
    params: Optional[Dict[str, Any]] = None
    cv_folds: int = 5

@router.post("/cross-validate")
def cross_validate(req: CVRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Perform cross-validation"""
    session = get_session(x_session_id)
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet")

    try:
        X_train, _, _ = get_selected_training_frames(session)
        result = cross_validate_model(
            X_train, session.y_train,
            model_type=req.model_type,
            params=req.params,
            cv_folds=req.cv_folds,
        )
        return safe_metrics(result) if isinstance(result, dict) else {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== HYPERPARAMETER TUNING ====================

class TuneRequest(BaseModel):
    model_type: str
    method: str = "Grid Search"  # Grid Search, Random Search, Optuna, Bayesian Optimization
    cv_folds: int = 5
    n_trials: int = 50
    auto_train_best: bool = False
    early_stopping_rounds: Optional[int] = None


@router.post("/tune")
def tune_model(req: TuneRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Tune hyperparameters and optionally train with best params."""
    session = get_session(x_session_id)
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet")

    try:
        X_train, X_valid, X_test = get_selected_training_frames(session)
        tuning = hyperparameter_tuning(
            X_train,
            session.y_train,
            model_type=req.model_type,
            method=req.method,
            cv_folds=req.cv_folds,
            n_trials=req.n_trials,
        )

        safe_tuning = safe_json({k: v for k, v in tuning.items() if k != "best_estimator"})
        session.tuning_results[req.model_type] = safe_tuning

        response = {
            "message": f"Tuning completed for {req.model_type} with {req.method}",
            "model_type": req.model_type,
            "method": req.method,
            "best_params": safe_tuning.get("best_params", {}),
            "best_score": safe_tuning.get("best_score"),
            "top_results": safe_tuning.get("top_results", []),
            "total_fits": safe_tuning.get("total_fits"),
        }

        if req.auto_train_best:
            best_params = map_tuned_params_for_training(req.model_type, safe_tuning.get("best_params", {}))
            model, metrics = train_model(
                X_train, session.y_train,
                X_test, session.y_test,
                model_type=req.model_type,
                params=best_params,
                X_valid=X_valid,
                y_valid=session.y_valid,
                early_stopping_rounds=req.early_stopping_rounds,
            )
            session.model = model
            session.model_type = f"{req.model_type} (Tuned)"
            session.model_metrics = metrics

            import datetime
            session.trained_models.append({
                "model_type": f"{req.model_type} (Tuned: {req.method})",
                "metrics": safe_metrics(metrics),
                "params": safe_json(best_params),
                "tuning_method": req.method,
                "tuning_best_score": safe_tuning.get("best_score"),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })
            session.trained_model_objects.append(model)

            response["trained_model_index"] = len(session.trained_models) - 1
            response["trained_metrics"] = safe_metrics(metrics)
            response["message"] = f"Tuning completed and trained best {req.model_type} model"

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class TuneStackingRequest(BaseModel):
    base_models: List[str]
    meta_model: str = "Random Forest"
    tuning_method: str = "Grid Search"  # Grid Search, Random Search, Default
    cv_folds: int = 5
    base_models_config: Optional[Dict[str, Dict[str, Any]]] = None
    meta_model_params: Optional[Dict[str, Any]] = None


@router.post("/tune-stacking")
def tune_stacking(req: TuneStackingRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Tune stacking with OOF and train final stacking model."""
    session = get_session(x_session_id)
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet")

    try:
        X_train, _, X_test = get_selected_training_frames(session)
        base_config: Dict[str, Dict[str, Any]] = {}
        provided_base = req.base_models_config or {}
        for model_key in req.base_models:
            cfg = provided_base.get(model_key, {})
            base_config[model_key] = cfg if isinstance(cfg, dict) else {}

        meta_params = req.meta_model_params or {}
        model, metrics, tuning_info = tune_stacking_with_oof(
            X_train,
            session.y_train,
            X_test,
            session.y_test,
            base_models_config=base_config,
            meta_model=req.meta_model,
            tuning_method=req.tuning_method,
            n_folds=req.cv_folds,
            params={"meta_model_params": meta_params},
        )

        session.model = model
        session.model_type = "Stacking (Tuned)"
        session.model_metrics = metrics
        safe_tuning_info = safe_json(tuning_info)
        session.stacking_tuning_results = safe_tuning_info

        import datetime
        session.trained_models.append({
            "model_type": f"Stacking Tuned ({req.tuning_method})",
            "metrics": safe_metrics(metrics),
            "params": {
                "base_models": req.base_models,
                "meta_model": req.meta_model,
                "cv_folds": req.cv_folds,
                "base_models_config": safe_json(base_config),
                "meta_model_params": safe_json(meta_params),
            },
            "tuning_info": safe_tuning_info,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })
        session.trained_model_objects.append(model)

        return {
            "message": "Stacking OOF tuning completed",
            "metrics": safe_metrics(metrics),
            "tuning_info": safe_tuning_info,
            "used_config": {
                "base_models_config": safe_json(base_config),
                "meta_model_params": safe_json(meta_params),
            },
            "model_index": len(session.trained_models) - 1,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tuning-results")
def get_tuning_results(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get cached tuning results for current session."""
    session = get_session(x_session_id)
    return {
        "single_model_tuning": safe_json(session.tuning_results),
        "stacking_tuning": safe_json(session.stacking_tuning_results),
    }


# ==================== MODEL HISTORY ====================

@router.get("/history")
def get_model_history(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get list of trained models"""
    session = get_session(x_session_id)
    models = []
    for idx, item in enumerate(session.trained_models):
        row = dict(item)
        row["index"] = idx
        row["is_approved"] = session.approved_model_index == idx
        models.append(safe_json(row))
    return {
        "models": models,
        "approved_model_index": session.approved_model_index,
    }


class SelectModelRequest(BaseModel):
    model_index: int

@router.post("/select")
def select_model(req: SelectModelRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Select a previously trained model as the active model"""
    session = get_session(x_session_id)
    if req.model_index < 0 or req.model_index >= len(session.trained_models):
        raise HTTPException(status_code=400, detail="Invalid model index")

    if req.model_index >= len(session.trained_model_objects) or session.trained_model_objects[req.model_index] is None:
        raise HTTPException(status_code=400, detail="Selected model object is not available in memory. Retrain this model first.")

    session.model = session.trained_model_objects[req.model_index]
    selected = session.trained_models[req.model_index]
    session.model_type = selected.get("model_type")
    session.model_metrics = selected.get("metrics", {})
    return {"message": f"Selected model: {selected['model_type']}", "metrics": selected["metrics"]}


@router.get("/current")
def get_current_model(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get info about the currently active model"""
    session = get_session(x_session_id)
    if session.model is None:
        raise HTTPException(status_code=404, detail="No model trained yet")

    return {
        "model_type": session.model_type,
        "metrics": safe_metrics(session.model_metrics),
        "features": session.selected_features,
        "n_features": len(session.selected_features),
        "approved_model_index": session.approved_model_index,
    }


# ==================== MODEL APPROVAL ====================

class ApprovalRequest(BaseModel):
    model_index: int
    decision: str = "approved"  # approved, conditional, rejected
    notes: Optional[str] = None
    approved_by: Optional[str] = None


@router.post("/approve")
def approve_model(req: ApprovalRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Approve/reject a trained model for downstream usage."""
    session = get_session(x_session_id)
    if req.model_index < 0 or req.model_index >= len(session.trained_models):
        raise HTTPException(status_code=400, detail="Invalid model index")

    decision = (req.decision or "").strip().lower()
    if decision not in {"approved", "conditional", "rejected"}:
        raise HTTPException(status_code=400, detail="Decision must be one of: approved, conditional, rejected")

    model_item = session.trained_models[req.model_index]

    import datetime
    record = {
        "model_index": req.model_index,
        "model_type": model_item.get("model_type"),
        "decision": decision,
        "notes": req.notes or "",
        "approved_by": req.approved_by or "unknown",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "metrics": safe_json(model_item.get("metrics", {})),
    }
    session.model_approvals.append(record)

    if decision == "approved":
        session.approved_model_index = req.model_index
        if req.model_index < len(session.trained_model_objects):
            obj = session.trained_model_objects[req.model_index]
            if obj is not None:
                session.model = obj
                session.model_type = model_item.get("model_type")
                session.model_metrics = model_item.get("metrics", {})

    return {
        "message": f"Model #{req.model_index} marked as {decision}",
        "approval": safe_json(record),
        "approved_model_index": session.approved_model_index,
    }


@router.get("/approvals")
def get_approvals(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get approval history and current approved model."""
    session = get_session(x_session_id)
    approved_model = None
    if (
        session.approved_model_index is not None
        and 0 <= session.approved_model_index < len(session.trained_models)
    ):
        approved_model = safe_json({
            "index": session.approved_model_index,
            **session.trained_models[session.approved_model_index],
        })

    return {
        "approved_model_index": session.approved_model_index,
        "approved_model": approved_model,
        "approvals": safe_json(session.model_approvals),
    }
