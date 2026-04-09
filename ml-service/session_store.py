"""
In-Memory Session Store for ML Service
Stores datasets, models, SHAP explainers etc. in memory, keyed by session_id.
Spring Boot backend will pass a session_id header with each request.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


class SessionData:
    """Stores all state for one user session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()

        # Data
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.selected_features: list = []

        # Train/Valid/Test splits
        self.X_train: Optional[pd.DataFrame] = None
        self.X_valid: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None

        # Model
        self.model = None
        self.model_type: Optional[str] = None
        self.model_metrics: dict = {}
        self.trained_models: list = []
        self.trained_model_objects: list = []
        self.tuning_results: dict = {}
        self.stacking_tuning_results: dict = {}
        self.model_approvals: list = []
        self.approved_model_index: Optional[int] = None

        # SHAP
        self.shap_explainer = None
        self.shap_values = None
        self.shap_X_explained: Optional[pd.DataFrame] = None
        self.shap_feature_importance: Optional[pd.DataFrame] = None
        self.shap_expected_value = None

        # Pipeline (fitted transformers)
        self.pipeline = None

        # Feature engineering configs
        self.missing_config: dict = {}
        self.encoding_config: dict = {}
        self.scaling_config: dict = {}
        self.outlier_config: dict = {}
        self.binning_config: dict = {}

        # Fitted objects
        self.label_encoders: dict = {}
        self.onehot_encoder = None
        self.scaler = None
        self.feature_stats: dict = {}

        # Balance info
        self.balance_info: dict = {}

        # AI Analysis cache
        self.ai_analysis: Optional[str] = None
        self.eda_summary: Optional[str] = None

    def clear_data(self):
        """Reset all data-dependent state (when new data is uploaded)"""
        self.processed_data = None
        self.target_column = None
        self.selected_features = []
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.model = None
        self.model_type = None
        self.model_metrics = {}
        self.trained_models = []
        self.trained_model_objects = []
        self.tuning_results = {}
        self.stacking_tuning_results = {}
        self.model_approvals = []
        self.approved_model_index = None
        self.shap_explainer = None
        self.shap_values = None
        self.shap_X_explained = None
        self.shap_feature_importance = None
        self.shap_expected_value = None
        self.pipeline = None
        self.missing_config = {}
        self.encoding_config = {}
        self.scaling_config = {}
        self.outlier_config = {}
        self.binning_config = {}
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = None
        self.feature_stats = {}
        self.balance_info = {}
        self.ai_analysis = None
        self.eda_summary = None


class SessionStore:
    """Global in-memory store for all sessions"""

    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}

    def get_or_create(self, session_id: str) -> SessionData:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData(session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionData]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> list:
        return [
            {"session_id": s.session_id, "created_at": s.created_at.isoformat(),
             "has_data": s.raw_data is not None, "has_model": s.model is not None}
            for s in self._sessions.values()
        ]


# Singleton store
store = SessionStore()
