"""
=============================================================================
SHAP EXPLAINER MODULE - GIẢI THÍCH MÔ HÌNH VỚI SHAP
=============================================================================
Mô tả:
    Module tính toán SHAP (SHapley Additive exPlanations) values để giải thích
    quyết định của model. SHAP dựa trên lý thuyết game Shapley values.

SHAP values cho biết:
    - Feature nào đóng góp nhiều nhất vào dự đoán
    - Chiều hướng tác động (tăng/giảm xác suất default)
    - Mức độ tác động của từng giá trị feature

Các loại SHAP Explainer:
    1. TreeExplainer: Cho tree-based models (RF, XGBoost, LightGBM, CatBoost)
       - Nhanh và chính xác
    2. LinearExplainer: Cho linear models (Logistic Regression)
       - Exact attribution
    3. KernelExplainer: Cho bất kỳ model nào
       - Chậm hơn, dùng làm fallback

Các chức năng chính:
    1. SHAPExplainer class:
       - compute_shap_values(): Tính SHAP values cho dataset
       - get_feature_importance(): Global importance (mean |SHAP|)
       - get_local_explanation(): Local explanation cho 1 sample
    
    2. Utility functions:
       - initialize_shap_explainer(): Khởi tạo + tính SHAP
       - compute_shap_for_sample(): Tính SHAP cho 1 sample

Lưu ý hiệu năng:
    - Với dataset lớn, sử dụng sample (500 background, 1000 explain)
    - TreeExplainer nhanh nhất (~O(TL2^M)) với T trees, L leaves, M depth
=============================================================================
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Optional, Tuple, List


# =============================================================================
# CLASS SHAP EXPLAINER
# =============================================================================


class SHAPExplainer:
    """Class để tính toán và quản lý SHAP values"""
    
    def __init__(self, model, X_background: pd.DataFrame, model_type: str = "tree"):
        """
        Khởi tạo SHAP Explainer
        
        Args:
            model: Trained model
            X_background: Background data for SHAP (typically training data sample)
            model_type: Type of model ("tree", "linear", "kernel")
        """
        self.model = model
        self.X_background = X_background
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = list(X_background.columns)
        
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            # For tree-based models (Random Forest, XGBoost, LightGBM, etc.)
            if self.model_type in ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]:
                self.explainer = shap.TreeExplainer(self.model)
            # For linear models (Logistic Regression)
            elif self.model_type == "Logistic Regression":
                self.explainer = shap.LinearExplainer(self.model, self.X_background)
            else:
                # Fallback to KernelExplainer (slower but works for any model)
                # Use a sample of background data for efficiency
                background_sample = shap.sample(self.X_background, min(100, len(self.X_background)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background_sample)
            
            self.expected_value = self.explainer.expected_value
            
            # For binary classification, get the positive class expected value
            if isinstance(self.expected_value, (list, np.ndarray)) and len(self.expected_value) > 1:
                self.expected_value = self.expected_value[1]
                
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            raise
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for given data
        
        Args:
            X: Data to compute SHAP values for
            
        Returns:
            SHAP values array
        """
        try:
            shap_values = self.explainer.shap_values(X)
            
            # For binary classification, get SHAP values for positive class
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
            
            self.shap_values = shap_values
            return shap_values
            
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call compute_shap_values first.")
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def get_local_explanation(self, sample_idx: int, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get local explanation for a specific sample
        
        Args:
            sample_idx: Index of sample to explain
            X: Full dataset
            
        Returns:
            Dictionary with local explanation data
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call compute_shap_values first.")
        
        # Get SHAP values for specific sample
        sample_shap = self.shap_values[sample_idx]
        sample_data = X.iloc[sample_idx]
        
        # Create DataFrame with feature contributions
        contributions = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': sample_data.values,
            'SHAP_Value': sample_shap
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        # Calculate prediction
        base_value = float(self.expected_value) if isinstance(self.expected_value, (int, float, np.number)) else float(self.expected_value)
        prediction = base_value + sample_shap.sum()
        
        return {
            'sample_idx': sample_idx,
            'base_value': base_value,
            'prediction': prediction,
            'shap_values': sample_shap,
            'feature_values': sample_data.values,
            'contributions': contributions
        }
    
    def get_shap_summary_data(self) -> Dict[str, Any]:
        """
        Get data for SHAP summary plots
        
        Returns:
            Dictionary with summary data
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed yet. Call compute_shap_values first.")
        
        return {
            'shap_values': self.shap_values,
            'feature_names': self.feature_names,
            'expected_value': self.expected_value
        }


def initialize_shap_explainer(model, X_train: pd.DataFrame, model_type: str) -> Tuple[SHAPExplainer, np.ndarray]:
    """
    Initialize SHAP explainer and compute SHAP values
    
    Args:
        model: Trained model
        X_train: Training data (features only)
        model_type: Type of model
        
    Returns:
        Tuple of (SHAPExplainer, shap_values)
    """
    # Use a sample for background if data is too large
    if len(X_train) > 500:
        background_sample = X_train.sample(n=500, random_state=42)
    else:
        background_sample = X_train
    
    # Initialize explainer
    explainer = SHAPExplainer(model, background_sample, model_type)
    
    # Compute SHAP values for all training data (or a sample for large datasets)
    if len(X_train) > 1000:
        X_to_explain = X_train.sample(n=1000, random_state=42)
    else:
        X_to_explain = X_train
    
    shap_values = explainer.compute_shap_values(X_to_explain)
    
    return explainer, shap_values, X_to_explain


def compute_shap_for_sample(explainer: SHAPExplainer, X_sample: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for a single sample or small batch
    
    Args:
        explainer: Initialized SHAPExplainer
        X_sample: Sample data (single row or small batch)
        
    Returns:
        SHAP values for the sample
    """
    shap_values = explainer.explainer.shap_values(X_sample)
    
    # For binary classification, get SHAP values for positive class
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]
    
    return shap_values
