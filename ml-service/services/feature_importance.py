"""
Feature Importance Module
Tính toán độ quan trọng của các features sử dụng nhiều phương pháp khác nhau
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def calculate_feature_importance(
    X_train,
    y_train,
    method="Random Forest",
    top_n=15,
    task_type="auto"
):
    """
    Tính feature importance sử dụng nhiều phương pháp khác nhau
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or np.array
        Training target
    method : str
        Phương pháp tính importance:
        - "Random Forest"
        - "LightGBM"
        - "XGBoost"
        - "Logistic Regression (Coef)"
    top_n : int
        Số lượng top features cần trả về
    task_type : str
        "classification", "regression", hoặc "auto" (tự động phát hiện)
    
    Returns:
    --------
    dict : {
        'feature_names': list of str,
        'importance_scores': list of float,
        'method': str,
        'task_type': str,
        'model_params': dict
    }
    """
    
    # Auto detect task type
    if task_type == "auto":
        unique_values = len(np.unique(y_train))
        if unique_values <= 10:
            task_type = "classification"
        else:
            task_type = "regression"
    
    feature_names = X_train.columns.tolist()
    
    # Handle missing values
    X_train_processed = X_train.fillna(X_train.median())
    
    try:
        if method == "Random Forest":
            importance_scores, model_params = _random_forest_importance(
                X_train_processed, y_train, task_type
            )
        
        elif method == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM chưa được cài đặt. Vui lòng cài: pip install lightgbm")
            importance_scores, model_params = _lightgbm_importance(
                X_train_processed, y_train, task_type
            )
        
        elif method == "XGBoost":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost chưa được cài đặt. Vui lòng cài: pip install xgboost")
            importance_scores, model_params = _xgboost_importance(
                X_train_processed, y_train, task_type
            )
        
        elif method == "Logistic Regression (Coef)":
            importance_scores, model_params = _logistic_coef_importance(
                X_train_processed, y_train, task_type
            )
        
        else:
            raise ValueError(f"Phương pháp không hợp lệ: {method}")
        
        # Normalize importance scores to [0, 1]
        importance_scores = np.array(importance_scores)
        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.sum()
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        # Get top N features
        top_indices = sorted_indices[:top_n]
        
        return {
            'feature_names': [feature_names[i] for i in top_indices],
            'importance_scores': importance_scores[top_indices].tolist(),
            'all_feature_names': feature_names,
            'all_importance_scores': importance_scores.tolist(),
            'method': method,
            'task_type': task_type,
            'model_params': model_params,
            'n_features': len(feature_names),
            'top_n': top_n
        }
    
    except Exception as e:
        raise Exception(f"Lỗi khi tính feature importance: {str(e)}")


def _random_forest_importance(X, y, task_type):
    """Random Forest feature importance"""
    
    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X, y)
    importance_scores = model.feature_importances_
    
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'criterion': model.criterion
    }
    
    return importance_scores, model_params


def _lightgbm_importance(X, y, task_type):
    """LightGBM feature importance"""
    
    if task_type == "classification":
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=10,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    
    model.fit(X, y)
    importance_scores = model.feature_importances_
    
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'num_leaves': 31,
        'learning_rate': 0.1
    }
    
    return importance_scores, model_params


def _xgboost_importance(X, y, task_type):
    """XGBoost feature importance"""
    
    if task_type == "classification":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
    
    model.fit(X, y)
    importance_scores = model.feature_importances_
    
    model_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    return importance_scores, model_params


def _logistic_coef_importance(X, y, task_type):
    """Logistic/Linear Regression coefficients as importance"""
    
    # Standardize features for fair coefficient comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if task_type == "classification":
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
    else:
        model = LinearRegression()
    
    model.fit(X_scaled, y)
    
    # Use absolute coefficients as importance
    if hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 1:
            importance_scores = np.abs(model.coef_)
        else:
            # Multi-class: average across classes
            importance_scores = np.abs(model.coef_).mean(axis=0)
    else:
        raise ValueError("Model không có coefficients")
    
    model_params = {
        'max_iter': 1000,
        'solver': model.solver if hasattr(model, 'solver') else 'default',
        'scaled': True
    }
    
    return importance_scores, model_params


def get_feature_importance_recommendations(importance_results, threshold=0.01):
    """
    Đưa ra gợi ý về features nên giữ lại dựa trên importance scores
    
    Parameters:
    -----------
    importance_results : dict
        Kết quả từ calculate_feature_importance
    threshold : float
        Ngưỡng importance tối thiểu
    
    Returns:
    --------
    dict : {
        'recommended_features': list,
        'removed_features': list,
        'threshold': float,
        'reason': str
    }
    """
    
    all_features = importance_results['all_feature_names']
    all_scores = importance_results['all_importance_scores']
    
    recommended = []
    removed = []
    
    for feat, score in zip(all_features, all_scores):
        if score >= threshold:
            recommended.append(feat)
        else:
            removed.append(feat)
    
    return {
        'recommended_features': recommended,
        'removed_features': removed,
        'n_recommended': len(recommended),
        'n_removed': len(removed),
        'threshold': threshold,
        'reason': f"Giữ lại {len(recommended)} features có importance >= {threshold:.3f}"
    }


def compare_importance_methods(X_train, y_train, methods=None, top_n=10):
    """
    So sánh feature importance từ nhiều phương pháp khác nhau
    
    Parameters:
    -----------
    X_train : pd.DataFrame
    y_train : pd.Series
    methods : list of str
        Danh sách phương pháp cần so sánh
    top_n : int
    
    Returns:
    --------
    dict : {
        'method_name': importance_results,
        ...
    }
    """
    
    if methods is None:
        methods = ["Random Forest", "LightGBM", "XGBoost", "Logistic Regression (Coef)"]
    
    results = {}
    
    for method in methods:
        try:
            result = calculate_feature_importance(
                X_train, y_train,
                method=method,
                top_n=top_n
            )
            results[method] = result
        except Exception as e:
            print(f"Lỗi với phương pháp {method}: {str(e)}")
            continue
    
    return results
