"""
=============================================================================
MODEL TRAINING MODULE - HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING
=============================================================================
Mô tả:
    Module xử lý toàn bộ quá trình huấn luyện các mô hình Machine Learning
    cho bài toán Credit Scoring (phân loại nhị phân).

Các thuật toán hỗ trợ:
    1. Logistic Regression - Hồi quy logistic
    2. Random Forest - Rừng ngẫu nhiên  
    3. Gradient Boosting - Tăng cường gradient
    4. XGBoost - Extreme Gradient Boosting
    5. LightGBM - Light Gradient Boosting Machine
    6. CatBoost - Categorical Boosting
    7. Stacking Ensemble - Kết hợp nhiều models

Các chức năng chính:
    - train_model(): Huấn luyện model đơn lẻ với early stopping
    - train_stacking_model(): Huấn luyện Stacking Classifier
    - tune_stacking_with_oof(): Tune Stacking với Out-of-Fold (tránh data leakage)
    - cross_validate_model(): Cross-validation K-Fold
    - hyperparameter_tuning(): Grid Search / Random Search
    - optuna_hyperparameter_tuning(): Bayesian Optimization với Optuna

NGUYÊN TẮC QUAN TRỌNG:
    - Test set KHÔNG BAO GIỜ được dùng cho training hoặc early stopping
    - Validation set được dùng cho early stopping và monitoring
    - Tách biệt rõ ràng train/valid/test để tránh data leakage
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


# =============================================================================
# PHẦN 1: HUẤN LUYỆN MODEL ĐƠN LẺ (SINGLE MODEL TRAINING)
# =============================================================================

def train_model(X_train, y_train, X_test, y_test, model_type, params=None,
                X_valid=None, y_valid=None, early_stopping_rounds=None):
    """
    Train a machine learning model based on the specified type and parameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features (holdout set - NEVER used for training or early stopping)
    y_test : pd.Series
        Test target
    model_type : str
        Type of model to train
    params : dict
        Model parameters
    X_valid : pd.DataFrame, optional
        Validation features (used for early stopping and monitoring, separate from test)
    y_valid : pd.Series, optional
        Validation target
    early_stopping_rounds : int, optional
        Number of rounds for early stopping (boosting models only)
        
    Returns:
    --------
    model : object
        Trained model object
    metrics : dict
        Dictionary containing train_metrics, valid_metrics, test_metrics
    """
    if params is None:
        params = {}

    model = None
    early_stopped_iteration = None
    
    # Helper function to calculate metrics
    def calculate_metrics(model, X, y, dataset_name=""):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else 0.5,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
    
    try:
        # =====================================================================
        # LOGISTIC REGRESSION - Hồi quy Logistic
        # ---------------------------------------------------------------------
        # Ưu điểm: Nhanh, dễ giải thích (interpretable), phù hợp dữ liệu tuyến tính
        # Tham số chính:
        #   - C: Regularization strength (1/lambda). C càng nhỏ = regularization càng mạnh
        #   - max_iter: Số vòng lặp tối đa để hội tụ
        # =====================================================================
        if model_type == "Logistic Regression":
            model = LogisticRegression(
                C=params.get('C', 1.0),                    # Mặc định: 1.0 (cân bằng)
                max_iter=params.get('max_iter', 200),      # Mặc định: 200 iterations
                random_state=params.get('random_state', 42)
            )
            model.fit(X_train, y_train)
        
        # =====================================================================
        # RANDOM FOREST - Rừng Ngẫu Nhiên
        # ---------------------------------------------------------------------
        # Ưu điểm: Mạnh mẽ, chống overfitting tốt, xử lý được dữ liệu phi tuyến
        # Tham số chính:
        #   - n_estimators: Số cây trong rừng (càng nhiều càng ổn định nhưng chậm)
        #   - max_depth: Độ sâu tối đa mỗi cây (None = không giới hạn)
        #   - min_samples_split: Số mẫu tối thiểu để tách node
        # =====================================================================
        elif model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),     # Mặc định: 100 cây
                max_depth=params.get('max_depth', None),          # Mặc định: không giới hạn
                min_samples_split=params.get('min_samples_split', 2),
                random_state=params.get('random_state', 42)
            )
            model.fit(X_train, y_train)
        
        # =====================================================================
        # GRADIENT BOOSTING - Tăng cường Gradient (sklearn)
        # ---------------------------------------------------------------------
        # Ưu điểm: Hiệu suất cao, học từ sai số của model trước
        # Tham số chính:
        #   - n_estimators: Số cây boosting
        #   - learning_rate: Tốc độ học (nhỏ hơn = học chậm nhưng chính xác hơn)
        #   - max_depth: Độ sâu tối đa (thường nhỏ: 3-7)
        #   - subsample: Tỷ lệ mẫu dùng cho mỗi cây (<1.0 giúp chống overfit)
        # =====================================================================
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),     # Mặc định: 100 cây
                learning_rate=params.get('learning_rate', 0.1),   # Mặc định: 0.1
                max_depth=params.get('max_depth', 3),             # Mặc định: 3 (shallow)
                subsample=params.get('subsample', 1.0),           # Mặc định: 100% mẫu
                random_state=params.get('random_state', 42)
            )
            model.fit(X_train, y_train)
        
        # =====================================================================
        # XGBOOST - Extreme Gradient Boosting
        # ---------------------------------------------------------------------
        # Ưu điểm: Nhanh, hiệu suất cao, hỗ trợ GPU, xử lý missing values tự động
        # Hỗ trợ Early Stopping: Dừng sớm nếu validation không cải thiện
        # Tham số chính:
        #   - n_estimators: Số cây boosting
        #   - learning_rate (eta): Tốc độ học
        #   - max_depth: Độ sâu tối đa (mặc định 6, cao hơn sklearn)
        # =====================================================================
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),             # XGBoost mặc định sâu hơn
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'                             # Metric đánh giá: Log Loss
            )
            
            # EARLY STOPPING: Dừng sớm dựa trên Validation Set (KHÔNG dùng Test Set!)
            # Giúp tránh overfitting bằng cách dừng khi validation không cải thiện
            if X_valid is not None and y_valid is not None and early_stopping_rounds:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],    # Dùng validation set để theo dõi
                    verbose=False
                )
                model.set_params(early_stopping_rounds=early_stopping_rounds)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False
                )
                early_stopped_iteration = model.best_iteration if hasattr(model, 'best_iteration') else None
            else:
                model.fit(X_train, y_train)
        
        # =====================================================================
        # LIGHTGBM - Light Gradient Boosting Machine
        # ---------------------------------------------------------------------
        # Ưu điểm: Cực nhanh, bộ nhớ thấp, xử lý dữ liệu lớn tốt
        # Sử dụng histogram-based algorithm và leaf-wise tree growth
        # Tham số chính:
        #   - max_depth: -1 = không giới hạn (khác với XGBoost)
        # =====================================================================
        elif model_type == "LightGBM":
            model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', -1),            # -1 = không giới hạn
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=-1                                        # Tắt log output
            )
            
            # EARLY STOPPING cho LightGBM
            if X_valid is not None and y_valid is not None and early_stopping_rounds:
                callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=callbacks
                )
                early_stopped_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else None
            else:
                model.fit(X_train, y_train)
        
        # =====================================================================
        # CATBOOST - Categorical Boosting
        # ---------------------------------------------------------------------
        # Ưu điểm: Xử lý biến phân loại tự động (không cần encoding thủ công)
        # Ít cần tiền xử lý dữ liệu, chống overfitting tốt
        # Tham số chính:
        #   - iterations: Số cây (tương đương n_estimators)
        #   - depth: Độ sâu (tương đương max_depth)
        # =====================================================================
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(
                iterations=params.get('n_estimators', 100),       # CatBoost gọi là iterations
                learning_rate=params.get('learning_rate', 0.1),
                depth=params.get('max_depth', 6),                 # CatBoost gọi là depth
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=0,                                        # Tắt log
                allow_writing_files=False                         # Không ghi cache ra file
            )
            
            # EARLY STOPPING cho CatBoost
            if X_valid is not None and y_valid is not None and early_stopping_rounds:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),                  # CatBoost dùng tuple, không list
                    early_stopping_rounds=early_stopping_rounds
                )
                early_stopped_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else None
            else:
                model.fit(X_train, y_train)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate metrics on all datasets (no data leakage - each set is separate)
        train_metrics = calculate_metrics(model, X_train, y_train, "train")
        test_metrics = calculate_metrics(model, X_test, y_test, "test")
        
        # Validation metrics only if validation set provided
        valid_metrics = None
        if X_valid is not None and y_valid is not None:
            valid_metrics = calculate_metrics(model, X_valid, y_valid, "validation")
        
        # Combine all metrics
        metrics = {
            'accuracy': test_metrics['accuracy'],  # Primary metrics still from test set
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'confusion_matrix': test_metrics['confusion_matrix'],
            # Detailed breakdown for comparison
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics,
            'early_stopped_iteration': early_stopped_iteration
        }
        
        return model, metrics
        
    except Exception as e:
        raise Exception(f"Error training {model_type}: {str(e)}")


def train_stacking_model(X_train, y_train, X_test, y_test, 
                         base_models: list, meta_model: str, params=None):
    """
    Train Stacking Classifier with selected base models and meta model.
    Following the approach from "Credit-Risk-Scoring: A Stacking Generalization Approach"
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    base_models : list
        List of base model names. Options: ['LR', 'DT', 'SVM', 'KNN', 'RF', 'GB']
    meta_model : str
        Name of meta model (final estimator). Options: 'Random Forest', 'Logistic Regression', 'XGBoost'
    params : dict
        Additional parameters (e.g., random_state)
        
    Returns:
    --------
    model : StackingClassifier
        Trained stacking model
    metrics : dict
        Dictionary of evaluation metrics
    """
    if params is None:
        params = {}
    
    random_state = params.get('random_state', 42)
    
    
    # Get custom params for models if available
    base_model_params_config = params.get('base_model_params', {})
    meta_model_params_config = params.get('meta_model_params', {})
    
    # Define base model classes
    base_model_classes = {
        'LR': LogisticRegression,
        'DT': DecisionTreeClassifier,
        'SVM': SVC,
        'KNN': KNeighborsClassifier,
        'RF': RandomForestClassifier,
        'GB': GradientBoostingClassifier
    }
    
    # helper to merge default and custom params
    def get_model_instance(model_key, model_class, custom_params, default_params):
        # Start with defaults
        final_params = default_params.copy()
        # Update with custom params if any
        if model_key in custom_params:
            final_params.update(custom_params[model_key])
        return model_class(**final_params)

    # Build estimators list from selected base models
    estimators = []
    for model_key in base_models:
        if model_key in base_model_classes:
            model_cls = base_model_classes[model_key]
            
            # Define default params
            defaults = {'random_state': random_state} if 'random_state' in model_cls().get_params() else {}
            if model_key == 'LR': defaults.update({'max_iter': 200})
            if model_key == 'DT': defaults.update({'max_depth': 10})
            if model_key == 'SVM': defaults.update({'kernel': 'rbf', 'probability': True})
            if model_key == 'KNN': defaults = {'n_neighbors': 5} # KNN doesn't have random_state
            if model_key == 'RF': defaults.update({'n_estimators': 100})
            if model_key == 'GB': defaults.update({'n_estimators': 100})
            
            model_instance = get_model_instance(model_key, model_cls, base_model_params_config, defaults)
            estimators.append((model_key, model_instance))
    
    if len(estimators) < 2:
        raise ValueError("Stacking requires at least 2 base models")
    
    # Define meta model (final estimator)
    if meta_model == "Random Forest":
        meta_defaults = {'n_estimators': 100, 'warm_start': True, 'random_state': random_state}
        final_estimator = RandomForestClassifier(**{**meta_defaults, **meta_model_params_config})
    elif meta_model == "Logistic Regression":
        meta_defaults = {'max_iter': 200, 'random_state': random_state}
        final_estimator = LogisticRegression(**{**meta_defaults, **meta_model_params_config})
    elif meta_model == "XGBoost":
        meta_defaults = {'n_estimators': 100, 'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': random_state}
        final_estimator = xgb.XGBClassifier(**{**meta_defaults, **meta_model_params_config})
    else:
        # Default to Random Forest
        final_estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    try:
        # Create Stacking Classifier
        # stack_method='predict' as per paper (not predict_proba)
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            stack_method='predict',
            cv=5  # 5-fold CV for generating meta-features
        )
        
        # Train the stacking model
        stacking_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = stacking_model.predict(X_test)
        y_pred_proba = stacking_model.predict_proba(X_test)[:, 1] if hasattr(stacking_model, "predict_proba") else None
        
        # Predictions on train set (for overfitting detection)
        y_train_pred = stacking_model.predict(X_train)
        y_train_proba = stacking_model.predict_proba(X_train)[:, 1] if hasattr(stacking_model, "predict_proba") else None
        
        # Calculate train metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'auc': roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else 0.5
        }
        
        # Calculate test metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5
        }
        
        # Calculate metrics (primary = test metrics)
        metrics = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'base_models': base_models,
            'meta_model': meta_model,
            # For overfitting detection
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'valid_metrics': None
        }
        
        return stacking_model, metrics
        
    except Exception as e:
        raise Exception(f"Error training Stacking model: {str(e)}")


def tune_stacking_with_oof(X_train, y_train, X_test, y_test,
                           base_models_config: dict,
                           meta_model: str,
                           tuning_method: str = "Grid Search",
                           n_folds: int = 5,
                           params=None):
    """
    Tune Stacking với OOF (Out-of-Fold) để tránh data leakage.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    base_models_config : dict
        Config cho từng base model với param ranges, ví dụ:
        {
            'LR': {'C': [0.1, 1.0, 10.0], 'max_iter': [200]},
            'DT': {'max_depth': [5, 10, 15]},
        }
    meta_model : str
        Name of meta model
    tuning_method : str
        'Grid Search', 'Random Search', or 'Default' (no tuning)
    n_folds : int
        Number of folds for CV
        
    Returns:
    --------
    stacking_model, metrics, tuning_info
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
    import warnings
    warnings.filterwarnings('ignore')
    
    if params is None:
        params = {}
    
    random_state = params.get('random_state', 42)
    
    # Base model class mapping
    base_model_classes = {
        'LR': LogisticRegression,
        'DT': DecisionTreeClassifier,
        'SVM': SVC,
        'KNN': KNeighborsClassifier,
        'RF': RandomForestClassifier,
        'GB': GradientBoostingClassifier
    }
    
    # Default params for each model
    default_params = {
        'LR': {'max_iter': 200, 'random_state': random_state},
        'DT': {'random_state': random_state},
        'SVM': {'probability': True, 'random_state': random_state},
        'KNN': {},
        'RF': {'random_state': random_state},
        'GB': {'random_state': random_state}
    }
    
    tuning_results = {}
    best_params_per_model = {}
    
    try:
        # Step 1: Tune each base model using CV on training data
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for model_key, param_grid in base_models_config.items():
            if model_key not in base_model_classes:
                continue
            
            model_class = base_model_classes[model_key]
            base_params = default_params.get(model_key, {})
            
            if tuning_method == "Default" or not param_grid:
                # No tuning, use default or provided fixed params
                if isinstance(param_grid, dict) and not any(isinstance(v, list) for v in param_grid.values()):
                    # Fixed params provided
                    best_params_per_model[model_key] = {**base_params, **param_grid}
                else:
                    best_params_per_model[model_key] = base_params
                tuning_results[model_key] = {'best_params': best_params_per_model[model_key], 'best_score': None}
                
            elif tuning_method == "Grid Search":
                model = model_class(**base_params)
                grid_search = GridSearchCV(
                    model, param_grid, cv=kfold, scoring='roc_auc', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_params_per_model[model_key] = {**base_params, **grid_search.best_params_}
                tuning_results[model_key] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                
            elif tuning_method == "Random Search":
                model = model_class(**base_params)
                random_search = RandomizedSearchCV(
                    model, param_grid, cv=kfold, scoring='roc_auc', 
                    n_iter=min(10, np.prod([len(v) if isinstance(v, list) else 1 for v in param_grid.values()])),
                    n_jobs=-1, random_state=random_state
                )
                random_search.fit(X_train, y_train)
                best_params_per_model[model_key] = {**base_params, **random_search.best_params_}
                tuning_results[model_key] = {
                    'best_params': random_search.best_params_,
                    'best_score': random_search.best_score_
                }
        
        # Step 2: Generate OOF predictions using tuned base models
        n_samples = len(X_train)
        n_models = len(base_models_config)
        oof_predictions = np.zeros((n_samples, n_models))
        oof_probabilities = np.zeros((n_samples, n_models))
        
        model_order = list(base_models_config.keys())
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            
            for model_idx, model_key in enumerate(model_order):
                if model_key not in base_model_classes:
                    continue
                    
                model_class = base_model_classes[model_key]
                model_params = best_params_per_model.get(model_key, {})
                model = model_class(**model_params)
                model.fit(X_tr, y_tr)
                
                oof_predictions[val_idx, model_idx] = model.predict(X_val)
                if hasattr(model, 'predict_proba'):
                    oof_probabilities[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
        
        # Step 3: Tune meta model using OOF predictions as features
        meta_model_params = params.get('meta_model_params', {})
        best_meta_params = {}
        
        # Define base meta model class and params
        meta_base_params = {'random_state': random_state}
        
        if meta_model == "Random Forest":
            meta_model_class = RandomForestClassifier
            meta_base_params['warm_start'] = True
        elif meta_model == "Logistic Regression":
            meta_model_class = LogisticRegression
            meta_base_params['max_iter'] = 200
        elif meta_model == "XGBoost":
            meta_model_class = xgb.XGBClassifier
            meta_base_params['use_label_encoder'] = False
            meta_base_params['eval_metric'] = 'logloss'
        else:
            meta_model_class = RandomForestClassifier
        
        # Tune meta model on OOF predictions
        if meta_model_params and tuning_method != "Default":
            meta_model_instance = meta_model_class(**meta_base_params)
            
            if tuning_method == "Grid Search":
                meta_grid_search = GridSearchCV(
                    meta_model_instance, meta_model_params, 
                    cv=kfold, scoring='roc_auc', n_jobs=-1
                )
                meta_grid_search.fit(oof_predictions, y_train)
                best_meta_params = {**meta_base_params, **meta_grid_search.best_params_}
                tuning_results['META_MODEL'] = {
                    'best_params': meta_grid_search.best_params_,
                    'best_score': meta_grid_search.best_score_
                }
            elif tuning_method == "Random Search":
                meta_random_search = RandomizedSearchCV(
                    meta_model_instance, meta_model_params, 
                    cv=kfold, scoring='roc_auc',
                    n_iter=min(10, np.prod([len(v) if isinstance(v, list) else 1 for v in meta_model_params.values()])),
                    n_jobs=-1, random_state=random_state
                )
                meta_random_search.fit(oof_predictions, y_train)
                best_meta_params = {**meta_base_params, **meta_random_search.best_params_}
                tuning_results['META_MODEL'] = {
                    'best_params': meta_random_search.best_params_,
                    'best_score': meta_random_search.best_score_
                }
        else:
            best_meta_params = meta_base_params
            tuning_results['META_MODEL'] = {'best_params': meta_base_params, 'best_score': None}
        
        # Step 4: Build final stacking model with tuned params
        estimators = []
        for model_key in model_order:
            if model_key in base_model_classes:
                model_class = base_model_classes[model_key]
                model_params = best_params_per_model.get(model_key, {})
                estimators.append((model_key, model_class(**model_params)))
        
        # Create final estimator with tuned params
        final_estimator = meta_model_class(**best_meta_params)
        
        # Create Stacking Classifier
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            stack_method='predict',
            cv=n_folds
        )
        
        # Train stacking model
        stacking_model.fit(X_train, y_train)
        
        # Predictions on test set
        y_pred = stacking_model.predict(X_test)
        y_pred_proba = stacking_model.predict_proba(X_test)[:, 1] if hasattr(stacking_model, "predict_proba") else None
        
        # Predictions on train set (for overfitting detection)
        y_train_pred = stacking_model.predict(X_train)
        y_train_proba = stacking_model.predict_proba(X_train)[:, 1] if hasattr(stacking_model, "predict_proba") else None
        
        # Calculate train metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'auc': roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else 0.5
        }
        
        # Calculate test metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5
        }
        
        # Calculate metrics (primary = test metrics)
        metrics = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'base_models': model_order,
            'meta_model': meta_model,
            # For overfitting detection
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'valid_metrics': None
        }
        
        tuning_info = {
            'tuning_method': tuning_method,
            'n_folds': n_folds,
            'best_params_per_model': best_params_per_model,
            'tuning_results': tuning_results
        }
        
        return stacking_model, metrics, tuning_info
        
    except Exception as e:
        raise Exception(f"Error tuning Stacking model: {str(e)}")


def cross_validate_model(X, y, model_type, params=None, cv_folds=5):
    """
    Perform cross-validation on a model.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_type : str
        Type of model to train
    params : dict
        Model parameters
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    cv_results : dict
        Dictionary containing CV scores and statistics
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    if params is None:
        params = {}
    
    model = None
    
    try:
        if model_type == "Logistic Regression":
            model = LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 200),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42)
            )
            
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        elif model_type == "LightGBM":
            model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', -1),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
            
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(
                iterations=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                depth=params.get('max_depth', 6),
                subsample=params.get('subsample', 1.0),
                random_state=params.get('random_state', 42),
                verbose=0,
                allow_writing_files=False
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Stratified K-Fold for classification
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores for multiple metrics
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
        auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        cv_results = {
            'cv_folds': cv_folds,
            'accuracy': {
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'scores': accuracy_scores.tolist()
            },
            'precision': {
                'mean': precision_scores.mean(),
                'std': precision_scores.std(),
                'scores': precision_scores.tolist()
            },
            'recall': {
                'mean': recall_scores.mean(),
                'std': recall_scores.std(),
                'scores': recall_scores.tolist()
            },
            'f1': {
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'scores': f1_scores.tolist()
            },
            'auc': {
                'mean': auc_scores.mean(),
                'std': auc_scores.std(),
                'scores': auc_scores.tolist()
            }
        }
        
        return cv_results
        
    except Exception as e:
        raise Exception(f"Error in cross-validation for {model_type}: {str(e)}")


def hyperparameter_tuning(X, y, model_type, method="Grid Search", cv_folds=5, n_trials=50):
    """
    Perform hyperparameter tuning on a model.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_type : str
        Type of model to tune
    method : str
        Tuning method: "Grid Search", "Random Search", "Optuna", or "Bayesian Optimization"
    cv_folds : int
        Number of cross-validation folds
    n_trials : int
        Number of trials for Optuna (default 50)
        
    Returns:
    --------
    tuning_results : dict
        Dictionary containing best parameters and scores
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
    import warnings
    warnings.filterwarnings('ignore')
    
    # Use Optuna for Bayesian Optimization or Optuna method
    if method in ["Optuna", "Bayesian Optimization"]:
        return optuna_hyperparameter_tuning(X, y, model_type, cv_folds, n_trials)
    
    try:
        # Define parameter grids for each model type
        if model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [100, 200, 500]
            }
            param_distributions = {
                'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                'max_iter': [100, 200, 300, 500]
            }
            
        elif model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
        elif model_type == "XGBoost":
            model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
        elif model_type == "LightGBM":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 10, -1],
                'num_leaves': [15, 31, 63, 127],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
            
        elif model_type == "CatBoost":
            model = cb.CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
            param_grid = {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            param_distributions = {
                'iterations': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'depth': [3, 4, 5, 6, 7, 8],
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Choose search method
        if method == "Grid Search":
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=skf,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
        elif method == "Random Search":
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=20,
                cv=skf,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported tuning method: {method}")
        
        # Fit the search
        search.fit(X, y)
        
        # Get results
        cv_results_df = pd.DataFrame(search.cv_results_)
        
        # Top 5 results
        top_results = cv_results_df.nsmallest(5, 'rank_test_score')[[
            'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
        ]].to_dict('records')
        
        tuning_results = {
            'method': method,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'top_results': top_results,
            'total_fits': len(cv_results_df)
        }
        
        return tuning_results
        
    except Exception as e:
        raise Exception(f"Error in hyperparameter tuning for {model_type}: {str(e)}")


def optuna_hyperparameter_tuning(X, y, model_type, cv_folds=5, n_trials=50):
    """
    Perform hyperparameter tuning using Optuna (Bayesian Optimization).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series  
        Target
    model_type : str
        Type of model to tune
    cv_folds : int
        Number of cross-validation folds
    n_trials : int
        Number of Optuna trials
        
    Returns:
    --------
    tuning_results : dict
        Dictionary containing best parameters and scores
    """
    try:
        import optuna
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Store all trial results
    all_trials = []
    
    def objective(trial):
        """Optuna objective function"""
        
        if model_type == "Logistic Regression":
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': 42
            }
            model = LogisticRegression(**params)
            
        elif model_type == "Random Forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
            
        elif model_type == "Gradient Boosting":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
            
        elif model_type == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_type == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': 42,
                'verbose': -1
            }
            model = lgb.LGBMClassifier(**params)
            
        elif model_type == "CatBoost":
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42,
                'verbose': 0,
                'allow_writing_files': False
            }
            model = cb.CatBoostClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Store trial info
        all_trials.append({
            'params': params.copy(),
            'mean_test_score': mean_score,
            'std_test_score': std_score
        })
        
        return mean_score
    
    # Create and run study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Get best params (clean up non-tunable params)
    best_params = study.best_params.copy()
    
    # Sort trials by score
    all_trials_sorted = sorted(all_trials, key=lambda x: x['mean_test_score'], reverse=True)
    
    # Format top results to match other methods
    top_results = []
    for i, trial in enumerate(all_trials_sorted[:5]):
        top_results.append({
            'params': trial['params'],
            'mean_test_score': trial['mean_test_score'],
            'std_test_score': trial['std_test_score'],
            'rank_test_score': i + 1
        })
    
    tuning_results = {
        'method': 'Optuna (TPE Bayesian)',
        'best_params': best_params,
        'best_score': study.best_value,
        'best_estimator': None,  # Will be trained separately
        'top_results': top_results,
        'total_fits': n_trials
    }
    
    return tuning_results

