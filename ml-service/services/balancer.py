"""
Data Balancing Module
Handles imbalanced datasets using various resampling techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def balance_data(
    data: pd.DataFrame,
    target_column: str,
    method: str = "SMOTE",
    random_state: int = 42,
    sampling_strategy: str = "auto",
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Balance imbalanced dataset using various resampling techniques - Cân bằng dữ liệu
    
    Các phương pháp hỗ trợ:
    1. SMOTE (Synthetic Minority Over-sampling Technique):
       - Sinh mẫu tổng hợp cho class thiểu số
       - Dựa trên K-Nearest Neighbors để tạo mẫu mới nằm giữa các mẫu có sẵn
    
    1b. ADASYN (Adaptive Synthetic Sampling):
       - Tương tự SMOTE nhưng tập trung sinh mẫu ở vùng khó học (near decision boundary)

    2. Random Over-sampling:
       - Nhân bản ngẫu nhiên các mẫu của class thiểu số
       - Dễ gây overfitting nếu không cẩn thận

    3. Random Under-sampling:
       - Loại bỏ ngẫu nhiên các mẫu của class đa số
       - Giảm dữ liệu -> Mất thông tin quan trọng
       - Tốt cho dataset quá lớn

    4. SMOTE-ENN:
       - Kết hợp SMOTE + Edited Nearest Neighbors để vừa tăng vừa làm sạch dữ liệu

    5. SMOTE-Tomek:
       - Kết hợp SMOTE + Tomek links để loại bỏ các cặp điểm nhiễu giữa 2 lớp
       
    Parameters:
    -----------
    data : pd.DataFrame - Dataset đầu vào (bao gồm Feature + Target)
    target_column : str - Tên cột target
    method : str - Phương pháp ("SMOTE", "ADASYN", "SMOTE-ENN", "SMOTE-Tomek", "Random Over-sampling", "Random Under-sampling")
    
    Returns:
    --------
    balanced_data : pd.DataFrame - Dữ liệu đã cân bằng
    info : Dict - Thông tin thống kê trước/sau khi cân bằng
    """
    
    try:
        # Validate inputs
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Normalize invalid numeric values before any resampling
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Handle categorical columns - encode them for SMOTE
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        label_encoders = {}
        
        if categorical_cols:
            from sklearn.preprocessing import LabelEncoder
            X_encoded = X.copy()
            for col in categorical_cols:
                le = LabelEncoder()
                # Handle NaN values
                X_encoded[col] = X_encoded[col].fillna('_MISSING_')
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
            X = X_encoded

        # Ensure SMOTE-family methods receive finite matrix (no NaN/Inf)
        # We impute remaining numeric NaN values with column median.
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        nan_fixes = {}
        for col in numeric_cols:
            missing_count = int(X[col].isna().sum())
            if missing_count > 0:
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X[col] = X[col].fillna(median_val)
                nan_fixes[col] = {
                    "filled_missing": missing_count,
                    "fill_strategy": "median",
                    "fill_value": float(median_val),
                }
        
        # Get original class distribution
        original_dist = y.value_counts().to_dict()
        
        info = {
            'method': method,
            'original_distribution': original_dist,
            'original_size': len(data)
        }
        if nan_fixes:
            info['nan_handling'] = {
                "columns_fixed": len(nan_fixes),
                "details": nan_fixes,
            }
        # Shared neighbor setting for SMOTE-like methods
        min_samples = y.value_counts().min()
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        
        # Random Over-sampling
        if method == "Random Over-sampling":
            from imblearn.over_sampling import RandomOverSampler
            
            ros = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
        # Random Under-sampling
        elif method == "Random Under-sampling":
            from imblearn.under_sampling import RandomUnderSampler
            
            rus = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
        # SMOTE
        elif method == "SMOTE":
            from imblearn.over_sampling import SMOTE
            
            if min_samples < 2:
                info['message'] = "Not enough samples for SMOTE. Using Random Over-sampling instead."
                from imblearn.over_sampling import RandomOverSampler
                ros = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
                X_resampled, y_resampled = ros.fit_resample(X, y)
            else:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=k_neighbors
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)

        # ADASYN
        elif method == "ADASYN":
            from imblearn.over_sampling import ADASYN, RandomOverSampler

            if min_samples < 2:
                info['message'] = "Not enough samples for ADASYN. Using Random Over-sampling instead."
                ros = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
                X_resampled, y_resampled = ros.fit_resample(X, y)
            else:
                adasyn = ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    n_neighbors=k_neighbors
                )
                X_resampled, y_resampled = adasyn.fit_resample(X, y)

        # SMOTE-ENN
        elif method == "SMOTE-ENN":
            from imblearn.combine import SMOTEENN
            from imblearn.over_sampling import SMOTE, RandomOverSampler

            if min_samples < 2:
                info['message'] = "Not enough samples for SMOTE-ENN. Using Random Over-sampling instead."
                ros = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
                X_resampled, y_resampled = ros.fit_resample(X, y)
            else:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=k_neighbors
                )
                smote_enn = SMOTEENN(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    smote=smote
                )
                X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        # SMOTE-Tomek
        elif method == "SMOTE-Tomek":
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE, RandomOverSampler

            if min_samples < 2:
                info['message'] = "Not enough samples for SMOTE-Tomek. Using Random Over-sampling instead."
                ros = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
                X_resampled, y_resampled = ros.fit_resample(X, y)
            else:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=k_neighbors
                )
                smote_tomek = SMOTETomek(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    smote=smote
                )
                X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Decode categorical columns back to original values
        if categorical_cols:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            for col in categorical_cols:
                le = label_encoders[col]
                X_resampled[col] = le.inverse_transform(X_resampled[col].astype(int))
                # Restore NaN values
                X_resampled[col] = X_resampled[col].replace('_MISSING_', np.nan)
        
        # Combine back into DataFrame
        balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
        
        # Get balanced class distribution
        balanced_dist = y_resampled.value_counts().to_dict()
        
        info['balanced_distribution'] = balanced_dist
        info['balanced_size'] = len(balanced_data)
        info['size_change'] = len(balanced_data) - len(data)
        info['message'] = f"Successfully balanced data using {method}"
        
        return balanced_data, info
        
    except ImportError as e:
        raise ImportError(
            f"imbalanced-learn library is required for data balancing. "
            f"Install it with: pip install imbalanced-learn. Error: {str(e)}"
        )
    except Exception as e:
        raise Exception(f"Error during data balancing: {str(e)}")


def get_class_distribution(data: pd.DataFrame, target_column: str) -> Dict:
    """
    Get class distribution for a target column
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target column
    
    Returns:
    --------
    distribution : Dict
        Dictionary containing class distribution info
    """
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    value_counts = data[target_column].value_counts()
    total = len(data)
    
    distribution = {
        'counts': value_counts.to_dict(),
        'percentages': {k: (v/total)*100 for k, v in value_counts.items()},
        'total': total,
        'n_classes': len(value_counts),
        'imbalance_ratio': value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1.0
    }
    
    return distribution


def check_imbalance(data: pd.DataFrame, target_column: str, threshold: float = 1.5) -> Dict:
    """
    Check if dataset is imbalanced
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target column
    threshold : float
        Imbalance ratio threshold (default: 1.5)
    
    Returns:
    --------
    result : Dict
        Dictionary containing imbalance check results
    """
    
    dist = get_class_distribution(data, target_column)
    
    is_imbalanced = dist['imbalance_ratio'] > threshold
    
    result = {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': dist['imbalance_ratio'],
        'threshold': threshold,
        'distribution': dist,
        'recommendation': _get_balancing_recommendation(dist['imbalance_ratio'])
    }
    
    return result


def _get_balancing_recommendation(imbalance_ratio: float) -> str:
    """Get balancing method recommendation based on imbalance ratio"""
    
    if imbalance_ratio < 1.5:
        return "No Balancing - Dataset is relatively balanced"
    elif imbalance_ratio < 3:
        return "SMOTE or Random Over-sampling - Mild imbalance"
    elif imbalance_ratio < 10:
        return "SMOTE recommended - Moderate imbalance"
    else:
        return "SMOTE + Random Under-sampling - Severe imbalance"
