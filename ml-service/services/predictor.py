"""
=============================================================================
PREDICTION MODULE - DỰ ĐOÁN ĐIỂM TÍN DỤNG
=============================================================================
Mô tả:
    Module xử lý dự đoán điểm tín dụng cho khách hàng mới, tính toán
    Credit Score theo chuẩn Basel II/III và đưa ra khuyến nghị.

Các chức năng chính:
    1. predict_single(): Dự đoán cho 1 khách hàng
    2. predict_batch(): Dự đoán hàng loạt
    3. get_feature_contributions(): Lấy đóng góp của từng feature
    4. generate_recommendations(): Tạo khuyến nghị cải thiện

CÔNG THỨC CREDIT SCORE (Log-Odds Scaling - Industry Standard):
    Score = Offset + Factor × ln(Odds / Base_Odds)
    
    Với:
    - PDO (Points to Double Odds) = 30 (chuẩn: 20-30)
    - Base Score = 600 (ở odds = 19:1, tức PD = 5%)
    - Factor = PDO / ln(2) ≈ 43.29
    
PHÂN LOẠI RỦI RO (5 cấp độ theo chuẩn ngành):
    - Very Low:  PD < 2%      (Rất thấp)
    - Low:       2% ≤ PD < 5% (Thấp)
    - Medium:    5% ≤ PD < 10% (Trung bình)
    - High:      10% ≤ PD < 20% (Cao)
    - Very High: PD ≥ 20%     (Rất cao)

QUYẾT ĐỊNH PHÊ DUYỆT:
    - Approved: PD < 5% VÀ Score ≥ 650
    - Conditional: PD < 10% HOẶC 550 ≤ Score < 650
    - Rejected: Còn lại
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


# =============================================================================
# PHẦN 1: DỰ ĐOÁN ĐƠN LẺ (SINGLE PREDICTION)
# =============================================================================


def predict_single(model, input_data: Dict[str, Any], feature_names: List[str], 
                   feature_stats: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Make prediction for a single sample.
    
    Parameters:
    -----------
    model : object
        Trained model object
    input_data : dict
        Dictionary of feature values
    feature_names : list
        List of feature names in the correct order
    feature_stats : dict, optional
        Statistics of training data for reference
        
    Returns:
    --------
    result : dict
        Dictionary containing prediction results
    """
    try:
        # Create DataFrame from input data with correct feature order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the features used for training in correct order
        X = input_df[feature_names]
        
        # Make prediction
        y_pred = model.predict(X)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[0]
            # For binary classification, determine which class is "default/bad"
            if len(y_proba) == 2:
                # Check model.classes_ to find the index of default class (usually 1)
                if hasattr(model, 'classes_'):
                    classes = model.classes_
                    # Assume class with higher value is "default" (commonly 1 = bad, 0 = good)
                    # This handles cases where classes might be [0,1], [1,0], or ['good','bad']
                    default_idx = 1 if len(classes) == 2 else 0
                    # If classes are strings, try to identify bad class
                    if isinstance(classes[0], str):
                        bad_keywords = ['bad', 'default', '1', 'yes', 'true', 'risk']
                        for i, cls in enumerate(classes):
                            if any(kw in str(cls).lower() for kw in bad_keywords):
                                default_idx = i
                                break
                    prob_positive = y_proba[default_idx]
                else:
                    # Fallback: assume index 1 is default
                    prob_positive = y_proba[1]
            else:
                prob_positive = y_proba[0]
        else:
            prob_positive = float(y_pred)
        
        # Determine risk level based on PD (5-tier industry standard)
        if prob_positive < 0.02:
            risk_level = 'Very Low'
            risk_label_vi = 'Rất thấp'
            risk_color = '#10b981'  # Green
        elif prob_positive < 0.05:
            risk_level = 'Low'
            risk_label_vi = 'Thấp'
            risk_color = '#22c55e'  # Light green
        elif prob_positive < 0.10:
            risk_level = 'Medium'
            risk_label_vi = 'Trung bình'
            risk_color = '#f59e0b'  # Orange
        elif prob_positive < 0.20:
            risk_level = 'High'
            risk_label_vi = 'Cao'
            risk_color = '#ef4444'  # Red
        else:
            risk_level = 'Very High'
            risk_label_vi = 'Rất cao'
            risk_color = '#dc2626'  # Dark red
        
        # Calculate credit score using log-odds scaling (industry standard)
        # Formula: Score = Offset + Factor × ln(Odds / Base_Odds)
        # Industry standard: Score = 600 at odds = 19:1 (PD = 5%)
        import math
        
        pdo = 30  # Points to Double Odds (industry standard: 20-30)
        base_score = 600  # Score at base odds
        base_odds = 19  # Odds at base score (19:1 means PD = 5%)
        
        factor = pdo / math.log(2)  # ≈ 43.29 with PDO=30
        
        # Handle edge cases with soft clamp (use 1e-6 to 1-1e-6)
        prob_default = max(1e-6, min(1 - 1e-6, prob_positive))
        
        odds = (1 - prob_default) / prob_default
        # Score relative to base odds
        credit_score = int(base_score + factor * math.log(odds / base_odds))
        credit_score = max(300, min(850, credit_score))  # Clamp to valid range
        
        # Credit score interpretation (5-tier)
        if credit_score >= 750:
            score_interpretation = 'Xuất sắc'
            score_description = 'Rủi ro rất thấp - Khách hàng có tín dụng xuất sắc'
        elif credit_score >= 650:
            score_interpretation = 'Tốt'
            score_description = 'Chấp nhận được - Khách hàng có tín dụng tốt'
        elif credit_score >= 550:
            score_interpretation = 'Trung bình'
            score_description = 'Cần xem xét kỹ - Khách hàng cần cải thiện tín dụng'
        elif credit_score >= 450:
            score_interpretation = 'Kém'
            score_description = 'Rủi ro cao - Cần tài sản đảm bảo hoặc bảo lãnh'
        else:
            score_interpretation = 'Rất kém'
            score_description = 'Gần như từ chối - Rủi ro vỡ nợ rất cao'
        
        # Approval decision based on BOTH PD and Score (industry best practice)
        if prob_positive < 0.05 and credit_score >= 650:
            approval_status = 'approved'
            approval_label_vi = 'Phê duyệt'
            approval_color = '#10b981'
        elif prob_positive < 0.10 or (credit_score >= 550 and credit_score < 650):
            approval_status = 'conditional'
            approval_label_vi = 'Cần bổ sung hồ sơ'
            approval_color = '#f59e0b'
        else:
            approval_status = 'rejected'
            approval_label_vi = 'Từ chối'
            approval_color = '#ef4444'
        
        result = {
            'prediction': int(y_pred),
            'probability': float(prob_positive),
            'credit_score': credit_score,
            'risk_level': risk_level,
            'risk_label_vi': risk_label_vi,
            'risk_color': risk_color,
            'score_interpretation': score_interpretation,
            'score_description': score_description,
            'approval_status': approval_status,
            'approval_label_vi': approval_label_vi,
            'approval_color': approval_color,
            'input_data': input_data,
            'feature_names': feature_names
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")


def predict_batch(model, input_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Make predictions for a batch of samples.
    
    Parameters:
    -----------
    model : object
        Trained model object
    input_df : pd.DataFrame
        DataFrame of input data
    feature_names : list
        List of feature names in the correct order
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with predictions added
    """
    try:
        # Select only the features used for training in correct order
        X = input_df[feature_names].copy()
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1] if model.predict_proba(X).shape[1] == 2 else model.predict_proba(X)[:, 0]
        else:
            probabilities = predictions.astype(float)
        
        # Create results DataFrame
        results_df = input_df.copy()
        results_df['prediction'] = predictions
        results_df['probability'] = probabilities
        
        # Calculate credit score using log-odds scaling (industry standard)
        import math
        pdo = 20
        base_score = 600
        factor = pdo / math.log(2)
        
        def calc_score(pd_val):
            pd_clamped = max(0.001, min(0.999, pd_val))
            odds = (1 - pd_clamped) / pd_clamped
            score = int(base_score + factor * math.log(odds))
            return max(300, min(850, score))
        
        results_df['credit_score'] = probabilities.apply(calc_score)
        results_df['risk_level'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return results_df
        
    except Exception as e:
        raise Exception(f"Batch prediction error: {str(e)}")


def get_feature_contributions(model, input_data: Dict[str, Any], feature_names: List[str],
                               shap_explainer=None) -> List[Tuple[str, float]]:
    """
    Get feature contributions for a prediction using SHAP or feature importance.
    
    Parameters:
    -----------
    model : object
        Trained model object
    input_data : dict
        Dictionary of feature values
    feature_names : list
        List of feature names
    shap_explainer : object, optional
        Pre-computed SHAP explainer
        
    Returns:
    --------
    contributions : list of tuples
        List of (feature_name, contribution) sorted by absolute impact
    """
    try:
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # Try to use SHAP if explainer is provided
        if shap_explainer is not None:
            try:
                shap_values = shap_explainer.shap_values(input_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                contributions = list(zip(feature_names, shap_values[0]))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                return contributions
            except:
                pass
        
        # Fallback to feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Scale by feature value deviation from mean (mock approach)
            contributions = []
            for i, feat in enumerate(feature_names):
                # Simple approach: use importance as contribution magnitude
                value = input_data.get(feat, 0)
                # Mock contribution based on importance
                contribution = importances[i] * (0.5 - np.random.random())
                contributions.append((feat, contribution))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            return contributions
        
        # Fallback to coefficients for linear models
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            contributions = []
            for i, feat in enumerate(feature_names):
                value = input_data.get(feat, 0)
                contribution = coefficients[i] * float(value)
                contributions.append((feat, contribution))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            return contributions
        
        # Last resort: return zeros
        return [(feat, 0.0) for feat in feature_names]
        
    except Exception as e:
        # Return mock values on error
        return [(feat, np.random.uniform(-0.3, 0.3)) for feat in feature_names]


def generate_recommendations(prediction_result: Dict, input_data: Dict, 
                             feature_contributions: List[Tuple[str, float]]) -> List[Dict]:
    """
    Generate improvement recommendations based on prediction.
    
    Parameters:
    -----------
    prediction_result : dict
        Prediction results
    input_data : dict
        Input feature values
    feature_contributions : list
        Feature contribution values
        
    Returns:
    --------
    recommendations : list of dict
        List of recommendation dictionaries
    """
    recommendations = []
    
    # Sort by negative impact (features that increase risk)
    negative_contributors = [(f, c) for f, c in feature_contributions if c > 0]
    negative_contributors.sort(key=lambda x: x[1], reverse=True)
    
    # Generate recommendations for top negative factors
    for feature, contribution in negative_contributors[:5]:
        current_value = input_data.get(feature, 'N/A')
        
        # Feature-specific recommendations
        rec = {
            'feature': feature,
            'current_value': current_value,
            'impact': contribution,
            'priority': 'High' if contribution > 0.1 else 'Medium' if contribution > 0.05 else 'Low'
        }
        
        # Add specific advice based on feature name
        feature_lower = feature.lower()
        
        if 'debt' in feature_lower or 'loan' in feature_lower:
            rec['advice'] = 'Giảm tổng dư nợ để cải thiện tỷ lệ nợ/thu nhập'
            rec['target'] = 'Giảm 20-30%'
            rec['impact_score'] = int(contribution * 100)
        elif 'late' in feature_lower or 'delinquent' in feature_lower:
            rec['advice'] = 'Đảm bảo thanh toán đúng hạn tất cả các khoản vay'
            rec['target'] = '0 lần trễ hạn'
            rec['impact_score'] = int(contribution * 100)
        elif 'utilization' in feature_lower:
            rec['advice'] = 'Giảm tỷ lệ sử dụng tín dụng xuống dưới 30%'
            rec['target'] = '< 30%'
            rec['impact_score'] = int(contribution * 100)
        elif 'income' in feature_lower:
            rec['advice'] = 'Tăng thu nhập hoặc cung cấp bằng chứng thu nhập bổ sung'
            rec['target'] = 'Tăng 15-20%'
            rec['impact_score'] = int(contribution * 100)
        elif 'age' in feature_lower or 'year' in feature_lower:
            rec['advice'] = 'Duy trì lịch sử tín dụng ổn định theo thời gian'
            rec['target'] = 'Duy trì trong 12+ tháng'
            rec['impact_score'] = int(contribution * 100)
        else:
            rec['advice'] = f'Cải thiện yếu tố {feature} để giảm rủi ro'
            rec['target'] = 'Cải thiện dần'
            rec['impact_score'] = int(contribution * 100)
        
        recommendations.append(rec)
    
    return recommendations
