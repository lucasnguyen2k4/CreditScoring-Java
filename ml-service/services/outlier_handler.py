"""
Outlier Handler - Backend xử lý outliers cho dữ liệu
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional


class OutlierHandler:
    """
    Class xử lý outliers với nhiều phương pháp khác nhau
    """
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.outlier_info = {}
        
    def detect_outliers_iqr(self, data: pd.Series, multiplier: float = 1.5) -> Dict:
        """
        Phát hiện outliers bằng phương pháp IQR
        
        Args:
            data: Series dữ liệu cần kiểm tra
            multiplier: Hệ số nhân cho IQR (mặc định 1.5)
            
        Returns:
            Dict chứa thông tin outliers
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers_mask = (data < lower_bound) | (data > upper_bound)
        outliers_count = outliers_mask.sum()
        
        return {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_count': outliers_count,
            'outliers_percentage': (outliers_count / len(data) * 100) if len(data) > 0 else 0,
            'outliers_mask': outliers_mask,
            'outliers_indices': data[outliers_mask].index.tolist(),
            'outliers_values': data[outliers_mask].values.tolist()
        }
    
    def detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> Dict:
        """
        Phát hiện outliers bằng phương pháp Z-Score
        
        Args:
            data: Series dữ liệu cần kiểm tra
            threshold: Ngưỡng Z-score (mặc định 3.0)
            
        Returns:
            Dict chứa thông tin outliers
        """
        z_scores = np.abs(stats.zscore(data.dropna()))
        outliers_mask = pd.Series(False, index=data.index)
        outliers_mask[data.dropna().index] = z_scores > threshold
        
        outliers_count = outliers_mask.sum()
        
        return {
            'threshold': threshold,
            'mean': data.mean(),
            'std': data.std(),
            'outliers_count': outliers_count,
            'outliers_percentage': (outliers_count / len(data) * 100) if len(data) > 0 else 0,
            'outliers_mask': outliers_mask,
            'outliers_indices': data[outliers_mask].index.tolist(),
            'outliers_values': data[outliers_mask].values.tolist(),
            'z_scores': z_scores
        }
    
    def handle_outliers_winsorization(
        self, 
        data: pd.DataFrame, 
        columns: List[str], 
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Xử lý outliers bằng phương pháp Winsorization
        
        Args:
            data: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý outliers
            lower_percentile: Phân vị dưới (mặc định 0.05 = 5%)
            upper_percentile: Phân vị trên (mặc định 0.95 = 95%)
            
        Returns:
            Tuple (DataFrame đã xử lý, Dict thông tin xử lý)
        """
        processed_data = data.copy()
        info = {}
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Tính các phân vị
            lower_bound = data[col].quantile(lower_percentile)
            upper_bound = data[col].quantile(upper_percentile)
            
            # Đếm số outliers
            outliers_lower = (data[col] < lower_bound).sum()
            outliers_upper = (data[col] > upper_bound).sum()
            total_outliers = outliers_lower + outliers_upper
            
            # Thực hiện Winsorization
            processed_data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Lưu thông tin
            info[col] = {
                'method': 'Winsorization',
                'lower_percentile': lower_percentile,
                'upper_percentile': upper_percentile,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_lower': outliers_lower,
                'outliers_upper': outliers_upper,
                'total_outliers': total_outliers,
                'outliers_percentage': (total_outliers / len(data) * 100) if len(data) > 0 else 0,
                'original_min': data[col].min(),
                'original_max': data[col].max(),
                'new_min': processed_data[col].min(),
                'new_max': processed_data[col].max()
            }
        
        return processed_data, info
    
    def handle_outliers_iqr(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        multiplier: float = 1.5,
        method: str = 'clip'  # 'clip', 'remove', 'nan'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Xử lý outliers bằng phương pháp IQR
        
        Args:
            data: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý outliers
            multiplier: Hệ số nhân cho IQR (mặc định 1.5)
            method: Phương pháp xử lý ('clip': cắt giá trị, 'remove': xóa dòng, 'nan': thay bằng NaN)
            
        Returns:
            Tuple (DataFrame đã xử lý, Dict thông tin xử lý)
        """
        processed_data = data.copy()
        info = {}
        removed_indices = set()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Phát hiện outliers
            outlier_info = self.detect_outliers_iqr(data[col], multiplier)
            
            # Xử lý outliers theo phương pháp được chọn
            if method == 'clip':
                processed_data[col] = data[col].clip(
                    lower=outlier_info['lower_bound'],
                    upper=outlier_info['upper_bound']
                )
            elif method == 'remove':
                removed_indices.update(outlier_info['outliers_indices'])
            elif method == 'nan':
                processed_data.loc[outlier_info['outliers_mask'], col] = np.nan
            
            # Lưu thông tin
            info[col] = {
                'method': f'IQR Method ({method})',
                'multiplier': multiplier,
                'Q1': outlier_info['Q1'],
                'Q3': outlier_info['Q3'],
                'IQR': outlier_info['IQR'],
                'lower_bound': outlier_info['lower_bound'],
                'upper_bound': outlier_info['upper_bound'],
                'outliers_count': outlier_info['outliers_count'],
                'outliers_percentage': outlier_info['outliers_percentage'],
                'original_min': data[col].min(),
                'original_max': data[col].max()
            }
            
            if method == 'clip':
                info[col]['new_min'] = processed_data[col].min()
                info[col]['new_max'] = processed_data[col].max()
        
        # Nếu method là 'remove', xóa các dòng có outliers
        if method == 'remove' and removed_indices:
            processed_data = processed_data.drop(index=list(removed_indices))
            for col in info:
                info[col]['removed_rows'] = len(removed_indices)
        
        return processed_data, info
    
    def handle_outliers_zscore(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        threshold: float = 3.0,
        method: str = 'clip'  # 'clip', 'remove', 'nan'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Xử lý outliers bằng phương pháp Z-Score
        
        Args:
            data: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý outliers
            threshold: Ngưỡng Z-score (mặc định 3.0)
            method: Phương pháp xử lý ('clip', 'remove', 'nan')
            
        Returns:
            Tuple (DataFrame đã xử lý, Dict thông tin xử lý)
        """
        processed_data = data.copy()
        info = {}
        removed_indices = set()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Phát hiện outliers
            outlier_info = self.detect_outliers_zscore(data[col], threshold)
            
            # Tính bounds dựa trên mean và std
            mean = outlier_info['mean']
            std = outlier_info['std']
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Xử lý outliers theo phương pháp được chọn
            if method == 'clip':
                processed_data[col] = data[col].clip(
                    lower=lower_bound,
                    upper=upper_bound
                )
            elif method == 'remove':
                removed_indices.update(outlier_info['outliers_indices'])
            elif method == 'nan':
                processed_data.loc[outlier_info['outliers_mask'], col] = np.nan
            
            # Lưu thông tin
            info[col] = {
                'method': f'Z-Score ({method})',
                'threshold': threshold,
                'mean': mean,
                'std': std,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': outlier_info['outliers_count'],
                'outliers_percentage': outlier_info['outliers_percentage'],
                'original_min': data[col].min(),
                'original_max': data[col].max()
            }
            
            if method == 'clip':
                info[col]['new_min'] = processed_data[col].min()
                info[col]['new_max'] = processed_data[col].max()
        
        # Nếu method là 'remove', xóa các dòng có outliers
        if method == 'remove' and removed_indices:
            processed_data = processed_data.drop(index=list(removed_indices))
            for col in info:
                info[col]['removed_rows'] = len(removed_indices)
        
        return processed_data, info
    
    def keep_all(self, data: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Giữ nguyên tất cả dữ liệu, không xử lý outliers
        
        Args:
            data: DataFrame
            columns: Danh sách cột (chỉ để thống nhất interface)
            
        Returns:
            Tuple (DataFrame nguyên gốc, Dict thông tin)
        """
        info = {}
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Phát hiện outliers để thông tin
            outlier_info = self.detect_outliers_iqr(data[col])
            
            info[col] = {
                'method': 'Keep All (No Processing)',
                'outliers_detected': outlier_info['outliers_count'],
                'outliers_percentage': outlier_info['outliers_percentage'],
                'action': 'No action taken - all data preserved',
                'min': data[col].min(),
                'max': data[col].max()
            }
        
        return data.copy(), info
    
    def apply_outlier_handling(
        self,
        data: pd.DataFrame,
        method: str,
        columns: List[str],
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Áp dụng phương pháp xử lý outliers được chọn
        
        Args:
            data: DataFrame cần xử lý
            method: Phương pháp ('Winsorization', 'IQR Method', 'Z-Score', 'Keep All')
            columns: Danh sách cột cần xử lý
            **kwargs: Tham số bổ sung cho từng phương pháp
            
        Returns:
            Tuple (DataFrame đã xử lý, Dict thông tin xử lý)
        """
        self.original_data = data.copy()
        
        if method == "Winsorization":
            lower_percentile = kwargs.get('lower_percentile', 0.05)
            upper_percentile = kwargs.get('upper_percentile', 0.95)
            self.processed_data, self.outlier_info = self.handle_outliers_winsorization(
                data, columns, lower_percentile, upper_percentile
            )
            
        elif method == "IQR Method":
            multiplier = kwargs.get('multiplier', 1.5)
            action = kwargs.get('action', 'clip')  # clip, remove, nan
            self.processed_data, self.outlier_info = self.handle_outliers_iqr(
                data, columns, multiplier, action
            )
            
        elif method == "Z-Score":
            threshold = kwargs.get('threshold', 3.0)
            action = kwargs.get('action', 'clip')  # clip, remove, nan
            self.processed_data, self.outlier_info = self.handle_outliers_zscore(
                data, columns, threshold, action
            )
            
        elif method == "Keep All":
            self.processed_data, self.outlier_info = self.keep_all(data, columns)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.processed_data, self.outlier_info
    
    def get_summary_report(self) -> str:
        """
        Tạo báo cáo tóm tắt về xử lý outliers
        
        Returns:
            String báo cáo
        """
        if not self.outlier_info:
            return "No outlier processing has been performed yet."
        
        report = "=" * 60 + "\n"
        report += "OUTLIER PROCESSING SUMMARY REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for col, info in self.outlier_info.items():
            report += f"Column: {col}\n"
            report += f"Method: {info['method']}\n"
            
            if 'outliers_count' in info:
                report += f"Outliers Found: {info['outliers_count']} ({info['outliers_percentage']:.2f}%)\n"
            
            if 'lower_bound' in info and 'upper_bound' in info:
                report += f"Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]\n"
            
            if 'original_min' in info and 'original_max' in info:
                report += f"Original Range: [{info['original_min']:.2f}, {info['original_max']:.2f}]\n"
            
            if 'new_min' in info and 'new_max' in info:
                report += f"New Range: [{info['new_min']:.2f}, {info['new_max']:.2f}]\n"
            
            if 'removed_rows' in info:
                report += f"Removed Rows: {info['removed_rows']}\n"
            
            report += "-" * 60 + "\n"
        
        return report


# Convenience functions for easy import
def handle_outliers(
    data: pd.DataFrame,
    method: str,
    columns: List[str],
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function để xử lý outliers
    
    Args:
        data: DataFrame cần xử lý
        method: Phương pháp xử lý
        columns: Danh sách cột
        **kwargs: Tham số bổ sung
        
    Returns:
        Tuple (DataFrame đã xử lý, Dict thông tin)
    """
    handler = OutlierHandler()
    return handler.apply_outlier_handling(data, method, columns, **kwargs)
