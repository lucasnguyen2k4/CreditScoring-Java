"""
=============================================================================
PREPROCESSING PIPELINE - QUẢN LÝ TIỀN XỬ LÝ DỮ LIỆU
=============================================================================
Mô tả:
    Module quản lý toàn bộ quy trình tiền xử lý dữ liệu một cách nhất quán.
    
NGUYÊN TẮC QUAN TRỌNG - TRÁNH DATA LEAKAGE:
    - TẤT CẢ các transformer đều được FIT trên TRAIN DATA
    - Sau đó TRANSFORM trên Train, Validation, Test data
    - Điều này đảm bảo không có thông tin từ test/valid "rò rỉ" vào train

Các chức năng chính:
    1. fit_imputer() / transform_imputation(): Xử lý giá trị thiếu
    2. fit_scaler() / transform_scaling(): Chuẩn hóa dữ liệu số
    3. fit_encoder() / transform_encoding(): Mã hóa biến categorical
    4. fit_outlier_bounds() / transform_outliers(): Xử lý outliers

Workflow sử dụng:
    pipeline = PreprocessingPipeline()
    
    # Bước 1: FIT trên train data
    pipeline.fit_imputer(train_data, 'age', 'Mean Imputation')
    pipeline.fit_scaler(train_data, ['income', 'age'], 'StandardScaler')
    
    # Bước 2: TRANSFORM trên tất cả datasets
    train_processed = pipeline.transform_imputation(train_data, 'age')
    valid_processed = pipeline.transform_imputation(valid_data, 'age')
    test_processed = pipeline.transform_imputation(test_data, 'age')
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CLASS PREPROCESSING PIPELINE
# =============================================================================
class PreprocessingPipeline:
    """
    Class quản lý preprocessing pipeline.
    
    Mục đích:
        - Đảm bảo FIT trên train data, TRANSFORM trên tất cả datasets
        - Lưu trữ các transformer đã fit để tái sử dụng
        - Theo dõi các cột đã được xử lý
    
    Attributes:
        scalers: Dict lưu scalers đã fit cho từng nhóm cột {key: scaler_object}
        imputers: Dict lưu thông tin imputation cho từng cột {col: {method, value}}
        encoders: Dict lưu encoders đã fit cho từng cột {col: {encoder, mapping}}
        outlier_bounds: Dict lưu ngưỡng outliers cho từng cột {col: {lower, upper}}
        fitted_columns: Dict theo dõi cột nào đã được xử lý bước nào
        
    CƠ CHẾ HOẠT ĐỘNG:
    1. FIT: Tính toán tham số từ dữ liệu TRAIN (ví dụ: tính Mean, tính Min/Max)
    2. TRANSFORM: Áp dụng tham số đó vào dữ liệu (Train/Valid/Test)
    -> Đảm bảo tính nhất quán và tránh Data Leakage.
    """
    
    def __init__(self):
        """
        Khởi tạo PreprocessingPipeline.
        Tạo các dict rỗng để lưu trữ transformers và tracking.
        """
        # =====================================================================
        # LƯU TRỮ CÁC TRANSFORMERS ĐÃ FIT
        # =====================================================================
        self.scalers: Dict[str, Any] = {}           # {cols_key: {scaler, method, columns}}
        self.imputers: Dict[str, Dict] = {}         # {col: {method, fill_value}}
        self.encoders: Dict[str, Dict] = {}         # {col: {method, mapping/encoder}}
        self.outlier_bounds: Dict[str, Dict] = {}   # {col: {lower, upper, method}}
        
        # =====================================================================
        # THEO DÕI CÁC CỘT ĐÃ ĐƯỢC FIT
        # =====================================================================
        # Giúp biết cột nào đã được xử lý với loại transform nào
        self.fitted_columns: Dict[str, List[str]] = {
            'scaling': [],      # Các cột đã fit scaler
            'imputation': [],   # Các cột đã fit imputer
            'encoding': [],     # Các cột đã fit encoder
            'outliers': []      # Các cột đã fit outlier bounds
        }
    
    # =========================================================================
    # PHẦN 1: XỬ LÝ GIÁ TRỊ THIẾU (MISSING VALUE HANDLING)
    # =========================================================================
    # Các phương pháp hỗ trợ:
    # - Mean Imputation: Điền bằng giá trị trung bình
    # - Median Imputation: Điền bằng giá trị trung vị
    # - Mode Imputation: Điền bằng giá trị xuất hiện nhiều nhất
    # - Constant Value: Điền bằng giá trị cố định
    # - Forward Fill: Điền bằng giá trị trước đó
    # - Backward Fill: Điền bằng giá trị sau đó
    # - Interpolation: Nội suy giá trị
    # =========================================================================
    
    def fit_imputer(
        self, 
        train_data: pd.DataFrame, 
        column: str, 
        method: str,
        constant_value: Any = None
    ) -> Dict:
        """
        Fit imputer trên train data.
        
        Args:
            train_data: DataFrame train
            column: Tên cột
            method: Phương pháp (Mean, Median, Mode, Constant, etc.)
            constant_value: Giá trị điền nếu method là Constant
            
        Returns:
            Dict chứa thông tin fit
        """
        col_data = train_data[column].dropna()
        
        fill_value = None
        
        if method == "Mean Imputation":
            fill_value = col_data.mean()
        elif method == "Median Imputation":
            fill_value = col_data.median()
        elif method == "Mode Imputation":
            mode_result = col_data.mode()
            fill_value = mode_result[0] if len(mode_result) > 0 else 0
        elif method == "Constant Value":
            fill_value = constant_value
        elif method in ["Forward Fill", "Backward Fill", "Interpolation"]:
            # These methods don't need pre-computed values
            fill_value = None
        
        self.imputers[column] = {
            'method': method,
            'fill_value': fill_value,
            'original_missing_count': train_data[column].isnull().sum()
        }
        
        if column not in self.fitted_columns['imputation']:
            self.fitted_columns['imputation'].append(column)
        
        return self.imputers[column]
    
    # -------------------------------------------------------------------------
    # TRANSFORM IMPUTATION
    # Áp dụng logic đã học từ bước FIT vào dữ liệu mới
    # -------------------------------------------------------------------------
    def transform_imputation(
        self, 
        data: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        """
        Apply imputation đã fit lên data.
        
        Args:
            data: DataFrame cần transform
            column: Tên cột
            
        Returns:
            DataFrame đã được impute
        """
        if column not in self.imputers:
            raise ValueError(f"Column {column} chưa được fit. Hãy gọi fit_imputer trước.")
        
        result = data.copy()
        imputer_info = self.imputers[column]
        method = imputer_info['method']
        fill_value = imputer_info['fill_value']
        
        if method in ["Mean Imputation", "Median Imputation", "Mode Imputation", "Constant Value"]:
            result[column].fillna(fill_value, inplace=True)
        elif method == "Forward Fill":
            result[column].fillna(method='ffill', inplace=True)
        elif method == "Backward Fill":
            result[column].fillna(method='bfill', inplace=True)
        elif method == "Interpolation":
            result[column] = result[column].interpolate()
        elif method == "Drop Rows":
            result = result[result[column].notna()]
        
        return result
    
    # ==================== SCALING ====================
    
    def fit_scaler(
        self, 
        train_data: pd.DataFrame, 
        columns: List[str], 
        method: str
    ) -> Dict:
        """
        Fit scaler trên train data.
        
        Args:
            train_data: DataFrame train
            columns: Danh sách cột cần scale
            method: Phương pháp scaling
            
        Returns:
            Dict chứa thông tin fit
        """
        # Select scaler
        if "StandardScaler" in method:
            scaler = StandardScaler()
        elif "MinMaxScaler" in method:
            scaler = MinMaxScaler()
        elif "RobustScaler" in method:
            scaler = RobustScaler()
        elif "MaxAbsScaler" in method:
            scaler = MaxAbsScaler()
        elif "Normalizer" in method:
            scaler = Normalizer()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on train data
        scaler.fit(train_data[columns])
        
        # Create a key for this group of columns
        cols_key = "_".join(sorted(columns))
        
        self.scalers[cols_key] = {
            'scaler': scaler,
            'method': method,
            'columns': columns
        }
        
        for col in columns:
            if col not in self.fitted_columns['scaling']:
                self.fitted_columns['scaling'].append(col)
        
        return {
            'method': method,
            'columns': columns,
            'fitted': True
        }
    
    # -------------------------------------------------------------------------
    # TRANSFORM SCALING
    # Sử dụng Scaler đã fit (lưu trong self.scalers) để biến đổi dữ liệu
    # -------------------------------------------------------------------------
    def transform_scaling(
        self, 
        data: pd.DataFrame, 
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply scaling đã fit lên data.
        
        Args:
            data: DataFrame cần transform
            columns: Danh sách cột
            
        Returns:
            DataFrame đã được scale
        """
        cols_key = "_".join(sorted(columns))
        
        if cols_key not in self.scalers:
            raise ValueError(f"Columns {columns} chưa được fit. Hãy gọi fit_scaler trước.")
        
        result = data.copy()
        scaler_info = self.scalers[cols_key]
        scaler = scaler_info['scaler']
        
        # Transform
        result[columns] = scaler.transform(result[columns])
        
        return result
    
    # ==================== ENCODING ====================
    
    def fit_encoder(
        self, 
        train_data: pd.DataFrame, 
        column: str, 
        method: str,
        **kwargs
    ) -> Dict:
        """
        Fit encoder trên train data.
        
        Args:
            train_data: DataFrame train
            column: Tên cột
            method: Phương pháp encoding
            **kwargs: Tham số bổ sung (target_column, ordinal_mapping, etc.)
            
        Returns:
            Dict chứa thông tin fit
        """
        col_data = train_data[column].dropna()
        
        if method == "Label Encoding":
            le = LabelEncoder()
            le.fit(col_data)
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            
            self.encoders[column] = {
                'method': method,
                'encoder': le,
                'mapping': mapping,
                'classes': le.classes_.tolist()
            }
            
        elif method == "One-Hot Encoding":
            categories = col_data.unique().tolist()
            drop_first = kwargs.get('drop_first', False)
            
            self.encoders[column] = {
                'method': method,
                'categories': categories,
                'drop_first': drop_first
            }
            
        elif method == "Target Encoding":
            target_column = kwargs.get('target_column')
            smoothing = kwargs.get('smoothing', 1.0)
            
            if not target_column:
                raise ValueError("Target Encoding requires target_column")
            
            # Calculate target mean for each category
            global_mean = train_data[target_column].mean()
            target_means = train_data.groupby(column)[target_column].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothed_means = (
                (target_means['count'] * target_means['mean'] + smoothing * global_mean) / 
                (target_means['count'] + smoothing)
            )
            
            self.encoders[column] = {
                'method': method,
                'mapping': smoothed_means.to_dict(),
                'global_mean': global_mean,
                'smoothing': smoothing
            }
            
        elif method == "Frequency Encoding":
            freq_mapping = (col_data.value_counts() / len(train_data)).to_dict()
            
            self.encoders[column] = {
                'method': method,
                'mapping': freq_mapping
            }
            
        elif method == "Ordinal Encoding":
            ordinal_mapping = kwargs.get('ordinal_mapping')
            if ordinal_mapping:
                mapping = {cat: idx for idx, cat in enumerate(ordinal_mapping)}
            else:
                categories = sorted(col_data.unique())
                mapping = {cat: idx for idx, cat in enumerate(categories)}
            
            self.encoders[column] = {
                'method': method,
                'mapping': mapping
            }
        
        if column not in self.fitted_columns['encoding']:
            self.fitted_columns['encoding'].append(column)
        
        return self.encoders[column]
    
    # -------------------------------------------------------------------------
    # TRANSFORM ENCODING
    # Sử dụng Encoder/Mapping đã fit để biến đổi biến phân loại thành số
    # Lưu ý: Handle trường hợp giá trị mới (Unseen categories) trong Test set
    # -------------------------------------------------------------------------
    def transform_encoding(
        self, 
        data: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        """
        Apply encoding đã fit lên data.
        
        Args:
            data: DataFrame cần transform
            column: Tên cột
            
        Returns:
            DataFrame đã được encode
        """
        if column not in self.encoders:
            raise ValueError(f"Column {column} chưa được fit. Hãy gọi fit_encoder trước.")
        
        result = data.copy()
        encoder_info = self.encoders[column]
        method = encoder_info['method']
        
        if method == "Label Encoding":
            le = encoder_info['encoder']
            mapping = encoder_info['mapping']
            # Use mapping directly to handle potential unknown values
            result[column] = result[column].map(mapping)
            
        elif method == "One-Hot Encoding":
            categories = encoder_info['categories']
            drop_first = encoder_info['drop_first']
            
            dummies = pd.get_dummies(result[column], prefix=column, drop_first=drop_first, dtype=int)
            
            # Add missing columns (categories in train but not in this data)
            for cat in categories:
                col_name = f"{column}_{cat}"
                if drop_first and cat == categories[0]:
                    continue
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            result = result.drop(columns=[column])
            result = pd.concat([result, dummies], axis=1)
            
        elif method in ["Target Encoding", "Frequency Encoding", "Ordinal Encoding"]:
            mapping = encoder_info['mapping']
            default_val = encoder_info.get('global_mean', 0)
            result[column] = result[column].map(mapping).fillna(default_val)
        
        return result
    
    # ==================== OUTLIER HANDLING ====================
    
    def fit_outlier_bounds(
        self, 
        train_data: pd.DataFrame, 
        column: str, 
        method: str,
        **params
    ) -> Dict:
        """
        Fit outlier bounds trên train data.
        
        Args:
            train_data: DataFrame train
            column: Tên cột
            method: Phương pháp (Winsorization, IQR, Z-Score)
            **params: Tham số bổ sung
            
        Returns:
            Dict chứa bounds
        """
        col_data = train_data[column].dropna()
        
        if method == "Winsorization":
            lower_pct = params.get('lower_percentile', 0.05)
            upper_pct = params.get('upper_percentile', 0.95)
            lower = col_data.quantile(lower_pct)
            upper = col_data.quantile(upper_pct)
            
        elif method == "IQR Method":
            multiplier = params.get('iqr_multiplier', 1.5)
            Q1, Q3 = col_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
        elif method == "Z-Score":
            threshold = params.get('z_threshold', 3.0)
            mean = col_data.mean()
            std = col_data.std()
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            lower = col_data.min()
            upper = col_data.max()
        
        self.outlier_bounds[column] = {
            'method': method,
            'lower': lower,
            'upper': upper,
            'params': params
        }
        
        if column not in self.fitted_columns['outliers']:
            self.fitted_columns['outliers'].append(column)
        
        return self.outlier_bounds[column]
    
    def transform_outliers(
        self, 
        data: pd.DataFrame, 
        column: str,
        action: str = 'clip'
    ) -> pd.DataFrame:
        """
        Apply outlier handling đã fit lên data.
        
        Args:
            data: DataFrame cần transform
            column: Tên cột
            action: Hành động (clip, nan, remove)
            
        Returns:
            DataFrame đã được xử lý outliers
        """
        if column not in self.outlier_bounds:
            raise ValueError(f"Column {column} chưa được fit. Hãy gọi fit_outlier_bounds trước.")
        
        result = data.copy()
        bounds = self.outlier_bounds[column]
        lower = bounds['lower']
        upper = bounds['upper']
        
        if action == 'clip':
            result[column] = result[column].clip(lower, upper)
        elif action == 'nan':
            mask = (result[column] < lower) | (result[column] > upper)
            result.loc[mask, column] = np.nan
        elif action == 'remove':
            result = result[(result[column] >= lower) & (result[column] <= upper)]
        
        return result
    
    # ==================== UTILITY METHODS ====================
    
    def get_summary(self) -> Dict:
        """Lấy tóm tắt các transformations đã fit."""
        return {
            'scaling': {
                'columns': self.fitted_columns['scaling'],
                'count': len(self.fitted_columns['scaling'])
            },
            'imputation': {
                'columns': self.fitted_columns['imputation'],
                'count': len(self.fitted_columns['imputation'])
            },
            'encoding': {
                'columns': self.fitted_columns['encoding'],
                'count': len(self.fitted_columns['encoding'])
            },
            'outliers': {
                'columns': self.fitted_columns['outliers'],
                'count': len(self.fitted_columns['outliers'])
            }
        }
    
    def is_fitted(self, transform_type: str, column: str) -> bool:
        """Kiểm tra xem column đã được fit cho transform_type chưa."""
        return column in self.fitted_columns.get(transform_type, [])


# Convenience function
def create_pipeline():
    """Tạo mới PreprocessingPipeline instance."""
    return PreprocessingPipeline()
