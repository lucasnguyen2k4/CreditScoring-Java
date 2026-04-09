"""
Encoder - Backend mã hóa biến phân loại
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')


class CategoricalEncoder:
    """
    Class mã hóa biến phân loại với nhiều phương pháp khác nhau
    """
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        self.encoding_info = {}
        self.encoders = {}  # Store fitted encoders for inverse transform
        
    def one_hot_encoding(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        drop_first: bool = False,
        handle_unknown: str = 'ignore'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Mã hóa One-Hot Encoding
        
        Args:
            data: DataFrame cần mã hóa
            columns: Danh sách cột cần mã hóa
            drop_first: Có drop cột đầu tiên để tránh multicollinearity
            handle_unknown: Cách xử lý giá trị mới ('ignore' hoặc 'error')
            
        Returns:
            Tuple (DataFrame đã mã hóa, Dict thông tin)
        """
        processed_data = data.copy()
        info = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Get unique values
            unique_values = data[col].dropna().unique()
            n_categories = len(unique_values)
            
            # Perform one-hot encoding
            dummies = pd.get_dummies(
                data[col], 
                prefix=col,
                drop_first=drop_first,
                dtype=int
            )
            
            # Remove original column and add dummy columns
            processed_data = processed_data.drop(columns=[col])
            processed_data = pd.concat([processed_data, dummies], axis=1)
            
            # Store info
            info[col] = {
                'method': 'One-Hot Encoding',
                'original_column': col,
                'n_categories': n_categories,
                'categories': unique_values.tolist(),
                'new_columns': dummies.columns.tolist(),
                'n_new_columns': len(dummies.columns),
                'drop_first': drop_first,
                'handle_unknown': handle_unknown
            }
            
            # Store encoder mapping for potential inverse transform
            self.encoders[col] = {
                'type': 'one_hot',
                'categories': unique_values.tolist(),
                'new_columns': dummies.columns.tolist()
            }
        
        return processed_data, info
    
    def label_encoding(
        self, 
        data: pd.DataFrame, 
        columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Mã hóa Label Encoding (0, 1, 2, ...)
        
        Args:
            data: DataFrame cần mã hóa
            columns: Danh sách cột cần mã hóa
            
        Returns:
            Tuple (DataFrame đã mã hóa, Dict thông tin)
        """
        processed_data = data.copy()
        info = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Initialize LabelEncoder
            le = LabelEncoder()
            
            # Handle missing values
            col_data = data[col].copy()
            mask_null = col_data.isnull()
            
            if mask_null.any():
                # Encode non-null values
                col_data_clean = col_data[~mask_null]
                encoded_clean = le.fit_transform(col_data_clean)
                
                # Create encoded series with NaN preserved
                encoded = pd.Series(np.nan, index=col_data.index)
                encoded[~mask_null] = encoded_clean
                processed_data[col] = encoded
            else:
                # No missing values
                processed_data[col] = le.fit_transform(col_data)
            
            # Get mapping
            classes = le.classes_.tolist()
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            
            # Store info
            info[col] = {
                'method': 'Label Encoding',
                'original_column': col,
                'n_categories': len(classes),
                'categories': classes,
                'mapping': mapping,
                'encoded_range': f'[0, {len(classes)-1}]'
            }
            
            # Store encoder for inverse transform
            self.encoders[col] = {
                'type': 'label',
                'encoder': le,
                'mapping': mapping
            }
        
        return processed_data, info
    
    def target_encoding(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        target_column: str,
        smoothing: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Mã hóa Target Encoding (dựa trên target mean)
        
        Args:
            data: DataFrame cần mã hóa
            columns: Danh sách cột cần mã hóa
            target_column: Tên cột target
            smoothing: Hệ số smoothing để tránh overfitting
            
        Returns:
            Tuple (DataFrame đã mã hóa, Dict thông tin)
        """
        processed_data = data.copy()
        info = {}
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Calculate global mean
        global_mean = data[target_column].mean()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Calculate target mean for each category
            target_means = data.groupby(col)[target_column].agg(['mean', 'count'])
            
            # Apply smoothing
            # Formula: (count * category_mean + smoothing * global_mean) / (count + smoothing)
            smoothed_means = (
                (target_means['count'] * target_means['mean'] + smoothing * global_mean) / 
                (target_means['count'] + smoothing)
            )
            
            # Map to data
            processed_data[col] = data[col].map(smoothed_means)
            
            # Handle unseen categories with global mean
            processed_data[col].fillna(global_mean, inplace=True)
            
            # Store info
            mapping_dict = smoothed_means.to_dict()
            
            info[col] = {
                'method': 'Target Encoding',
                'original_column': col,
                'target_column': target_column,
                'n_categories': len(target_means),
                'global_mean': global_mean,
                'smoothing': smoothing,
                'mapping': mapping_dict,
                'min_encoded': smoothed_means.min(),
                'max_encoded': smoothed_means.max()
            }
            
            # Store encoder
            self.encoders[col] = {
                'type': 'target',
                'mapping': mapping_dict,
                'global_mean': global_mean
            }
        
        return processed_data, info
    
    def ordinal_encoding(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        ordinal_mappings: Dict[str, List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Mã hóa Ordinal Encoding (theo thứ tự có ý nghĩa)
        
        Args:
            data: DataFrame cần mã hóa
            columns: Danh sách cột cần mã hóa
            ordinal_mappings: Dict mapping cho từng cột
                Ví dụ: {'education': ['High School', 'Bachelor', 'Master', 'PhD']}
            
        Returns:
            Tuple (DataFrame đã mã hóa, Dict thông tin)
        """
        processed_data = data.copy()
        info = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Get categories in order
            if ordinal_mappings and col in ordinal_mappings:
                categories = ordinal_mappings[col]
            else:
                # Auto-detect order (alphabetical if not specified)
                categories = sorted(data[col].dropna().unique())
            
            # Create mapping
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            
            # Apply mapping
            processed_data[col] = data[col].map(mapping)
            
            # Store info
            info[col] = {
                'method': 'Ordinal Encoding',
                'original_column': col,
                'n_categories': len(categories),
                'categories': categories,
                'mapping': mapping,
                'encoded_range': f'[0, {len(categories)-1}]',
                'order': 'specified' if (ordinal_mappings and col in ordinal_mappings) else 'alphabetical'
            }
            
            # Store encoder
            self.encoders[col] = {
                'type': 'ordinal',
                'mapping': mapping,
                'categories': categories
            }
        
        return processed_data, info
    
    def frequency_encoding(
        self, 
        data: pd.DataFrame, 
        columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Mã hóa Frequency Encoding (theo tần suất xuất hiện)
        
        Args:
            data: DataFrame cần mã hóa
            columns: Danh sách cột cần mã hóa
            
        Returns:
            Tuple (DataFrame đã mã hóa, Dict thông tin)
        """
        processed_data = data.copy()
        info = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Calculate frequencies
            value_counts = data[col].value_counts()
            freq_mapping = (value_counts / len(data)).to_dict()
            
            # Apply mapping
            processed_data[col] = data[col].map(freq_mapping)
            
            # Store info
            info[col] = {
                'method': 'Frequency Encoding',
                'original_column': col,
                'n_categories': len(value_counts),
                'mapping': freq_mapping,
                'min_frequency': min(freq_mapping.values()),
                'max_frequency': max(freq_mapping.values()),
                'encoded_range': f'[{min(freq_mapping.values()):.4f}, {max(freq_mapping.values()):.4f}]'
            }
            
            # Store encoder
            self.encoders[col] = {
                'type': 'frequency',
                'mapping': freq_mapping
            }
        
        return processed_data, info
    
    def apply_encoding(
        self,
        data: pd.DataFrame,
        method: str,
        columns: List[str],
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Áp dụng phương pháp mã hóa được chọn
        
        Args:
            data: DataFrame cần mã hóa
            method: Phương pháp ('One-Hot Encoding', 'Label Encoding', 'Target Encoding', 'Ordinal Encoding')
            columns: Danh sách cột cần mã hóa
            **kwargs: Tham số bổ sung cho từng phương pháp
            
        Returns:
            Tuple (DataFrame đã mã hóa, Dict thông tin mã hóa)
        """
        self.original_data = data.copy()
        
        if method == "One-Hot Encoding":
            drop_first = kwargs.get('drop_first', False)
            handle_unknown = kwargs.get('handle_unknown', 'ignore')
            self.processed_data, self.encoding_info = self.one_hot_encoding(
                data, columns, drop_first, handle_unknown
            )
            
        elif method == "Label Encoding":
            self.processed_data, self.encoding_info = self.label_encoding(
                data, columns
            )
            
        elif method == "Target Encoding":
            target_column = kwargs.get('target_column')
            smoothing = kwargs.get('smoothing', 1.0)
            
            if not target_column:
                raise ValueError("Target Encoding requires 'target_column' parameter")
            
            self.processed_data, self.encoding_info = self.target_encoding(
                data, columns, target_column, smoothing
            )
            
        elif method == "Ordinal Encoding":
            ordinal_mappings = kwargs.get('ordinal_mappings', None)
            self.processed_data, self.encoding_info = self.ordinal_encoding(
                data, columns, ordinal_mappings
            )
            
        elif method == "Frequency Encoding":
            self.processed_data, self.encoding_info = self.frequency_encoding(
                data, columns
            )
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return self.processed_data, self.encoding_info
    
    def get_summary_report(self) -> str:
        """
        Tạo báo cáo tóm tắt về mã hóa
        
        Returns:
            String báo cáo
        """
        if not self.encoding_info:
            return "No encoding has been performed yet."
        
        report = "=" * 60 + "\n"
        report += "CATEGORICAL ENCODING SUMMARY REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for col, info in self.encoding_info.items():
            report += f"Column: {col}\n"
            report += f"Method: {info['method']}\n"
            report += f"Categories: {info['n_categories']}\n"
            
            if 'new_columns' in info:
                report += f"New Columns Created: {info['n_new_columns']}\n"
                report += f"Columns: {', '.join(info['new_columns'][:5])}"
                if len(info['new_columns']) > 5:
                    report += f"... (+{len(info['new_columns'])-5} more)"
                report += "\n"
            
            if 'mapping' in info and info['method'] != 'One-Hot Encoding':
                report += f"Encoding Range: {info.get('encoded_range', 'N/A')}\n"
            
            report += "-" * 60 + "\n"
        
        return report
    
    def recommend_encoding_method(
        self,
        data: pd.DataFrame,
        column: str,
        target_column: Optional[str] = None,
        max_categories_for_onehot: int = 10
    ) -> Dict[str, str]:
        """
        Gợi ý phương pháp mã hóa phù hợp cho cột
        
        Args:
            data: DataFrame
            column: Tên cột cần gợi ý
            target_column: Tên cột target (nếu có)
            max_categories_for_onehot: Số categories tối đa để dùng One-Hot
            
        Returns:
            Dict chứa recommendation và lý do
        """
        if column not in data.columns:
            return {'recommendation': 'N/A', 'reason': 'Column not found'}
        
        n_categories = data[column].nunique()
        
        # Binary categorical
        if n_categories == 2:
            return {
                'recommendation': 'Label Encoding',
                'reason': f'Binary variable ({n_categories} categories) - Label Encoding is sufficient'
            }
        
        # Low cardinality
        elif n_categories <= max_categories_for_onehot:
            return {
                'recommendation': 'One-Hot Encoding',
                'reason': f'Low cardinality ({n_categories} categories) - One-Hot Encoding avoids ordinal assumption'
            }
        
        # High cardinality with target
        elif target_column and n_categories > max_categories_for_onehot:
            return {
                'recommendation': 'Target Encoding',
                'reason': f'High cardinality ({n_categories} categories) - Target Encoding captures relationship with target'
            }
        
        # High cardinality without target
        elif n_categories > max_categories_for_onehot:
            return {
                'recommendation': 'Frequency Encoding',
                'reason': f'High cardinality ({n_categories} categories) - Frequency Encoding reduces dimensionality'
            }
        
        else:
            return {
                'recommendation': 'One-Hot Encoding',
                'reason': 'Default recommendation'
            }


# Convenience functions for easy import
def encode_categorical(
    data: pd.DataFrame,
    method: str,
    columns: List[str],
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function để mã hóa biến phân loại
    
    Args:
        data: DataFrame cần mã hóa
        method: Phương pháp mã hóa
        columns: Danh sách cột
        **kwargs: Tham số bổ sung
        
    Returns:
        Tuple (DataFrame đã mã hóa, Dict thông tin)
    """
    encoder = CategoricalEncoder()
    return encoder.apply_encoding(data, method, columns, **kwargs)


def recommend_encoding(
    data: pd.DataFrame,
    column: str,
    target_column: Optional[str] = None
) -> Dict[str, str]:
    """
    Gợi ý phương pháp mã hóa phù hợp
    
    Args:
        data: DataFrame
        column: Tên cột
        target_column: Tên cột target (nếu có)
        
    Returns:
        Dict chứa recommendation
    """
    encoder = CategoricalEncoder()
    return encoder.recommend_encoding_method(data, column, target_column)
