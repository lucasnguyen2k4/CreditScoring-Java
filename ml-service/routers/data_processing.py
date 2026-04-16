"""
Data Processing Router
Endpoints for file upload, EDA, preprocessing, encoding, outliers, balancing, scaling.
"""

from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import json

from session_store import store
from services.encoder import CategoricalEncoder, encode_categorical, recommend_encoding
from services.outlier_handler import OutlierHandler, handle_outliers
from services.balancer import balance_data, get_class_distribution, check_imbalance
from services.preprocessing_pipeline import PreprocessingPipeline, create_pipeline
from services.data_generator import generate_sample_data

router = APIRouter()


def get_session(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required")
    return store.get_or_create(session_id)


def _mark_step_completed(session, step_key: str):
    """Persist feature-engineering step completion in session state."""
    try:
        session.completed_steps.add(step_key)
    except Exception:
        session.completed_steps = set([step_key])


def df_to_json(df: pd.DataFrame, max_rows: int = 500) -> dict:
    """Convert DataFrame to JSON-safe dict with limited rows"""
    preview = df.head(max_rows)
    return {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": list(df.shape),
        "data": json.loads(preview.to_json(orient="records", date_format="iso")),
    }


def to_finite_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric and drop non-finite values as NaN."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def to_finite_float(value: Any) -> Optional[float]:
    """Convert value to float and return None for NaN/inf."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def safe_round(value: Any, digits: int = 4) -> Optional[float]:
    """Round finite floats and return None for invalid values."""
    finite = to_finite_float(value)
    return round(finite, digits) if finite is not None else None


def to_jsonable(value: Any):
    """Convert nested values into JSON-serializable python primitives."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ========================= UPLOAD =========================

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Upload a CSV file and store in session"""
    session = get_session(x_session_id)

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        session.raw_data = df
        session.clear_data()  # Reset dependent state
        session.raw_data = df  # Re-set after clear

        return {
            "message": f"Uploaded {file.filename}: {df.shape[0]} rows, {df.shape[1]} columns",
            "preview": df_to_json(df, max_rows=100),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")


@router.post("/generate-sample")
def generate_sample(
    records: int = 100,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Generate sample credit scoring data and store in session"""
    session = get_session(x_session_id)
    try:
        df = generate_sample_data(n_records=records)
        session.raw_data = df
        session.clear_data()
        session.raw_data = df
        
        return {
            "message": f"Generated sample data: {df.shape[0]} rows, {df.shape[1]} columns",
            "preview": df_to_json(df, max_rows=100),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating data: {str(e)}")


# ========================= EDA =========================

@router.get("/info")
def get_data_info(
    use_raw: bool = False,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Get basic info about loaded data. When use_raw=true, always use the uploaded raw_data (e.g. for cleanup UI after split)."""
    session = get_session(x_session_id)
    if use_raw:
        df = session.raw_data
    elif session.X_train is not None:
        if session.target_column and session.y_train is not None:
            df = pd.concat([session.X_train, pd.Series(session.y_train, name=session.target_column, index=session.X_train.index)], axis=1)
        else:
            df = session.X_train
    elif session.processed_data is not None:
        df = session.processed_data
    else:
        df = session.raw_data

    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded. Upload a file first.")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    unique_counts = {col: int(df[col].nunique(dropna=True)) for col in df.columns}

    return {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": {
            col: {"count": int(missing[col]), "percentage": float(missing_pct[col])}
            for col in df.columns if missing[col] > 0
        },
        "unique_counts": unique_counts,
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
    }


@router.get("/preview")
def get_data_preview(
    rows: int = 100,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Get data preview (first N rows)"""
    session = get_session(x_session_id)
    
    if session.X_train is not None:
        if session.target_column and session.y_train is not None:
            df = pd.concat([session.X_train, pd.Series(session.y_train, name=session.target_column, index=session.X_train.index)], axis=1)
        else:
            df = session.X_train
    elif session.processed_data is not None:
        df = session.processed_data
    else:
        df = session.raw_data

    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    return df_to_json(df, max_rows=rows)


@router.get("/stats")
def get_statistics(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get descriptive statistics for all numeric columns"""
    session = get_session(x_session_id)
    if session.X_train is not None:
        if session.target_column and session.y_train is not None:
            df = pd.concat([session.X_train, pd.Series(session.y_train, name=session.target_column, index=session.X_train.index)], axis=1)
        else:
            df = session.X_train
    elif session.processed_data is not None:
        df = session.processed_data
    else:
        df = session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    numeric_df = df.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric_df.columns:
        col_data = to_finite_numeric(df[col]).dropna()
        if len(col_data) == 0:
            continue
        q1 = float(col_data.quantile(0.25))
        q3 = float(col_data.quantile(0.75))
        iqr = q3 - q1
        outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
        stats[col] = {
            "count": int(col_data.count()),
            "mean": to_finite_float(col_data.mean()),
            "median": to_finite_float(col_data.median()),
            "std": to_finite_float(col_data.std()),
            "min": to_finite_float(col_data.min()),
            "max": to_finite_float(col_data.max()),
            "q1": to_finite_float(q1), "q3": to_finite_float(q3), "iqr": to_finite_float(iqr),
            "outliers_count": len(outliers),
            "skewness": to_finite_float(col_data.skew()),
            "kurtosis": to_finite_float(col_data.kurtosis()),
        }

    return {"numeric_stats": stats}


@router.get("/distribution")
def get_column_distribution(
    column: str,
    processed: bool = False,
    bins: int = 20,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Get chart-friendly distribution data for a specific column."""
    session = get_session(x_session_id)
    if session.X_train is not None:
        if session.target_column and session.y_train is not None:
            df = pd.concat([session.X_train, pd.Series(session.y_train, name=session.target_column, index=session.X_train.index)], axis=1)
        else:
            df = session.X_train
    elif session.processed_data is not None:
        df = session.processed_data
    else:
        df = session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        clean = to_finite_numeric(series).dropna()
        if clean.empty:
            raise HTTPException(status_code=400, detail=f"Column '{column}' has no finite numeric values")

        hist_counts, bin_edges = np.histogram(clean, bins=max(5, min(int(bins), 40)))
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_series = clean[(clean < lower_bound) | (clean > upper_bound)]
        outliers_count = int(outliers_series.shape[0])
        outlier_values = [round(float(v), 4) for v in outliers_series.head(200).tolist()]
        quantiles = {
            "p01": round(float(clean.quantile(0.01)), 4),
            "p05": round(float(clean.quantile(0.05)), 4),
            "p25": round(q1, 4),
            "p50": round(float(clean.quantile(0.50)), 4),
            "p75": round(q3, 4),
            "p95": round(float(clean.quantile(0.95)), 4),
            "p99": round(float(clean.quantile(0.99)), 4),
        }
        mean_val = to_finite_float(clean.mean())
        std_val = to_finite_float(clean.std())
        cv = (std_val / mean_val * 100.0) if (std_val is not None and mean_val not in (None, 0.0)) else None

        histogram = []
        for i, count in enumerate(hist_counts):
            start = float(bin_edges[i])
            end = float(bin_edges[i + 1])
            histogram.append({
                "bin_start": round(start, 4),
                "bin_end": round(end, 4),
                "label": f"{start:.2f} - {end:.2f}",
                "count": int(count),
            })

        return {
            "column": column,
            "data_type": "numeric",
            "data_source": "processed" if processed and session.processed_data is not None else "raw",
            "histogram": histogram,
            "outlier_values": outlier_values,
            "box_stats": {
                "min": safe_round(clean.min(), 4),
                "q1": safe_round(q1, 4),
                "median": safe_round(clean.median(), 4),
                "q3": safe_round(q3, 4),
                "max": safe_round(clean.max(), 4),
                "lower_bound": safe_round(lower_bound, 4),
                "upper_bound": safe_round(upper_bound, 4),
            },
            "summary": {
                "count": int(clean.count()),
                "mean": safe_round(mean_val, 4),
                "median": safe_round(clean.median(), 4),
                "std": safe_round(std_val, 4),
                "min": safe_round(clean.min(), 4),
                "q1": safe_round(q1, 4),
                "q3": safe_round(q3, 4),
                "max": safe_round(clean.max(), 4),
                "skewness": safe_round(clean.skew(), 4),
                "kurtosis": safe_round(clean.kurtosis(), 4),
                "range": safe_round(clean.max() - clean.min(), 4),
                "cv_pct": safe_round(cv, 4) if cv is not None else None,
                "iqr": safe_round(iqr, 4),
                "lower_bound": safe_round(lower_bound, 4),
                "upper_bound": safe_round(upper_bound, 4),
                "outliers_count": outliers_count,
                "outliers_ratio_pct": round((outliers_count / max(1, int(clean.count()))) * 100.0, 4),
                "quantiles": quantiles,
            },
        }

    clean = series.dropna()
    if clean.empty:
        raise HTTPException(status_code=400, detail=f"Column '{column}' has no non-null values")

    counts = clean.astype(str).value_counts().head(15)
    return {
        "column": column,
        "data_type": "categorical",
        "data_source": "processed" if processed and session.processed_data is not None else "raw",
        "categories": [
            {"label": idx, "count": int(value)}
            for idx, value in counts.items()
        ],
        "summary": {
            "count": int(clean.count()),
            "unique": int(clean.nunique()),
            "top": str(counts.index[0]) if not counts.empty else None,
            "top_count": int(counts.iloc[0]) if not counts.empty else 0,
        },
    }


@router.get("/categorical-summary")
def get_categorical_summary(
    processed: bool = False,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Get summary statistics for all categorical columns."""
    session = get_session(x_session_id)
    if session.X_train is not None:
        if session.target_column and session.y_train is not None:
            df = pd.concat([session.X_train, pd.Series(session.y_train, name=session.target_column, index=session.X_train.index)], axis=1)
        else:
            df = session.X_train
    elif session.processed_data is not None:
        df = session.processed_data
    else:
        df = session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    rows = []

    for col in cat_cols:
        series = df[col]
        non_null = series.dropna().astype(str)
        value_counts = non_null.value_counts()
        most_common = str(value_counts.index[0]) if not value_counts.empty else None
        top_frequency = int(value_counts.iloc[0]) if not value_counts.empty else 0
        missing_count = int(series.isna().sum())
        total_count = int(series.shape[0])

        rows.append({
            "column_name": col,
            "unique_values": int(series.nunique(dropna=True)),
            "most_common": most_common,
            "top_frequency": top_frequency,
            "missing": missing_count,
            "missing_ratio_pct": round((missing_count / max(1, total_count)) * 100.0, 4),
        })

    return {
        "total_rows": int(df.shape[0]),
        "total_categorical_columns": len(rows),
        "columns": rows,
    }


class RemoveCategoricalRequest(BaseModel):
    columns: Optional[List[str]] = None
    processed: bool = False
    apply_on_splits: bool = False


@router.post("/remove-categorical")
def remove_categorical_columns(req: RemoveCategoricalRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Remove specified columns from dataset (EDA cleanup utility)."""
    session = get_session(x_session_id)
    target_col = session.target_column

    if req.apply_on_splits:
        if session.X_train is None:
            raise HTTPException(status_code=400, detail="Data not split yet. Call /split first.")

        if req.columns is not None:
            cols_to_drop = [c for c in req.columns if c in session.X_train.columns]
        else:
            cols_to_drop = session.X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        if target_col:
            cols_to_drop = [c for c in cols_to_drop if c != target_col]

        session.X_train = session.X_train.drop(columns=cols_to_drop, errors="ignore")
        if session.X_valid is not None:
            session.X_valid = session.X_valid.drop(columns=cols_to_drop, errors="ignore")
        if session.X_test is not None:
            session.X_test = session.X_test.drop(columns=cols_to_drop, errors="ignore")
        session.selected_features = session.X_train.columns.tolist()
        _mark_step_completed(session, "cleanup")

        return {
            "message": f"Removed {len(cols_to_drop)} columns from train/valid/test splits",
            "removed_columns": cols_to_drop,
            "train_shape": list(session.X_train.shape),
            "valid_shape": list(session.X_valid.shape) if session.X_valid is not None else None,
            "test_shape": list(session.X_test.shape) if session.X_test is not None else None,
        }

    df = session.processed_data if req.processed and session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    if req.columns is not None:
        cols_to_drop = [c for c in req.columns if c in df.columns]
    else:
        cols_to_drop = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_col:
        cols_to_drop = [c for c in cols_to_drop if c != target_col]

    result_df = df.drop(columns=cols_to_drop, errors="ignore").copy()

    # Reset dependent state to avoid stale train/model/shap state.
    session.clear_data()
    session.raw_data = result_df
    _mark_step_completed(session, "cleanup")

    return {
        "message": f"Removed {len(cols_to_drop)} columns",
        "removed_columns": cols_to_drop,
        "new_shape": list(result_df.shape),
    }


class CleanInvalidNumbersRequest(BaseModel):
    columns: Optional[List[str]] = None
    strategy: str = "drop_rows"  # drop_rows | fill_median
    processed: bool = False
    apply_on_splits: bool = False


@router.post("/clean-invalid-numbers")
def clean_invalid_numbers(req: CleanInvalidNumbersRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Convert selected columns to numeric and handle invalid numeric values."""
    session = get_session(x_session_id)

    if req.apply_on_splits:
        if session.X_train is None:
            raise HTTPException(status_code=400, detail="Data not split yet. Call /split first.")

        split_frames = {
            "train": session.X_train.copy(),
            "valid": session.X_valid.copy() if session.X_valid is not None else None,
            "test": session.X_test.copy() if session.X_test is not None else None,
        }

        default_cols = split_frames["train"].select_dtypes(include=[np.number]).columns.tolist()
        cols = req.columns if req.columns is not None else default_cols
        cols = [c for c in cols if c in split_frames["train"].columns]
        if not cols:
            raise HTTPException(status_code=400, detail="No valid columns provided")

        invalid_summary: Dict[str, Dict[str, Any]] = {}
        for col in cols:
            details = {"converted_to_numeric": True}
            total_invalid = 0
            for split_name in ["train", "valid", "test"]:
                frame = split_frames[split_name]
                if frame is None or col not in frame.columns:
                    details[f"{split_name}_invalid"] = 0
                    continue

                series = frame[col]
                converted = pd.to_numeric(series, errors="coerce")
                invalid_mask = series.notna() & converted.isna()
                invalid_count = int(invalid_mask.sum())
                frame[col] = converted
                details[f"{split_name}_invalid"] = invalid_count
                total_invalid += invalid_count

            details["invalid_count"] = total_invalid
            invalid_summary[col] = details

        dropped_rows = {"train": 0, "valid": 0, "test": 0}
        if req.strategy == "drop_rows":
            for split_name, y_attr in [("train", "y_train"), ("valid", "y_valid"), ("test", "y_test")]:
                frame = split_frames[split_name]
                if frame is None:
                    continue
                before = int(frame.shape[0])
                keep_mask = frame[cols].notna().all(axis=1)
                filtered = frame.loc[keep_mask].copy()
                split_frames[split_name] = filtered
                dropped_rows[split_name] = before - int(filtered.shape[0])

                y_vals = getattr(session, y_attr)
                if y_vals is not None:
                    try:
                        setattr(session, y_attr, y_vals.loc[filtered.index])
                    except Exception:
                        keep_idx = np.where(keep_mask.to_numpy())[0]
                        setattr(session, y_attr, np.asarray(y_vals)[keep_idx])
        elif req.strategy == "fill_median":
            train_fill_values = {}
            for col in cols:
                median = split_frames["train"][col].median()
                if pd.isna(median):
                    median = 0.0
                train_fill_values[col] = float(median)

            for split_name in ["train", "valid", "test"]:
                frame = split_frames[split_name]
                if frame is None:
                    continue
                for col in cols:
                    frame[col] = frame[col].fillna(train_fill_values[col])
                    invalid_summary[col][f"fill_value_from_train_median"] = train_fill_values[col]
        else:
            raise HTTPException(status_code=400, detail="Invalid strategy. Use 'drop_rows' or 'fill_median'.")

        session.X_train = split_frames["train"]
        session.X_valid = split_frames["valid"]
        session.X_test = split_frames["test"]
        session.selected_features = session.X_train.columns.tolist()
        _mark_step_completed(session, "cleanup")

        total_invalid = int(sum(v["invalid_count"] for v in invalid_summary.values()))
        return {
            "message": f"Invalid number cleanup completed on splits using '{req.strategy}'",
            "columns": cols,
            "total_invalid_values": total_invalid,
            "dropped_rows": dropped_rows,
            "details": invalid_summary,
            "train_shape": list(session.X_train.shape),
            "valid_shape": list(session.X_valid.shape) if session.X_valid is not None else None,
            "test_shape": list(session.X_test.shape) if session.X_test is not None else None,
        }

    df = session.processed_data if req.processed and session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    work_df = df.copy()
    cols = req.columns or work_df.columns.tolist()
    cols = [c for c in cols if c in work_df.columns]
    if not cols:
        raise HTTPException(status_code=400, detail="No valid columns provided")

    invalid_summary = {}
    for col in cols:
        series = work_df[col]
        converted = pd.to_numeric(series, errors="coerce")
        invalid_mask = series.notna() & converted.isna()
        invalid_count = int(invalid_mask.sum())
        work_df[col] = converted
        invalid_summary[col] = {
            "invalid_count": invalid_count,
            "converted_to_numeric": True,
        }

    dropped_rows = 0
    if req.strategy == "drop_rows":
        before = int(work_df.shape[0])
        work_df = work_df.dropna(subset=cols)
        dropped_rows = before - int(work_df.shape[0])
    elif req.strategy == "fill_median":
        for col in cols:
            median = work_df[col].median()
            if pd.isna(median):
                median = 0.0
            work_df[col] = work_df[col].fillna(median)
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy. Use 'drop_rows' or 'fill_median'.")

    session.clear_data()
    session.raw_data = work_df
    _mark_step_completed(session, "cleanup")

    total_invalid = int(sum(v["invalid_count"] for v in invalid_summary.values()))
    return {
        "message": f"Invalid number cleanup completed using '{req.strategy}'",
        "columns": cols,
        "total_invalid_values": total_invalid,
        "dropped_rows": dropped_rows,
        "details": invalid_summary,
        "new_shape": list(work_df.shape),
    }


@router.get("/correlation")
def get_correlation_matrix(
    processed: bool = False,
    method: str = "pearson",
    threshold: float = 0.8,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Get numeric correlation matrix and high-correlation feature pairs."""
    session = get_session(x_session_id)
    df = session.processed_data if processed and session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric_df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric columns")

    method = (method or "pearson").lower()
    if method not in {"pearson", "spearman", "kendall"}:
        raise HTTPException(status_code=400, detail="Correlation method must be pearson, spearman, or kendall")

    corr = numeric_df.corr(method=method).fillna(0.0)
    columns = corr.columns.tolist()

    matrix = [
        {
            "column": row_col,
            "values": [round(float(v), 6) for v in corr.loc[row_col, columns].tolist()],
        }
        for row_col in columns
    ]

    high_pairs = []
    thr = max(0.0, min(float(threshold), 1.0))
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            val = float(corr.iloc[i, j])
            if abs(val) >= thr:
                high_pairs.append({
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "correlation": round(val, 6),
                    "abs_correlation": round(abs(val), 6),
                })
    high_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

    return {
        "method": method,
        "threshold": thr,
        "columns": columns,
        "matrix": matrix,
        "high_pairs": high_pairs,
    }


@router.get("/joint-distribution")
def get_joint_distribution(
    column_x: str,
    column_y: str,
    processed: bool = False,
    bins: int = 8,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Get 2-variable distribution table (cross-tab with optional numeric binning)."""
    session = get_session(x_session_id)
    df = session.processed_data if processed and session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    if column_x not in df.columns or column_y not in df.columns:
        raise HTTPException(status_code=404, detail="One or both columns not found")

    pair_df = df[[column_x, column_y]].copy()
    if pd.api.types.is_numeric_dtype(df[column_x]):
        pair_df[column_x] = to_finite_numeric(pair_df[column_x])
    if pd.api.types.is_numeric_dtype(df[column_y]):
        pair_df[column_y] = to_finite_numeric(pair_df[column_y])
    pair_df = pair_df.dropna()
    if pair_df.empty:
        raise HTTPException(status_code=400, detail="No valid paired values for selected columns")

    n_bins = max(3, min(int(bins), 20))

    def prep_axis(series: pd.Series):
        if pd.api.types.is_numeric_dtype(series):
            binned = pd.cut(series.astype(float), bins=n_bins, include_lowest=True, duplicates="drop")
            labels = [str(v) for v in binned.cat.categories]
            return binned.astype(str), labels, "numeric_binned"
        text_series = series.astype(str)
        counts = text_series.value_counts()
        top = counts.head(25).index.tolist()
        has_other = (~text_series.isin(top)).any()
        reduced = text_series.where(text_series.isin(top), "__OTHER__")
        labels = top + (["__OTHER__"] if has_other else [])
        return reduced, labels, "categorical"

    x_series, x_labels, x_type = prep_axis(pair_df[column_x])
    y_series, y_labels, y_type = prep_axis(pair_df[column_y])

    x_cat = pd.Categorical(x_series, categories=x_labels, ordered=True)
    y_cat = pd.Categorical(y_series, categories=y_labels, ordered=True)
    table = pd.crosstab(x_cat, y_cat, dropna=False)

    rows = []
    for x_label in x_labels:
        counts = [int(table.loc[x_label, y_label]) for y_label in y_labels]
        row_total = int(sum(counts))
        row_ratio = [round((c / max(1, row_total)) * 100.0, 4) for c in counts]
        rows.append({
            "x_value": x_label,
            "counts": counts,
            "row_ratio_pct": row_ratio,
            "total": row_total,
        })

    col_totals = [int(table[y_label].sum()) for y_label in y_labels]
    total_count = int(table.values.sum())

    return {
        "column_x": column_x,
        "column_y": column_y,
        "x_type": x_type,
        "y_type": y_type,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "rows": rows,
        "column_totals": col_totals,
        "total_count": total_count,
        "bins_used": n_bins,
    }


class ScatterMatrixRequest(BaseModel):
    columns: List[str]
    processed: bool = False
    max_points: int = 800


@router.post("/scatter-matrix")
def get_scatter_matrix(req: ScatterMatrixRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Return sampled numeric rows for scatter-plot matrix rendering."""
    session = get_session(x_session_id)
    df = session.processed_data if req.processed and session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    unique_cols = []
    for c in req.columns:
        if c in df.columns and c not in unique_cols:
            unique_cols.append(c)
    if len(unique_cols) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 valid columns")
    if len(unique_cols) > 5:
        unique_cols = unique_cols[:5]

    numeric_cols = [c for c in unique_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric columns")

    work = df[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise HTTPException(status_code=400, detail="No finite paired numeric values for selected columns")

    max_points = max(100, min(int(req.max_points), 5000))
    total_rows = int(work.shape[0])
    if total_rows > max_points:
        work = work.sample(n=max_points, random_state=42)

    corr = work.corr().fillna(0.0)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "correlation": round(float(corr.iloc[i, j]), 4),
            })

    return {
        "columns": numeric_cols,
        "rows": json.loads(work.to_json(orient="records")),
        "total_rows": total_rows,
        "sampled_rows": int(work.shape[0]),
        "correlations": sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True),
    }


@router.get("/grouped-analysis")
def get_grouped_analysis(
    value_column: str,
    group_column: str,
    processed: bool = False,
    top_groups: int = 10,
    max_points: int = 2500,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """Grouped distribution data for box/violin/strip style visualizations."""
    session = get_session(x_session_id)
    df = session.processed_data if processed and session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    if value_column not in df.columns or group_column not in df.columns:
        raise HTTPException(status_code=404, detail="Value or group column not found")

    value_series = to_finite_numeric(df[value_column])
    group_series = df[group_column].astype(str)
    pair_df = pd.DataFrame({"group": group_series, "value": value_series}).dropna(subset=["group", "value"])
    if pair_df.empty:
        raise HTTPException(status_code=400, detail="No valid grouped numeric values")

    n_top = max(2, min(int(top_groups), 30))
    group_counts = pair_df["group"].value_counts()
    shown_groups = group_counts.head(n_top).index.tolist()
    filtered = pair_df[pair_df["group"].isin(shown_groups)].copy()

    stats_df = filtered.groupby("group")["value"].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    ).reset_index()
    q1 = filtered.groupby("group")["value"].quantile(0.25).rename("q1")
    q3 = filtered.groupby("group")["value"].quantile(0.75).rename("q3")
    stats_df = stats_df.merge(q1, on="group").merge(q3, on="group")
    stats_df["order"] = stats_df["group"].map({g: i for i, g in enumerate(shown_groups)})
    stats_df = stats_df.sort_values("order")

    safe_stats = []
    for _, row in stats_df.iterrows():
        safe_stats.append({
            "group": str(row["group"]),
            "count": int(row["count"]),
            "mean": round(float(row["mean"]), 6),
            "median": round(float(row["median"]), 6),
            "std": None if pd.isna(row["std"]) else round(float(row["std"]), 6),
            "min": round(float(row["min"]), 6),
            "q1": round(float(row["q1"]), 6),
            "q3": round(float(row["q3"]), 6),
            "max": round(float(row["max"]), 6),
        })

    max_n_points = max(200, min(int(max_points), 10000))
    group_limit = max(20, int(np.ceil(max_n_points / max(1, len(shown_groups)))))
    sampled_parts = []
    for g in shown_groups:
        g_df = filtered[filtered["group"] == g]
        if len(g_df) > group_limit:
            g_df = g_df.sample(n=group_limit, random_state=42)
        sampled_parts.append(g_df)
    sampled = pd.concat(sampled_parts, axis=0) if sampled_parts else filtered.head(0)

    points = [
        {"group": str(g), "value": round(float(v), 6)}
        for g, v in sampled[["group", "value"]].itertuples(index=False, name=None)
    ]

    total_unique_groups = int(group_counts.shape[0])
    warning = None
    if total_unique_groups > len(shown_groups):
        warning = f"Column '{group_column}' has {total_unique_groups} groups. Showing top {len(shown_groups)} most frequent groups."

    return {
        "value_column": value_column,
        "group_column": group_column,
        "total_groups": total_unique_groups,
        "shown_groups": shown_groups,
        "top_groups": len(shown_groups),
        "warning": warning,
        "stats": safe_stats,
        "points": points,
        "total_pairs": int(filtered.shape[0]),
        "sampled_points": len(points),
    }


@router.get("/scatter-2d")
def get_scatter_2d(
    column_x: str,
    column_y: str,
    processed: bool = False,
    hue_column: Optional[str] = None,
    max_points: int = 2500,
    bins: int = 12,
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """2D scatter payload with marginal histograms, correlation and optional coloring."""
    session = get_session(x_session_id)
    df = session.processed_data if processed and session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    if column_x not in df.columns or column_y not in df.columns:
        raise HTTPException(status_code=404, detail="X or Y column not found")

    if not pd.api.types.is_numeric_dtype(df[column_x]) or not pd.api.types.is_numeric_dtype(df[column_y]):
        raise HTTPException(status_code=400, detail="Scatter 2D requires numeric X and Y columns")

    use_hue = bool(hue_column and hue_column in df.columns and hue_column not in {column_x, column_y})
    cols = [column_x, column_y] + ([hue_column] if use_hue else [])
    work = df[cols].copy()
    work[column_x] = to_finite_numeric(work[column_x])
    work[column_y] = to_finite_numeric(work[column_y])
    work = work.dropna(subset=[column_x, column_y])
    if work.empty:
        raise HTTPException(status_code=400, detail="No finite paired numeric values for selected columns")

    if use_hue:
        work[hue_column] = work[hue_column].fillna("__MISSING__").astype(str)
    else:
        work["__hue__"] = "All"
        hue_column = "__hue__"

    total_points = int(work.shape[0])
    n_max = max(200, min(int(max_points), 12000))
    if total_points > n_max:
        parts = []
        for _, gdf in work.groupby(hue_column):
            n_take = max(1, int(round((len(gdf) / total_points) * n_max)))
            if len(gdf) > n_take:
                gdf = gdf.sample(n=n_take, random_state=42)
            parts.append(gdf)
        work = pd.concat(parts, axis=0).head(n_max)

    x_values = work[column_x].to_numpy(dtype=float)
    y_values = work[column_y].to_numpy(dtype=float)
    if x_values.size < 2 or np.std(x_values) == 0 or np.std(y_values) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x_values, y_values)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    trendline = None
    if x_values.size >= 2 and np.std(x_values) > 0:
        try:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            trendline = {
                "slope": float(slope),
                "intercept": float(intercept),
                "x_min": float(np.min(x_values)),
                "x_max": float(np.max(x_values)),
                "y_min": float(slope * np.min(x_values) + intercept),
                "y_max": float(slope * np.max(x_values) + intercept),
            }
        except Exception:
            trendline = None

    n_bins = max(6, min(int(bins), 40))
    x_counts, x_edges = np.histogram(x_values, bins=n_bins)
    y_counts, y_edges = np.histogram(y_values, bins=n_bins)
    x_hist = []
    y_hist = []
    for i, c in enumerate(x_counts):
        x_hist.append({
            "bin_start": round(float(x_edges[i]), 6),
            "bin_end": round(float(x_edges[i + 1]), 6),
            "label": f"{x_edges[i]:.2f}-{x_edges[i + 1]:.2f}",
            "count": int(c),
        })
    for i, c in enumerate(y_counts):
        y_hist.append({
            "bin_start": round(float(y_edges[i]), 6),
            "bin_end": round(float(y_edges[i + 1]), 6),
            "label": f"{y_edges[i]:.2f}-{y_edges[i + 1]:.2f}",
            "count": int(c),
        })

    points = []
    for _, row in work.iterrows():
        points.append({
            "x": round(float(row[column_x]), 6),
            "y": round(float(row[column_y]), 6),
            "hue": str(row[hue_column]),
        })

    hue_levels = work[hue_column].value_counts().index.tolist()
    return {
        "column_x": column_x,
        "column_y": column_y,
        "hue_column": None if hue_column == "__hue__" else hue_column,
        "hue_levels": hue_levels,
        "points": points,
        "x_hist": x_hist,
        "y_hist": y_hist,
        "pearson_correlation": round(corr, 6),
        "trendline": trendline,
        "total_points": total_points,
        "sampled_points": int(work.shape[0]),
    }


# ==================== TARGET COLUMN ====================

class TargetRequest(BaseModel):
    target_column: str

@router.post("/set-target")
def set_target(req: TargetRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Set the target column for the dataset"""
    session = get_session(x_session_id)
    if session.raw_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    if req.target_column not in session.raw_data.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_column}' not found")

    session.target_column = req.target_column
    dist = session.raw_data[req.target_column].value_counts().to_dict()
    return {"message": f"Target set to '{req.target_column}'", "distribution": {str(k): v for k, v in dist.items()}}


# ==================== MISSING VALUE HANDLING ====================

class MissingValueRequest(BaseModel):
    method: str  # Mean Imputation, Median Imputation, Mode Imputation, Constant Value, Drop Rows, Forward Fill, Backward Fill
    columns: List[str]
    constant_value: Optional[float] = None

@router.post("/handle-missing")
def handle_missing_values(req: MissingValueRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Handle missing values in specified columns"""
    session = get_session(x_session_id)
    if req.method == "Constant Value" and req.constant_value is None:
        raise HTTPException(status_code=400, detail="constant_value is required when method is 'Constant Value'")

    # Determine which data to work on
    if session.X_train is not None:
        # Data is already split — fit on train, transform all (prevent leakage)
        if session.pipeline is None:
            session.pipeline = create_pipeline()

        results = {}
        for col in req.columns:
            if col not in session.X_train.columns:
                continue
            # Fit on train
            info = session.pipeline.fit_imputer(session.X_train, col, req.method, constant_value=req.constant_value)
            # Transform all splits
            session.X_train = session.pipeline.transform_imputation(session.X_train, col)
            session.X_valid = session.pipeline.transform_imputation(session.X_valid, col)
            session.X_test = session.pipeline.transform_imputation(session.X_test, col)
            results[col] = {"method": req.method, "fill_value": str(info.get("fill_value", "N/A"))}

        _mark_step_completed(session, "missing")
        return {
            "message": f"Missing values handled ({req.method}) on {len(results)} columns (fit on train, transform all splits)",
            "info": results,
            "train_shape": list(session.X_train.shape),
        }
    else:
        # Data not yet split — apply on raw/processed data
        df = session.processed_data if session.processed_data is not None else session.raw_data
        if df is None:
            raise HTTPException(status_code=404, detail="No data loaded")

        result_df = df.copy()
        results = {}
        for col in req.columns:
            if col not in result_df.columns:
                continue
            missing_before = int(result_df[col].isnull().sum())
            if req.method == "Drop Rows":
                result_df = result_df[result_df[col].notna()]
            elif req.method == "Mean Imputation":
                fill = result_df[col].mean()
                result_df[col].fillna(fill, inplace=True)
            elif req.method == "Median Imputation":
                fill = result_df[col].median()
                result_df[col].fillna(fill, inplace=True)
            elif req.method == "Mode Imputation":
                mode = result_df[col].mode()
                fill = mode[0] if len(mode) > 0 else 0
                result_df[col].fillna(fill, inplace=True)
            elif req.method == "Constant Value":
                result_df[col].fillna(req.constant_value, inplace=True)
            elif req.method == "Forward Fill":
                result_df[col].fillna(method='ffill', inplace=True)
            elif req.method == "Backward Fill":
                result_df[col].fillna(method='bfill', inplace=True)

            missing_after = int(result_df[col].isnull().sum())
            results[col] = {"before": missing_before, "after": missing_after}

        session.processed_data = result_df
        _mark_step_completed(session, "missing")
        return {
            "message": f"Missing values handled ({req.method}) on {len(results)} columns",
            "info": results,
            "new_shape": list(result_df.shape),
        }


# ==================== ENCODING ====================

class EncodingRequest(BaseModel):
    method: str  # One-Hot, Label, Target, Ordinal, Frequency
    columns: List[str]
    target_column: Optional[str] = None
    drop_first: bool = False
    smoothing: float = 1.0
    ordinal_mappings: Optional[Dict[str, List[str]]] = None

@router.post("/encode")
def encode_columns(req: EncodingRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Apply categorical encoding to specified columns"""
    session = get_session(x_session_id)
    df = session.processed_data if session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    kwargs = {}
    if req.method == "Target Encoding":
        kwargs["target_column"] = req.target_column or session.target_column
    if req.method == "One-Hot Encoding":
        kwargs["drop_first"] = req.drop_first
    if req.method == "Ordinal Encoding" and req.ordinal_mappings:
        kwargs["ordinal_mappings"] = req.ordinal_mappings
    if req.method == "Target Encoding":
        kwargs["smoothing"] = req.smoothing

    try:
        if session.X_train is not None:
            if session.pipeline is None: session.pipeline = create_pipeline()
            for col in req.columns:
                if col not in session.X_train.columns: continue
                # We inject target_column explicitly into train_data for target encoding
                if req.method == "Target Encoding":
                    temp_train = session.X_train.copy()
                    temp_train[kwargs["target_column"]] = session.y_train
                    session.pipeline.fit_encoder(temp_train, col, req.method, **kwargs)
                else:
                    session.pipeline.fit_encoder(session.X_train, col, req.method, **kwargs)

                session.X_train = session.pipeline.transform_encoding(session.X_train, col)
                if session.X_valid is not None and col in session.X_valid.columns:
                    session.X_valid = session.pipeline.transform_encoding(session.X_valid, col)
                if session.X_test is not None and col in session.X_test.columns:
                    session.X_test = session.pipeline.transform_encoding(session.X_test, col)
            session.selected_features = session.X_train.columns.tolist()
            _mark_step_completed(session, "encoding")
            return {"message": "Encoding applied to split datasets", "info": {"columns": req.columns}, "new_shape": list(session.X_train.shape)}
        else:
            processed, info = encode_categorical(df, req.method, req.columns, **kwargs)
            session.processed_data = processed
            session.encoding_config[f"{req.method}_{','.join(req.columns)}"] = info
            safe_info = {}
            for k, v in info.items():
                safe_v = {}
                for k2, v2 in v.items():
                    try:
                        json.dumps(v2)
                        safe_v[k2] = v2
                    except (TypeError, ValueError):
                        safe_v[k2] = str(v2)
                safe_info[k] = safe_v
            _mark_step_completed(session, "encoding")
            return {"message": "Encoding applied", "info": safe_info, "new_shape": list(processed.shape)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== OUTLIERS ====================

class OutlierRequest(BaseModel):
    method: str  # Winsorization, IQR Method, Z-Score, Keep All
    columns: List[str]
    multiplier: float = 1.5
    threshold: float = 3.0
    lower_percentile: float = 0.05
    upper_percentile: float = 0.95
    action: str = "clip"  # clip, remove, nan

@router.post("/outliers")
def handle_outliers_endpoint(req: OutlierRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Handle outliers in specified columns"""
    session = get_session(x_session_id)
    df = session.processed_data if session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    kwargs = {"action": req.action}
    if req.method == "Winsorization":
        kwargs["lower_percentile"] = req.lower_percentile
        kwargs["upper_percentile"] = req.upper_percentile
    elif req.method == "IQR Method":
        kwargs["multiplier"] = req.multiplier
    elif req.method == "Z-Score":
        kwargs["threshold"] = req.threshold

    try:
        if session.X_train is not None:
            if session.pipeline is None: session.pipeline = create_pipeline()
            # For actions that REMOVE rows, we must ALSO update y_train!
            if req.action == "remove":
                processed, _ = handle_outliers(session.X_train, req.method, req.columns, **kwargs)
                kept_indices = processed.index
                session.X_train = processed
                session.y_train = session.y_train.loc[kept_indices]
                # we don't automatically drop validation/test rows to prevent leakage, typically clip them
            else:
                for col in req.columns:
                    if col not in session.X_train.columns: continue
                    session.pipeline.fit_outlier_bounds(session.X_train, col, req.method, **kwargs)
                    session.X_train = session.pipeline.transform_outliers(session.X_train, col, action=req.action)
                    if session.X_valid is not None and col in session.X_valid.columns:
                        session.X_valid = session.pipeline.transform_outliers(session.X_valid, col, action='clip') # Usually clip valid
                    if session.X_test is not None and col in session.X_test.columns:
                        session.X_test = session.pipeline.transform_outliers(session.X_test, col, action='clip')
            
            _mark_step_completed(session, "outliers")
            return {"message": "Outliers handled on split datasets", "info": {}, "new_shape": list(session.X_train.shape)}
        else:
            processed, info = handle_outliers(df, req.method, req.columns, **kwargs)
            session.processed_data = processed
            session.outlier_config[f"{req.method}_{','.join(req.columns)}"] = info
            safe_info = {}
            for k, v in info.items():
                safe_v = {k2: (v2 if isinstance(v2, (str, int, float, bool, type(None))) else str(v2)) for k2, v2 in v.items() if k2 != "outliers_mask"}
                safe_info[k] = safe_v
            _mark_step_completed(session, "outliers")
            return {"message": "Outliers handled", "info": safe_info, "new_shape": list(processed.shape)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== SKEWNESS / DISTRIBUTION TRANSFORM ====================

class SkewnessTransformRequest(BaseModel):
    method: str  # Log, Sqrt, Box-Cox, Yeo-Johnson, Reciprocal
    columns: List[str]

@router.post("/transform-skewness")
def transform_skewness(req: SkewnessTransformRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Apply distribution transformation to reduce skewness"""
    session = get_session(x_session_id)
    is_split = session.X_train is not None

    if is_split:
        df_target = session.X_train
    else:
        df_target = session.processed_data if session.processed_data is not None else session.raw_data
        if df_target is None:
            raise HTTPException(status_code=404, detail="No data loaded")

    from scipy import stats as scipy_stats

    results = {}
    
    def apply_transform(data_series, method, min_val):
        if method == "Log":
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            return np.log1p(data_series + shift)
        elif method == "Sqrt":
            shift = abs(min_val) if min_val < 0 else 0
            return np.sqrt(data_series + shift)
        elif method == "Reciprocal":
            return 1.0 / (data_series.replace(0, np.nan).fillna(1e-6))
        return data_series

    # Calculate transformations and immediately apply them
    if is_split:
        for col in req.columns:
            if col not in session.X_train.columns: continue
            if not pd.api.types.is_numeric_dtype(session.X_train[col]):
                results[col] = {"error": "Not a numeric column"}
                continue
            
            col_data = session.X_train[col].dropna()
            skew_before = float(col_data.skew())
            min_val = col_data.min()

            try:
                if req.method in ["Log", "Sqrt", "Reciprocal"]:
                    session.X_train[col] = apply_transform(session.X_train[col], req.method, min_val)
                    if session.X_valid is not None and col in session.X_valid.columns:
                        session.X_valid[col] = apply_transform(session.X_valid[col], req.method, min_val)
                    if session.X_test is not None and col in session.X_test.columns:
                        session.X_test[col] = apply_transform(session.X_test[col], req.method, min_val)
                        
                elif req.method == "Box-Cox":
                    shift = abs(min_val) + 1 if min_val <= 0 else 0
                    transformed, lmbda = scipy_stats.boxcox(col_data + shift)
                    session.X_train.loc[session.X_train[col].notna(), col] = transformed
                    # Use lambda on valid/test
                    if session.X_valid is not None and col in session.X_valid.columns:
                        session.X_valid.loc[session.X_valid[col].notna(), col] = scipy_stats.boxcox(session.X_valid.loc[session.X_valid[col].notna(), col] + shift, lmbda=lmbda)
                    if session.X_test is not None and col in session.X_test.columns:
                        session.X_test.loc[session.X_test[col].notna(), col] = scipy_stats.boxcox(session.X_test.loc[session.X_test[col].notna(), col] + shift, lmbda=lmbda)
                    results[col] = {"skew_before": round(skew_before, 4), "lambda": round(float(lmbda), 4)}

                elif req.method == "Yeo-Johnson":
                    transformed, lmbda = scipy_stats.yeojohnson(col_data)
                    session.X_train.loc[session.X_train[col].notna(), col] = transformed
                    if session.X_valid is not None and col in session.X_valid.columns:
                        session.X_valid.loc[session.X_valid[col].notna(), col] = scipy_stats.yeojohnson(session.X_valid.loc[session.X_valid[col].notna(), col], lmbda=lmbda)
                    if session.X_test is not None and col in session.X_test.columns:
                        session.X_test.loc[session.X_test[col].notna(), col] = scipy_stats.yeojohnson(session.X_test.loc[session.X_test[col].notna(), col], lmbda=lmbda)
                    results[col] = {"skew_before": round(skew_before, 4), "lambda": round(float(lmbda), 4)}

                skew_after = float(session.X_train[col].dropna().skew())
                if col not in results: results[col] = {"skew_before": round(skew_before, 4)}
                results[col]["skew_after"] = round(skew_after, 4)
                results[col]["method"] = req.method
                results[col]["improved"] = abs(skew_after) < abs(skew_before)

            except Exception as e:
                results[col] = {"error": str(e)}

        session.selected_features = session.X_train.columns.tolist()
        _mark_step_completed(session, "skewness")
        return {
            "message": f"Skewness transformation ({req.method}) applied on {len(results)} columns on split data",
            "results": results,
            "new_shape": list(session.X_train.shape),
        }
    else:
        result_df = df_target.copy()
        for col in req.columns:
            if col not in result_df.columns: continue
            if not pd.api.types.is_numeric_dtype(result_df[col]):
                results[col] = {"error": "Not a numeric column"}
                continue
            
            col_data = result_df[col].dropna()
            skew_before = float(col_data.skew())

            try:
                if req.method == "Log":
                    min_val = col_data.min()
                    shift = abs(min_val) + 1 if min_val <= 0 else 0
                    result_df[col] = np.log1p(result_df[col] + shift)
                elif req.method == "Sqrt":
                    min_val = col_data.min()
                    shift = abs(min_val) if min_val < 0 else 0
                    result_df[col] = np.sqrt(result_df[col] + shift)
                elif req.method == "Box-Cox":
                    min_val = col_data.min()
                    shift = abs(min_val) + 1 if min_val <= 0 else 0
                    transformed, lmbda = scipy_stats.boxcox(col_data + shift)
                    result_df.loc[result_df[col].notna(), col] = transformed
                    results[col] = {"skew_before": round(skew_before, 4), "lambda": round(float(lmbda), 4)}
                elif req.method == "Yeo-Johnson":
                    transformed, lmbda = scipy_stats.yeojohnson(col_data)
                    result_df.loc[result_df[col].notna(), col] = transformed
                    results[col] = {"skew_before": round(skew_before, 4), "lambda": round(float(lmbda), 4)}
                elif req.method == "Reciprocal":
                    result_df[col] = 1.0 / (result_df[col].replace(0, np.nan).fillna(1e-6))

                skew_after = float(result_df[col].dropna().skew())
                if col not in results: results[col] = {"skew_before": round(skew_before, 4)}
                results[col]["skew_after"] = round(skew_after, 4)
                results[col]["method"] = req.method
                results[col]["improved"] = abs(skew_after) < abs(skew_before)

            except Exception as e:
                results[col] = {"error": str(e)}

        session.processed_data = result_df
        _mark_step_completed(session, "skewness")
        return {
            "message": f"Skewness transformation ({req.method}) applied on {len(results)} columns",
            "results": results,
            "new_shape": list(result_df.shape),
        }


# ==================== WoE / IV BINNING ====================

class BinningRequest(BaseModel):
    columns: List[str]
    max_n_bins: int = 10
    min_bin_size: float = 0.05
    target_column: Optional[str] = None
    monotonic_trend: Optional[str] = "auto"
    new_column_name: Optional[str] = None

@router.post("/binning")
def apply_binning(req: BinningRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Apply Optimal Binning (WoE/IV) on specified columns"""
    session = get_session(x_session_id)
    target = req.target_column or session.target_column
    if not target: raise HTTPException(status_code=400, detail="Target column not set")

    is_split = session.X_train is not None
    if is_split:
        df_target = pd.concat([session.X_train, pd.Series(session.y_train, name=target, index=session.X_train.index)], axis=1)
        y_fit = session.y_train
        df_fit = session.X_train
    else:
        df_target = session.processed_data if session.processed_data is not None else session.raw_data
        if df_target is None: raise HTTPException(status_code=404, detail="No data loaded")
        y_fit = df_target[target]
        df_fit = df_target

    if target not in df_target.columns: raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in data")

    unique_vals = y_fit.dropna().unique()
    if len(unique_vals) != 2:
        raise HTTPException(status_code=400, detail=f"Target must be binary (0/1). Found {len(unique_vals)} unique values: {list(unique_vals[:5])}")

    try:
        from optbinning import OptimalBinning
    except ImportError:
        raise HTTPException(status_code=500, detail="optbinning not installed. Run: pip install optbinning")

    results = {}
    total_iv = 0
    transformers = {}

    for col in req.columns:
        if col not in df_fit.columns or col == target:
            continue
        try:
            x = df_fit[col].values
            dtype = "numerical" if pd.api.types.is_numeric_dtype(df_fit[col]) else "categorical"

            optb = OptimalBinning(name=col, dtype=dtype, max_n_bins=req.max_n_bins, min_bin_size=req.min_bin_size, solver="cp", monotonic_trend=req.monotonic_trend)
            optb.fit(x, y_fit.values)
            transformers[col] = optb

            binning_table = optb.binning_table.build()
            iv = float(binning_table["IV"].iloc[-1]) if "IV" in binning_table.columns else 0.0
            total_iv += iv

            safe_table = []
            for _, row in binning_table.iterrows():
                safe_row = {}
                for k, v in row.items():
                    try:
                        if pd.isna(v): safe_row[k] = None
                        elif isinstance(v, (int, float, np.integer, np.floating)): safe_row[k] = round(float(v), 4)
                        else: safe_row[k] = str(v)
                    except: safe_row[k] = str(v)
                safe_table.append(safe_row)

            results[col] = {
                "iv": round(iv, 4),
                "n_bins": len(binning_table) - 2,
                "predictive_power": "Strong" if iv > 0.3 else "Medium" if iv > 0.1 else "Weak" if iv > 0.02 else "Useless",
                "table": safe_table,
            }
        except Exception as e:
            results[col] = {"error": str(e)}

    if is_split:
        for col, optb in transformers.items():
            out_col = req.new_column_name if (req.new_column_name and len(req.columns) == 1) else f"{col}_woe"
            session.X_train[out_col] = optb.transform(session.X_train[col].values, metric="woe")
            if session.X_valid is not None and col in session.X_valid.columns:
                session.X_valid[out_col] = optb.transform(session.X_valid[col].values, metric="woe")
            if session.X_test is not None and col in session.X_test.columns:
                session.X_test[out_col] = optb.transform(session.X_test[col].values, metric="woe")
            session.binning_config[out_col] = {"iv": results[col].get("iv", 0), "n_bins": results[col].get("n_bins", 0)}
        session.selected_features = session.X_train.columns.tolist()
        _mark_step_completed(session, "binning")
        return {
            "message": f"WoE/IV binning applied on {len(transformers)} columns on split datasets.",
            "results": results,
            "new_shape": list(session.X_train.shape),
        }
    else:
        result_df = df_target.copy()
        for col, optb in transformers.items():
            out_col = req.new_column_name if (req.new_column_name and len(req.columns) == 1) else f"{col}_woe"
            result_df[out_col] = optb.transform(result_df[col].values, metric="woe")
            session.binning_config[out_col] = {"iv": results[col].get("iv", 0), "n_bins": results[col].get("n_bins", 0)}
        session.processed_data = result_df
        _mark_step_completed(session, "binning")
        return {
            "message": f"WoE/IV binning applied on {len(transformers)} columns. Total IV: {round(total_iv, 4)}",
            "results": results,
            "new_shape": list(result_df.shape),
        }


# ==================== WoE ANALYSIS ====================

class WoeAnalysisRequest(BaseModel):
    columns: Optional[List[str]] = None  # None = auto-detect numeric columns
    target_column: Optional[str] = None

@router.post("/woe-analysis")
def woe_analysis(req: WoeAnalysisRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Analyze features using WoE/IV to classify predictive power (no data transformation)"""
    session = get_session(x_session_id)

    # Use split data if available, otherwise processed/raw
    if session.X_train is not None:
        df = pd.concat([session.X_train, pd.Series(session.y_train, name=session.target_column, index=session.X_train.index)], axis=1)
    else:
        df = session.processed_data if session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    target = req.target_column or session.target_column
    if not target or target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not set or not found")

    y = df[target]
    unique_vals = y.dropna().unique()
    if len(unique_vals) != 2:
        raise HTTPException(status_code=400, detail=f"Target must be binary. Found {len(unique_vals)} unique values.")

    try:
        from optbinning import OptimalBinning
    except ImportError:
        raise HTTPException(status_code=500, detail="optbinning not installed")

    cols = req.columns or [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]

    results = {}
    for col in cols:
        if col not in df.columns or col == target:
            continue
        try:
            x = df[col].values
            dtype = "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
            optb = OptimalBinning(name=col, dtype=dtype, solver="cp")
            optb.fit(x, y.values)

            bt = optb.binning_table.build()
            iv = float(bt["IV"].iloc[-1]) if "IV" in bt.columns else 0.0

            # Extract WoE per bin for analysis (not transforming data)
            woe_bins = []
            for _, row in bt.iterrows():
                try:
                    woe_bins.append({
                        "bin": str(row.get("Bin", "")),
                        "count": int(row["Count"]) if pd.notna(row.get("Count")) else 0,
                        "woe": round(float(row["WoE"]), 4) if pd.notna(row.get("WoE")) else None,
                        "iv": round(float(row["IV"]), 4) if pd.notna(row.get("IV")) else None,
                    })
                except:
                    pass

            if iv > 0.3:
                power = "Strong"
                recommendation = "Excellent predictor — keep"
            elif iv > 0.1:
                power = "Medium"
                recommendation = "Good predictor — keep"
            elif iv > 0.02:
                power = "Weak"
                recommendation = "Marginal predictor — consider removing"
            else:
                power = "Useless"
                recommendation = "No predictive power — remove"

            results[col] = {
                "iv": round(iv, 4),
                "predictive_power": power,
                "recommendation": recommendation,
                "bins": woe_bins,
            }
        except Exception as e:
            results[col] = {"iv": 0, "predictive_power": "Error", "recommendation": str(e), "bins": []}

    # Sort by IV descending
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1].get("iv", 0), reverse=True))

    _mark_step_completed(session, "woe_analysis")
    return {
        "message": f"WoE/IV analysis completed on {len(results)} features",
        "results": sorted_results,
        "summary": {
            "strong": sum(1 for v in results.values() if v.get("predictive_power") == "Strong"),
            "medium": sum(1 for v in results.values() if v.get("predictive_power") == "Medium"),
            "weak": sum(1 for v in results.values() if v.get("predictive_power") == "Weak"),
            "useless": sum(1 for v in results.values() if v.get("predictive_power") == "Useless"),
        },
    }


# ==================== MULTICOLLINEARITY DETECTION ====================

class MulticollinearityRequest(BaseModel):
    vif_threshold: float = 10.0
    corr_threshold: float = 0.8
    auto_remove: bool = False  # If True, removes features with VIF > threshold

@router.post("/multicollinearity")
def check_multicollinearity(req: MulticollinearityRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Detect and optionally handle multicollinearity using VIF and correlation"""
    session = get_session(x_session_id)

    # Use split data if available
    if session.X_train is not None:
        df_numeric = session.X_train.select_dtypes(include=[np.number])
    else:
        df = session.processed_data if session.processed_data is not None else session.raw_data
        if df is None:
            raise HTTPException(status_code=404, detail="No data loaded")
        target = session.target_column
        df_work = df.drop(columns=[target]) if target and target in df.columns else df
        df_numeric = df_work.select_dtypes(include=[np.number])

    if df_numeric.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric features")

    # Drop columns with NaN for VIF calculation
    df_clean = df_numeric.dropna(axis=1, how='any')
    if df_clean.shape[1] < 2:
        df_clean = df_numeric.fillna(df_numeric.median())

    # Calculate VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = []
    try:
        for i, col in enumerate(df_clean.columns):
            vif = variance_inflation_factor(df_clean.values, i)
            vif_data.append({
                "feature": col,
                "vif": round(float(vif), 2) if np.isfinite(vif) else 999.99,
                "high_vif": bool(vif > req.vif_threshold),
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"VIF calculation error: {str(e)}")

    vif_data.sort(key=lambda x: x["vif"], reverse=True)

    # High-correlation pairs
    corr_matrix = df_clean.corr().abs()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val >= req.corr_threshold:
                high_corr_pairs.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": round(float(val), 4),
                })
    high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)

    # Auto-remove if requested
    removed_features = []
    if req.auto_remove:
        to_remove = [v["feature"] for v in vif_data if v["high_vif"]]
        if to_remove and session.X_train is not None:
            session.X_train = session.X_train.drop(columns=[c for c in to_remove if c in session.X_train.columns], errors='ignore')
            session.X_valid = session.X_valid.drop(columns=[c for c in to_remove if c in session.X_valid.columns], errors='ignore')
            session.X_test = session.X_test.drop(columns=[c for c in to_remove if c in session.X_test.columns], errors='ignore')
            removed_features = to_remove
        elif to_remove and session.processed_data is not None:
            session.processed_data = session.processed_data.drop(columns=[c for c in to_remove if c in session.processed_data.columns], errors='ignore')
            removed_features = to_remove

    high_vif_count = sum(1 for v in vif_data if v["high_vif"])
    _mark_step_completed(session, "multicollinearity")

    return {
        "message": f"Multicollinearity analysis: {high_vif_count} features with VIF > {req.vif_threshold}, {len(high_corr_pairs)} highly correlated pairs",
        "vif": vif_data,
        "high_correlation_pairs": high_corr_pairs,
        "removed_features": removed_features,
        "summary": {
            "total_features": len(vif_data),
            "high_vif_count": high_vif_count,
            "high_corr_pairs": len(high_corr_pairs),
        },
    }


# ==================== BALANCING ====================

class BalanceRequest(BaseModel):
    method: str = "SMOTE"
    target_column: Optional[str] = None

@router.post("/balance")
def balance_dataset(req: BalanceRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Balance the dataset using SMOTE or other methods"""
    session = get_session(x_session_id)
    df = session.processed_data if session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    target = req.target_column or session.target_column
    if not target:
        raise HTTPException(status_code=400, detail="Target column not set")

    try:
        balanced_df, info = balance_data(df, target, method=req.method)
        session.processed_data = balanced_df
        safe_info = to_jsonable(info)
        session.balance_info = safe_info
        _mark_step_completed(session, "balance")
        return {"message": safe_info.get("message", "Balanced"), "info": safe_info, "new_shape": list(balanced_df.shape)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== TRAIN/TEST SPLIT ====================

class SplitRequest(BaseModel):
    target_column: Optional[str] = None
    test_size: float = 0.2
    valid_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

@router.post("/split")
def split_data(req: SplitRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Split data into train/valid/test sets"""
    from sklearn.model_selection import train_test_split

    session = get_session(x_session_id)
    df = session.processed_data if session.processed_data is not None else session.raw_data
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    target = req.target_column or session.target_column
    if not target:
        raise HTTPException(status_code=400, detail="Target column not set")

    try:
        X = df.drop(columns=[target])
        y = df[target]

        train_ratio = 1.0 - req.test_size - req.valid_size
        if train_ratio <= 0:
            raise HTTPException(status_code=400, detail="Invalid split ratios: train ratio must be > 0")

        # First split: train vs (valid+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            train_size=train_ratio,
            random_state=req.random_state,
            stratify=y if req.stratify else None,
        )

        # Second split: valid vs test from remaining subset
        temp_total = req.valid_size + req.test_size
        valid_ratio_in_temp = req.valid_size / temp_total if temp_total > 0 else 0.5
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=valid_ratio_in_temp,
            random_state=req.random_state,
            stratify=y_temp if req.stratify else None,
        )

        session.X_train = X_train
        session.X_valid = X_valid
        session.X_test = X_test
        session.y_train = y_train
        session.y_valid = y_valid
        session.y_test = y_test
        session.target_column = target
        session.selected_features = X_train.columns.tolist()
        _mark_step_completed(session, "split")

        return {
            "message": "Data split successfully",
            "train_size": len(X_train),
            "valid_size": len(X_valid),
            "test_size": len(X_test),
            "features": X_train.columns.tolist(),
            "n_features": len(X_train.columns),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== SCALING ====================

class ScalingRequest(BaseModel):
    method: str  # StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
    columns: Optional[List[str]] = None

@router.post("/scale")
def scale_data(req: ScalingRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Scale numeric columns using the preprocessing pipeline"""
    session = get_session(x_session_id)
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet. Call /split first.")

    columns = req.columns or session.X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not columns:
        raise HTTPException(status_code=400, detail="No numeric columns to scale")

    try:
        if session.pipeline is None:
            session.pipeline = create_pipeline()

        session.pipeline.fit_scaler(session.X_train, columns, req.method)
        session.X_train = session.pipeline.transform_scaling(session.X_train, columns)
        session.X_valid = session.pipeline.transform_scaling(session.X_valid, columns)
        session.X_test = session.pipeline.transform_scaling(session.X_test, columns)
        _mark_step_completed(session, "scaling")
        return {"message": f"Scaling applied ({req.method}) on {len(columns)} columns", "columns_scaled": columns}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== FEATURE IMPORTANCE ====================

class FeatureImportanceRequest(BaseModel):
    method: str = "Random Forest"
    top_n: int = 15

@router.post("/feature-importance")
def calculate_importance(req: FeatureImportanceRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Calculate feature importance"""
    from services.feature_importance import calculate_feature_importance

    session = get_session(x_session_id)
    if session.X_train is None or session.y_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet")

    try:
        result = calculate_feature_importance(session.X_train, session.y_train, method=req.method, top_n=req.top_n)
        _mark_step_completed(session, "importance")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class SelectedFeaturesRequest(BaseModel):
    columns: List[str]


@router.post("/selected-features")
def set_selected_features(req: SelectedFeaturesRequest, x_session_id: str = Header(..., alias="X-Session-ID")):
    """Persist selected feature columns for downstream model training/prediction."""
    session = get_session(x_session_id)
    if session.X_train is None:
        raise HTTPException(status_code=400, detail="Data not split yet")

    available_cols = set(session.X_train.columns.tolist())
    selected = []
    for col in req.columns:
        if col in available_cols and col not in selected:
            selected.append(col)

    if not selected:
        raise HTTPException(status_code=400, detail="No valid columns provided")

    session.selected_features = selected
    return {
        "message": f"Selected {len(selected)} features for model training",
        "selected_features": selected,
        "n_features": len(selected),
    }


@router.get("/session-info")
def get_session_info(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Get current session state summary"""
    session = get_session(x_session_id)
    split_missing = {}
    split_feature_columns: List[str] = []
    split_numeric_columns: List[str] = []
    split_categorical_columns: List[str] = []
    split_shape = None
    split_sizes = None
    if session.X_train is not None:
        split_shape = list(session.X_train.shape)
        split_sizes = {
            "train": len(session.X_train),
            "valid": len(session.X_valid) if session.X_valid is not None else 0,
            "test": len(session.X_test) if session.X_test is not None else 0,
        }
        split_feature_columns = session.X_train.columns.tolist()
        split_numeric_columns = session.X_train.select_dtypes(include=[np.number]).columns.tolist()
        split_categorical_columns = session.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        missing_counts = session.X_train.isnull().sum()
        missing_pct = (missing_counts / max(1, len(session.X_train)) * 100).round(2)
        split_missing = {
            col: {"count": int(missing_counts[col]), "percentage": float(missing_pct[col])}
            for col in session.X_train.columns if int(missing_counts[col]) > 0
        }
    return {
        "has_data": session.raw_data is not None,
        "has_processed_data": session.processed_data is not None,
        "has_model": session.model is not None,
        "has_splits": session.X_train is not None,
        "target_column": session.target_column,
        "n_features": len(session.selected_features),
        "data_shape": list(session.raw_data.shape) if session.raw_data is not None else None,
        "processed_shape": list(session.processed_data.shape) if session.processed_data is not None else None,
        "n_trained_models": len(session.trained_models),
        "split_shape": split_shape,
        "split_sizes": split_sizes,
        "split_feature_columns": split_feature_columns,
        "split_numeric_columns": split_numeric_columns,
        "split_categorical_columns": split_categorical_columns,
        "split_missing": split_missing,
        "balance_info": session.balance_info,
        "selected_features": session.selected_features,
        "completed_steps": sorted(list(session.completed_steps)) if hasattr(session, "completed_steps") else [],
    }
