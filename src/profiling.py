"""
Profiling Module for DataGuard AI.
Performs deterministic column-by-column profiling, statistical distribution calculations,
and schema inference using pure Pandas and NumPy.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .models import (
    DatasetMetadata,
    DatasetProfile,
    ColumnProfile,
    ColumnStatistics,
    InferredDataType,
)

def infer_column_type(series: pd.Series) -> InferredDataType:
    """
    Deterministically infers semantic data type for a Pandas Series.
    """
    # Drop NAs for type inspection
    non_null = series.dropna()
    if len(non_null) == 0:
        return InferredDataType.UNKNOWN

    dtype = series.dtype

    # Boolean detection
    if pd.api.types.is_bool_dtype(dtype):
        return InferredDataType.BOOLEAN

    # Check for string booleans (e.g. True/False, Y/N, Yes/No)
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        unique_vals = set(str(v).strip().lower() for v in non_null.unique()[:20])
        if unique_vals.issubset({"true", "false", "y", "n", "yes", "no", "1", "0"}):
            return InferredDataType.BOOLEAN

    # Numeric detection
    if pd.api.types.is_numeric_dtype(dtype):
        # Check if integer with low cardinality might be categorical
        if pd.api.types.is_integer_dtype(dtype) and series.nunique() <= 5 and len(series) > 50:
            return InferredDataType.CATEGORICAL
        return InferredDataType.NUMERIC

    # Datetime detection
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return InferredDataType.DATETIME

    # Try string-to-datetime parsing on a small sample
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        sample = non_null.head(20).astype(str)
        try:
            # Check if strings look like dates (length >= 8 and separators present)
            if all("-" in s or "/" in s or ":" in s for s in sample if len(s) >= 8):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= len(sample) * 0.8:
                    return InferredDataType.DATETIME
        except Exception:
            pass

    # Categorical vs Free Text
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        total_non_null = len(non_null)
        unique_count = series.nunique()
        # If low distinct ratio or max length is small, treat as categorical
        max_str_len = non_null.astype(str).str.len().max()
        if (unique_count / max(1, total_non_null) < 0.20 or unique_count <= 50) and max_str_len < 100:
            return InferredDataType.CATEGORICAL
        return InferredDataType.TEXT

    return InferredDataType.CATEGORICAL

def calculate_column_statistics(series: pd.Series, inferred_type: InferredDataType) -> ColumnStatistics:
    """
    Computes numerical boundaries and top frequency categories for a column.
    """
    stats = ColumnStatistics()
    non_null = series.dropna()

    # Sample values (sanitized up to 5 distinct values)
    distinct_sample = [str(x) for x in non_null.unique()[:5]]
    stats.sample_values = distinct_sample

    if inferred_type == InferredDataType.NUMERIC:
        numeric_series = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric_series) > 0:
            stats.min_val = float(numeric_series.min())
            stats.max_val = float(numeric_series.max())
            stats.mean_val = float(numeric_series.mean())
            stats.median_val = float(numeric_series.median())
            stats.std_val = float(numeric_series.std()) if len(numeric_series) > 1 else 0.0

            q1 = float(numeric_series.quantile(0.25))
            q3 = float(numeric_series.quantile(0.75))
            iqr = q3 - q1
            stats.q1_val = q1
            stats.q3_val = q3
            stats.iqr_val = iqr
            stats.lower_bound = q1 - 1.5 * iqr
            stats.upper_bound = q3 + 1.5 * iqr

    if inferred_type in (InferredDataType.CATEGORICAL, InferredDataType.TEXT, InferredDataType.BOOLEAN):
        counts = non_null.astype(str).value_counts().head(10).to_dict()
        stats.top_categories = {str(k): int(v) for k, v in counts.items()}

    return stats

def generate_dataset_profile(df: pd.DataFrame, metadata: DatasetMetadata) -> DatasetProfile:
    """
    Generates a full DatasetProfile covering all columns.
    """
    columns: Dict[str, ColumnProfile] = {}
    warnings: List[str] = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        # Missing count includes null, NaN, and whitespace-only strings
        if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
            is_empty_str = series.astype(str).str.strip().isin(["", "nan", "none", "null", "nat"])
            null_count = int(series.isna().sum() + is_empty_str.sum() - series.isna().sum())
            null_count = int(is_empty_str.sum())
        else:
            null_count = int(series.isna().sum())

        null_percentage = round((null_count / max(1, total_rows)) * 100, 2)
        unique_count = int(series.nunique())
        unique_percentage = round((unique_count / max(1, total_rows)) * 100, 2)

        inferred_type = infer_column_type(series)
        stats = calculate_column_statistics(series, inferred_type)

        # Detect potential business keys (e.g. name ends in _id, _code, or 100% unique)
        col_lower = str(col).lower()
        is_id_name = any(col_lower.endswith(suffix) for suffix in ["_id", "id", "_key", "code", "no", "number", "email"])
        is_business_key = (is_id_name and unique_percentage >= 95.0) or (unique_percentage == 100.0 and total_rows > 10)

        columns[str(col)] = ColumnProfile(
            column_name=str(col),
            inferred_type=inferred_type,
            total_count=total_rows,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            is_business_key=is_business_key,
            statistics=stats
        )

    if total_rows == 0:
        warnings.append("Dataset contains 0 rows.")

    return DatasetProfile(
        metadata=metadata,
        columns=columns,
        warnings=warnings
    )

profile_dataset = generate_dataset_profile
