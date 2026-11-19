"""
Unit Tests for Profiling Module (src/profiling.py).
Validates dynamic type inference, numerical statistics (IQR bounds, quantiles),
and categorical distribution summaries.
"""
import pytest
import pandas as pd
from src.profiling import infer_column_type, calculate_column_statistics, generate_dataset_profile, profile_dataset
from src.models import InferredDataType, DatasetMetadata

def test_infer_column_type_numeric():
    s = pd.Series([10.5, 20.1, 30.8, 40.2, 50.0])
    assert infer_column_type(s) == InferredDataType.NUMERIC

def test_infer_column_type_boolean():
    s_bool = pd.Series([True, False, True, False])
    assert infer_column_type(s_bool) == InferredDataType.BOOLEAN

    s_str_bool = pd.Series(["yes", "no", "yes", "yes"])
    assert infer_column_type(s_str_bool) == InferredDataType.BOOLEAN

def test_infer_column_type_datetime():
    s_dt = pd.Series(["2026-01-01", "2026-02-15", "2026-03-30"])
    assert infer_column_type(s_dt) == InferredDataType.DATETIME

def test_infer_column_type_categorical():
    s_cat = pd.Series(["USA", "Canada", "Mexico", "USA", "USA"] * 20)
    assert infer_column_type(s_cat) == InferredDataType.CATEGORICAL

def test_calculate_column_statistics_numeric():
    s = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    stats = calculate_column_statistics(s, InferredDataType.NUMERIC)
    assert stats.min_val == 10.0
    assert stats.max_val == 100.0
    assert stats.mean_val == 55.0
    assert stats.iqr_val > 0
    assert len(stats.sample_values) <= 5

def test_generate_dataset_profile(dirty_dataframe, sample_metadata):
    profile = generate_dataset_profile(dirty_dataframe, sample_metadata)
    assert profile.metadata.filename == "test_dataset.csv"
    assert len(profile.columns) == len(dirty_dataframe.columns)
    
    # Check that age column is numeric
    age_col = profile.columns.get("age")
    assert age_col is not None
    assert age_col.inferred_type == InferredDataType.NUMERIC
    assert age_col.null_count >= 0

    # Test alias
    p2 = profile_dataset(dirty_dataframe, sample_metadata)
    assert len(p2.columns) == len(profile.columns)
