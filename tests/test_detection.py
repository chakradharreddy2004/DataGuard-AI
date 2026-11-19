"""
Unit Tests for Detection Modules (src/missing_values.py, duplicates.py, outliers.py, inconsistencies.py).
Tests deterministic detection across dirty benchmark data.
"""
import pytest
import pandas as pd
from src.missing_values import detect_missing_values
from src.duplicates import detect_duplicates
from src.outliers import detect_numeric_outliers
from src.inconsistencies import detect_inconsistencies
from src.models import (
    IssueCategory,
    IssueSeverity,
    DatasetMetadata,
)
from src.profiling import generate_dataset_profile

def test_detect_missing_values():
    df = pd.DataFrame({
        "col_a": ["valid", "", "   ", None, "ok"],
        "col_b": [1.0, 2.0, None, 4.0, 5.0],
        "col_c": ["a", "b", "c", "d", "e"]
    })
    issues = detect_missing_values(df)
    
    missing_cols = [i.column_name for i in issues]
    assert "col_a" in missing_cols
    assert "col_b" in missing_cols
    assert "col_c" not in missing_cols

    col_a_issue = next(i for i in issues if i.column_name == "col_a")
    assert col_a_issue.affected_count == 3  # "", "   ", None

def test_detect_exact_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["A", "B", "B", "C"],
        "city": ["NY", "SF", "SF", "LA"]
    })
    issues = detect_duplicates(df)
    exact_issues = [i for i in issues if i.category == IssueCategory.EXACT_DUPLICATE_ROW]
    assert len(exact_issues) == 1
    assert exact_issues[0].affected_count == 1

def test_detect_business_key_duplicates():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C2", "C3"],
        "email": ["c1@a.com", "c2_first@a.com", "c2_different@b.com", "c3@c.com"],
        "city": ["NY", "SF", "LA", "Chicago"]
    })
    issues = detect_duplicates(df, business_keys=["customer_id"])
    key_issues = [i for i in issues if i.category == IssueCategory.KEY_DUPLICATE]
    assert len(key_issues) == 1
    assert key_issues[0].column_name == "customer_id"
    assert key_issues[0].affected_count == 2

def test_detect_numeric_outliers():
    # 20 standard values, 2 extreme outliers
    data = [50] * 20 + [5000, 10000]
    df = pd.DataFrame({"amount": data})
    meta = DatasetMetadata("data.csv", 100, len(df), 1, "")
    profile = generate_dataset_profile(df, meta)

    issues = detect_numeric_outliers(df, profile, method="iqr")
    assert len(issues) == 1
    assert issues[0].category == IssueCategory.NUMERIC_OUTLIER
    assert issues[0].is_advisory is True
    assert issues[0].affected_count >= 1

def test_detect_inconsistencies():
    df = pd.DataFrame({
        "city": ["  New York  ", "New York", "new york", "NEW YORK", "San Francisco"]
    })
    meta = DatasetMetadata("data.csv", 100, len(df), 1, "")
    profile = generate_dataset_profile(df, meta)

    issues = detect_inconsistencies(df, profile)
    categories = [i.category for i in issues]
    assert IssueCategory.WHITESPACE_INCONSISTENCY in categories
    assert IssueCategory.CASING_INCONSISTENCY in categories
