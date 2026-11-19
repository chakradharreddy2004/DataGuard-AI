"""
Missing-Value Detection Module for DataGuard AI.
Detects standard nulls, empty strings (""), and whitespace-only strings ("   ").
Purely deterministic execution using Pandas.
"""
from typing import List, Optional
import pandas as pd
from .models import (
    DatasetProfile,
    QualityIssue,
    IssueCategory,
    IssueSeverity,
    CleaningActionType,
)

def detect_missing_values(df: pd.DataFrame, profile: Optional[DatasetProfile] = None) -> List[QualityIssue]:
    """
    Identifies missing, null, empty, and whitespace-only values across all columns.
    """
    issues: List[QualityIssue] = []
    total_rows = max(1, len(df))

    for col in df.columns:
        series = df[col]
        # Identify standard nulls
        is_na = series.isna()

        # Identify whitespace and empty strings for text/object types
        is_whitespace = pd.Series(False, index=df.index)
        if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
            str_series = series.astype(str)
            # Empty or whitespace only or common string literals
            is_whitespace = str_series.str.strip().isin(["", "nan", "none", "null", "nat"])

        missing_mask = is_na | is_whitespace
        missing_count = int(missing_mask.sum())

        if missing_count > 0:
            pct = round((missing_count / total_rows) * 100, 2)
            sample_indices = df[missing_mask].index[:5].tolist()

            # Severity calculation
            if pct > 40.0:
                severity = IssueSeverity.CRITICAL
            elif pct > 15.0:
                severity = IssueSeverity.HIGH
            elif pct > 5.0:
                severity = IssueSeverity.MEDIUM
            else:
                severity = IssueSeverity.LOW

            issues.append(QualityIssue(
                issue_id=f"missing_{col}",
                category=IssueCategory.MISSING_VALUE,
                column_name=str(col),
                severity=severity,
                affected_count=missing_count,
                affected_percentage=pct,
                sample_values=["<NULL / EMPTY>"],
                sample_row_indices=sample_indices,
                explanation=f"Column '{col}' has {missing_count} missing or empty values ({pct}% of rows).",
                suggested_action=CleaningActionType.IMPUTE_MISSING if pct < 30.0 else CleaningActionType.DROP_MISSING_ROWS
            ))

    return issues
