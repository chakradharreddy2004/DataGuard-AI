"""
Inconsistency Detection Module for DataGuard AI.
Detects casing variations, leading/trailing whitespace, and formatting inconsistencies in text/categorical columns.
"""
from typing import List, Set, Optional
import pandas as pd
from .models import (
    DatasetProfile,
    QualityIssue,
    IssueCategory,
    IssueSeverity,
    CleaningActionType,
    InferredDataType,
)

def detect_inconsistencies(df: pd.DataFrame, profile: Optional[DatasetProfile] = None) -> List[QualityIssue]:
    """
    Identifies casing inconsistencies, whitespace issues, and format anomalies.
    """
    issues: List[QualityIssue] = []
    total_rows = max(1, len(df))

    if profile is None:
        from .profiling import generate_dataset_profile
        from .models import DatasetMetadata
        meta = DatasetMetadata(filename="data.csv", file_size_bytes=1, row_count=len(df), column_count=len(df.columns), ingestion_timestamp="")
        profile = generate_dataset_profile(df, meta)

    for col, col_prof in profile.columns.items():
        if col_prof.inferred_type not in (InferredDataType.CATEGORICAL, InferredDataType.TEXT):
            continue

        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        # 1. Whitespace Inconsistency (Leading, Trailing, or Double Spaces)
        has_whitespace = series.str.startswith(" ") | series.str.endswith(" ") | series.str.contains(r"\s{2,}", regex=True)
        ws_count = int(has_whitespace.sum())

        if ws_count > 0:
            pct = round((ws_count / total_rows) * 100, 2)
            sample_indices = series[has_whitespace].index[:5].tolist()
            sample_values = [f"'{v}'" for v in series[has_whitespace].head(3).tolist()]

            issues.append(QualityIssue(
                issue_id=f"whitespace_{col}",
                category=IssueCategory.WHITESPACE_INCONSISTENCY,
                column_name=str(col),
                severity=IssueSeverity.LOW,
                affected_count=ws_count,
                affected_percentage=pct,
                sample_values=sample_values,
                sample_row_indices=sample_indices,
                explanation=f"Column '{col}' contains {ws_count} values ({pct}% of rows) with superfluous leading, trailing, or double whitespace.",
                suggested_action=CleaningActionType.TRIM_WHITESPACE
            ))

        # 2. Casing Inconsistency
        # Unique original values vs unique lowercase values
        orig_unique = set(series.str.strip().unique())
        lower_map: dict[str, Set[str]] = {}
        for val in orig_unique:
            if not val:
                continue
            lowered = val.lower()
            lower_map.setdefault(lowered, set()).add(val)

        casing_conflicts = {k: v for k, v in lower_map.items() if len(v) > 1}
        if casing_conflicts:
            # Count affected rows with conflicting casing
            conflicting_variants = set.union(*casing_conflicts.values())
            casing_mask = series.str.strip().isin(conflicting_variants)
            casing_count = int(casing_mask.sum())
            pct = round((casing_count / total_rows) * 100, 2)

            sample_indices = series[casing_mask].index[:5].tolist()
            sample_examples = [list(variants) for variants in list(casing_conflicts.values())[:3]]

            issues.append(QualityIssue(
                issue_id=f"casing_{col}",
                category=IssueCategory.CASING_INCONSISTENCY,
                column_name=str(col),
                severity=IssueSeverity.MEDIUM if pct > 5.0 else IssueSeverity.LOW,
                affected_count=casing_count,
                affected_percentage=pct,
                sample_values=sample_examples,
                sample_row_indices=sample_indices,
                explanation=f"Column '{col}' contains {len(casing_conflicts)} value groups with inconsistent casing (e.g. {sample_examples[:2]}), affecting {casing_count} rows.",
                suggested_action=CleaningActionType.STANDARDIZE_CASING
            ))

    return issues
