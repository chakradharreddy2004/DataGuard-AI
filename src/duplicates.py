"""
Duplicate Detection Module for DataGuard AI.
Implements dual-tier duplicate detection:
1. Exact full-row duplicates across 100% of columns.
2. Business-key duplicates where keys collide with conflicting row attributes.
Does NOT assume repeated values in ordinary categorical columns are duplicates.
"""
from typing import List, Optional
import pandas as pd
from .models import (
    QualityIssue,
    IssueCategory,
    IssueSeverity,
    CleaningActionType,
)

def detect_duplicates(
    df: pd.DataFrame,
    business_keys: Optional[List[str]] = None
) -> List[QualityIssue]:
    """
    Detects exact full-row duplicates and conflicting business-key duplicates.
    """
    issues: List[QualityIssue] = []
    total_rows = max(1, len(df))

    # 1. Exact full-row duplicates (100% identical across all columns)
    exact_dup_mask = df.duplicated(keep="first")
    exact_dup_count = int(exact_dup_mask.sum())

    if exact_dup_count > 0:
        pct = round((exact_dup_count / total_rows) * 100, 2)
        sample_indices = df[exact_dup_mask].index[:5].tolist()
        sample_records = df.loc[sample_indices].head(3).to_dict(orient="records")

        issues.append(QualityIssue(
            issue_id="exact_duplicate_rows",
            category=IssueCategory.EXACT_DUPLICATE_ROW,
            column_name=None,
            severity=IssueSeverity.HIGH if pct > 10.0 else IssueSeverity.MEDIUM,
            affected_count=exact_dup_count,
            affected_percentage=pct,
            sample_values=sample_records,
            sample_row_indices=sample_indices,
            explanation=f"Found {exact_dup_count} exact duplicate rows ({pct}% of dataset) that are 100% identical across all columns.",
            suggested_action=CleaningActionType.DROP_EXACT_DUPLICATES
        ))

    # 2. Business Key Duplicates (Scoped to user-selected identifier columns)
    if business_keys:
        valid_keys = [k for k in business_keys if k in df.columns]
        if valid_keys:
            key_dup_mask = df.duplicated(subset=valid_keys, keep=False)
            # Find subset where keys are duplicate but full row is NOT identical (conflicting attributes)
            conflicting_mask = key_dup_mask & (~df.duplicated(keep=False))
            conflicting_count = int(conflicting_mask.sum())

            if conflicting_count > 0:
                pct = round((conflicting_count / total_rows) * 100, 2)
                sample_indices = df[conflicting_mask].index[:5].tolist()
                sample_records = df.loc[sample_indices, valid_keys].head(3).to_dict(orient="records")

                key_names = ", ".join(valid_keys)
                issues.append(QualityIssue(
                    issue_id=f"business_key_duplicates_{'_'.join(valid_keys)}",
                    category=IssueCategory.BUSINESS_KEY_DUPLICATE,
                    column_name=key_names,
                    severity=IssueSeverity.CRITICAL if pct > 5.0 else IssueSeverity.HIGH,
                    affected_count=conflicting_count,
                    affected_percentage=pct,
                    sample_values=sample_records,
                    sample_row_indices=sample_indices,
                    explanation=(
                        f"Found {conflicting_count} records ({pct}% of rows) sharing identical business keys "
                        f"({key_names}) but containing conflicting data in other columns. "
                        "Requires explicit resolution strategy (Keep First, Keep Last, or Manual Triage)."
                    ),
                    suggested_action=CleaningActionType.RESOLVE_KEY_DUPLICATES
                ))

    return issues
