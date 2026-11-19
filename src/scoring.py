"""
Quality Scoring Engine for DataGuard AI.
Implements DAMA-DMBOK aligned multidimensional data quality scoring with
dynamic proportional re-weighting for non-applicable dimensions.
Deterministic, reproducible, and executed purely in Python.
"""
from typing import Dict, List, Optional
import pandas as pd
from .models import (
    DatasetProfile,
    QualityIssue,
    QualityScorecard,
    DimensionScore,
    IssueCategory,
    InferredDataType,
)

DEFAULT_WEIGHTS = {
    "completeness": 0.35,
    "validity": 0.30,
    "consistency": 0.25,
    "uniqueness": 0.10,
}

def compute_quality_scorecard(
    df: pd.DataFrame,
    profile: DatasetProfile,
    issues: List[QualityIssue],
    business_keys: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None
) -> QualityScorecard:
    """
    Computes deterministic quality scores (0-100) per column and for the overall dataset.
    
    Rules:
    - Completeness: Evaluates null and empty values.
    - Validity: Evaluates numerical/format validity. Non-applicable on free text.
    - Consistency: Evaluates casing, whitespace, spelling variations, and format consistency.
    - Uniqueness: Evaluates key duplicates. Non-applicable on regular categorical/dimension columns.
    - Dynamic re-weighting: Non-applicable dimensions are excluded and remaining active weights normalized to 100%.
    """
    active_weights_base = weights if weights else DEFAULT_WEIGHTS.copy()
    total_rows = max(1, len(df))
    
    # Map issues by column
    issues_by_col: Dict[str, List[QualityIssue]] = {col: [] for col in df.columns}
    dataset_level_issues: List[QualityIssue] = []

    for issue in issues:
        if issue.column_name and issue.column_name in issues_by_col:
            issues_by_col[issue.column_name].append(issue)
        else:
            dataset_level_issues.append(issue)

    column_scores: Dict[str, float] = {}
    column_dimensions: Dict[str, DimensionScore] = {}
    deduction_reasons: List[str] = []

    for col_name, col_profile in profile.columns.items():
        # 1. Completeness Dimension (always applicable)
        missing_count = col_profile.null_count
        missing_issues = [i for i in issues_by_col.get(col_name, []) if i.category == IssueCategory.MISSING_VALUE]
        if missing_issues:
            missing_count = max(missing_count, sum(i.affected_count for i in missing_issues))

        null_ratio = missing_count / total_rows
        completeness_score = max(0.0, 100.0 - (null_ratio * 100.0))
        if completeness_score < 100.0:
            deduction_reasons.append(
                f"Column '{col_name}': Completeness score reduced to {completeness_score:.1f}% "
                f"({missing_count} missing/empty values, {(missing_count/total_rows)*100:.1f}% of rows)."
            )

        # 2. Validity Dimension
        # Applicable primarily to numeric, datetime, and boolean columns
        is_validity_applicable = col_profile.inferred_type in (
            InferredDataType.NUMERIC, 
            InferredDataType.DATETIME, 
            InferredDataType.BOOLEAN
        )
        validity_score = 100.0
        if is_validity_applicable:
            # Check for format anomalies or confirmed invalid values
            val_issues = [
                i for i in issues_by_col.get(col_name, []) 
                if i.category in (IssueCategory.FORMAT_ANOMALY, IssueCategory.UNKNOWN_COLUMN)
            ]
            for i in val_issues:
                penalty = min(50.0, (i.affected_count / total_rows) * 100.0)
                validity_score = max(0.0, validity_score - penalty)
                deduction_reasons.append(
                    f"Column '{col_name}': Validity reduced by {penalty:.1f}% due to {i.category.value}."
                )

        # 3. Consistency Dimension (always applicable)
        consistency_score = 100.0
        cons_issues = [
            i for i in issues_by_col.get(col_name, [])
            if i.category in (
                IssueCategory.CASING_INCONSISTENCY,
                IssueCategory.WHITESPACE_INCONSISTENCY,
                IssueCategory.ALIAS_INCONSISTENCY,
                IssueCategory.SPELLING_INCONSISTENCY
            )
        ]
        for i in cons_issues:
            # Minor penalty for cosmetic, higher for spelling/aliases
            rate = i.affected_count / total_rows
            penalty = min(40.0, rate * 50.0)
            consistency_score = max(0.0, consistency_score - penalty)
            deduction_reasons.append(
                f"Column '{col_name}': Consistency reduced by {penalty:.1f}% due to {i.category.value}."
            )

        # 4. Uniqueness Dimension
        # Applicable ONLY if designated business key or explicit unique column
        is_uniqueness_applicable = col_profile.is_business_key or (
            business_keys and col_name in business_keys
        )
        uniqueness_score = 100.0
        if is_uniqueness_applicable:
            dup_issues = [
                i for i in issues_by_col.get(col_name, [])
                if i.category in (IssueCategory.EXACT_DUPLICATE_ROW, IssueCategory.BUSINESS_KEY_DUPLICATE)
            ]
            for i in dup_issues:
                penalty = min(60.0, (i.affected_count / total_rows) * 100.0)
                uniqueness_score = max(0.0, uniqueness_score - penalty)
                deduction_reasons.append(
                    f"Column '{col_name}' (Business Key): Uniqueness reduced by {penalty:.1f}% due to duplicate keys."
                )

        # Dynamic Proportional Re-weighting
        active_weights: Dict[str, float] = {}
        active_weights["completeness"] = active_weights_base["completeness"]
        active_weights["consistency"] = active_weights_base["consistency"]

        if is_validity_applicable:
            active_weights["validity"] = active_weights_base["validity"]
        if is_uniqueness_applicable:
            active_weights["uniqueness"] = active_weights_base["uniqueness"]

        weight_sum = sum(active_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in active_weights.items()}

        col_score = (
            completeness_score * normalized_weights.get("completeness", 0.0) +
            consistency_score * normalized_weights.get("consistency", 0.0) +
            validity_score * normalized_weights.get("validity", 0.0) +
            uniqueness_score * normalized_weights.get("uniqueness", 0.0)
        )

        column_scores[col_name] = round(col_score, 1)
        column_dimensions[col_name] = DimensionScore(
            completeness=round(completeness_score, 1),
            validity=round(validity_score, 1),
            consistency=round(consistency_score, 1),
            uniqueness=round(uniqueness_score, 1),
            active_weights={k: round(v * 100, 1) for k, v in normalized_weights.items()}
        )

    # Dataset-level Uniqueness Score (Exact duplicate rows)
    exact_dup_issues = [i for i in issues if i.category == IssueCategory.EXACT_DUPLICATE_ROW]
    exact_dup_count = sum(i.affected_count for i in exact_dup_issues)
    dataset_uniqueness_score = max(0.0, 100.0 - ((exact_dup_count / total_rows) * 100.0))
    if dataset_uniqueness_score < 100.0:
        deduction_reasons.append(
            f"Dataset Uniqueness: Reduced to {dataset_uniqueness_score:.1f}% due to {exact_dup_count} exact duplicate rows."
        )

    # Compute overall composite dataset score
    if column_scores:
        avg_column_score = sum(column_scores.values()) / len(column_scores)
        overall = (0.90 * avg_column_score) + (0.10 * dataset_uniqueness_score)
    else:
        overall = 100.0

    return QualityScorecard(
        overall_score=round(overall, 1),
        dataset_uniqueness_score=round(dataset_uniqueness_score, 1),
        column_scores=column_scores,
        column_dimensions=column_dimensions,
        deduction_reasons=deduction_reasons
    )

calculate_dataset_scorecard = compute_quality_scorecard
