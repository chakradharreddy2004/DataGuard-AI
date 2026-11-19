"""
Safe Cleaning & Transformation Module for DataGuard AI.
Adheres to Constitution Principles V & VI:
- Preserves the raw uploaded dataset in an immutable state.
- Executes cleaning operations strictly on the working copy.
- Generates side-by-side diff previews prior to applying changes.
- Requires explicit user approval for every transformation.
"""
from typing import Tuple, List, Optional
import pandas as pd
from .models import (
    CleaningProposal,
    CleaningActionType,
    ValueMapping,
    ProposalStatus,
    QualityIssue,
    IssueCategory,
)

def generate_cleaning_proposals(
    df: pd.DataFrame,
    issues: List[QualityIssue],
    value_mappings: Optional[List[ValueMapping]] = None
) -> List[CleaningProposal]:
    """
    Constructs candidate cleaning proposals from detected anomalies.
    Does NOT mutate any data.
    """
    proposals: List[CleaningProposal] = []
    seen_actions = set()

    for issue in issues:
        if issue.category == IssueCategory.EXACT_DUPLICATE_ROW:
            key = ("exact_dups", None)
            if key not in seen_actions:
                proposals.append(CleaningProposal(
                    proposal_id="proposal_drop_exact_dups",
                    action_type=CleaningActionType.DROP_EXACT_DUPLICATES,
                    column_name=None,
                    parameters={"keep": "first"},
                    affected_rows=issue.affected_count,
                    status=ProposalStatus.PENDING
                ))
                seen_actions.add(key)

        elif issue.category == IssueCategory.WHITESPACE_INCONSISTENCY and issue.column_name:
            key = ("trim_ws", issue.column_name)
            if key not in seen_actions:
                proposals.append(CleaningProposal(
                    proposal_id=f"proposal_trim_{issue.column_name}",
                    action_type=CleaningActionType.TRIM_WHITESPACE,
                    column_name=issue.column_name,
                    parameters={},
                    affected_rows=issue.affected_count,
                    status=ProposalStatus.PENDING
                ))
                seen_actions.add(key)

        elif issue.category == IssueCategory.CASING_INCONSISTENCY and issue.column_name:
            key = ("casing", issue.column_name)
            if key not in seen_actions:
                proposals.append(CleaningProposal(
                    proposal_id=f"proposal_casing_{issue.column_name}",
                    action_type=CleaningActionType.STANDARDIZE_CASING,
                    column_name=issue.column_name,
                    parameters={"casing": "title"},
                    affected_rows=issue.affected_count,
                    status=ProposalStatus.PENDING
                ))
                seen_actions.add(key)

        elif issue.category == IssueCategory.MISSING_VALUE and issue.column_name:
            key = ("missing", issue.column_name)
            if key not in seen_actions and issue.affected_percentage < 30.0:
                proposals.append(CleaningProposal(
                    proposal_id=f"proposal_impute_{issue.column_name}",
                    action_type=CleaningActionType.IMPUTE_MISSING,
                    column_name=issue.column_name,
                    parameters={"strategy": "mode" if df[issue.column_name].dtype == "object" else "median"},
                    affected_rows=issue.affected_count,
                    status=ProposalStatus.PENDING
                ))
                seen_actions.add(key)

    # Add Value Standardizations if provided
    if value_mappings:
        col_mappings: dict[str, List[ValueMapping]] = {}
        for m in value_mappings:
            # We map target column from context or default
            pass

    return proposals

def preview_proposal(
    working_df: pd.DataFrame,
    proposal: CleaningProposal,
    sample_size: int = 5
) -> pd.DataFrame:
    """
    Computes a side-by-side diff comparison showing:
    [Row Index, Column, Before Value, After Value]
    Does NOT mutate working_df.
    """
    diff_records = []
    col = proposal.column_name

    if proposal.action_type == CleaningActionType.DROP_EXACT_DUPLICATES:
        dup_mask = working_df.duplicated(keep="first")
        sample_dups = working_df[dup_mask].head(sample_size)
        for idx, row in sample_dups.iterrows():
            diff_records.append({
                "Row Index": idx,
                "Column": "Entire Row",
                "Before Value": f"Duplicate Record (Row {idx})",
                "After Value": "<Row Dropped>"
            })

    elif proposal.action_type == CleaningActionType.TRIM_WHITESPACE and col and col in working_df.columns:
        series = working_df[col].astype(str)
        ws_mask = series.str.startswith(" ") | series.str.endswith(" ") | series.str.contains(r"\s{2,}", regex=True)
        affected = working_df[ws_mask].head(sample_size)
        for idx, row in affected.iterrows():
            orig_val = str(row[col])
            cleaned_val = " ".join(orig_val.split())
            diff_records.append({
                "Row Index": idx,
                "Column": col,
                "Before Value": f"'{orig_val}'",
                "After Value": f"'{cleaned_val}'"
            })

    elif proposal.action_type == CleaningActionType.STANDARDIZE_CASING and col and col in working_df.columns:
        series = working_df[col].astype(str)
        affected = working_df.head(sample_size)
        for idx, row in affected.iterrows():
            orig_val = str(row[col])
            cleaned_val = orig_val.strip().title()
            if orig_val != cleaned_val:
                diff_records.append({
                    "Row Index": idx,
                    "Column": col,
                    "Before Value": orig_val,
                    "After Value": cleaned_val
                })

    elif proposal.action_type == CleaningActionType.IMPUTE_MISSING and col and col in working_df.columns:
        missing_mask = working_df[col].isna() | working_df[col].astype(str).str.strip().isin(["", "nan", "none", "null"])
        affected = working_df[missing_mask].head(sample_size)
        # Compute fill value
        if pd.api.types.is_numeric_dtype(working_df[col].dtype):
            fill_val = working_df[col].median()
        else:
            mode_series = working_df[col].dropna().mode()
            fill_val = mode_series[0] if len(mode_series) > 0 else "UNKNOWN"

        for idx in affected.index:
            diff_records.append({
                "Row Index": idx,
                "Column": col,
                "Before Value": "<NULL / EMPTY>",
                "After Value": str(fill_val)
            })

    elif proposal.action_type == CleaningActionType.MAP_ALIASES and col and col in working_df.columns:
        for mapping in proposal.value_mappings[:sample_size]:
            diff_records.append({
                "Row Index": "All Matches",
                "Column": col,
                "Before Value": mapping.original_value,
                "After Value": mapping.replacement_value
            })

    if not diff_records:
        return pd.DataFrame(columns=["Row Index", "Column", "Before Value", "After Value"])

    return pd.DataFrame(diff_records)

def apply_proposal(
    working_df: pd.DataFrame,
    proposal: CleaningProposal
) -> Tuple[pd.DataFrame, int]:
    """
    Executes an approved cleaning proposal strictly on working_df.
    Original raw_df is NEVER modified.
    
    Returns:
        Tuple of (updated working_df, rows_affected_count).
    """
    df_copy = working_df.copy()
    col = proposal.column_name
    rows_affected = 0

    if proposal.action_type == CleaningActionType.DROP_EXACT_DUPLICATES:
        initial_len = len(df_copy)
        df_copy = df_copy.drop_duplicates(keep="first").reset_index(drop=True)
        rows_affected = initial_len - len(df_copy)

    elif proposal.action_type == CleaningActionType.TRIM_WHITESPACE and col and col in df_copy.columns:
        series = df_copy[col].astype(str)
        ws_mask = series.str.startswith(" ") | series.str.endswith(" ") | series.str.contains(r"\s{2,}", regex=True)
        rows_affected = int(ws_mask.sum())
        df_copy[col] = df_copy[col].astype(str).apply(lambda x: " ".join(x.split()) if pd.notna(x) else x)

    elif proposal.action_type == CleaningActionType.STANDARDIZE_CASING and col and col in df_copy.columns:
        series = df_copy[col].astype(str)
        casing = proposal.parameters.get("casing", "title")
        if casing == "title":
            df_copy[col] = series.str.strip().str.title()
        elif casing == "lower":
            df_copy[col] = series.str.strip().str.lower()
        elif casing == "upper":
            df_copy[col] = series.str.strip().str.upper()
        rows_affected = len(df_copy)

    elif proposal.action_type == CleaningActionType.IMPUTE_MISSING and col and col in df_copy.columns:
        missing_mask = df_copy[col].isna() | df_copy[col].astype(str).str.strip().isin(["", "nan", "none", "null"])
        rows_affected = int(missing_mask.sum())
        if pd.api.types.is_numeric_dtype(df_copy[col].dtype):
            val = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(val)
        else:
            mode_s = df_copy[col].dropna().mode()
            val = mode_s[0] if len(mode_s) > 0 else "UNKNOWN"
            df_copy[col] = df_copy[col].replace(["", "nan", "none", "null", None], val)
            df_copy[col] = df_copy[col].fillna(val)

    elif proposal.action_type in (CleaningActionType.MAP_ALIASES, CleaningActionType.FUZZY_STANDARDIZE) and col and col in df_copy.columns:
        rep_dict = {m.original_value: m.replacement_value for m in proposal.value_mappings if m.is_approved}
        if rep_dict:
            mask = df_copy[col].isin(rep_dict.keys())
            rows_affected = int(mask.sum())
            df_copy[col] = df_copy[col].replace(rep_dict)

    proposal.status = ProposalStatus.APPLIED
    return df_copy, rows_affected
