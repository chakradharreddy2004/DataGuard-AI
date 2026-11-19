"""
Unit Tests for Cleaning Module (src/cleaning.py).
Validates raw_df immutability, side-by-side diff previews,
and safe execution of transformations exclusively on working_df.
"""
import pytest
import pandas as pd
from src.models import (
    CleaningActionType,
    CleaningProposal,
    ProposalStatus,
    ValueMapping,
    ConfidenceTier,
    QualityIssue,
    IssueCategory,
    IssueSeverity,
)
from src.cleaning import (
    generate_cleaning_proposals,
    preview_proposal,
    apply_proposal,
)

def test_raw_df_is_strictly_immutable():
    raw_df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": ["  alice  ", "Bob", "Bob", "Charlie"],
        "city": ["NYC", "SF", "SF", "LA"]
    })
    raw_copy = raw_df.copy(deep=True)
    working_df = raw_df.copy(deep=True)

    proposal = CleaningProposal(
        proposal_id="drop_dups",
        action_type=CleaningActionType.DROP_EXACT_DUPLICATES,
        column_name=None,
        parameters={"keep": "first"},
        affected_rows=1,
        status=ProposalStatus.PENDING
    )

    new_working_df, affected_count = apply_proposal(working_df, proposal)

    # working_df changed
    assert len(new_working_df) == 3
    assert affected_count == 1

    # raw_df remains 100% UNCHANGED
    pd.testing.assert_frame_equal(raw_df, raw_copy)
    assert len(raw_df) == 4

def test_preview_proposal_diff():
    df = pd.DataFrame({
        "name": ["   Alice", "Bob   ", "Charlie"]
    })
    proposal = CleaningProposal(
        proposal_id="trim_name",
        action_type=CleaningActionType.TRIM_WHITESPACE,
        column_name="name",
        parameters={},
        affected_rows=2,
        status=ProposalStatus.PENDING
    )
    diff_df = preview_proposal(df, proposal, sample_size=5)

    assert isinstance(diff_df, pd.DataFrame)
    assert "Row Index" in diff_df.columns
    assert "Before Value" in diff_df.columns
    assert "After Value" in diff_df.columns
    assert len(diff_df) == 2

def test_apply_proposal_trim_whitespace():
    df = pd.DataFrame({"city": ["  New York  ", "San Francisco  "]})
    prop = CleaningProposal("p_ws", CleaningActionType.TRIM_WHITESPACE, "city", {}, 2, ProposalStatus.PENDING)
    cleaned_df, affected = apply_proposal(df, prop)
    
    assert list(cleaned_df["city"]) == ["New York", "San Francisco"]
    assert affected == 2

def test_apply_proposal_standardize_casing():
    df = pd.DataFrame({"city": ["new york", "SAN FRANCISCO"]})
    prop = CleaningProposal("p_case", CleaningActionType.STANDARDIZE_CASING, "city", {"casing": "title"}, 2, ProposalStatus.PENDING)
    cleaned_df, affected = apply_proposal(df, prop)
    
    assert list(cleaned_df["city"]) == ["New York", "San Francisco"]

def test_apply_proposal_impute_missing_median():
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0, None]})
    prop = CleaningProposal("p_imp", CleaningActionType.IMPUTE_MISSING, "age", {"strategy": "median"}, 1, ProposalStatus.PENDING)
    cleaned_df, affected = apply_proposal(df, prop)
    
    assert cleaned_df["age"].isna().sum() == 0
    assert cleaned_df["age"].iloc[3] == 30.0  # Median of 20, 30, 40

def test_apply_proposal_map_aliases():
    df = pd.DataFrame({"city": ["NYC", "New York", "NYC"]})
    mapping = ValueMapping(
        original_value="NYC",
        replacement_value="New York",
        confidence_tier=ConfidenceTier.HIGH_CONFIDENCE,
        similarity_score=95.0,
        frequency=2,
        is_approved=True
    )
    prop = CleaningProposal(
        "p_alias",
        CleaningActionType.MAP_ALIASES,
        "city",
        {},
        2,
        ProposalStatus.PENDING,
        value_mappings=[mapping]
    )
    cleaned_df, affected = apply_proposal(df, prop)
    
    assert list(cleaned_df["city"]) == ["New York", "New York", "New York"]
    assert affected == 2
