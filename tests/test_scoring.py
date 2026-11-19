"""
Unit Tests for Scoring Module (src/scoring.py).
Validates DAMA-DMBOK dimensional quality scoring, dynamic proportional re-weighting,
and score deduction calculations.
"""
import pytest
import pandas as pd
from src.models import (
    DatasetMetadata,
    DatasetProfile,
    QualityIssue,
    IssueCategory,
    IssueSeverity,
    CleaningActionType,
    InferredDataType,
)
from src.profiling import generate_dataset_profile
from src.scoring import compute_quality_scorecard, calculate_dataset_scorecard

def test_clean_dataset_scores_100(clean_dataframe):
    meta = DatasetMetadata("clean.csv", 100, len(clean_dataframe), len(clean_dataframe.columns), "")
    profile = generate_dataset_profile(clean_dataframe, meta)
    scorecard = compute_quality_scorecard(clean_dataframe, profile, issues=[])
    
    assert scorecard.overall_score == 100.0
    assert scorecard.dataset_uniqueness_score == 100.0
    assert len(scorecard.deduction_reasons) == 0

def test_scoring_with_missing_values(clean_dataframe):
    meta = DatasetMetadata("clean.csv", 100, len(clean_dataframe), len(clean_dataframe.columns), "")
    profile = generate_dataset_profile(clean_dataframe, meta)
    
    # Simulate a missing value issue in 'age' (25% missing)
    issue = QualityIssue(
        issue_id="missing_age",
        category=IssueCategory.MISSING_VALUE,
        column_name="age",
        severity=IssueSeverity.HIGH,
        affected_count=25,
        affected_percentage=25.0,
        sample_values=[],
        explanation="25% missing in age",
        suggested_action=CleaningActionType.IMPUTE_MISSING
    )
    scorecard = compute_quality_scorecard(clean_dataframe, profile, issues=[issue])
    
    assert scorecard.overall_score < 100.0
    assert scorecard.column_scores["age"] < 100.0
    assert any("age" in r.lower() for r in scorecard.deduction_reasons)

def test_scoring_advisory_outlier_does_not_deduct(clean_dataframe):
    meta = DatasetMetadata("clean.csv", 100, len(clean_dataframe), len(clean_dataframe.columns), "")
    profile = generate_dataset_profile(clean_dataframe, meta)
    
    advisory_outlier = QualityIssue(
        issue_id="outlier_score",
        category=IssueCategory.NUMERIC_OUTLIER,
        column_name="score",
        severity=IssueSeverity.ADVISORY,
        affected_count=5,
        affected_percentage=5.0,
        sample_values=[999.0],
        explanation="Statistical outlier",
        is_advisory=True,
        suggested_action=CleaningActionType.CAP_OUTLIERS
    )
    scorecard = compute_quality_scorecard(clean_dataframe, profile, issues=[advisory_outlier])
    
    # Advisory anomaly should NOT deduct from overall score
    assert scorecard.overall_score == 100.0

def test_scoring_dynamic_reweighting(clean_dataframe):
    meta = DatasetMetadata("clean.csv", 100, len(clean_dataframe), len(clean_dataframe.columns), "")
    profile = generate_dataset_profile(clean_dataframe, meta)
    
    scorecard = calculate_dataset_scorecard(clean_dataframe, profile, issues=[])
    # Check that for text/categorical column 'city', validity is non-applicable
    city_dims = scorecard.column_dimensions["city"]
    assert "validity" not in city_dims.active_weights
    assert "completeness" in city_dims.active_weights
    assert "consistency" in city_dims.active_weights
    # Sum of active weights should normalize to 100%
    assert round(sum(city_dims.active_weights.values()), 0) == 100.0
