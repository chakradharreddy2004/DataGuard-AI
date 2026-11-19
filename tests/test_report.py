"""
Unit Tests for Report Generator Module (src/report_generator.py).
Validates executive Markdown and HTML report generation, quality scorecards,
and transformation audit logs.
"""
import pytest
import pandas as pd
from src.models import (
    DatasetMetadata,
    DatasetProfile,
    QualityIssue,
    QualityScorecard,
    CleaningProposal,
    CleaningActionType,
    ProposalStatus,
    ProviderSource,
    IssueCategory,
    IssueSeverity,
)
from src.profiling import generate_dataset_profile
from src.scoring import compute_quality_scorecard
from src.report_generator import generate_markdown_report, generate_html_report

def test_generate_markdown_report(clean_dataframe):
    meta = DatasetMetadata("sales.csv", 1024, len(clean_dataframe), len(clean_dataframe.columns), "2026-11-19T10:00:00")
    profile = generate_dataset_profile(clean_dataframe, meta)
    scorecard = compute_quality_scorecard(clean_dataframe, profile, issues=[])

    prop = CleaningProposal("p1", CleaningActionType.TRIM_WHITESPACE, "city", {}, 5, ProposalStatus.APPLIED)

    md_report = generate_markdown_report(
        profile,
        scorecard,
        issues=[],
        applied_proposals=[prop],
        active_provider=ProviderSource.GROQ,
        active_model="openai/gpt-oss-120b"
    )

    assert "# DataGuard AI - Quality Audit & Health Report" in md_report
    assert "sales.csv" in md_report
    assert "100.0 / 100" in md_report or "100 / 100" in md_report
    assert "Groq (openai/gpt-oss-120b)" in md_report
    assert "trim_whitespace" in md_report.lower()

def test_generate_html_report(clean_dataframe):
    meta = DatasetMetadata("customers.csv", 2048, len(clean_dataframe), len(clean_dataframe.columns), "2026-11-19T10:00:00")
    profile = generate_dataset_profile(clean_dataframe, meta)
    scorecard = compute_quality_scorecard(clean_dataframe, profile, issues=[])

    html_report = generate_html_report(
        profile,
        scorecard,
        issues=[],
        applied_proposals=[],
        active_provider=ProviderSource.GEMINI,
        active_model="gemini-2.5-flash"
    )

    assert "<!DOCTYPE html>" in html_report
    assert "DataGuard AI" in html_report
    assert "customers.csv" in html_report
    assert "score-circle" in html_report
    assert "Gemini (gemini-2.5-flash)" in html_report
