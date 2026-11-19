"""
Unit Tests for Value Standardization Module (src/value_standardization.py).
Tests RapidFuzz fuzzy matching, alias dictionary synonyms, and confidence tier classification.
"""
import pytest
import pandas as pd
from src.value_standardization import (
    load_alias_dictionaries,
    analyze_standardization,
    detect_alias_and_spelling_issues,
)
from src.models import ConfidenceTier, IssueCategory

def test_load_alias_dictionaries():
    aliases = load_alias_dictionaries()
    assert isinstance(aliases, dict)
    assert len(aliases) > 0
    # NYC -> New York should be present in config
    assert aliases.get("nyc") == "New York"
    assert aliases.get("sf") == "San Francisco"

def test_analyze_standardization_aliases():
    df = pd.DataFrame({
        "city": ["New York", "New York", "NYC", "New York City", "Chicago", "Chicagoo"]
    })
    aliases = {"nyc": "New York", "new york city": "New York"}
    mappings = analyze_standardization(df, "city", alias_dict=aliases, similarity_threshold=80.0)

    orig_values = [m.original_value for m in mappings]
    assert "NYC" in orig_values
    
    # Check that NYC maps to New York with High Confidence
    nyc_mapping = next(m for m in mappings if m.original_value == "NYC")
    assert nyc_mapping.replacement_value == "New York"
    assert nyc_mapping.confidence_tier == ConfidenceTier.HIGH_CONFIDENCE

def test_analyze_standardization_fuzzy_spelling():
    df = pd.DataFrame({
        "city": ["San Francisco"] * 10 + ["San Fran", "San Francico"]
    })
    mappings = analyze_standardization(df, "city", alias_dict={}, similarity_threshold=75.0)
    orig_values = [m.original_value for m in mappings]
    assert "San Francico" in orig_values
    
    francico_mapping = next(m for m in mappings if m.original_value == "San Francico")
    assert francico_mapping.replacement_value == "San Francisco"
    assert francico_mapping.confidence_tier in (ConfidenceTier.HIGH_CONFIDENCE, ConfidenceTier.MEDIUM_CONFIDENCE)

def test_detect_alias_and_spelling_issues():
    df = pd.DataFrame({
        "city": ["New York"] * 5 + ["NYC", "New York City"]
    })
    aliases = {"nyc": "New York", "new york city": "New York"}
    issues = detect_alias_and_spelling_issues(df, "city", alias_dict=aliases)
    assert len(issues) == 1
    assert issues[0].column_name == "city"
    assert issues[0].affected_count == 2
