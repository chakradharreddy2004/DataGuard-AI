"""
Value Standardization Module for DataGuard AI.
Uses RapidFuzz and configurable alias dictionaries to categorize text variations into confidence tiers:
- Cosmetic (whitespace / casing)
- High Confidence (>= 90% / known aliases)
- Medium Confidence (80% - 89% fuzzy spelling candidates)
- Low Confidence (< 80% kept distinct)
Requires explicit human review and approval before execution.
"""
import json
import os
from typing import Dict, List, Optional
import pandas as pd
from rapidfuzz import fuzz, process
from .models import (
    ConfidenceTier,
    ValueMapping,
    QualityIssue,
    IssueCategory,
    IssueSeverity,
    CleaningActionType,
)

DEFAULT_ALIAS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "aliases.json")

def load_alias_dictionaries(config_path: str = DEFAULT_ALIAS_FILE) -> Dict[str, str]:
    """
    Loads known alias mappings from JSON configuration.
    """
    merged_aliases: Dict[str, str] = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for category, mapping in data.items():
                    if isinstance(mapping, dict):
                        for k, v in mapping.items():
                            merged_aliases[k.lower().strip()] = v
        except Exception:
            pass
    return merged_aliases

def analyze_standardization(
    df: pd.DataFrame,
    column_name: str,
    alias_dict: Optional[Dict[str, str]] = None,
    similarity_threshold: float = 80.0
) -> List[ValueMapping]:
    """
    Analyzes distinct string values in a column, detects spelling variations and known synonyms,
    and returns a structured list of candidate ValueMappings categorized by confidence tier.
    """
    if column_name not in df.columns:
        return []

    series = df[column_name].dropna().astype(str).str.strip()
    if len(series) == 0:
        return []

    value_counts = series.value_counts().to_dict()
    distinct_values = list(value_counts.keys())
    if len(distinct_values) <= 1:
        return []

    aliases = alias_dict if alias_dict is not None else load_alias_dictionaries()
    mappings: List[ValueMapping] = []
    processed_originals = set()

    # Step 1: Check Known Reference Aliases
    for val in distinct_values:
        val_clean = val.lower().strip()
        if val_clean in aliases:
            target = aliases[val_clean]
            if val != target:
                mappings.append(ValueMapping(
                    original_value=val,
                    replacement_value=target,
                    confidence_tier=ConfidenceTier.HIGH_CONFIDENCE,
                    similarity_score=95.0,
                    frequency=value_counts[val],
                    is_approved=False
                ))
                processed_originals.add(val)

    # Step 2: Fuzzy Matching on remaining distinct values
    # Sort distinct values by frequency descending (canonical target is the most frequent variant)
    sorted_by_freq = sorted(distinct_values, key=lambda v: value_counts[v], reverse=True)

    for i, target_val in enumerate(sorted_by_freq):
        # We search for other less frequent values that are similar to target_val
        for other_val in sorted_by_freq[i + 1:]:
            if other_val in processed_originals:
                continue

            # Skip if casing difference only (handled in inconsistencies module)
            if target_val.lower() == other_val.lower():
                continue

            # Calculate RapidFuzz token sort ratio and weighted ratio
            score = fuzz.token_sort_ratio(target_val.lower(), other_val.lower())

            if score >= similarity_threshold:
                if score >= 90.0:
                    tier = ConfidenceTier.HIGH_CONFIDENCE
                else:
                    tier = ConfidenceTier.MEDIUM_CONFIDENCE

                mappings.append(ValueMapping(
                    original_value=other_val,
                    replacement_value=target_val,
                    confidence_tier=tier,
                    similarity_score=round(score, 1),
                    frequency=value_counts[other_val],
                    is_approved=False
                ))
                processed_originals.add(other_val)

    return mappings

def detect_alias_and_spelling_issues(
    df: pd.DataFrame,
    column_name: str,
    alias_dict: Optional[Dict[str, str]] = None
) -> List[QualityIssue]:
    """
    Wraps analyze_standardization into QualityIssue instances for reporting and scoring.
    """
    mappings = analyze_standardization(df, column_name, alias_dict)
    if not mappings:
        return []

    total_rows = max(1, len(df))
    affected_count = sum(m.frequency for m in mappings)
    pct = round((affected_count / total_rows) * 100, 2)
    sample_pairs = [f"{m.original_value} -> {m.replacement_value} ({m.confidence_tier.value}, {m.similarity_score}%)" for m in mappings[:3]]

    return [QualityIssue(
        issue_id=f"standardization_{column_name}",
        category=IssueCategory.SPELLING_INCONSISTENCY if any(m.confidence_tier == ConfidenceTier.MEDIUM_CONFIDENCE for m in mappings) else IssueCategory.ALIAS_INCONSISTENCY,
        column_name=column_name,
        severity=IssueSeverity.MEDIUM if pct > 5.0 else IssueSeverity.LOW,
        affected_count=affected_count,
        affected_percentage=pct,
        sample_values=sample_pairs,
        sample_row_indices=[],
        explanation=(
            f"Column '{column_name}' contains {len(mappings)} potential spelling or alias variations "
            f"affecting {affected_count} rows ({pct}%). Review and approve suggested canonical values."
        ),
        confidence_tier=ConfidenceTier.HIGH_CONFIDENCE if all(m.confidence_tier == ConfidenceTier.HIGH_CONFIDENCE for m in mappings) else ConfidenceTier.MEDIUM_CONFIDENCE,
        suggested_action=CleaningActionType.FUZZY_STANDARDIZE
    )]
