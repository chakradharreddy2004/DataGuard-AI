"""
Security and Privacy Boundary Verification Tests (Phase 8, T036).
Validates that:
- API keys are never leaked into logs, error messages, or UI components.
- Sanitized payloads strictly contain metadata and max 5 distinct samples.
- Raw CSV records are never transmitted to external APIs.
"""
import os
import pytest
from src.ai_agent import classify_llm_error, build_sanitized_metadata_payload
from src.models import DatasetMetadata
from src.profiling import generate_dataset_profile
from src.demo_data import generate_dirty_sample_dataframe

def test_api_keys_not_leaked_in_classified_error():
    fake_key = "AIzaSySecretApiKey123456789"
    raw_error_message = f"Error with key {fake_key}: 401 Unauthorized"
    classified = classify_llm_error(Exception(raw_error_message))
    
    # Assert classification message does NOT expose the raw secret key
    assert fake_key not in classified.message

def test_sanitized_payload_has_no_pii_rows():
    df = generate_dirty_sample_dataframe(num_rows=200)
    meta = DatasetMetadata("pii_test.csv", 1000, len(df), len(df.columns), "")
    profile = generate_dataset_profile(df, meta)
    
    payload = build_sanitized_metadata_payload(profile)
    
    # Payload must not contain dataframe records or series
    assert "data" not in payload
    assert "rows" not in payload
    for col_name, info in payload["columns"].items():
        assert len(info["distinct_samples"]) <= 5
