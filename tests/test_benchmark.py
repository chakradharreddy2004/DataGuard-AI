"""
End-to-End Performance Benchmark Test (Phase 8, T035).
Validates that ingestion, dynamic profiling, multi-category anomaly detection,
and DAMA quality scoring of a 10,000-row CSV executes in < 5.0 seconds locally.
"""
import time
import pytest
import pandas as pd
from src.demo_data import generate_dirty_sample_dataframe
from src.models import DatasetMetadata
from src.profiling import generate_dataset_profile
from src.missing_values import detect_missing_values
from src.duplicates import detect_duplicates
from src.outliers import detect_numeric_outliers
from src.inconsistencies import detect_inconsistencies
from src.scoring import compute_quality_scorecard

def test_10k_row_benchmark_under_5_seconds():
    # 1. Generate 10,000 rows of realistic dirty data
    start_gen = time.perf_counter()
    df_10k = generate_dirty_sample_dataframe(num_rows=10000, seed=999)
    csv_bytes = df_10k.to_csv(index=False).encode("utf-8")
    
    # 2. Benchmark complete pipeline execution
    pipeline_start = time.perf_counter()

    # Ingestion & Metadata
    metadata = DatasetMetadata(
        filename="benchmark_10k.csv",
        file_size_bytes=len(csv_bytes),
        row_count=len(df_10k),
        column_count=len(df_10k.columns),
        ingestion_timestamp="2026-11-19T12:00:00"
    )

    # Dynamic Profiling
    profile = generate_dataset_profile(df_10k, metadata)

    # Multi-category Anomaly Detection
    issues = []
    issues.extend(detect_missing_values(df_10k, profile))
    issues.extend(detect_duplicates(df_10k, business_keys=["customer_id"]))
    issues.extend(detect_numeric_outliers(df_10k, profile, method="iqr"))
    issues.extend(detect_inconsistencies(df_10k, profile))

    # Scoring
    scorecard = compute_quality_scorecard(df_10k, profile, issues, business_keys=["customer_id"])

    elapsed = time.perf_counter() - pipeline_start

    print(f"\n[BENCHMARK] Processed {len(df_10k):,} rows, {len(df_10k.columns)} columns in {elapsed:.3f} seconds.")
    print(f"[BENCHMARK] Overall Quality Score: {scorecard.overall_score}/100, Anomalies Detected: {len(issues)}")

    # Assert SLA target < 5.0 seconds
    assert elapsed < 5.0, f"Pipeline execution took {elapsed:.3f}s, exceeding 5.0s SLA."
    assert 0.0 <= scorecard.overall_score <= 100.0
    assert len(issues) > 0
    assert len(profile.columns) == len(df_10k.columns)
