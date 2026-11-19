"""
Shared Pytest Fixtures and Benchmark Test Data Generators for DataGuard AI.
Provides synthetic dirty and clean datasets adhering to Constitution testing requirements.
"""
import pytest
import pandas as pd
import numpy as np
from src.demo_data import generate_dirty_sample_dataframe
from src.models import DatasetMetadata
from src.profiling import generate_dataset_profile

@pytest.fixture
def dirty_dataframe() -> pd.DataFrame:
    """Returns a realistic 1,000-row DataFrame seeded with known anomalies."""
    return generate_dirty_sample_dataframe(num_rows=1000, seed=123)

@pytest.fixture
def dirty_csv_bytes(dirty_dataframe) -> bytes:
    """Returns the dirty dataframe as UTF-8 encoded CSV bytes."""
    return dirty_dataframe.to_csv(index=False).encode("utf-8")

@pytest.fixture
def clean_dataframe() -> pd.DataFrame:
    """Returns a clean 100-row DataFrame with zero anomalies."""
    return pd.DataFrame({
        "customer_id": [f"CUST-{i}" for i in range(1, 101)],
        "name": [f"Customer {i}" for i in range(1, 101)],
        "city": ["New York" if i % 2 == 0 else "San Francisco" for i in range(1, 101)],
        "age": [30 + (i % 20) for i in range(1, 101)],
        "score": [10.0 + (i * 0.5) for i in range(1, 101)]
    })

@pytest.fixture
def sample_metadata(dirty_dataframe) -> DatasetMetadata:
    """Returns valid DatasetMetadata for the dirty dataframe."""
    return DatasetMetadata(
        filename="test_dataset.csv",
        file_size_bytes=int(dirty_dataframe.memory_usage(deep=True).sum()),
        row_count=len(dirty_dataframe),
        column_count=len(dirty_dataframe.columns),
        ingestion_timestamp="2026-11-19T12:00:00"
    )

@pytest.fixture
def sample_profile(dirty_dataframe, sample_metadata) -> DatasetProfile:
    """Returns a pre-computed profile of the dirty dataset."""
    return generate_dataset_profile(dirty_dataframe, sample_metadata)
