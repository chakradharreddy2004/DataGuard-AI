"""
Unit Tests for Data Loader Module (src/data_loader.py).
Tests CSV ingestion, encoding resilience, row boundary checks, and error handling.
"""
import io
import pytest
import pandas as pd
from src.data_loader import load_csv, load_csv_data
from src.models import DatasetMetadata

def test_load_csv_from_bytes(dirty_csv_bytes):
    df, meta = load_csv(dirty_csv_bytes, filename="test.csv")
    assert isinstance(df, pd.DataFrame)
    assert isinstance(meta, DatasetMetadata)
    assert len(df) > 900
    assert meta.filename == "test.csv"
    assert meta.column_count == len(df.columns)

def test_load_csv_from_buffer(dirty_csv_bytes):
    buf = io.BytesIO(dirty_csv_bytes)
    df, meta = load_csv(buf, filename="stream.csv")
    assert len(df) > 0
    assert meta.filename == "stream.csv"

def test_load_csv_empty_file_raises_error():
    empty_bytes = b""
    with pytest.raises(ValueError, match="empty"):
        load_csv(empty_bytes)

def test_load_csv_no_rows_raises_error():
    header_only = b"col1,col2,col3\n"
    with pytest.raises(ValueError, match="no data rows"):
        load_csv(header_only)

def test_load_csv_encoding_latin1():
    content = "id,name,city\n1,René,München\n2,André,Köln\n"
    latin1_bytes = content.encode("latin1")
    df, meta = load_csv(latin1_bytes, filename="latin1.csv")
    assert len(df) == 2
    assert "René" in df["name"].values

def test_load_csv_max_rows_cutoff():
    rows = ["id,val"] + [f"{i},{i*10}" for i in range(100)]
    csv_bytes = "\n".join(rows).encode("utf-8")
    df, meta = load_csv(csv_bytes, max_rows=30)
    assert len(df) == 30
    assert meta.row_count == 30

def test_load_csv_alias_works(dirty_csv_bytes):
    df, meta = load_csv_data(dirty_csv_bytes)
    assert len(df) > 0
