"""
Data Loader Module for DataGuard AI.
Handles CSV upload, encoding detection, structure validation, and bounds checking.
"""
import io
import os
from datetime import datetime
from typing import Tuple, Union, Any, BinaryIO
import pandas as pd
from .models import DatasetMetadata

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "cp1252"]

def load_csv(
    file_source: Union[str, bytes, BinaryIO, io.BytesIO, io.StringIO, Any],
    max_rows: int = 15000,
    filename: str = "uploaded_dataset.csv"
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """
    Loads, decodes, and validates a tabular CSV file.
    
    Args:
        file_source: File path, bytes, or file-like buffer.
        max_rows: Soft upper bound for dataset size.
        filename: Optional filename override.
        
    Returns:
        Tuple of (pd.DataFrame, DatasetMetadata).
        
    Raises:
        ValueError: If file is empty, non-tabular, or corrupted.
    """
    file_bytes: bytes = b""
    actual_filename = filename

    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise ValueError(f"File not found: {file_source}")
        actual_filename = os.path.basename(file_source)
        with open(file_source, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    elif hasattr(file_source, "read"):
        if hasattr(file_source, "name") and file_source.name:
            actual_filename = os.path.basename(file_source.name)
        file_bytes = file_source.read()
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode("utf-8")
        if hasattr(file_source, "seek"):
            file_source.seek(0)
    else:
        raise ValueError("Unsupported file source format.")

    file_size_bytes = len(file_bytes)
    if file_size_bytes == 0:
        raise ValueError("The uploaded CSV file is empty (0 bytes). At least one row of data is required.")

    # Attempt decoding across candidate encodings
    df: Optional[pd.DataFrame] = None
    decoded_text: Optional[str] = None
    used_encoding: Optional[str] = None

    for enc in SUPPORTED_ENCODINGS:
        try:
            decoded_text = file_bytes.decode(enc)
            # Try reading with pandas
            string_buffer = io.StringIO(decoded_text)
            temp_df = pd.read_csv(string_buffer, sep=None, engine="python")
            df = temp_df
            used_encoding = enc
            break
        except Exception:
            continue

    if df is None or decoded_text is None:
        raise ValueError(
            "Could not parse CSV file. Please verify that the file is a valid delimited text file "
            "with consistent columns and standard encoding (UTF-8, Latin-1, or CP1252)."
        )

    # Validate structural requirements
    if df.empty:
        raise ValueError("The uploaded CSV contains no data rows. At least one row of data is required.")

    if len(df.columns) == 0:
        raise ValueError("The uploaded CSV contains no detectable columns.")

    # Soft bound check
    row_count = len(df)
    if row_count > max_rows:
        # We process up to max_rows to maintain local responsiveness
        df = df.iloc[:max_rows].copy()
        row_count = len(df)

    metadata = DatasetMetadata(
        filename=actual_filename,
        file_size_bytes=file_size_bytes,
        row_count=row_count,
        column_count=len(df.columns),
        ingestion_timestamp=datetime.now().isoformat()
    )

    return df, metadata

load_csv_data = load_csv
