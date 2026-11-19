# Detection Module Contracts

## Module: `data_loader.py`
```python
def load_csv(file_source: Any, max_rows: int = 10000) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """
    Loads, decodes, and validates an uploaded CSV file.
    
    Args:
        file_source: File path (str) or file-like buffer (io.BytesIO / io.StringIO).
        max_rows: Soft upper bound for the hackathon baseline.
        
    Returns:
        Tuple of (loaded DataFrame, DatasetMetadata).
        
    Raises:
        ValueError: If file is empty, corrupt, or exceeds memory safety boundaries.
    """
```

## Module: `profiling.py`
```python
def generate_dataset_profile(df: pd.DataFrame, metadata: DatasetMetadata) -> DatasetProfile:
    """
    Dynamically infers data types, calculates missing/unique counts, and computes
    statistical distributions (IQR, mean, std) for numerical and categorical columns.
    
    Deterministic: Uses pure Pandas & NumPy. No LLM involvement.
    """
```

## Module: `missing_values.py`
```python
def detect_missing_values(df: pd.DataFrame, profile: DatasetProfile) -> List[QualityIssue]:
    """
    Detects null values, empty strings, and whitespace-only entries across all columns.
    Returns QualityIssue instances with exact counts and sample row indices.
    """
```

## Module: `duplicates.py`
```python
def detect_duplicates(df: pd.DataFrame, business_keys: Optional[List[str]] = None) -> List[QualityIssue]:
    """
    Implements dual-tier duplicate detection:
    1. Exact row duplicates across 100% of columns.
    2. Duplicate values in designated business-key columns where remaining columns conflict.
    """
```

## Module: `outliers.py`
```python
def detect_numeric_outliers(df: pd.DataFrame, profile: DatasetProfile, method: str = "iqr") -> List[QualityIssue]:
    """
    Computes IQR and optional Z-score anomaly bounds for numeric columns.
    Classified as 'advisory' anomalies with zero automatic validity penalties.
    """
```

## Module: `inconsistencies.py`
```python
def detect_inconsistencies(df: pd.DataFrame, profile: DatasetProfile) -> List[QualityIssue]:
    """
    Identifies casing inconsistencies, leading/trailing whitespace, and mixed format anomalies.
    """
```

## Module: `value_standardization.py`
```python
def analyze_standardization(
    df: pd.DataFrame, 
    column_name: str, 
    alias_dict: Optional[Dict[str, str]] = None,
    similarity_threshold: float = 80.0
) -> List[ValueMapping]:
    """
    Uses RapidFuzz and alias reference mappings to detect spelling variations and known synonyms.
    Tiers into Cosmetic (100%), High (>=90%), Medium (80-89%), and Low (<80%).
    """
```
