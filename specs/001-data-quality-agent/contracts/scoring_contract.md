# Quality Scoring Contract

## Module: `scoring.py`

```python
from typing import Dict, List, Optional
import pandas as pd
from ..models import DatasetProfile, QualityIssue, QualityScorecard, DimensionScore

DEFAULT_WEIGHTS = {
    "completeness": 0.35,
    "validity": 0.30,
    "consistency": 0.25,
    "uniqueness": 0.10,
}

def compute_quality_scorecard(
    df: pd.DataFrame,
    profile: DatasetProfile,
    issues: List[QualityIssue],
    business_keys: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None
) -> QualityScorecard:
    """
    Computes deterministic, reproducible quality scores (0-100) for every column
    and the overall dataset using proportional dynamic re-weighting.
    
    Rules:
    1. For each column, evaluate applicable dimensions.
    2. If Uniqueness or Validity does not apply, dynamically redistribute its weight
       proportionally across active dimensions to equal 100%.
    3. Dataset-level Uniqueness score is evaluated based on exact full duplicate rows
       and designated business keys.
    4. Generates human-readable point-deduction explanations.
    
    Returns:
        Fully populated QualityScorecard instance.
    """
```
