# Cleaning & Transformation Contract

## Module: `cleaning.py`

```python
from typing import Tuple, List, Optional
import pandas as pd
from ..models import CleaningProposal, ValueMapping, CleaningActionType

def generate_cleaning_proposals(
    df: pd.DataFrame,
    issues: List[QualityIssue],
    value_mappings: Optional[List[ValueMapping]] = None
) -> List[CleaningProposal]:
    """
    Constructs candidate cleaning proposals from detected anomalies.
    Does NOT mutate any data.
    """

def preview_proposal(
    working_df: pd.DataFrame,
    proposal: CleaningProposal,
    sample_size: int = 5
) -> pd.DataFrame:
    """
    Computes a side-by-side diff comparison of affected rows before and after the action.
    Returns a dataframe displaying: [row_idx, column, before_value, after_value].
    """

def apply_proposal(
    working_df: pd.DataFrame,
    proposal: CleaningProposal
) -> Tuple[pd.DataFrame, int]:
    """
    Executes an approved cleaning proposal strictly on working_df.
    Original raw_df is never touched.
    
    Returns:
        Tuple of (updated working_df, rows_affected_count).
    """
```
