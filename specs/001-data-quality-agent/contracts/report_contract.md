# Report Generator Contract

## Module: `report_generator.py`

```python
from ..models import DatasetProfile, QualityScorecard, QualityIssue, CleaningProposal

def generate_markdown_report(
    profile: DatasetProfile,
    scorecard: QualityScorecard,
    issues: List[QualityIssue],
    applied_cleanings: List[CleaningProposal]
) -> str:
    """
    Renders an executive Markdown data quality audit report with scores,
    issue breakdown, and transformation history.
    """

def generate_html_report(
    profile: DatasetProfile,
    scorecard: QualityScorecard,
    issues: List[QualityIssue],
    applied_cleanings: List[CleaningProposal]
) -> str:
    """
    Renders a standalone styled HTML report ready for download and stakeholder sharing.
    """
```
