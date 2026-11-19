"""
Outlier Detection Module for DataGuard AI.
Implements IQR and Z-score numerical outlier detection.
Adheres to Clarification 4: Outliers are classified as 'Statistical Anomalies (Advisory)',
never automatically assumed to be errors or automatically penalized in scoring.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from .models import (
    DatasetProfile,
    QualityIssue,
    IssueCategory,
    IssueSeverity,
    CleaningActionType,
    InferredDataType,
)

def detect_numeric_outliers(
    df: pd.DataFrame,
    profile: Optional[DatasetProfile] = None,
    method: str = "iqr",
    z_threshold: float = 3.0
) -> List[QualityIssue]:
    """
    Detects statistical outliers in numeric columns using IQR or Z-score.
    """
    issues: List[QualityIssue] = []
    total_rows = max(1, len(df))

    if profile is None:
        from .profiling import generate_dataset_profile
        from .models import DatasetMetadata
        meta = DatasetMetadata(filename="data.csv", file_size_bytes=1, row_count=len(df), column_count=len(df.columns), ingestion_timestamp="")
        profile = generate_dataset_profile(df, meta)

    for col, col_prof in profile.columns.items():
        if col_prof.inferred_type != InferredDataType.NUMERIC:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 10:
            continue  # Insufficient sample for statistical outlier detection

        outlier_indices = []
        outlier_values = []
        lower_bound = 0.0
        upper_bound = 0.0

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                std = series.std()
                if std > 0:
                    mean = series.mean()
                    lower_bound = float(mean - 2.5 * std)
                    upper_bound = float(mean + 2.5 * std)
                    mask = (series < lower_bound) | (series > upper_bound)
                    outlier_indices = series[mask].index.tolist()
                    outlier_values = series[mask].tolist()
                else:
                    continue
            else:
                lower_bound = float(q1 - 1.5 * iqr)
                upper_bound = float(q3 + 1.5 * iqr)
                mask = (series < lower_bound) | (series > upper_bound)
                outlier_indices = series[mask].index.tolist()
                outlier_values = series[mask].tolist()

        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0 or np.isnan(std):
                continue

            z_scores = (series - mean).abs() / std
            mask = z_scores > z_threshold
            outlier_indices = series[mask].index.tolist()
            outlier_values = series[mask].tolist()
            lower_bound = float(mean - z_threshold * std)
            upper_bound = float(mean + z_threshold * std)

        outlier_count = len(outlier_indices)
        if outlier_count > 0:
            pct = round((outlier_count / total_rows) * 100, 2)
            sample_vals = [round(float(v), 2) for v in outlier_values[:5]]

            issues.append(QualityIssue(
                issue_id=f"outlier_{col}_{method}",
                category=IssueCategory.NUMERIC_OUTLIER,
                column_name=str(col),
                severity=IssueSeverity.ADVISORY,
                affected_count=outlier_count,
                affected_percentage=pct,
                sample_values=sample_vals,
                sample_row_indices=outlier_indices[:5],
                explanation=(
                    f"Column '{col}' has {outlier_count} statistical outliers ({pct}% of rows) "
                    f"outside bounds [{lower_bound:.2f}, {upper_bound:.2f}] using {method.upper()} method. "
                    "Classified as an Advisory Anomaly—review before deciding whether to cap or keep as legitimate data."
                ),
                is_advisory=True,
                suggested_action=CleaningActionType.CAP_OUTLIERS
            ))

    return issues

detect_numerical_outliers = detect_numeric_outliers
