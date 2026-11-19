"""
Report Generator Module for DataGuard AI.
Generates comprehensive executive audit reports in Markdown and standalone styled HTML.
Includes dataset metrics, quality scorecard breakdown, detected issues, and transformation provenance.
"""
from datetime import datetime
from typing import List, Optional
from .models import (
    DatasetProfile,
    QualityScorecard,
    QualityIssue,
    CleaningProposal,
    ProposalStatus,
    ProviderSource,
)

def generate_markdown_report(
    profile: DatasetProfile,
    scorecard: QualityScorecard,
    issues: List[QualityIssue],
    applied_proposals: List[CleaningProposal],
    active_provider: ProviderSource = ProviderSource.PYTHON_TEMPLATE,
    active_model: str = "deterministic-rule-template"
) -> str:
    """
    Renders an executive Markdown data quality audit report.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# DataGuard AI - Quality Audit & Health Report",
        f"**Generated**: {now_str} | **Dataset**: `{profile.metadata.filename}` | **AI Engine**: `{active_provider.value} ({active_model})`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        f"- **Overall Quality Score**: **{scorecard.overall_score} / 100**",
        f"- **Dataset Uniqueness Score**: {scorecard.dataset_uniqueness_score} / 100",
        f"- **Total Rows Evaluated**: {profile.metadata.row_count:,}",
        f"- **Total Columns**: {profile.metadata.column_count}",
        f"- **Total Anomaly Categories Detected**: {len(issues)}",
        f"- **Approved Cleanings Applied**: {len([p for p in applied_proposals if p.status == ProposalStatus.APPLIED])}",
        "",
        "### Score Deductions & Rationale",
    ]

    if scorecard.deduction_reasons:
        for reason in scorecard.deduction_reasons[:10]:
            lines.append(f"- {reason}")
    else:
        lines.append("- Zero quality deductions. Dataset meets all health thresholds.")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Column-by-Column Health Breakdown",
        "| Column | Inferred Type | Quality Score | Null % | Unique % |",
        "|---|---|---|---|---|"
    ])

    for col_name, col_prof in profile.columns.items():
        score = scorecard.column_scores.get(col_name, 100.0)
        lines.append(
            f"| `{col_name}` | {col_prof.inferred_type.value} | **{score:.1f}%** | {col_prof.null_percentage}% | {col_prof.unique_percentage}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detected Quality Anomalies & Issues",
    ])

    if issues:
        for idx, issue in enumerate(issues, start=1):
            target = f"`{issue.column_name}`" if issue.column_name else "Entire Dataset"
            lines.extend([
                f"### {idx}. {issue.category.value.replace('_', ' ').title()} - {target}",
                f"- **Severity**: `{issue.severity.value.upper()}`" + (" *(Advisory Anomaly)*" if issue.is_advisory else ""),
                f"- **Affected Records**: {issue.affected_count:,} ({issue.affected_percentage}% of rows)",
                f"- **Explanation**: {issue.explanation}",
                f"- **Sample Values**: `{issue.sample_values[:3]}`",
                ""
            ])
    else:
        lines.append("- No data quality anomalies detected.")

    lines.extend([
        "---",
        "",
        "## 4. Applied Data Transformations",
    ])

    applied = [p for p in applied_proposals if p.status == ProposalStatus.APPLIED]
    if applied:
        for idx, prop in enumerate(applied, start=1):
            col_target = f"on column `{prop.column_name}`" if prop.column_name else "on entire dataset"
            lines.append(
                f"- **Action {idx}**: `{prop.action_type.value}` {col_target} (Affected {prop.affected_rows} records)."
            )
    else:
        lines.append("- No transformations were applied. Dataset remains in original raw state.")

    lines.extend([
        "",
        "---",
        "*Report generated automatically by DataGuard AI. All deterministic computations executed with Python, Pandas & NumPy.*"
    ])

    return "\n".join(lines)

def generate_html_report(
    profile: DatasetProfile,
    scorecard: QualityScorecard,
    issues: List[QualityIssue],
    applied_proposals: List[CleaningProposal],
    active_provider: ProviderSource = ProviderSource.PYTHON_TEMPLATE,
    active_model: str = "deterministic-rule-template"
) -> str:
    """
    Renders a standalone styled HTML report ready for download.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score_color = "#10b981" if scorecard.overall_score >= 80 else ("#f59e0b" if scorecard.overall_score >= 60 else "#ef4444")

    col_rows = []
    for col_name, col_prof in profile.columns.items():
        score = scorecard.column_scores.get(col_name, 100.0)
        col_rows.append(f"""
        <tr>
            <td><strong>{col_name}</strong></td>
            <td><span class="badge badge-type">{col_prof.inferred_type.value}</span></td>
            <td><strong>{score:.1f}%</strong></td>
            <td>{col_prof.null_percentage}%</td>
            <td>{col_prof.unique_percentage}%</td>
        </tr>
        """)

    issues_cards = []
    for issue in issues:
        target = issue.column_name or "Entire Dataset"
        advisory_tag = '<span class="badge badge-advisory">Advisory</span>' if issue.is_advisory else ""
        issues_cards.append(f"""
        <div class="card">
            <h4>{issue.category.value.replace('_', ' ').title()} - {target} {advisory_tag}</h4>
            <p><strong>Severity:</strong> <span class="badge badge-sev">{issue.severity.value.upper()}</span> | 
               <strong>Affected:</strong> {issue.affected_count:,} records ({issue.affected_percentage}%)</p>
            <p>{issue.explanation}</p>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DataGuard AI - Quality Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #1e293b; background: #f8fafc; padding: 40px; margin: 0; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 20px; }}
        .score-circle {{ width: 100px; height: 100px; border-radius: 50%; background: {score_color}; color: #ffffff; display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 28px; font-weight: bold; }}
        .score-label {{ font-size: 11px; text-transform: uppercase; font-weight: normal; }}
        .badge {{ padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
        .badge-type {{ background: #e0f2fe; color: #0369a1; }}
        .badge-sev {{ background: #fee2e2; color: #b91c1c; }}
        .badge-advisory {{ background: #fef3c7; color: #b45309; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; }}
        .card {{ background: #f8fafc; border-left: 4px solid #3b82f6; padding: 16px; margin-bottom: 12px; border-radius: 0 8px 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🛡️ DataGuard AI Quality Audit</h1>
                <p>Dataset: <strong>{profile.metadata.filename}</strong> | Generated: {now_str}</p>
                <p>AI Engine: <strong>{active_provider.value} ({active_model})</strong></p>
            </div>
            <div class="score-circle">
                {scorecard.overall_score}
                <span class="score-label">Quality Score</span>
            </div>
        </div>

        <h2>Summary Metrics</h2>
        <p>Evaluated <strong>{profile.metadata.row_count:,} rows</strong> and <strong>{profile.metadata.column_count} columns</strong>. Identified <strong>{len(issues)} data defect categories</strong>.</p>

        <h2>Column Quality Health</h2>
        <table>
            <thead>
                <tr>
                    <th>Column Name</th>
                    <th>Type</th>
                    <th>Quality Score</th>
                    <th>Missing %</th>
                    <th>Unique %</th>
                </tr>
            </thead>
            <tbody>
                {''.join(col_rows)}
            </tbody>
        </table>

        <h2>Detected Quality Anomalies</h2>
        {''.join(issues_cards) if issues_cards else '<p>No data defects detected!</p>'}

        <footer style="margin-top: 40px; text-align: center; color: #64748b; font-size: 13px;">
            Generated by DataGuard AI • Deterministic Data Quality & Anomaly Detection
        </footer>
    </div>
</body>
</html>
"""
    return html
