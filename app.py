"""
DataGuard AI – Intelligent Data Quality & Anomaly Detection Agent
Main Streamlit Application Entrypoint.
Complies with Constitution v1.0.0 and System Architecture:
- Multi-provider LLM failover (Gemini -> Groq -> Deterministic Python Templates)
- Immutable raw dataset (raw_df) vs working copy (working_df)
- Deterministic profiling and scoring using Pandas and NumPy
- Human-in-the-loop cleaning studio with before-and-after diffs
- Grounded AI Chat Assistant with zero CSV raw data leakage
"""
import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.models import (
    DatasetMetadata,
    DatasetProfile,
    QualityIssue,
    QualityScorecard,
    IssueCategory,
    IssueSeverity,
    ConfidenceTier,
    CleaningActionType,
    CleaningProposal,
    ValueMapping,
    ProposalStatus,
    ProviderSource,
    InferredDataType,
)
from src.data_loader import load_csv
from src.profiling import generate_dataset_profile
from src.missing_values import detect_missing_values
from src.duplicates import detect_duplicates
from src.outliers import detect_numeric_outliers
from src.inconsistencies import detect_inconsistencies
from src.value_standardization import (
    analyze_standardization,
    detect_alias_and_spelling_issues,
    load_alias_dictionaries,
)
from src.scoring import compute_quality_scorecard
from src.ai_agent import (
    get_llm_manager,
    explain_single_issue,
    explain_issues,
    infer_unknown_columns,
    answer_user_query,
)
from src.cleaning import (
    generate_cleaning_proposals,
    preview_proposal,
    apply_proposal,
)
from src.report_generator import generate_markdown_report, generate_html_report
from src.demo_data import generate_dirty_sample_dataframe

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataGuard AI – Quality & Anomaly Detection Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .badge-provider {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-gemini { background-color: #DEF7EC; color: #03543F; border: 1px solid #31C48D; }
    .badge-groq { background-color: #EDEBFE; color: #5521B5; border: 1px solid #9061F9; }
    .badge-template { background-color: #FEF08A; color: #713F12; border: 1px solid #EAB308; }
    .stat-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-value-good { color: #059669; font-weight: 700; font-size: 1.8rem; }
    .metric-value-warning { color: #D97706; font-weight: 700; font-size: 1.8rem; }
    .metric-value-bad { color: #DC2626; font-weight: 700; font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "working_df" not in st.session_state:
    st.session_state["working_df"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "profile" not in st.session_state:
    st.session_state["profile"] = None
if "issues" not in st.session_state:
    st.session_state["issues"] = []
if "scorecard" not in st.session_state:
    st.session_state["scorecard"] = None
if "business_keys" not in st.session_state:
    st.session_state["business_keys"] = []
if "applied_proposals" not in st.session_state:
    st.session_state["applied_proposals"] = []
if "applied_history" not in st.session_state:
    st.session_state["applied_history"] = []
if "triage_outliers" not in st.session_state:
    st.session_state["triage_outliers"] = {}  # col -> is_whitelisted bool
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "ai_explanations" not in st.session_state:
    st.session_state["ai_explanations"] = {}
if "column_inferences" not in st.session_state:
    st.session_state["column_inferences"] = []
if "custom_value_mappings" not in st.session_state:
    st.session_state["custom_value_mappings"] = []

alias_dict = load_alias_dictionaries()
llm_mgr = get_llm_manager()

# -----------------------------------------------------------------------------
# 3. Deterministic Pipeline Execution Helper
# -----------------------------------------------------------------------------
def analyze_current_dataset():
    """
    Executes deterministic profiling, multi-category anomaly detection,
    and DAMA quality scoring on working_df.
    """
    if st.session_state["working_df"] is None or st.session_state["working_df"].empty:
        return

    df = st.session_state["working_df"]
    meta = st.session_state["metadata"]
    b_keys = st.session_state["business_keys"]

    # 1. Profile
    profile = generate_dataset_profile(df, meta)
    st.session_state["profile"] = profile

    # 2. Detect Anomalies
    issues: List[QualityIssue] = []
    issues.extend(detect_missing_values(df, profile))
    issues.extend(detect_duplicates(df, business_keys=b_keys))
    issues.extend(detect_numeric_outliers(df, profile, method="iqr"))
    issues.extend(detect_inconsistencies(df, profile))

    for col in df.columns:
        if profile.columns[col].inferred_type in (InferredDataType.CATEGORICAL, InferredDataType.TEXT):
            std_issues = detect_alias_and_spelling_issues(df, col, alias_dict=alias_dict)
            issues.extend(std_issues)

    # 3. Respect Outlier Triage
    for issue in issues:
        if issue.category == IssueCategory.NUMERIC_OUTLIER and issue.column_name:
            if st.session_state["triage_outliers"].get(issue.column_name, False):
                issue.is_advisory = True
                # Whitelisted by user -> no score deduction

    st.session_state["issues"] = issues

    # 4. Score
    scorecard = compute_quality_scorecard(df, profile, issues, business_keys=b_keys)
    st.session_state["scorecard"] = scorecard

def load_new_dataframe(df: pd.DataFrame, filename: str):
    """Initializes raw_df and working_df, ensuring raw_df is strictly immutable."""
    st.session_state["raw_df"] = df.copy(deep=True)
    st.session_state["working_df"] = df.copy(deep=True)
    st.session_state["metadata"] = DatasetMetadata(
        filename=filename,
        file_size_bytes=int(df.memory_usage(deep=True).sum()),
        row_count=len(df),
        column_count=len(df.columns),
        ingestion_timestamp=datetime.now().isoformat()
    )
    # Automatically infer business keys if suitable column exists
    potential_keys = [c for c in df.columns if c.lower().endswith("id") or "code" in c.lower() or "key" in c.lower()]
    st.session_state["business_keys"] = [potential_keys[0]] if potential_keys else []
    st.session_state["applied_proposals"] = []
    st.session_state["applied_history"] = []
    st.session_state["triage_outliers"] = {}
    st.session_state["ai_explanations"] = {}
    st.session_state["column_inferences"] = []
    st.session_state["custom_value_mappings"] = []
    st.session_state["chat_history"] = []
    analyze_current_dataset()

# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Multi-Provider Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ DataGuard AI")
    st.caption("Intelligent Data Quality & Anomaly Detection Agent")
    st.markdown("---")

    # Multi-Provider Status Pill
    status = llm_mgr.get_status()
    st.markdown("### 🤖 AI Engine Provenance")
    
    if status.active_provider == ProviderSource.GEMINI:
        badge_html = f'<span class="badge-provider badge-gemini">Primary: Gemini ({status.active_model})</span>'
    elif status.active_provider == ProviderSource.GROQ:
        badge_html = f'<span class="badge-provider badge-groq">Secondary Failover: Groq ({status.active_model})</span>'
    else:
        badge_html = f'<span class="badge-provider badge-template">Fallback: Python Templates ({status.active_model})</span>'

    st.markdown(badge_html, unsafe_allow_html=True)
    
    with st.expander("Provider Configuration & Health", expanded=False):
        st.write(f"**Gemini Primary**: {'🟢 Ready' if status.gemini_healthy else '⚪ Offline / Unconfigured'}")
        st.write(f"**Groq Secondary**: {'🟢 Ready' if status.groq_healthy else '⚪ Offline / Unconfigured'}")
        st.write(f"**Deterministic Fallback**: 🟢 Always Active")
        st.caption("Automatic failover operates with bounded exponential backoff. Zero raw CSV records are ever transmitted.")

    st.markdown("---")

    # Dataset Ingestion Controls
    st.markdown("### 📂 Ingest Dataset")
    uploaded_file = st.file_uploader("Upload CSV File (up to 10k rows)", type=["csv"], key="csv_uploader")
    if uploaded_file is not None:
        if st.session_state["metadata"] is None or st.session_state["metadata"].filename != uploaded_file.name:
            try:
                loaded_df, meta = load_csv(uploaded_file, filename=uploaded_file.name)
                load_new_dataframe(loaded_df, uploaded_file.name)
                st.success(f"Loaded '{uploaded_file.name}' successfully!")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    # Demo Dataset Button
    if st.button("🧪 Load Dirty Customer Demo Dataset (1,000 rows)", use_container_width=True):
        demo_df = generate_dirty_sample_dataframe(num_rows=1000)
        load_new_dataframe(demo_df, "dirty_customer_demo.csv")
        st.success("Loaded synthetic benchmark dataset with seeded real-world anomalies!")
        st.rerun()

    # Business Key Selector
    if st.session_state["working_df"] is not None:
        st.markdown("---")
        st.markdown("### 🔑 Business Key Configuration")
        all_cols = list(st.session_state["working_df"].columns)
        default_keys = [k for k in st.session_state["business_keys"] if k in all_cols]
        selected_keys = st.multiselect(
            "Designate Unique Key Column(s):",
            options=all_cols,
            default=default_keys,
            help="Select columns that should identify unique business entities (e.g., customer_id)."
        )
        if selected_keys != st.session_state["business_keys"]:
            st.session_state["business_keys"] = selected_keys
            analyze_current_dataset()
            st.rerun()

    # Grounded AI Chat Assistant in Sidebar
    st.markdown("---")
    st.markdown("### 💬 DataGuard AI Assistant")
    st.caption("Ask questions grounded strictly in computed metrics (Gemini & Groq).")

    # Show last 2 chat interactions
    for msg in st.session_state["chat_history"][-3:]:
        if msg["role"] == "user":
            st.markdown(f"**You**: {msg['content']}")
        else:
            st.markdown(f"**DataGuard AI**: {msg['content']}")
            st.caption(f"Source: {msg.get('source', 'System')}")

    chat_query = st.text_input("Ask a question about the dataset:", placeholder="e.g. Which column has the lowest score?", key="sidebar_chat_input")
    if st.button("Send Query", use_container_width=True) and chat_query.strip():
        if st.session_state["scorecard"] and st.session_state["profile"]:
            with st.spinner("Analyzing quality context..."):
                ans, prov, model = answer_user_query(
                    chat_query,
                    st.session_state["scorecard"],
                    st.session_state["profile"],
                    st.session_state["issues"],
                    manager=llm_mgr
                )
                st.session_state["chat_history"].append({"role": "user", "content": chat_query})
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": ans,
                    "source": f"{prov.value} ({model})"
                })
                st.rerun()

# -----------------------------------------------------------------------------
# 5. Main Dashboard View
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ DataGuard AI – Quality & Anomaly Detection Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated data health profiling, multi-category anomaly detection, DAMA scoring, and human-in-the-loop cleaning.</div>', unsafe_allow_html=True)

# Guard against un-ingested state
if st.session_state["working_df"] is None:
    st.info("👋 Welcome to DataGuard AI! Ingest a CSV dataset or click **'Load Dirty Customer Demo Dataset'** in the sidebar to begin analysis.")
    st.stop()

# Top KPI Metric Cards Ribbon
df_curr = st.session_state["working_df"]
scorecard = st.session_state["scorecard"]
issues = st.session_state["issues"]
meta = st.session_state["metadata"]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    score = scorecard.overall_score if scorecard else 100.0
    color_class = "metric-value-good" if score >= 80 else ("metric-value-warning" if score >= 60 else "metric-value-bad")
    st.markdown(f'<div class="stat-card"><div>Overall Health Score</div><div class="{color_class}">{score} / 100</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="stat-card"><div>Total Rows</div><div style="font-size:1.8rem; font-weight:700;">{len(df_curr):,}</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="stat-card"><div>Total Columns</div><div style="font-size:1.8rem; font-weight:700;">{len(df_curr.columns)}</div></div>', unsafe_allow_html=True)

with c4:
    issue_count = len(issues)
    st.markdown(f'<div class="stat-card"><div>Active Anomalies</div><div style="font-size:1.8rem; font-weight:700; color:#EF4444;">{issue_count}</div></div>', unsafe_allow_html=True)

with c5:
    applied_count = len(st.session_state["applied_history"])
    st.markdown(f'<div class="stat-card"><div>Applied Cleanings</div><div style="font-size:1.8rem; font-weight:700; color:#3B82F6;">{applied_count}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Tabbed User Interface
# -----------------------------------------------------------------------------
tab_overview, tab_anomalies, tab_standardization, tab_cleaning, tab_reports = st.tabs([
    "📊 Overview & Quality Scorecard",
    f"⚠️ Detected Anomalies ({len(issues)})",
    "🔤 Value Standardization",
    "🧹 Safe Cleaning Studio",
    "📑 Audit Reports & Export"
])

# -----------------------------------------------------------------------------
# Tab 1: Overview & Quality Scorecard
# -----------------------------------------------------------------------------
with tab_overview:
    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        # Plotly Radial Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scorecard.overall_score if scorecard else 100.0,
            title={"text": "Overall Data Quality Score", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "#2563EB"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 59], "color": "#FEE2E2"},
                    {"range": [60, 79], "color": "#FEF3C7"},
                    {"range": [80, 100], "color": "#DCFCE7"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 60
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart_right:
        # DAMA Dimension Breakdown Bar Chart
        comp_scores = [d.completeness for d in scorecard.column_dimensions.values()]
        val_scores = [d.validity for d in scorecard.column_dimensions.values() if "validity" in d.active_weights]
        cons_scores = [d.consistency for d in scorecard.column_dimensions.values()]
        
        avg_dims = {
            "Completeness": round(sum(comp_scores) / max(1, len(comp_scores)), 1),
            "Validity": round(sum(val_scores) / max(1, len(val_scores)), 1) if val_scores else 100.0,
            "Consistency": round(sum(cons_scores) / max(1, len(cons_scores)), 1),
            "Uniqueness": scorecard.dataset_uniqueness_score
        }

        fig_dims = px.bar(
            x=list(avg_dims.keys()),
            y=list(avg_dims.values()),
            labels={"x": "DAMA Quality Dimension", "y": "Score (0 - 100)"},
            color=list(avg_dims.values()),
            color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
            range_y=[0, 100],
            title="DAMA Dimension Quality Breakdown"
        )
        fig_dims.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig_dims, use_container_width=True)

    # Score Deductions & Health Rationale
    with st.expander("🔍 Score Deductions & Quality Rationale", expanded=True):
        if scorecard.deduction_reasons:
            for reason in scorecard.deduction_reasons:
                st.markdown(f"- ⚠️ {reason}")
        else:
            st.success("🎉 Zero quality deductions detected! Dataset achieves a perfect 100/100 score.")

    st.markdown("---")

    # Column Health & Statistics Explorer Table
    st.markdown("### 📋 Dynamic Column Health Explorer")
    prof = st.session_state["profile"]
    col_rows = []
    for c_name, c_prof in prof.columns.items():
        c_score = scorecard.column_scores.get(c_name, 100.0)
        col_rows.append({
            "Column Name": c_name,
            "Inferred Type": c_prof.inferred_type.value,
            "Quality Score": f"{c_score:.1f}%",
            "Null Count": c_prof.null_count,
            "Null %": f"{c_prof.null_percentage}%",
            "Unique Values": c_prof.unique_count,
            "Sample Values": ", ".join(c_prof.statistics.sample_values[:3]),
            "Is Key?": "🔑 Yes" if c_prof.is_business_key else "No"
        })
    st.dataframe(pd.DataFrame(col_rows), use_container_width=True)

    # Deep Column Distribution Inspector
    st.markdown("#### 🔬 Inspect Column Distribution")
    selected_inspect_col = st.selectbox("Select column to inspect:", options=list(prof.columns.keys()))
    if selected_inspect_col:
        c_meta = prof.columns[selected_inspect_col]
        series = df_curr[selected_inspect_col]
        
        ci1, ci2 = st.columns([1, 1])
        with ci1:
            st.write(f"**Inferred Type**: `{c_meta.inferred_type.value}` | **Unique Count**: `{c_meta.unique_count}`")
            if c_meta.inferred_type == InferredDataType.NUMERIC:
                stats = c_meta.statistics
                st.write(f"- Min: `{stats.min_val}` | Max: `{stats.max_val}`")
                st.write(f"- Mean: `{round(stats.mean_val or 0, 2)}` | Median: `{stats.median_val}` | Std: `{round(stats.std_val or 0, 2)}`")
                st.write(f"- IQR: `{stats.iqr_val}` (Bounds: [{round(stats.lower_bound or 0, 2)}, {round(stats.upper_bound or 0, 2)}])")
            else:
                st.write("**Top Frequency Values:**")
                st.write(c_meta.statistics.top_categories)

        with ci2:
            if c_meta.inferred_type == InferredDataType.NUMERIC:
                fig_num = px.histogram(df_curr, x=selected_inspect_col, marginal="box", title=f"Distribution of {selected_inspect_col}")
                fig_num.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_num, use_container_width=True)
            else:
                top_data = c_meta.statistics.top_categories
                if top_data:
                    fig_cat = px.bar(x=list(top_data.keys()), y=list(top_data.values()), labels={"x": "Value", "y": "Count"}, title=f"Frequencies in {selected_inspect_col}")
                    fig_cat.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_cat, use_container_width=True)

    # AI Column Understanding
    st.markdown("---")
    st.markdown("### 🧠 AI Semantic Understanding & Rules")
    if not st.session_state["column_inferences"]:
        if st.button("Generate AI Column Understanding & Semantic Rules", key="btn_col_understanding"):
            with st.spinner("Analyzing column semantics across providers..."):
                infs, prov, model = infer_unknown_columns(st.session_state["profile"], manager=llm_mgr)
                st.session_state["column_inferences"] = infs
                st.rerun()
    
    if st.session_state["column_inferences"]:
        inf_cols = st.columns(min(3, len(st.session_state["column_inferences"])))
        for idx, inf in enumerate(st.session_state["column_inferences"][:6]):
            with inf_cols[idx % len(inf_cols)]:
                badge_color = "green" if inf.confidence >= 80 else "orange"
                st.markdown(f"""
                <div style="border:1px solid #CBD5E1; border-radius:6px; padding:0.75rem; margin-bottom:0.75rem; background:#F8FAFC;">
                    <div style="font-weight:700; font-size:1.05rem;">{inf.column_name}</div>
                    <div style="font-size:0.85rem; color:#475569;">{inf.suggested_meaning}</div>
                    <div style="margin-top:0.4rem; font-size:0.8rem; color:{badge_color};">Confidence: <b>{inf.confidence}%</b></div>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 2: Detected Anomalies & Outlier Triage
# -----------------------------------------------------------------------------
with tab_anomalies:
    st.markdown("### ⚠️ Multi-Category Anomaly Inspector")
    st.caption("Deterministic issue classification with AI remediation explanations and triage controls.")

    if not issues:
        st.success("🎉 No anomalies detected in current dataset!")
    else:
        # Category filter pills
        categories = list(set(i.category.value for i in issues))
        selected_cat = st.radio("Filter by Anomaly Category:", options=["All"] + categories, horizontal=True)

        filtered_issues = issues if selected_cat == "All" else [i for i in issues if i.category.value == selected_cat]

        for idx, issue in enumerate(filtered_issues):
            severity_colors = {
                IssueSeverity.CRITICAL: "#DC2626",
                IssueSeverity.HIGH: "#EA580C",
                IssueSeverity.MEDIUM: "#D97706",
                IssueSeverity.LOW: "#2563EB",
                IssueSeverity.ADVISORY: "#6B7280"
            }
            sev_color = severity_colors.get(issue.severity, "#4B5563")
            col_target = f"Column: `{issue.column_name}`" if issue.column_name else "Entire Dataset"

            with st.expander(f"{idx+1}. {issue.category.value.replace('_', ' ').title()} ({col_target}) – Affected: {issue.affected_count:,} ({issue.affected_percentage}%)", expanded=(idx == 0)):
                st.markdown(f"""
                <div style="display:flex; gap:10px; align-items:center; margin-bottom:10px;">
                    <span style="background:{sev_color}; color:white; padding:3px 8px; border-radius:4px; font-weight:700; font-size:0.8rem;">
                        {issue.severity.value.upper()}
                    </span>
                    <span style="color:#64748B; font-size:0.9rem;">Suggested Action: <code>{issue.suggested_action.value}</code></span>
                </div>
                """, unsafe_allow_html=True)

                st.write(f"**Deterministic Explanation**: {issue.explanation}")

                # Advisory Outlier Triage Controls (Principle IV)
                if issue.category == IssueCategory.NUMERIC_OUTLIER and issue.column_name:
                    is_whitelisted = st.session_state["triage_outliers"].get(issue.column_name, False)
                    col_triage_1, col_triage_2 = st.columns([1, 3])
                    with col_triage_1:
                        if is_whitelisted:
                            if st.button("Flag as Quality Issue", key=f"btn_flag_{issue.issue_id}"):
                                st.session_state["triage_outliers"][issue.column_name] = False
                                analyze_current_dataset()
                                st.rerun()
                        else:
                            if st.button("Whitelist as Normal Variation", key=f"btn_whitelist_{issue.issue_id}"):
                                st.session_state["triage_outliers"][issue.column_name] = True
                                analyze_current_dataset()
                                st.rerun()
                    with col_triage_2:
                        if is_whitelisted:
                            st.info("ℹ️ Currently whitelisted as expected business distribution (zero quality score penalty).")
                        else:
                            st.caption("Advisory anomaly: you may whitelist this variation if expected in business operations.")

                # Sample Affected Rows
                if issue.sample_row_indices and len(df_curr) > 0:
                    st.write("**Affected Sample Records:**")
                    valid_indices = [i for i in issue.sample_row_indices if i in df_curr.index]
                    if valid_indices:
                        st.dataframe(df_curr.loc[valid_indices].head(5), use_container_width=True)

                # AI Explanation Section with Failover Attribution
                st.markdown("---")
                st.markdown("#### 🧠 AI Explanation & Recommended Remediation")

                cached_exp = st.session_state["ai_explanations"].get(issue.issue_id)
                if not cached_exp:
                    if st.button(f"Generate AI Explanation for this Issue", key=f"btn_exp_{issue.issue_id}"):
                        with st.spinner("Generating explanation via AI failover manager..."):
                            exp = explain_single_issue(issue, st.session_state["profile"], manager=llm_mgr)
                            st.session_state["ai_explanations"][issue.issue_id] = exp
                            st.rerun()
                else:
                    st.markdown(f"**Plain-English Summary**: {cached_exp.plain_english_summary}")
                    st.markdown(f"**Business Impact**: {cached_exp.business_impact}")
                    st.markdown(f"**Recommended Fix**: {cached_exp.recommended_fix}")
                    if cached_exp.suggested_python_code:
                        st.code(cached_exp.suggested_python_code, language="python")
                    
                    prov_badge = f"Generated by **{cached_exp.provider_source.value}** (`{cached_exp.model_name}`)"
                    st.caption(f"Provenance: {prov_badge}")

# -----------------------------------------------------------------------------
# Tab 3: Value Standardization (RapidFuzz & Synonym Dictionaries)
# -----------------------------------------------------------------------------
with tab_standardization:
    st.markdown("### 🔤 Fuzzy Value Standardization")
    st.caption("Group text variations and map them to canonical values using RapidFuzz and configurable alias dictionaries.")

    string_cols = [c for c, p in prof.columns.items() if p.inferred_type in (InferredDataType.CATEGORICAL, InferredDataType.TEXT)]

    if not string_cols:
        st.info("No categorical or text columns detected for standardization.")
    else:
        selected_std_col = st.selectbox("Select Column for Standardization Review:", options=string_cols, key="std_col_select")
        
        mappings = analyze_standardization(df_curr, selected_std_col, alias_dict=alias_dict, similarity_threshold=80.0)

        if not mappings:
            st.success(f"No spelling or alias variations detected in column '{selected_std_col}'. Values are clean!")
        else:
            st.write(f"Detected **{len(mappings)} potential variations** in `{selected_std_col}`:")

            tier_badges = {
                ConfidenceTier.COSMETIC: "🟢 Cosmetic (100%)",
                ConfidenceTier.HIGH_CONFIDENCE: "🔵 High Confidence (≥90%)",
                ConfidenceTier.MEDIUM_CONFIDENCE: "🟡 Medium Confidence (80-89%)",
                ConfidenceTier.LOW_CONFIDENCE: "⚪ Low Confidence (<80%)"
            }

            rows_to_display = []
            for m in mappings:
                rows_to_display.append({
                    "Original Value": m.original_value,
                    "Frequency": m.frequency,
                    "Suggested Target": m.replacement_value,
                    "Similarity %": f"{m.similarity_score}%",
                    "Confidence Tier": tier_badges.get(m.confidence_tier, m.confidence_tier.value)
                })

            st.dataframe(pd.DataFrame(rows_to_display), use_container_width=True)

            col_btn_1, col_btn_2 = st.columns([1, 2])
            with col_btn_1:
                if st.button("Send All High-Confidence Mappings to Cleaning Studio", key="btn_send_high_conf"):
                    approved_maps = [m for m in mappings if m.confidence_tier in (ConfidenceTier.COSMETIC, ConfidenceTier.HIGH_CONFIDENCE)]
                    for m in approved_maps:
                        m.is_approved = True
                    st.session_state["custom_value_mappings"].extend(approved_maps)
                    st.success(f"Added {len(approved_maps)} approved mappings for '{selected_std_col}' to Cleaning Studio!")
                    st.rerun()

# -----------------------------------------------------------------------------
# Tab 4: Safe Cleaning Studio (Human Approval & Live Diff Preview)
# -----------------------------------------------------------------------------
with tab_cleaning:
    st.markdown("### 🧹 Human-in-the-Loop Cleaning Studio")
    st.info("🛡️ **Constitution Principle V & VI**: The original dataset (`raw_df`) is strictly immutable. All transformations are previewed before approval and executed exclusively on `working_df`.")

    # Generate Candidate Proposals
    candidate_proposals = generate_cleaning_proposals(df_curr, issues, st.session_state["custom_value_mappings"])

    # If user sent custom value mappings from tab 3, add them as proposals
    if st.session_state["custom_value_mappings"]:
        cols_with_maps = set()
        for m in st.session_state["custom_value_mappings"]:
            # find which column this belongs to
            for c in df_curr.columns:
                if m.original_value in df_curr[c].astype(str).values:
                    cols_with_maps.add(c)
        for c in cols_with_maps:
            c_maps = [m for m in st.session_state["custom_value_mappings"] if m.original_value in df_curr[c].astype(str).values]
            candidate_proposals.append(CleaningProposal(
                proposal_id=f"proposal_alias_{c}",
                action_type=CleaningActionType.MAP_ALIASES,
                column_name=c,
                parameters={"mappings_count": len(c_maps)},
                affected_rows=sum(m.frequency for m in c_maps),
                status=ProposalStatus.PENDING,
                value_mappings=c_maps
            ))

    if not candidate_proposals:
        st.success("No automated cleaning proposals pending. Data is in optimal state!")
    else:
        st.markdown(f"#### Pending Proposals ({len(candidate_proposals)})")

        approved_proposal_ids = []

        for p_idx, prop in enumerate(candidate_proposals):
            target_str = f"on column `{prop.column_name}`" if prop.column_name else "on entire dataset"
            action_title = prop.action_type.value.replace("_", " ").title()

            with st.container():
                st.markdown(f"**{p_idx + 1}. {action_title}** {target_str} (Affects: **{prop.affected_rows:,} rows**)")

                col_chk, col_prev = st.columns([1, 4])
                with col_chk:
                    is_checked = st.checkbox("Approve", key=f"chk_{prop.proposal_id}_{p_idx}", value=True)
                    if is_checked:
                        approved_proposal_ids.append(prop.proposal_id)

                with col_prev:
                    with st.expander("🔍 Preview Side-by-Side Diff Comparison", expanded=False):
                        diff_df = preview_proposal(df_curr, prop, sample_size=5)
                        if diff_df.empty:
                            st.write("No row differences to preview.")
                        else:
                            st.dataframe(diff_df, use_container_width=True)

                st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_exec_1, col_exec_2 = st.columns([1, 1])

        with col_exec_1:
            if st.button("🚀 Apply Approved Transformations", type="primary", use_container_width=True):
                if not approved_proposal_ids:
                    st.warning("Please check at least one approved transformation.")
                else:
                    proposals_to_run = [p for p in candidate_proposals if p.proposal_id in approved_proposal_ids]
                    total_modified = 0
                    current_work_df = st.session_state["working_df"].copy()

                    for p in proposals_to_run:
                        current_work_df, mod_rows = apply_proposal(current_work_df, p)
                        total_modified += mod_rows
                        st.session_state["applied_history"].append({
                            "Action": p.action_type.value,
                            "Column": p.column_name or "All",
                            "Rows Affected": mod_rows,
                            "Timestamp": datetime.now().strftime("%H:%M:%S")
                        })

                    st.session_state["working_df"] = current_work_df
                    st.session_state["custom_value_mappings"] = []
                    analyze_current_dataset()
                    st.success(f"Successfully applied {len(proposals_to_run)} transformations! Modified {total_modified:,} rows.")
                    st.rerun()

        with col_exec_2:
            if st.button("🔄 Reset Working Copy to Original Raw Data", use_container_width=True):
                st.session_state["working_df"] = st.session_state["raw_df"].copy(deep=True)
                st.session_state["applied_history"] = []
                st.session_state["custom_value_mappings"] = []
                analyze_current_dataset()
                st.info("Reset working copy to original immutable uploaded state.")
                st.rerun()

    # Applied Transformation History Audit Log
    if st.session_state["applied_history"]:
        st.markdown("---")
        st.markdown("#### 📜 Audit History of Applied Transformations")
        st.dataframe(pd.DataFrame(st.session_state["applied_history"]), use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 5: Audit Reports & Clean Data Export
# -----------------------------------------------------------------------------
with tab_reports:
    st.markdown("### 📑 Quality Audit Reporting & Export")
    st.caption("Generate downloadable executive audit reports (Markdown/HTML) and export cleaned dataset.")

    c_rep1, c_rep2, c_rep3 = st.columns(3)

    # 1. Cleaned CSV Download
    with c_rep1:
        st.markdown("#### 📥 Cleaned CSV")
        cleaned_csv_data = df_curr.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV",
            data=cleaned_csv_data,
            file_name=f"cleaned_{meta.filename}",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"Reflects all {len(st.session_state['applied_history'])} approved transformations.")

    # 2. Executive HTML Audit Report Download
    with c_rep2:
        st.markdown("#### 🌐 HTML Audit Report")
        html_report_str = generate_html_report(
            prof,
            scorecard,
            issues,
            st.session_state["applied_proposals"],
            active_provider=status.active_provider,
            active_model=status.active_model
        )
        st.download_button(
            label="Download HTML Audit Report",
            data=html_report_str.encode("utf-8"),
            file_name=f"DataGuard_Quality_Report_{meta.filename}.html",
            mime="text/html",
            use_container_width=True
        )
        st.caption("Standalone styled HTML document with scorecards, issues, and change history.")

    # 3. Executive Markdown Audit Report Download
    with c_rep3:
        st.markdown("#### 📝 Markdown Audit Report")
        md_report_str = generate_markdown_report(
            prof,
            scorecard,
            issues,
            st.session_state["applied_proposals"],
            active_provider=status.active_provider,
            active_model=status.active_model
        )
        st.download_button(
            label="Download Markdown Audit Report",
            data=md_report_str.encode("utf-8"),
            file_name=f"DataGuard_Quality_Report_{meta.filename}.md",
            mime="text/markdown",
            use_container_width=True
        )
        st.caption("Standard GitHub-flavored Markdown for engineering documentation.")

    # Live Report Preview
    st.markdown("---")
    st.markdown("#### 👁️ Report Preview")
    preview_tab_html, preview_tab_md = st.tabs(["HTML Preview", "Markdown Preview"])
    with preview_tab_html:
        st.components.v1.html(html_report_str, height=600, scrolling=True)
    with preview_tab_md:
        st.markdown(md_report_str)
