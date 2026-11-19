# Quickstart & Verification Guide

**Feature**: DataGuard AI – Intelligent Data Quality & Anomaly Detection Agent  
**Branch**: `001-data-quality-agent`  
**Date**: 2026-11-19  

## 1. Prerequisites
- **Python**: Python 3.10+ (tested with Python 3.14)
- **Git**: Installed
- **Optional**: Gemini API key for AI copilot narrative insights (system works in offline template mode if omitted).

---

## 2. Installation & Setup

```bash
# 1. Clone/navigate to repository root
cd path/to/hackathon

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# 3. Install required packages
pip install streamlit pandas numpy rapidfuzz plotly google-genai groq python-dotenv pytest

# 4. Configure environment
cp .env.example .env
# Edit .env and set credentials:
# GEMINI_API_KEY="your-gemini-api-key"
# GROQ_API_KEY="your-groq-api-key"
```

---

## 3. Running Unit Tests

All deterministic profiling, issue detection, scoring, cleaning algorithms, and LLM failover handlers must pass automated tests:

```bash
# Run complete test suite
pytest -v

# Run multi-provider failover and fallback unit tests specifically
pytest tests/test_ai_agent.py -v

# Run with coverage report
pytest --cov=src tests/
```

### Verified Failover Test Scenarios in `test_ai_agent.py`:
1. **Healthy Primary**: Gemini succeeds → Verifies result returned with `ProviderSource.GEMINI`.
2. **Transient Rate Limit (429)**: Gemini returns 429 → Verifies bounded exponential backoff retry, then clean failover to Groq with `ProviderSource.GROQ`.
3. **Authentication Failure (401/403)**: Gemini has invalid key → Verifies non-transient immediate failover to Groq without retry delay.
4. **Dual Provider Outage**: Both Gemini and Groq fail or time out → Verifies zero exceptions raised, deterministic template explanations generated with `ProviderSource.PYTHON_TEMPLATE`.
5. **Zero-LLM Pipeline Integrity**: All CSV upload, profiling, detection, scoring, and cleaning tests run with zero API keys and pass 100%.

---

## 4. Launching the Application

```bash
# Start Streamlit application
streamlit run app.py
```

---

## 5. End-to-End Verification Scenarios

### Scenario 1: Upload & Dynamic Profiling
1. Open the web UI at `http://localhost:8501`.
2. Drag and drop a test CSV (e.g., `sample_dirty.csv` with ~1,000 to 10,000 rows).
3. **Verify**: Total rows, column count, inferred data types, and null/unique percentages appear immediately in the top metric bar.

### Scenario 2: Deterministic Quality Scoring
1. Observe the **Quality Scorecard** gauge (0–100).
2. Inspect the dimension breakdown: Completeness (35%), Validity (30%), Consistency (25%), Uniqueness (10%).
3. **Verify**: Uniqueness is excluded from non-identifier columns, and active dimensions are proportionally re-weighted to 100%.

### Scenario 3: Anomaly & Inconsistency Inspection
1. Navigate to the **Detected Issues** tab.
2. Review Missing Values, Duplicates, and Outliers (flagged as *Advisory* with IQR thresholds).
3. Under **Inconsistent Values**, observe city variations (e.g., `Bengaluru`, `Bangalore`, `Banglore`, `bengaluru `) grouped into Cosmetic, High, and Medium confidence tiers.

### Scenario 4: AI Copilot, Multi-Provider Failover & Grounded Q&A
1. View the plain-English explanation generated for a detected issue.
2. Observe the provider attribution badge (`Gemini`, `Groq`, or `Python Template Fallback`).
3. If primary key or Gemini is unavailable or rate-limited, verify automatic failover to Groq (`openai/gpt-oss-120b`) or template fallback with zero crashes.
4. Open the **DataGuard Chat** sidebar.
5. Ask: *"Which column contains the most issues?"* or *"Why is my completeness score low?"*.
6. **Verify**: The answer cites exact metrics from the dataset scorecard and does not hallucinate non-existent columns.

### Scenario 5: Preview, Safe Cleaning & Export
1. Under **Cleaning Studio**, select a proposed action (e.g. *Standardize City Names* or *Trim Whitespace*).
2. Click **Preview Changes** to inspect the side-by-side diff table (Original vs. Proposed Cleaned).
3. Click **Approve & Apply**.
4. **Verify**: The working copy updates, the raw dataset remains intact, and a download button appears for the cleaned CSV and the HTML/Markdown audit report.
