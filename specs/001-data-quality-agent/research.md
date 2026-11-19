# Phase 0: Research & Technical Architecture Decisions

**Feature**: DataGuard AI – Intelligent Data Quality & Anomaly Detection Agent  
**Branch**: `001-data-quality-agent`  
**Date**: 2026-11-19  

## Overview
This document records technical investigations, architectural choices, and design rationale for DataGuard AI, aligning the technology stack (Streamlit, Pandas, NumPy, RapidFuzz, Plotly, Gemini API, Pytest) with the 14 non-negotiable principles of the DataGuard AI Constitution.

---

## 1. UI & State Architecture: Streamlit + In-Memory Dataset Isolation

### Decision
Use **Streamlit** with `st.session_state` storing an immutable `raw_df` and an isolated mutable `working_df`.

### Rationale
- **Constitution Alignment (Principles V & VI)**: The raw uploaded dataset must remain immutable. Storing `st.session_state["raw_df"] = df.copy(deep=True)` ensures that any cleaning transformation only modifies `st.session_state["working_df"]`.
- **Rapid Iteration for Hackathon Scope (Principle XIV)**: Streamlit provides native interactive components (file uploader, dataframes, metric cards, tabs, modal confirmation dialogs) without requiring separate frontend/backend hosting.
- **Plotly Integration**: Plotly charts (`st.plotly_chart`) render interactive distribution histograms, box plots for outliers, and quality score radial/gauge charts seamlessly.

### Alternatives Considered
- *FastAPI + React/Next.js*: Superior production separation, but significantly higher deployment and development overhead. Rejected per Principle XIV (Hackathon Scope: prioritize working end-to-end app over architectural sprawl).
- *Gradio*: Good for pure ML demos, but weaker layout customization for complex multi-tab audit workflows and side-by-side diff previews.

---

## 2. Deterministic Detection & Profiling: Pandas & NumPy

### Decision
Implement all profiling, anomaly detection, statistical boundaries, and scoring using pure **Python, Pandas, and NumPy** inside dedicated functional modules (`data_loader.py`, `profiling.py`, `missing_values.py`, `duplicates.py`, `outliers.py`, `inconsistencies.py`, `scoring.py`).

### Rationale
- **Constitution Alignment (Principle III - Non-Negotiable)**: LLMs are strictly forbidden from calculating actual counts or metrics. Deterministic computation guarantees 100% reproducible results.
- **Performance Benchmark (Principle XII)**: Vectorized Pandas and NumPy operations process 10,000 rows across 30 columns in < 250 milliseconds.
- **Modular Decoupling**: Pure Python functions accept a `pd.DataFrame` and configuration parameters, returning strongly typed dataclasses/dicts. They have zero dependencies on Streamlit, enabling 100% unit test coverage via Pytest (Principle VIII).

### Outlier Detection Algorithm
- **IQR (Interquartile Range)**: Default method. Computes $Q_1$, $Q_3$, and $IQR = Q_3 - Q_1$. Lower Bound = $Q_1 - 1.5 \times IQR$, Upper Bound = $Q_3 + 1.5 \times IQR$.
- **Z-Score (Standard Deviations)**: Optional secondary filter for Gaussian distributions: $Z = \frac{x - \mu}{\sigma}$, where $|Z| > 3.0$ flags extreme anomalies.
- **Triage Status**: Tagged as `Statistical Anomaly (Advisory)` per Clarification 4 to prevent penalizing valid extreme business values.

---

## 3. String Inconsistency & Value Standardization: RapidFuzz + Alias Dictionaries

### Decision
Use **RapidFuzz** (`token_sort_ratio` / `normalized_levenshtein`) paired with configurable reference dictionaries and regex for string standardization.

### Rationale
- **Performance**: RapidFuzz is written in C++ with SIMD optimizations, running up to 50x faster than legacy `fuzzywuzzy` or pure Python Levenshtein implementations—critical for pairwise comparison on ~10,000 rows.
- **Confidence Tiers (Clarification 1)**:
  1. *Cosmetic*: Whitespace trimming (`.strip()`) and casing normalization (`.title()` / `.str.lower()`).
  2. *High Confidence (≥90%)*: Known aliases from local dictionary lookup (e.g. `{"Bangalore": "Bengaluru", "Bombay": "Mumbai"}`) or near-identical string similarity.
  3. *Medium Confidence (80–89%)*: Fuzzy match suggestions flagged as candidate spelling mistakes.
  4. *Distinct (<80%)*: Maintained as separate entities.
- **Safety**: Transformations produce a `CleaningProposal` diff. No row is mutated without user checkbox confirmation (Principle V).

### Alternatives Considered
- *FuzzyWuzzy / Levenshtein*: Slower, licensing issues (GPL vs. MIT/Apache). RapidFuzz is MIT licensed and significantly more performant.
- *LLM-driven string clustering*: Slow, non-deterministic, and violates privacy/cost boundaries.

---

## 4. AI Copilot: Multi-Provider LLM Architecture with Automatic Failover

### Decision
Implement `ai_agent.py` using a pluggable `LLMProviderManager` orchestrating:
1. **Primary Provider (Gemini)**: Google GenAI SDK (`google-genai`), retrying up to 2 times on transient network glitches with bounded exponential backoff.
2. **Secondary Provider (Groq)**: Official Groq Python SDK (`groq`), utilizing `openai/gpt-oss-120b` (or environment override), engaged automatically upon Gemini rate limits (HTTP 429), timeouts, model unavailability, or persistent server errors.
3. **Fallback Engine (Deterministic Python Templates)**: Rule-based template engine that generates instant, structured explanations if both cloud providers fail, time out, or lack valid API keys.

### Technical Deep Dive

#### A. Error Classification Taxonomy
Exceptions caught across provider SDKs are categorized by `classify_llm_error()`:
- `AUTHENTICATION` (HTTP 401, 403, invalid/missing key): Non-transient. Skips retries and immediately attempts the next provider.
- `RATE_LIMIT` (HTTP 429, quota exhausted): Transient. Reads `Retry-After` header if available, otherwise initiates failover to secondary provider after 1 backoff attempt.
- `MODEL_UNAVAILABLE` (HTTP 404, model deprecated or not provisioned): Non-transient. Immediately fails over to secondary provider.
- `TIMEOUT` (Socket connect/read timeout): Transient. Retries once with backoff, then fails over.
- `SERVER_ERROR` (HTTP 500, 502, 503, 504): Transient. Retries with backoff before failover.

#### B. Bounded Exponential Backoff Strategy
For transient errors on the active provider:
$$\text{delay} = \min\left(\text{max\_backoff},\ \text{initial\_backoff} \times \text{factor}^{\text{attempt}}\right) + \text{uniform}(0, 0.5)$$
where defaults are $\text{initial\_backoff} = 1.0\text{s}$, $\text{factor} = 2.0$, $\text{max\_backoff} = 4.0\text{s}$, and $\text{max\_retries} = 2$.

#### C. Provider Status Reporting
`LLMProviderManager.get_status()` exposes a real-time `ProviderStatusReport` rendered in the Streamlit sidebar:
- Visual status pills: `🟢 Gemini (Primary)` / `🟡 Groq (Failover Active)` / `⚪ Python Template (Offline Mode)`
- Active model identifier and error category notifications (e.g. *"Gemini rate-limited; switched to Groq (openai/gpt-oss-120b)"*).
- Zero exposure of raw API keys.

### Rationale
- **Constitution Alignment (Principles IV, IX, X, XI)**:
  - *Principle IX (Security)*: Credentials (`GEMINI_API_KEY`, `GROQ_API_KEY`, `GEMINI_MODEL`, `GROQ_MODEL`) loaded strictly from environment variables via `python-dotenv`. Keys are scrubbed and never exposed in logs or UI.
  - *Principle X (Privacy)*: Transmit strictly metadata (types, null%, stats, max 5 sanitized samples). Zero full CSV records leave the local environment.
  - *Principle XI (Reliability)*: Automatic multi-provider failover guarantees high availability during hackathon presentations or rate limit spikes. The main application (profiling, detection, scoring, cleaning) functions deterministically and independently of all LLMs.
  - *Attribution & Transparency*: Every AI-generated output records its source provider (`Gemini`, `Groq`, or `Python Template Fallback`).
  - *Principle IV (AI as Assistant)*: Generates structured JSON responses validated via schemas. Never outputs executable arbitrary code that runs automatically.

---

## 5. Scoring Engine Architecture: Proportional Dynamic Re-weighting

### Decision
Implement the DAMA-DMBOK-aligned scoring formula in `scoring.py` with dynamic proportional re-weighting:

$$\text{Column Score} = \sum_{d \in D_{\text{active}}} \left( \frac{W_d}{\sum_{k \in D_{\text{active}}} W_k} \times S_d \right)$$

where default weights are:
- Completeness ($W = 35\%$)
- Validity ($W = 30\%$)
- Consistency ($W = 25\%$)
- Uniqueness ($W = 10\%$)

### Rationale
- **Clarification 2**: Categorical columns (like `Country` or `Status`) have no uniqueness requirement; free text columns have no numerical validity checks. Excluding non-applicable dimensions and normalizing active weights to 100% prevents false penalties.
- **Dataset-Level Score**:
$$\text{Overall Score} = 0.90 \times \left( \frac{1}{N} \sum_{i=1}^N \text{ColumnScore}_i \right) + 0.10 \times \text{DatasetUniquenessScore}$$
where $\text{DatasetUniquenessScore} = 100 \times \left( 1 - \frac{\text{DuplicateRowCount}}{\text{TotalRowCount}} \right)$.

---

## 6. Safe Cleaning & Data Preservation Pipeline

### Decision
Implement `cleaning.py` with an explicit before-and-after preview contract operating strictly on `st.session_state["working_df"]`.

### Rationale
- **Constitution Alignment (Principle V & VI)**: Original uploaded dataframe is stored in `raw_df` and remains immutable.
- **Workflow**:
  1. Generate `CleaningProposal` containing proposed action type, parameters, and affected row indices.
  2. Compute a preview dataframe displaying a 5-row diff (Original vs. Cleaned).
  3. Require user to click "Apply Transformation" before executing against `working_df`.
  4. Track transformation history to support "Undo Last Transformation".
  5. Export cleaned dataset as a downloadable CSV.
