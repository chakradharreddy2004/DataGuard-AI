# Feature Specification: DataGuard AI – Intelligent Data Quality & Anomaly Detection Agent

**Feature Branch**: `001-data-quality-agent`

**Created**: 2026-11-19

**Status**: Draft

**Input**: User description: "Build a web application called DataGuard AI – Intelligent Data Quality & Anomaly Detection Agent..."

## Clarifications

### Session 2026-11-19 (Initial Clarifications)
- Q: How should the standardization engine categorize text variations, establish fuzzy matching confidence thresholds, and present them for user review? → A: Tiered Matching & Review: Distinguish exact cosmetic fixes (whitespace trim / casing); map known dictionary aliases (≥90% confidence); flag fuzzy spelling similarities (80–89% confidence) as candidate suggestions; keep <80% distinct; require explicit user approval for all merges.
- Q: How should the scoring engine handle dimensions that do not apply to specific columns (e.g., Uniqueness on categorical columns, or Numerical Validity on text columns)? → A: Dynamic Re-weighting: Exclude non-applicable dimensions per column and scale the remaining active dimensions proportionally to 100% (e.g., a categorical column without uniqueness scales Completeness to 38.9%, Validity to 33.3%, Consistency to 27.8%); dataset-level Uniqueness (10%) scores full duplicate rows/keys.
- Q: How should DataGuard AI detect, separate, and present exact duplicate rows versus business identifier duplicates? → A: Dual-Tier Detection: Automatically identify 100% exact duplicate rows (safe 1-click deduplication); provide an optional user selector for Business Key columns (e.g., Customer_ID); if duplicate keys have conflicting attributes, flag them as "Conflicting Entity Records" and offer resolution rules (Keep First / Keep Last / Manual Review).
- Q: How should DataGuard AI classify, score, and handle numerical outliers detected via IQR / Z-score? → A: Advisory Anomaly with Triage: Classify outliers as "Statistical Anomalies (Advisory)" rather than errors; show statistical bounds (IQR Q1/Q3 thresholds); provide a 1-click triage ("Mark Legitimate" to whitelist without score penalty, or "Cap/Filter"); do not automatically deduct validity points unless flagged as invalid by the user.
- Q: What exact payload is permissible to send to the LLM, and how should DataGuard AI guarantee grounded Q&A and handle offline/unconfigured LLM states? → A: Minimized Metadata Payload & Deterministic Fallback: Send ONLY column metadata, inferred types, null/unique counts, summary stats, and max 5 distinct samples (masked/sanitized); ground Q&A strictly against the generated JSON profiling scorecard; if LLM is offline/key missing, gracefully fall back to pre-built deterministic rule templates for explanations.

### Session 2026-11-19 (Multi-Provider LLM Architecture & Automatic Failover)
- Q: How should LLM provider routing and failover be structured? → A: Multi-Provider with Automatic Fallback: Gemini as primary provider; Groq (`openai/gpt-oss-120b`) as secondary provider; deterministic Python templates as ultimate fallback. Attempt Gemini with limited retries for transient errors; fail over to Groq on timeouts, rate limits (HTTP 429), or server errors; fall back to Python templates if all LLMs fail. Clearly display provider attribution badge (`Gemini`, `Groq`, or `Python Template Fallback`) in the UI. Keep all API keys hidden and all deterministic scoring/detection unaffected.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Dataset Ingestion, Dynamic Profiling & Quality Scoring (Priority: P1) 🎯 MVP

As a data analyst or business user, I want to upload a CSV file and immediately view comprehensive structural statistics, column distributions, and a composite quality score so that I can gauge the overall health of my data at a glance without writing validation scripts.

**Why this priority**: Ingestion, dynamic schema profiling, and baseline quality scoring form the fundamental prerequisite for any data quality workflow. Implementing this story alone provides immediate business value as an automated data health dashboard.

**Independent Test**: Can be verified by uploading a 10,000-row CSV with diverse column types and verifying that the summary statistics, column data types, missing/unique value counts, and overall quality score render accurately within 5 seconds.

**Acceptance Scenarios**:

1. **Given** a user navigates to the application, **When** they upload a valid CSV file (up to ~10,000 rows), **Then** the application displays file metadata (name, total rows, total columns), inferred column types, missing/unique percentages, numerical distributions, and an overall dataset quality score (0–100) alongside individual column scores.
2. **Given** a user uploads an invalid, empty, or non-CSV file, **When** the file is processed, **Then** the application rejects the upload gracefully with a clear explanatory error message and leaves the system in a clean state.
3. **Given** a dataset where certain columns lack applicable dimensions (e.g., text columns have no numerical validity checks), **When** the quality scores are calculated, **Then** non-applicable dimensions are excluded from penalizing those columns, and clear explanations for score deductions are displayed.

---

### User Story 2 - Deterministic Multi-Category Issue & Anomaly Detection (Priority: P1) 🎯 MVP

As a data engineer or quality analyst, I want the system to deterministically detect data quality problems (missing values, exact/key duplicates, statistical outliers, casing anomalies, and formatting variations) with sample examples and exact counts so that I can inspect precise data defects without false positives.

**Why this priority**: Automated, accurate detection is the core analytical capability. Relying on deterministic computation guarantees that issue counts and affected row references are 100% verifiable and reproducible.

**Independent Test**: Can be verified by running detection on a benchmark CSV seeded with known errors (e.g., 50 nulls, 10 duplicate rows, 15 IQR outliers, and mixed-case city names) and asserting that the exact issue counts and sample records match ground truth.

**Acceptance Scenarios**:

1. **Given** a column with nulls, empty strings, and whitespace-only entries, **When** missing-value detection runs, **Then** all variations are flagged with exact counts, percentages, and representative sample row records.
2. **Given** a dataset with duplicate records, **When** duplicate detection runs, **Then** exact duplicate rows are identified; if business-key columns are designated by the user, duplicate keys are flagged without incorrectly flagging repeated categories in ordinary dimension columns.
3. **Given** a numerical column with extreme values, **When** outlier detection runs using IQR (and optional Z-score), **Then** statistically unusual values are isolated with their calculated boundary thresholds, without assuming all outliers are necessarily erroneous.
4. **Given** categorical columns with mixed casing, padding spaces, or minor spelling variations (e.g., "Bengaluru", "Bangalore", "bengaluru "), **When** inconsistency detection runs, **Then** the system distinguishes casing differences, whitespace padding, known aliases, and potential spelling mistakes.

---

### User Story 3 - AI Explanations, Column Semantic Understanding & Interactive Q&A (Priority: P2)

As a business user or analyst, I want an AI assistant to explain why detected issues matter, suggest semantic interpretations for unfamiliar column names, and answer questions about my dataset's health so that I can understand the business impact without consulting a data engineer.

**Why this priority**: Raw anomaly numbers can be overwhelming for non-technical stakeholders. AI adds semantic intelligence, translating numbers into actionable business context and remediation advice.

**Independent Test**: Can be verified by inspecting an anomaly report for an unfamiliar column (e.g., `TXN_LOC_CD`) and verifying that the AI suggests likely semantics, provides a plain-English explanation of why inconsistent values impact reporting, and answers grounded questions (e.g., "Which column has the lowest score?") correctly using only analysis metadata.

**Acceptance Scenarios**:

1. **Given** an unfamiliar or abbreviated column header, **When** AI semantic understanding is requested, **Then** the assistant evaluates column statistics, inferred data types, and limited distinct samples to suggest potential real-world meanings and candidate validation rules while clearly flagging uncertain interpretations.
2. **Given** a set of detected anomalies, **When** the user views the issue details, **Then** the AI assistant provides a plain-English rationale for why the issue matters, recommended remediation strategies, and illustrative Python/Pandas transformation snippets.
3. **Given** a user submits a natural-language question in the chat interface (e.g., "Why is my completeness score low?" or "Which cities have inconsistent values?"), **When** the assistant responds, **Then** the answer is strictly grounded in the actual profiling metrics and detected issues without hallucinating nonexistent columns or numbers.
4. **Given** Gemini is operational and configured as primary, **When** AI explanations or Q&A responses are generated, **Then** the UI displays the attribution badge `Gemini` alongside the response.
5. **Given** Gemini encounters transient network errors, rate limits (HTTP 429), timeouts, or model unavailability, **When** the retry threshold is reached, **Then** the system automatically and transparently switches to Groq (`openai/gpt-oss-120b`) and displays the attribution badge `Groq`.
6. **Given** both Gemini and Groq encounter authentication errors, missing keys, rate limits, or service outages, **When** issue explanations are requested, **Then** the application automatically substitutes deterministic Python rule templates, displays the attribution badge `Python Template Fallback`, and maintains uninterrupted operation of all ingestion, profiling, scoring, and cleaning workflows.
7. **Given** any interaction with external LLM providers, **When** configuration is loaded and requests are logged, **Then** API keys are strictly scrubbed and never rendered in UI elements, error banners, or log files.

---

### User Story 4 - Smart Standardization & Human-Approved Cleaning with Live Preview (Priority: P2)

As a data analyst or BI developer, I want to review recommended cleaning transformations, inspect side-by-side before-and-after previews, and explicitly approve changes so that I can obtain a clean dataset without risking unintended data loss or corruption.

**Why this priority**: Data cleaning must be safe and auditable. Providing before-and-after previews with mandatory user confirmation protects data integrity and guarantees human oversight.

**Independent Test**: Can be verified by proposing a standardization on city names and whitespace trimming, viewing the before-and-after preview table, approving the change, and confirming that the cleaned dataset reflects the fix while the original dataset remains unaltered.

**Acceptance Scenarios**:

1. **Given** recommended cleaning actions (e.g., trim whitespace, standardize casing, impute missing values, map known aliases), **When** the user selects an action, **Then** the system displays a side-by-side preview showing affected sample rows before and after the proposed change.
2. **Given** an uncertain value mapping or fuzzy match suggestion, **When** presented to the user, **Then** the system requires explicit user confirmation before applying the mapping and never executes unverified LLM-generated transformation code.
3. **Given** user approval of selected cleaning operations, **When** the transformations are applied, **Then** operations execute strictly on a separate copy of the data, leaving the original uploaded dataset completely intact.

---

### User Story 5 - Comprehensive Audit Reporting & Clean Data Export (Priority: P3)

As a data engineer or BI team member, I want to download an executive quality audit report and export the cleaned dataset so that I can share data quality evidence with stakeholders and feed clean data into downstream analytics pipelines.

**Why this priority**: Downstream teams need portable documentation of data health and verified clean data to maintain trust in dashboards and machine learning models.

**Independent Test**: Can be verified by approving a set of cleaning rules and downloading both the cleaned CSV file and the Markdown/HTML quality report, then validating that the report contains full provenance, scores, issue counts, and applied transformations.

**Acceptance Scenarios**:

1. **Given** an analyzed dataset, **When** the user requests a quality report, **Then** the system generates a structured, downloadable report (Markdown/HTML) detailing dataset summary, overall and column-wise quality scores, detected issues with frequencies and examples, and AI recommendations.
2. **Given** that one or more cleaning operations have been approved and applied, **When** the user clicks export, **Then** the system downloads the cleaned CSV file with all approved adjustments applied.

---

### Edge Cases

- **Empty or Header-Only CSV**: The system rejects 0-row or header-only files with an informative notification explaining that at least one data row is required.
- **Corrupt, Malformed, or Truncated CSV**: The system catches delimiter mismatches, uneven row lengths, and encoding issues, presenting clear diagnostic guidance instead of an unhandled crash.
- **Uniform Single-Value Columns**: Columns where all rows share an identical value are handled gracefully (e.g., zero variance in numerical columns does not trigger division-by-zero errors in outlier or scoring logic).
- **All-Null or High-Cardinality Columns**: Columns with 100% null values or unique identifiers across every row are handled without crashing scoring algorithms.
- **Conflicting Values in Standardization**: When two similar strings might represent genuinely different entities (e.g., "Springfield, IL" vs. "Springfield, MO", or "Cat" vs. "Cot"), the system requires explicit human confirmation and does not merge them automatically.
- **LLM Service Unavailability & Automatic Multi-Provider Failover**: If the primary LLM (Gemini) encounters transient network drops, rate limits (HTTP 429), timeouts, model unavailability, or authentication errors, the system retries transient issues (up to 2 attempts) and immediately fails over to the secondary provider (Groq: `openai/gpt-oss-120b`). If both providers fail or keys are unconfigured, the system automatically falls back to deterministic Python explanation templates without interrupting ingestion, profiling, scoring, or cleaning workflows. At no point may API keys be exposed in logs or user interfaces.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Ingestion & Profiling
- **FR-001**: The system MUST support user upload of CSV files up to approximately 10,000 rows and validate file format, readability, and structural encoding.
- **FR-002**: The system MUST dynamically profile any valid tabular CSV without requiring a predefined schema, extracting row count, column count, inferred data types, missing value counts/percentages, unique value counts, and numerical/categorical distribution summaries.
- **FR-003**: The system MUST preserve the original uploaded CSV dataset in an immutable state, executing all cleaning operations strictly on an isolated working copy.

#### Issue & Anomaly Detection
- **FR-004**: The system MUST detect missing values including standard nulls, empty strings (`""`), and whitespace-only entries (`"   "`), reporting exact affected columns, counts, and sample records.
- **FR-005**: The system MUST implement a Dual-Tier Duplicate Engine: (1) Exact Duplicate Detection: automatically identify full rows that are 100% identical across all columns for safe bulk deduplication; and (2) Business Key Duplicate Detection: provide an interface allowing users to designate one or more business-key columns (e.g., `Customer_ID`, `Transaction_ID`). When duplicate business keys contain conflicting attributes across other columns, the system MUST flag them as "Conflicting Entity Records" and require the user to choose an explicit resolution policy (Keep First, Keep Last, or Manual Review) rather than silently dropping rows.
- **FR-006**: The system MUST implement an Outlier Triage Engine using IQR and optional Z-score: (1) Classification: detected outliers MUST be categorized as "Statistical Anomalies (Advisory)" with explicit upper/lower boundary thresholds displayed alongside affected values; (2) Non-Penalizing Default: statistical outliers MUST NOT automatically deduct validity score points unless confirmed as erroneous by the user; and (3) Triage Actions: the interface MUST provide user controls to "Mark Legitimate" (whitelisting values without score penalty) or apply user-approved capping/filtering transformations.
- **FR-007**: The system MUST implement a Tiered Value Standardization Engine that categorizes text inconsistencies: (1) Cosmetic Fixes: whitespace trimming and casing normalization; (2) Known Aliases: dictionary/reference mappings classified as High Confidence (≥90%); (3) Spelling Variations: fuzzy string similarity matching classified as Medium Confidence (80–89%); and (4) Distinct Values: similarity <80% retained as separate entities. All suggested merges MUST require explicit user review and approval before execution.

#### AI Explanations & Semantic Insights
- **FR-008**: The system MUST employ an AI assistant to infer semantic meanings and propose candidate validation rules for unfamiliar or ambiguous column names using only inferred types, statistical distributions, and at most 5 distinct sanitized sample values, clearly distinguishing confident inferences from uncertain interpretations.
- **FR-009**: The system MUST generate plain-language explanations for all detected issues detailing why each problem matters to business reporting, suggested remediation strategies, and reproducible transformation logic.
- **FR-010**: The system MUST provide an interactive Q&A assistant capable of answering user inquiries regarding dataset quality, grounded strictly against the computed JSON profiling results and detected issues without fabricating uncomputed findings or hallucinating non-existent columns.
- **FR-011**: The system MUST enforce strict data privacy by transmitting ONLY column metadata, aggregated statistics, and a maximum of 5 sanitized sample values per column to external LLM services, never transmitting raw full CSV files or unmasked PII. API keys and secret credentials MUST NEVER be exposed in logs, error traces, or UI elements.
- **FR-017**: The system MUST support a pluggable Multi-Provider LLM Architecture implementing a shared `LLMProviderProtocol`, configuring Gemini as the primary provider and Groq (initially configured with `openai/gpt-oss-120b`, subject to account availability) as the secondary provider. Provider credentials, API endpoints, and model identifiers MUST be configurable via environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL`, `GROQ_API_KEY`, `GROQ_MODEL`).
- **FR-018**: The system MUST execute an Automatic Failover Engine for all AI requests: (1) Primary execution via Gemini with up to 2 retries for transient connection errors; (2) Immediate, transparent failover to Groq upon persistent timeouts, rate limit throttling (HTTP 429), model unavailability, or authentication errors (HTTP 401/403); and (3) Automatic fallback to deterministic Python rule templates if both LLM providers fail or credentials are not configured. The core application (upload, profiling, scoring, detection, cleaning, and reporting) MUST continue operating deterministically without interruption during any LLM failure.
- **FR-019**: The system MUST provide Provider Attribution & Transparency by visibly displaying which provider generated each explanation, recommendation, or Q&A response using an explicit badge or tag (`Gemini`, `Groq`, or `Python Template Fallback`).

#### Quality Scoring
- **FR-012**: The system MUST compute a deterministic quality score (0–100) for each column and an aggregate dataset quality score using documented dimensional weights: Completeness (35%), Validity (30%), Consistency (25%), and Uniqueness (10%).
- **FR-013**: The scoring engine MUST dynamically adjust for dimensional applicability using proportional re-weighting: whenever a dimension is non-applicable to a specific column (e.g., Uniqueness for non-identifier columns, Numerical Validity for free text), that dimension MUST be excluded and the remaining active dimensions normalized to 100%. Dataset-level Uniqueness (10%) MUST score exact duplicate rows or user-selected business keys. Clear point-deduction explanations MUST be generated for every score reduction.

#### Preview, Human Approval & Export
- **FR-014**: The system MUST present an interactive before-and-after preview for all recommended cleaning and standardization operations, requiring explicit user confirmation before applying any changes.
- **FR-015**: The system MUST generate and export a comprehensive quality report in downloadable Markdown and HTML formats summarizing dataset metrics, scores, detected issues, and AI recommendations.
- **FR-016**: The system MUST allow users to download the cleaned dataset as a CSV file once user-approved transformations have been executed.

---

### Key Entities *(include if feature involves data)*

- **Dataset Profile**: Represents the overall structural metadata of the uploaded file, including filename, file size, total row count, column count, and ingestion timestamp.
- **Column Profile**: Structural analysis of a specific column, including inferred data type (numerical, categorical, datetime, text, boolean), null count, null percentage, unique count, and statistical summary (mean, median, min, max, std dev, or top categorical frequencies).
- **Quality Issue / Anomaly**: A specific data defect instance containing issue category (Missing, Duplicate, Outlier, Inconsistency, Format), affected column(s), severity level, exact occurrence count, percentage of dataset affected, representative sample records, and detection rule metadata.
- **Quality Scorecard**: Comprehensive numerical health assessment containing aggregate score (0–100), dimensional sub-scores (Completeness, Validity, Consistency, Uniqueness with dynamic proportional re-weighting), column-level score breakdowns, and human-readable deduction reasons.
- **AI Explanation & Recommendation**: Contextual insight generated for an issue or column containing inferred semantic meaning, business risk explanation, suggested remediation action, confidence score, source provider attribution (`Gemini`, `Groq`, or `Python Template Fallback`), and model name.
- **Cleaning Proposal & Action**: Candidate transformation containing operation type (Impute, Trim, Standardize Case, Map Alias, Drop Duplicate, Filter), targeted rows/columns, confidence tier (Cosmetic, High ≥90%, Medium 80–89%), before-and-after data diffs, approval status (Pending, Approved, Rejected), and execution timestamp.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete the full journey from CSV upload to viewing profiling statistics, detected issues, and quality scores in under 5 seconds for datasets up to 10,000 rows.
- **SC-002**: 100% of issue counts, anomaly frequencies, and quality scores are verifiable, deterministic, and identical across repeated runs on the same input dataset.
- **SC-003**: 0% of full CSV datasets or sensitive unmasked records are transmitted across external network boundaries or LLM APIs during semantic inference.
- **SC-004**: 100% of data transformations and cleaning actions require explicit human approval prior to application; 0 unauthorized data mutations occur.
- **SC-005**: Users can review a before-and-after preview and successfully download both a cleaned CSV and a standalone quality report in under 3 clicks following analysis.
- **SC-006**: 90% of business users evaluated can identify their dataset's primary quality defects and understand the remediation steps without requiring technical consultation.
- **SC-007**: 100% of LLM service exceptions (timeouts, rate limits HTTP 429, authentication errors, model unavailability) resolve via automatic failover to the secondary provider or Python template fallback within 3 seconds without raising unhandled exceptions or crashing the user session.
- **SC-008**: 100% of AI-generated explanations and interactive answers display clear provider source attribution (`Gemini`, `Groq`, or `Python Template Fallback`) in the user interface.

---

## Assumptions

- **Target Audience**: Users have standard CSV datasets exported from business tools, databases, or CRM systems and understand their basic business domain even if they lack programming expertise.
- **Execution Environment**: The application runs in a web browser interface with a local or hosted Python execution backend capable of processing in-memory tabular data.
- **Dataset Boundaries**: Initial release scope is focused on single CSV files up to approximately 10,000 rows; multi-file joins, streaming data, and specialized binary formats (e.g., Parquet, Excel) are deferred to future milestones.
- **LLM Connectivity**: An external LLM provider API (or local Ollama/compatible endpoint) is accessible via standard environment variables (`.env`); if unavailable, the application gracefully degrades to deterministic rule-based explanations without failing.
- **Human Accountability**: The ultimate decision to standardize or merge ambiguous categorical aliases rests with the human user through explicit approval workflows.
