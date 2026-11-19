<!-- SYNC IMPACT REPORT
Version change: Template [0.0.0] → 1.0.0
Modified principles:
  - Initialized 14 non-negotiable engineering principles for DataGuard AI from user requirements.
Added sections:
  - Core Principles (14 principles: Modular Architecture, Separation of Concerns, Python-First Deterministic Detection, AI as an Assistant, Human Approval for Data Mutations, Data Preservation, Comprehensive Explainability, Testability & Unit Testing Discipline, Security & Credential Hygiene, Privacy & Data Minimization, Graceful Reliability & Input Resilience, Performance & Scalability Baseline, Maintainability & Code Quality, Hackathon Scope & Pragmatic Simplicity)
  - Technical & Operational Constraints
  - Development Workflow & Quality Gates
  - Governance
Removed sections:
  - Replaced all template placeholder tokens with concrete project rules.
Templates requiring updates:
  - .specify/templates/plan-template.md (✅ aligned with DataGuard AI principles)
  - .specify/templates/spec-template.md (✅ aligned with user scenarios and quality requirements)
  - .specify/templates/tasks-template.md (✅ aligned with modular and testing requirements)
Follow-up TODOs:
  - None; all principles and governance fields ratified.
-->

# DataGuard AI Constitution

## Core Principles

### I. Modular Architecture
Every major responsibility MUST be implemented in a dedicated, isolated module with clear and well-defined programmatic interfaces. Modules MUST interact through explicit contracts and function signatures rather than tightly coupled internal state.
*Rationale*: High cohesion and loose coupling allow independent development, targeted testing, and frictionless evolution of individual components.

### II. Separation of Concerns
Data profiling, issue detection, quality scoring, AI explanations, data cleaning, and reporting MUST remain strictly separated across distinct subsystems. No module may cross these functional boundaries (e.g., detection modules must not perform cleaning, and reporting modules must not compute quality scores).
*Rationale*: Mixing profiling, detection, scoring, and cleaning creates fragile pipelines, obscures data provenance, and severely complicates debugging.

### III. Python-First Deterministic Detection (NON-NEGOTIABLE)
Data profiling, anomaly detection, rule validation, and metrics calculation MUST be performed deterministically using Python, Pandas, and NumPy. The system MUST NEVER rely on an LLM to calculate actual issue counts, numerical distributions, or quality scores.
*Rationale*: LLMs are non-deterministic, prone to numerical hallucinations, and unsuited for computational verification. Critical data quality assessments demand 100% reproducible and verifiable mathematical accuracy.

### IV. AI as an Assistant
The LLM MUST act strictly as an intelligent assistant and advisor. The LLM SHOULD infer semantic column meanings, explain detected anomalies in clear human language, suggest potential remediation strategies, and recommend candidate validation or cleaning rules. The LLM MUST NOT replace deterministic rule validation or calculation engines.
*Rationale*: Combining generative AI for semantic reasoning and contextual narratives with deterministic Python code for numerical execution maximizes both intelligibility and accuracy.

### V. Human Approval for Data Mutations (NON-NEGOTIABLE)
The system MUST NEVER automatically modify, overwrite, drop, or delete user data without explicit user review and confirmation. All data cleaning and transformation operations MUST require explicit human approval prior to application.
*Rationale*: Automated destructive data manipulation risks permanent loss of critical records and undermines user trust in automated systems.

### VI. Data Preservation
The system MUST ALWAYS preserve the original uploaded dataset in an immutable state. All cleaning, transformation, imputation, and filtering operations MUST execute exclusively on an isolated, separate copy of the dataset.
*Rationale*: Non-destructive processing ensures full auditability, allowing users to compare original versus cleaned data side-by-side and revert changes at any time.

### VII. Comprehensive Explainability
Every detected issue, anomaly, or warning MUST provide a complete explanatory record containing: issue category, affected column(s), exact frequency count, representative sample examples, and a clear, human-understandable explanation of why the anomaly was flagged.
*Rationale*: Data quality findings are only actionable when domain users understand precisely what went wrong, where it occurred, and why it matters.

### VIII. Testability & Unit Testing Discipline
Every detection module, profiling routine, cleaning action, and quality scoring function MUST have automated unit tests covering nominal scenarios, edge cases, malformed data, and empty inputs.
*Rationale*: Rigorous automated testing of deterministic algorithms ensures stability, prevents regressions, and guarantees that quality scores remain trustworthy over time.

### IX. Security & Credential Hygiene
API keys, access tokens, and credentials MUST NEVER be hardcoded in source code, configuration files committed to source control, or test suites. All secrets MUST be loaded via environment variables and local `.env` files. The `.env` file MUST be explicitly listed in `.gitignore`.
*Rationale*: Credential leaks represent severe operational and security liabilities; strict environment isolation prevents accidental exposure.

### X. Privacy & Data Minimization
The complete CSV dataset MUST NEVER be transmitted to an external LLM or remote service. When invoking LLM capabilities, the system MUST transmit ONLY the minimum necessary metadata, column data types, statistical summaries, and limited, sanitized samples.
*Rationale*: Strict data minimization prevents data leakage, respects privacy boundaries, and significantly reduces API token usage and latency.

### XI. Graceful Reliability & Input Resilience
The application MUST handle missing columns, mixed data types, empty files, malformed CSV structures, corrupt rows, and unexpected schema anomalies gracefully. The system MUST present clear, actionable error notifications rather than crashing or throwing unhandled exceptions.
*Rationale*: Real-world tabular data is inherently messy; a resilient application maintains uptime and guides users through unexpected format problems.

### XII. Performance & Scalability Baseline
The initial application MUST smoothly load, profile, detect anomalies in, and score CSV files containing up to approximately 10,000 rows without memory crashes or UI freezing.
*Rationale*: Establishing a clear baseline scope ensures performant local execution and immediate responsiveness for hackathon delivery.

### XIII. Maintainability & Code Quality
Code MUST adhere to clean code practices: meaningful naming conventions, explicit Python type hints, reusable single-responsibility functions, and concise, clear docstrings for all public interfaces.
*Rationale*: Readable, self-documenting code accelerates feature iteration, simplifies code reviews, and eases maintenance.

### XIV. Hackathon Scope & Pragmatic Simplicity
The team MUST prioritize a complete, working end-to-end user journey over unnecessary architectural complexity or speculative abstractions. Delivering a seamless flow—from CSV upload through profiling, anomaly detection, scoring, AI explanation, approved cleaning, and reporting—takes precedence over over-engineering.
*Rationale*: Under hackathon time constraints, an operational, demonstrable product delivers vastly more value than complex, half-finished infrastructure.

## Technical & Operational Constraints

- **Language & Core Libraries**: Python 3.10+ using Pandas and NumPy for deterministic data processing.
- **LLM Integration**: Structured prompt templates passing minimized metadata and summary statistics with built-in fallbacks if LLM calls fail or keys are absent.
- **Data State Management**: In-memory dataset handling maintaining strict separation between immutable raw inputs and mutable working copies.
- **Environment & Configuration**: `.env` configuration managed via standard environment variable loaders, secured from git commits.

## Development Workflow & Quality Gates

1. **Gate 1 - Architecture & Separation**: Verify that profiling, detection, scoring, explanation, cleaning, and reporting reside in dedicated modules.
2. **Gate 2 - Determinism Audit**: Ensure no calculations, scores, or counts originate from LLM outputs.
3. **Gate 3 - Privacy & Security Check**: Confirm `.env` is git-ignored, API keys are externalized, and no raw full CSVs are sent over LLM APIs.
4. **Gate 4 - Mutation Safety**: Verify that cleaning operations require user confirmation and operate on cloned data.
5. **Gate 5 - Test Suite Verification**: All unit tests for detection modules and scoring calculations must pass before merging.

## Governance

- **Authority**: This constitution is the supreme architectural authority for DataGuard AI. It supersedes informal agreements, ad-hoc practices, and conflicting requirements.
- **Amendment Procedure**: Amendments to this constitution must be proposed explicitly with clear rationale. Any addition, modification, or retirement of principles requires consensus and an update to this document.
- **Versioning Policy**: Semantic versioning MUST be applied to this constitution:
  - **MAJOR**: Structural removals, redefinitions, or backward-incompatible changes to non-negotiable principles.
  - **MINOR**: Additions of new principles, new quality gates, or substantial extensions to existing guidelines.
  - **PATCH**: Wording improvements, clarifications, typo corrections, or non-semantic formatting updates.
- **Compliance Enforcement**: All specifications (`spec.md`), implementation plans (`plan.md`), and task lists (`tasks.md`) must explicitly verify compliance with this constitution during the "Constitution Check" gate.

**Version**: 1.0.0 | **Ratified**: 2026-11-19 | **Last Amended**: 2026-11-19
