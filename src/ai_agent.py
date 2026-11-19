"""
Multi-Provider AI Agent Layer for DataGuard AI.
Orchestrates:
1. Shared LLMProviderProtocol
2. GeminiProvider (Primary via google-genai)
3. GroqProvider (Secondary via groq SDK with openai/gpt-oss-120b)
4. LLMProviderManager (Bounded exponential backoff & automatic transparent failover)
5. TemplateExplanationEngine (100% deterministic Python fallback when all LLMs offline)
6. Privacy Payload Sanitizer (Metadata only + max 5 distinct samples)
"""
import os
import json
import time
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dotenv import load_dotenv

from .models import (
    DatasetProfile,
    QualityIssue,
    QualityScorecard,
    AIExplanation,
    ColumnSemanticInference,
    ProviderSource,
    LLMErrorCategory,
    ProviderStatusReport,
    IssueCategory,
)

# Load local environment configuration
load_dotenv()

# ---------------------------------------------------------------------------
# 1. Error Classification Taxonomy
# ---------------------------------------------------------------------------

class ClassifiedLLMError(Exception):
    def __init__(
        self,
        category: LLMErrorCategory,
        message: str,
        original_exception: Optional[Exception] = None,
        is_transient: bool = False,
        retry_after_seconds: Optional[float] = None
    ):
        super().__init__(message)
        self.category = category
        self.message = message
        self.original_exception = original_exception
        self.is_transient = is_transient
        self.retry_after_seconds = retry_after_seconds

def classify_llm_error(exc: Exception) -> ClassifiedLLMError:
    """
    Inspects exception types and messages to classify them into the error taxonomy.
    """
    msg = str(exc).lower()

    # Authentication / Key errors (Non-transient)
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg or "api key not valid" in msg or "invalid api key" in msg:
        return ClassifiedLLMError(LLMErrorCategory.AUTHENTICATION, "Authentication failed. Invalid or missing API key.", exc, is_transient=False)

    # Model not found / unavailable (Non-transient)
    if "404" in msg or "not found" in msg or "model not found" in msg or "deprecated" in msg:
        return ClassifiedLLMError(LLMErrorCategory.MODEL_UNAVAILABLE, f"Model unavailable: {str(exc)}", exc, is_transient=False)

    # Rate limiting / Quota exceeded (Transient)
    if "429" in msg or "rate limit" in msg or "quota exceeded" in msg or "too many requests" in msg:
        return ClassifiedLLMError(LLMErrorCategory.RATE_LIMIT, "Rate limit reached (HTTP 429).", exc, is_transient=True, retry_after_seconds=2.0)

    # Timeouts (Transient)
    if "timeout" in msg or "timed out" in msg or "deadline exceeded" in msg:
        return ClassifiedLLMError(LLMErrorCategory.TIMEOUT, "Request timed out.", exc, is_transient=True)

    # Server errors (Transient)
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg or "internal server error" in msg or "service unavailable" in msg:
        return ClassifiedLLMError(LLMErrorCategory.SERVER_ERROR, "Remote provider server error.", exc, is_transient=True)

    return ClassifiedLLMError(LLMErrorCategory.UNKNOWN, f"LLM error: {str(exc)}", exc, is_transient=False)

# ---------------------------------------------------------------------------
# 2. Shared Protocol & Provider Implementations
# ---------------------------------------------------------------------------

class LLMProviderProtocol:
    name: str
    model_name: str
    is_configured: bool

    def generate_json(self, prompt: str, schema_instruction: str) -> dict: ...
    def answer_query(self, system_prompt: str, user_query: str) -> str: ...

class GeminiProvider:
    """
    Primary LLM provider using the google-genai SDK.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        self.name = "Gemini"
        self.is_configured = bool(self.api_key)
        self._client = None

        if self.is_configured:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self.is_configured = False

    def generate_json(self, prompt: str, schema_instruction: str) -> dict:
        if not self.is_configured or not self._client:
            raise ClassifiedLLMError(LLMErrorCategory.AUTHENTICATION, "Gemini API key is not configured.", is_transient=False)
        try:
            full_prompt = f"{prompt}\n\nOutput ONLY a valid JSON object matching this schema:\n{schema_instruction}"
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            raw_text = response.text.strip()
            # Clean possible markdown wrapping
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return json.loads(raw_text.strip())
        except Exception as e:
            raise classify_llm_error(e)

    def answer_query(self, system_prompt: str, user_query: str) -> str:
        if not self.is_configured or not self._client:
            raise ClassifiedLLMError(LLMErrorCategory.AUTHENTICATION, "Gemini API key is not configured.", is_transient=False)
        try:
            full_prompt = f"System Instructions: {system_prompt}\n\nUser Question: {user_query}"
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            return response.text.strip()
        except Exception as e:
            raise classify_llm_error(e)

class GroqProvider:
    """
    Secondary LLM provider using the official Groq SDK.
    Configured with openai/gpt-oss-120b by default.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        self.name = "Groq"
        self.is_configured = bool(self.api_key)
        self._client = None

        if self.is_configured:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                self.is_configured = False

    def generate_json(self, prompt: str, schema_instruction: str) -> dict:
        if not self.is_configured or not self._client:
            raise ClassifiedLLMError(LLMErrorCategory.AUTHENTICATION, "Groq API key is not configured.", is_transient=False)
        try:
            messages = [
                {"role": "system", "content": f"You are a helpful DataGuard AI quality assistant. Output strictly valid JSON matching: {schema_instruction}"},
                {"role": "user", "content": prompt}
            ]
            chat_completion = self._client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                response_format={"type": "json_object"}
            )
            raw_text = chat_completion.choices[0].message.content.strip()
            return json.loads(raw_text)
        except Exception as e:
            raise classify_llm_error(e)

    def answer_query(self, system_prompt: str, user_query: str) -> str:
        if not self.is_configured or not self._client:
            raise ClassifiedLLMError(LLMErrorCategory.AUTHENTICATION, "Groq API key is not configured.", is_transient=False)
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            chat_completion = self._client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            raise classify_llm_error(e)

# ---------------------------------------------------------------------------
# 3. Deterministic Python Template Fallback Engine
# ---------------------------------------------------------------------------

class TemplateExplanationEngine:
    """
    100% deterministic rule templates when all cloud LLMs are offline.
    """
    @staticmethod
    def explain_issue(issue: QualityIssue, profile: DatasetProfile) -> AIExplanation:
        category = issue.category
        col = issue.column_name or "Dataset"
        affected = issue.affected_count

        if category == IssueCategory.MISSING_VALUE:
            summary = f"Column '{col}' has {affected} blank or null values."
            impact = "Missing data skews analytical aggregates and causes errors in downstream calculations or ML models."
            fix = "Apply median/mode imputation or drop incomplete records if missingness is isolated."
            code = f"# Impute or drop missing values in '{col}'\ndf['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])"
        elif category == IssueCategory.EXACT_DUPLICATE_ROW:
            summary = f"Found {affected} exact duplicate rows identical across all columns."
            impact = "Duplicate rows artificially inflate record counts and distort volume/revenue metrics."
            fix = "Safely deduplicate by retaining the first occurrence."
            code = "df = df.drop_duplicates(keep='first')"
        elif category == IssueCategory.BUSINESS_KEY_DUPLICATE:
            summary = f"Found {affected} conflicting duplicate records on business key '{col}'."
            impact = "Indicates uncoordinated updates or inconsistent transaction state across records."
            fix = "Apply business deduplication policy (e.g. Keep First or Keep Latest)."
            code = f"df = df.drop_duplicates(subset=['{col}'], keep='first')"
        elif category == IssueCategory.NUMERIC_OUTLIER:
            summary = f"Column '{col}' has {affected} values outside statistical IQR bounds."
            impact = "Outliers may be legitimate extreme events or sensor/entry errors; they significantly distort mean and variance."
            fix = "Review distribution; cap extreme values to 1.5*IQR bounds if confirmed erroneous."
            code = f"# Cap outliers to bounds\nq1 = df['{col}'].quantile(0.25)\nq3 = df['{col}'].quantile(0.75)\niqr = q3 - q1\ndf['{col}'] = df['{col}'].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)"
        elif category in (IssueCategory.CASING_INCONSISTENCY, IssueCategory.WHITESPACE_INCONSISTENCY):
            summary = f"Column '{col}' has {affected} entries with casing or whitespace inconsistencies."
            impact = "Creates fragmented groupings in BI dashboards (e.g., 'Active' vs 'active')."
            fix = "Normalize casing to Title Case and trim padding spaces."
            code = f"df['{col}'] = df['{col}'].astype(str).str.strip().str.title()"
        elif category in (IssueCategory.ALIAS_INCONSISTENCY, IssueCategory.SPELLING_INCONSISTENCY):
            summary = f"Column '{col}' contains spelling variations or known synonyms."
            impact = "Causes split categorical grouping (e.g., 'Bengaluru' vs 'Bangalore')."
            fix = "Standardize values using approved canonical alias mappings."
            code = f"df['{col}'] = df['{col}'].replace({{'Bangalore': 'Bengaluru', 'Banglore': 'Bengaluru'}})"
        else:
            summary = f"Detected issue in '{col}': {issue.explanation}"
            impact = "Reduces data reliability."
            fix = "Review sample records and standardize."
            code = f"# Inspect samples\nprint(df['{col}'].unique()[:10])"

        return AIExplanation(
            issue_id=issue.issue_id,
            plain_english_summary=summary,
            business_impact=impact,
            recommended_fix=fix,
            suggested_python_code=code,
            provider_source=ProviderSource.PYTHON_TEMPLATE,
            model_name="deterministic-rule-template",
            is_fallback=True
        )

    @staticmethod
    def infer_column(col_name: str, profile: DatasetProfile) -> ColumnSemanticInference:
        col_prof = profile.columns.get(col_name)
        col_lower = col_name.lower().strip()

        if any(term in col_lower for term in ["id", "key", "code", "num"]):
            meaning = "Unique business entity identifier or transaction key."
            rules = ["Must be unique", "Cannot contain nulls"]
        elif any(term in col_lower for term in ["date", "time", "created", "updated"]):
            meaning = "Temporal timestamp or event date."
            rules = ["Must follow ISO-8601 or YYYY-MM-DD", "Cannot be in future"]
        elif any(term in col_lower for term in ["price", "amount", "salary", "cost", "revenue", "fee"]):
            meaning = "Financial or monetary numerical measure."
            rules = ["Must be non-negative (>= 0)", "Decimal precision max 2 places"]
        elif any(term in col_lower for term in ["status", "state", "type", "category", "flag"]):
            meaning = "Categorical state or business workflow classification."
            rules = ["Must match controlled vocabulary"]
        elif any(term in col_lower for term in ["city", "country", "state", "address", "loc"]):
            meaning = "Geographic location or address attribute."
            rules = ["Standardize casing and canonical spellings"]
        else:
            meaning = f"Tabular attribute inferred as {col_prof.inferred_type.value if col_prof else 'general attribute'}."
            rules = ["Values should match column data type"]

        return ColumnSemanticInference(
            column_name=col_name,
            suggested_meaning=meaning,
            confidence=85.0,
            candidate_rules=rules,
            is_uncertain=False
        )

    @staticmethod
    def answer_query(query: str, scorecard: QualityScorecard, issues: List[QualityIssue]) -> str:
        q_lower = query.lower()
        if "score" in q_lower or "overall" in q_lower or "quality" in q_lower:
            return (
                f"Your overall dataset quality score is **{scorecard.overall_score}/100**.\n\n"
                f"**Score deductions**:\n" +
                ("\n".join(f"- {d}" for d in scorecard.deduction_reasons[:4]) if scorecard.deduction_reasons else "- No deductions!")
            )
        elif "worst" in q_lower or "most" in q_lower or "lowest" in q_lower:
            if scorecard.column_scores:
                lowest_col = min(scorecard.column_scores.items(), key=lambda x: x[1])
                return f"The column with the lowest score is **'{lowest_col[0]}'** with a score of **{lowest_col[1]}/100**."
            return "All columns have consistent scores."
        elif "issue" in q_lower or "anomaly" in q_lower or "problem" in q_lower:
            return (
                f"A total of **{len(issues)} data quality issue categories** were detected:\n" +
                "\n".join(f"- **{i.category.value}** in '{i.column_name or 'Dataset'}': {i.affected_count} affected records" for i in issues[:5])
            )
        elif "city" in q_lower or "cities" in q_lower:
            city_issues = [i for i in issues if "city" in (i.column_name or "").lower() or i.category == IssueCategory.SPELLING_INCONSISTENCY]
            if city_issues:
                return f"Inconsistent city values detected in '{city_issues[0].column_name}': {city_issues[0].explanation}"
            return "No city inconsistency issues were detected in this dataset."
        else:
            return (
                f"Dataset Health Summary:\n"
                f"- Overall Quality Score: **{scorecard.overall_score}/100**\n"
                f"- Total Detected Issue Types: **{len(issues)}**\n"
                f"Ask specifically about 'scores', 'lowest column', or 'detected issues' for deeper analysis."
            )

# ---------------------------------------------------------------------------
# 4. Multi-Provider Manager with Bounded Backoff & Automatic Failover
# ---------------------------------------------------------------------------

class LLMProviderManager:
    """
    Orchestrates provider selection, retries with backoff, and transparent failover:
    Primary (Gemini) -> Secondary (Groq) -> Deterministic Python Templates.
    """
    def __init__(
        self,
        primary_provider: Optional[LLMProviderProtocol] = None,
        secondary_provider: Optional[LLMProviderProtocol] = None,
        max_retries: int = 2,
        initial_backoff: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff: float = 3.0
    ):
        self.primary = primary_provider if primary_provider is not None else GeminiProvider()
        self.secondary = secondary_provider if secondary_provider is not None else GroqProvider()
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.last_status = ProviderStatusReport(
            active_provider=ProviderSource.GEMINI if self.primary.is_configured else (ProviderSource.GROQ if self.secondary.is_configured else ProviderSource.PYTHON_TEMPLATE),
            active_model=self.primary.model_name if self.primary.is_configured else (self.secondary.model_name if self.secondary.is_configured else "rule-template"),
            gemini_healthy=self.primary.is_configured,
            groq_healthy=self.secondary.is_configured
        )

    def execute_with_failover(
        self,
        call_fn: Callable[[LLMProviderProtocol], Any],
        fallback_fn: Callable[[], Any]
    ) -> Tuple[Any, ProviderSource, str]:
        """
        Executes call_fn with failover: Gemini -> Groq -> fallback_fn.
        Never raises exceptions; guarantees valid return.
        """
        # Step 1: Try Primary (Gemini)
        if self.primary.is_configured:
            for attempt in range(self.max_retries + 1):
                try:
                    result = call_fn(self.primary)
                    self.last_status.active_provider = ProviderSource.GEMINI
                    self.last_status.active_model = self.primary.model_name
                    self.last_status.gemini_healthy = True
                    return result, ProviderSource.GEMINI, self.primary.model_name
                except ClassifiedLLMError as err:
                    if not err.is_transient or attempt >= self.max_retries:
                        self.last_status.gemini_healthy = False
                        self.last_status.last_error_message = err.message
                        self.last_status.last_error_category = err.category
                        break
                    # Bounded exponential backoff with jitter
                    delay = min(self.max_backoff, self.initial_backoff * (self.backoff_factor ** attempt)) + random.uniform(0.0, 0.3)
                    time.sleep(delay)
                except Exception as ex:
                    self.last_status.gemini_healthy = False
                    break

        # Step 2: Try Secondary (Groq)
        if self.secondary.is_configured:
            try:
                result = call_fn(self.secondary)
                self.last_status.active_provider = ProviderSource.GROQ
                self.last_status.active_model = self.secondary.model_name
                self.last_status.groq_healthy = True
                return result, ProviderSource.GROQ, self.secondary.model_name
            except Exception as ex:
                self.last_status.groq_healthy = False
                self.last_status.last_error_message = str(ex)

        # Step 3: Fallback to Deterministic Python Templates
        self.last_status.active_provider = ProviderSource.PYTHON_TEMPLATE
        self.last_status.active_model = "deterministic-rule-template"
        result = fallback_fn()
        return result, ProviderSource.PYTHON_TEMPLATE, "deterministic-rule-template"

    def get_status(self) -> ProviderStatusReport:
        return self.last_status

# ---------------------------------------------------------------------------
# 5. Privacy Payload Sanitizer (Principle X)
# ---------------------------------------------------------------------------

def build_sanitized_metadata_payload(profile: DatasetProfile, issue: Optional[QualityIssue] = None) -> dict:
    """
    Constructs an anonymized, minimal JSON payload.
    TRANSMITS ZERO FULL CSV ROWS.
    Includes only: column names, types, null counts, stats, and max 5 distinct samples.
    """
    cols_summary = {}
    for name, p in profile.columns.items():
        cols_summary[name] = {
            "type": p.inferred_type.value,
            "null_pct": p.null_percentage,
            "unique_count": p.unique_count,
            "distinct_samples": p.statistics.sample_values[:5]
        }
        if p.statistics.min_val is not None:
            cols_summary[name]["min"] = p.statistics.min_val
            cols_summary[name]["max"] = p.statistics.max_val
            cols_summary[name]["mean"] = round(p.statistics.mean_val, 2) if p.statistics.mean_val else None

    payload = {
        "dataset": {
            "filename": profile.metadata.filename,
            "total_rows": profile.metadata.row_count,
            "total_columns": profile.metadata.column_count
        },
        "columns": cols_summary
    }

    if issue:
        payload["target_issue"] = {
            "category": issue.category.value,
            "column": issue.column_name,
            "count": issue.affected_count,
            "pct": issue.affected_percentage,
            "sample_examples": issue.sample_values[:3]
        }

    return payload

# ---------------------------------------------------------------------------
# 6. High-Level AI Functions
# ---------------------------------------------------------------------------

_GLOBAL_MANAGER: Optional[LLMProviderManager] = None

def get_llm_manager() -> LLMProviderManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = LLMProviderManager()
    return _GLOBAL_MANAGER

def explain_single_issue(
    issue: QualityIssue,
    profile: DatasetProfile,
    manager: Optional[LLMProviderManager] = None
) -> AIExplanation:
    mgr = manager or get_llm_manager()
    payload = build_sanitized_metadata_payload(profile, issue)

    schema = '{"summary": "string", "impact": "string", "fix": "string", "python_code": "string"}'
    prompt = f"Explain this data quality issue and suggest actionable remediation based on this metadata:\n{json.dumps(payload)}"

    def call_provider(prov: LLMProviderProtocol):
        data = prov.generate_json(prompt, schema)
        prov_name_lower = str(prov.name).lower()
        if "gemini" in prov_name_lower:
            src = ProviderSource.GEMINI
        elif "groq" in prov_name_lower:
            src = ProviderSource.GROQ
        else:
            src = ProviderSource.PYTHON_TEMPLATE

        return AIExplanation(
            issue_id=issue.issue_id,
            plain_english_summary=data.get("summary", issue.explanation),
            business_impact=data.get("impact", "Impairs data accuracy."),
            recommended_fix=data.get("fix", "Review and standardize."),
            suggested_python_code=data.get("python_code", "# review column\n"),
            provider_source=src,
            model_name=prov.model_name,
            is_fallback=False
        )

    def fallback():
        return TemplateExplanationEngine.explain_issue(issue, profile)

    result, source, model = mgr.execute_with_failover(call_provider, fallback)
    result.provider_source = source
    result.model_name = model
    return result

def explain_issues(
    issues: List[QualityIssue],
    profile: DatasetProfile,
    manager: Optional[LLMProviderManager] = None
) -> Tuple[Dict[str, AIExplanation], ProviderSource, str]:
    explanations: Dict[str, AIExplanation] = {}
    mgr = manager or get_llm_manager()

    for issue in issues[:8]:  # Explain top issues to stay responsive
        exp = explain_single_issue(issue, profile, mgr)
        explanations[issue.issue_id] = exp

    status = mgr.get_status()
    return explanations, status.active_provider, status.active_model

def infer_unknown_columns(
    profile: DatasetProfile,
    manager: Optional[LLMProviderManager] = None
) -> Tuple[List[ColumnSemanticInference], ProviderSource, str]:
    mgr = manager or get_llm_manager()
    payload = build_sanitized_metadata_payload(profile)

    schema = '{"inferences": [{"column": "name", "meaning": "desc", "confidence": 90, "rules": ["rule1"]}]}'
    prompt = f"Infer the real-world semantic meaning of these tabular columns from sanitized metadata:\n{json.dumps(payload)}"

    def call_provider(prov: LLMProviderProtocol):
        data = prov.generate_json(prompt, schema)
        inferences = []
        for item in data.get("inferences", []):
            inferences.append(ColumnSemanticInference(
                column_name=item.get("column", "unknown"),
                suggested_meaning=item.get("meaning", "General attribute"),
                confidence=float(item.get("confidence", 80.0)),
                candidate_rules=item.get("rules", []),
                is_uncertain=float(item.get("confidence", 80.0)) < 75.0
            ))
        return inferences

    def fallback():
        return [TemplateExplanationEngine.infer_column(col, profile) for col in profile.columns.keys()]

    results, source, model = mgr.execute_with_failover(call_provider, fallback)
    return results, source, model

def answer_user_query(
    user_query: str,
    scorecard: QualityScorecard,
    profile: DatasetProfile,
    issues: List[QualityIssue],
    manager: Optional[LLMProviderManager] = None
) -> Tuple[str, ProviderSource, str]:
    mgr = manager or get_llm_manager()

    # Build grounded system context
    system_context = (
        f"You are the DataGuard AI assistant. Ground your answer strictly in these calculated metrics:\n"
        f"- Overall Score: {scorecard.overall_score}/100\n"
        f"- Column Scores: {json.dumps(scorecard.column_scores)}\n"
        f"- Score Deductions: {scorecard.deduction_reasons[:5]}\n"
        f"- Detected Issues: {[{'category': i.category.value, 'col': i.column_name, 'count': i.affected_count} for i in issues[:5]]}\n"
        "NEVER invent numbers, columns, or findings not present in this context."
    )

    def call_provider(prov: LLMProviderProtocol):
        return prov.answer_query(system_context, user_query)

    def fallback():
        return TemplateExplanationEngine.answer_query(user_query, scorecard, issues)

    return mgr.execute_with_failover(call_provider, fallback)
