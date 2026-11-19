"""
Unit and Integration Tests for Multi-Provider AI Agent Layer (src/ai_agent.py).
Validates LLM error taxonomy, Gemini primary execution, Groq failover,
bounded retries, deterministic template fallback, and privacy payload sanitization.
"""
import pytest
from unittest.mock import MagicMock
from src.models import (
    DatasetMetadata,
    DatasetProfile,
    QualityIssue,
    QualityScorecard,
    IssueCategory,
    IssueSeverity,
    CleaningActionType,
    ProviderSource,
    LLMErrorCategory,
)
from src.profiling import generate_dataset_profile
from src.scoring import compute_quality_scorecard
from src.ai_agent import (
    classify_llm_error,
    ClassifiedLLMError,
    LLMProviderProtocol,
    LLMProviderManager,
    TemplateExplanationEngine,
    build_sanitized_metadata_payload,
    explain_single_issue,
    answer_user_query,
)

# -----------------------------------------------------------------------------
# 1. Error Taxonomy Classification Tests
# -----------------------------------------------------------------------------
def test_classify_llm_error_authentication():
    err = classify_llm_error(Exception("401 Unauthorized: Invalid API key"))
    assert err.category == LLMErrorCategory.AUTHENTICATION
    assert err.is_transient is False

def test_classify_llm_error_rate_limit():
    err = classify_llm_error(Exception("429 Too Many Requests - quota exceeded"))
    assert err.category == LLMErrorCategory.RATE_LIMIT
    assert err.is_transient is True

def test_classify_llm_error_model_unavailable():
    err = classify_llm_error(Exception("404 Model openai/gpt-oss-120b not found"))
    assert err.category == LLMErrorCategory.MODEL_UNAVAILABLE
    assert err.is_transient is False

def test_classify_llm_error_timeout():
    err = classify_llm_error(Exception("Deadline exceeded: connection timed out"))
    assert err.category == LLMErrorCategory.TIMEOUT
    assert err.is_transient is True

def test_classify_llm_error_server_error():
    err = classify_llm_error(Exception("503 Service Unavailable: Remote backend crash"))
    assert err.category == LLMErrorCategory.SERVER_ERROR
    assert err.is_transient is True

# -----------------------------------------------------------------------------
# 2. Privacy Payload Sanitizer Tests (Principle X)
# -----------------------------------------------------------------------------
def test_privacy_payload_sanitizer_transmits_zero_full_csv_rows(dirty_dataframe, sample_metadata):
    profile = generate_dataset_profile(dirty_dataframe, sample_metadata)
    issue = QualityIssue(
        issue_id="test_issue",
        category=IssueCategory.MISSING_VALUE,
        column_name="city",
        severity=IssueSeverity.MEDIUM,
        affected_count=10,
        affected_percentage=1.0,
        sample_values=["NYC", "San Fran"],
        explanation="Test explanation",
        suggested_action=CleaningActionType.IMPUTE_MISSING
    )
    payload = build_sanitized_metadata_payload(profile, issue)

    # Assert structural metadata only
    assert "dataset" in payload
    assert payload["dataset"]["total_rows"] == len(dirty_dataframe)
    assert "columns" in payload
    
    # Assert each column summary has at most 5 distinct samples
    for col_name, summary in payload["columns"].items():
        assert len(summary["distinct_samples"]) <= 5
        # Ensure raw full series is not present
        assert "full_data" not in summary

# -----------------------------------------------------------------------------
# 3. Failover Scenarios: Gemini Success, Groq Failover, Template Fallback
# -----------------------------------------------------------------------------
class MockProvider(LLMProviderProtocol):
    def __init__(self, name: str, model_name: str, configured: bool = True):
        self._name = name
        self._model = model_name
        self._configured = configured
        self.call_count = 0
        self.fail_with: list = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_configured(self) -> bool:
        return self._configured

    def generate_json(self, prompt: str, schema: str) -> dict:
        self.call_count += 1
        if self.fail_with:
            err = self.fail_with.pop(0)
            raise err
        return {
            "summary": f"Summary from {self._name}",
            "impact": "Test impact",
            "fix": "Test fix",
            "python_code": "# test code\n"
        }

    def answer_query(self, system_context: str, user_query: str) -> str:
        self.call_count += 1
        if self.fail_with:
            err = self.fail_with.pop(0)
            raise err
        return f"Answer from {self._name} for: {user_query}"

def test_failover_gemini_primary_success(dirty_dataframe, sample_metadata):
    profile = generate_dataset_profile(dirty_dataframe, sample_metadata)
    issue = QualityIssue("i1", IssueCategory.MISSING_VALUE, "city", IssueSeverity.LOW, 5, 0.5, [], "desc", CleaningActionType.IMPUTE_MISSING)

    gemini_mock = MockProvider("gemini", "gemini-2.5-flash", configured=True)
    groq_mock = MockProvider("groq", "openai/gpt-oss-120b", configured=True)

    mgr = LLMProviderManager(primary_provider=gemini_mock, secondary_provider=groq_mock)
    exp = explain_single_issue(issue, profile, manager=mgr)

    assert exp.provider_source == ProviderSource.GEMINI
    assert gemini_mock.call_count == 1
    assert groq_mock.call_count == 0

def test_failover_gemini_429_fails_over_to_groq(dirty_dataframe, sample_metadata):
    profile = generate_dataset_profile(dirty_dataframe, sample_metadata)
    issue = QualityIssue("i1", IssueCategory.MISSING_VALUE, "city", IssueSeverity.LOW, 5, 0.5, [], "desc", CleaningActionType.IMPUTE_MISSING)

    gemini_mock = MockProvider("gemini", "gemini-2.5-flash", configured=True)
    # Gemini fails with 429 rate limit
    gemini_mock.fail_with = [
        ClassifiedLLMError(LLMErrorCategory.RATE_LIMIT, "429 Rate limit exceeded", is_transient=True),
        ClassifiedLLMError(LLMErrorCategory.RATE_LIMIT, "429 Rate limit exceeded", is_transient=True),
        ClassifiedLLMError(LLMErrorCategory.RATE_LIMIT, "429 Rate limit exceeded", is_transient=True),
    ]

    groq_mock = MockProvider("groq", "openai/gpt-oss-120b", configured=True)

    mgr = LLMProviderManager(
        primary_provider=gemini_mock,
        secondary_provider=groq_mock,
        max_retries=1,
        initial_backoff=0.01,
        max_backoff=0.02
    )
    exp = explain_single_issue(issue, profile, manager=mgr)

    assert exp.provider_source == ProviderSource.GROQ
    assert exp.model_name == "openai/gpt-oss-120b"
    assert groq_mock.call_count == 1

def test_failover_gemini_401_immediate_switch_without_retries(dirty_dataframe, sample_metadata):
    profile = generate_dataset_profile(dirty_dataframe, sample_metadata)
    issue = QualityIssue("i1", IssueCategory.MISSING_VALUE, "city", IssueSeverity.LOW, 5, 0.5, [], "desc", CleaningActionType.IMPUTE_MISSING)

    gemini_mock = MockProvider("gemini", "gemini-2.5-flash", configured=True)
    # Non-transient 401 auth error
    gemini_mock.fail_with = [
        ClassifiedLLMError(LLMErrorCategory.AUTHENTICATION, "401 Unauthorized API key", is_transient=False)
    ]

    groq_mock = MockProvider("groq", "openai/gpt-oss-120b", configured=True)

    mgr = LLMProviderManager(primary_provider=gemini_mock, secondary_provider=groq_mock, max_retries=3)
    exp = explain_single_issue(issue, profile, manager=mgr)

    assert exp.provider_source == ProviderSource.GROQ
    # Should NOT have retried Gemini 3 times because 401 is non-transient
    assert gemini_mock.call_count == 1
    assert groq_mock.call_count == 1

def test_failover_dual_outage_template_fallback(dirty_dataframe, sample_metadata):
    profile = generate_dataset_profile(dirty_dataframe, sample_metadata)
    issue = QualityIssue("i1", IssueCategory.MISSING_VALUE, "city", IssueSeverity.LOW, 5, 0.5, [], "desc", CleaningActionType.IMPUTE_MISSING)

    # Both providers unconfigured or failing
    gemini_mock = MockProvider("gemini", "gemini-2.5-flash", configured=False)
    groq_mock = MockProvider("groq", "openai/gpt-oss-120b", configured=False)

    mgr = LLMProviderManager(primary_provider=gemini_mock, secondary_provider=groq_mock)
    exp = explain_single_issue(issue, profile, manager=mgr)

    assert exp.provider_source == ProviderSource.PYTHON_TEMPLATE
    assert exp.is_fallback is True
    assert "city" in exp.plain_english_summary.lower() or "missing" in exp.plain_english_summary.lower()
