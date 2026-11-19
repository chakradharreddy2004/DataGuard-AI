# AI Agent Integration Contract & Multi-Provider Architecture

## Module: `src/ai_agent.py`

This contract specifies the multi-provider LLM architecture, error classification taxonomy, bounded exponential backoff retry mechanism, automatic failover manager, and deterministic template fallback engine.

```python
from enum import Enum
from typing import Dict, List, Optional, Protocol, Any, Tuple, Callable
import time
from dataclasses import dataclass, field
from .models import (
    DatasetProfile, 
    QualityIssue, 
    QualityScorecard, 
    AIExplanation, 
    ColumnSemanticInference,
    ProviderSource
)

# ---------------------------------------------------------------------------
# 1. Error Classification Taxonomy
# ---------------------------------------------------------------------------

class LLMErrorCategory(str, Enum):
    AUTHENTICATION = "authentication"          # 401, 403, invalid/missing API key (Non-transient)
    RATE_LIMIT = "rate_limit"                  # 429 quota/throughput exceeded (Transient / Failover)
    MODEL_UNAVAILABLE = "model_unavailable"    # 404 model not found / deprecated (Non-transient)
    TIMEOUT = "timeout"                        # Network socket / API read timeout (Transient)
    SERVER_ERROR = "server_error"              # 500, 502, 503, 504 server failures (Transient)
    UNKNOWN = "unknown"

@dataclass
class ClassifiedLLMError:
    category: LLMErrorCategory
    original_exception: Exception
    status_code: Optional[int]
    is_transient: bool
    retry_after_seconds: Optional[float] = None

def classify_llm_error(exc: Exception) -> ClassifiedLLMError:
    """
    Inspects exceptions from google-genai, groq, or requests/urllib and categorizes them:
    - 401, 403 -> AUTHENTICATION (is_transient=False)
    - 404, 'model not found' -> MODEL_UNAVAILABLE (is_transient=False)
    - 429 -> RATE_LIMIT (is_transient=True)
    - Timeout / ConnectError -> TIMEOUT (is_transient=True)
    - 5xx -> SERVER_ERROR (is_transient=True)
    """

# ---------------------------------------------------------------------------
# 2. Shared Provider Protocol
# ---------------------------------------------------------------------------

class LLMProviderProtocol(Protocol):
    """Common interface implemented by GeminiProvider and GroqProvider."""
    name: str                 # "Gemini" or "Groq"
    model_name: str           # e.g. "gemini-2.5-flash" or "openai/gpt-oss-120b"
    is_configured: bool       # True if API key is present
    
    def generate_json(self, prompt: str, schema: dict) -> dict:
        """
        Sends structured prompt and receives parsed JSON matching schema.
        Raises ClassifiedLLMError on failure.
        """
        ...
        
    def answer_query(self, system_prompt: str, user_query: str) -> str:
        """
        Executes interactive grounded Q&A.
        Raises ClassifiedLLMError on failure.
        """
        ...

# ---------------------------------------------------------------------------
# 3. Provider Implementations
# ---------------------------------------------------------------------------

class GeminiProvider:
    """
    Primary provider using google-genai SDK.
    Configured via GEMINI_API_KEY and GEMINI_MODEL.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None): ...
    def generate_json(self, prompt: str, schema: dict) -> dict: ...
    def answer_query(self, system_prompt: str, user_query: str) -> str: ...

class GroqProvider:
    """
    Secondary provider using the official Groq SDK.
    Configured via GROQ_API_KEY and GROQ_MODEL (default: 'openai/gpt-oss-120b').
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None): ...
    def generate_json(self, prompt: str, schema: dict) -> dict: ...
    def answer_query(self, system_prompt: str, user_query: str) -> str: ...

# ---------------------------------------------------------------------------
# 4. Multi-Provider Failover Manager with Bounded Backoff
# ---------------------------------------------------------------------------

@dataclass
class ProviderStatusReport:
    active_provider: ProviderSource
    active_model: str
    gemini_healthy: bool
    groq_healthy: bool
    last_error_message: Optional[str] = None
    last_error_category: Optional[LLMErrorCategory] = None

class LLMProviderManager:
    """
    Orchestrates provider selection, bounded exponential backoff, and transparent failover:
    
    Flow:
    1. Check primary (Gemini). If configured:
       - Attempt call.
       - If transient error (TIMEOUT, SERVER_ERROR, RATE_LIMIT):
         Retry up to max_retries with bounded exponential backoff:
         delay = min(initial_backoff * (backoff_factor ** attempt) + jitter, max_backoff)
       - If retries exhausted or non-transient error: Proceed to secondary.
    2. Check secondary (Groq). If configured:
       - Attempt call (1 retry for transient error).
       - If failed or unconfigured: Proceed to fallback.
    3. Execute deterministic Python template fallback.
    
    All calls return: (result, ProviderStatusReport).
    """
    def __init__(
        self,
        primary_provider: Optional[LLMProviderProtocol] = None,
        secondary_provider: Optional[LLMProviderProtocol] = None,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 4.0
    ): ...

    def execute_with_failover(
        self,
        call_fn: Callable[[LLMProviderProtocol], Any],
        fallback_fn: Callable[[], Any]
    ) -> Tuple[Any, ProviderSource, str]:
        """
        Executes call_fn with failover chain: Primary -> Secondary -> fallback_fn.
        Returns: (result_data, provider_source, model_name).
        Guarantees zero unhandled exceptions to callers.
        """
        ...

    def get_status(self) -> ProviderStatusReport:
        """Returns current health status for Streamlit UI rendering."""
        ...

# ---------------------------------------------------------------------------
# 5. Deterministic Python Template Fallback Engine
# ---------------------------------------------------------------------------

class TemplateExplanationEngine:
    """
    Generates rule-based explanations and remediation recommendations using
    pure Python when all LLM providers are unavailable.
    """
    @staticmethod
    def explain_issue(issue: QualityIssue, profile: DatasetProfile) -> AIExplanation: ...
    
    @staticmethod
    def infer_column(column_name: str, profile: DatasetProfile) -> ColumnSemanticInference: ...
    
    @staticmethod
    def answer_query(query: str, scorecard: QualityScorecard, issues: List[QualityIssue]) -> str: ...

# ---------------------------------------------------------------------------
# 6. High-Level Public API (Used by Streamlit & Pipelines)
# ---------------------------------------------------------------------------

def infer_unknown_columns(
    profile: DatasetProfile,
    manager: Optional[LLMProviderManager] = None
) -> Tuple[List[ColumnSemanticInference], ProviderSource, str]:
    """Infers column meanings with automatic failover and privacy sanitization."""

def explain_issues(
    issues: List[QualityIssue],
    profile: DatasetProfile,
    manager: Optional[LLMProviderManager] = None
) -> Tuple[Dict[str, AIExplanation], ProviderSource, str]:
    """Generates plain-English issue explanations with automatic failover."""

def answer_user_query(
    user_query: str,
    scorecard: QualityScorecard,
    profile: DatasetProfile,
    issues: List[QualityIssue],
    manager: Optional[LLMProviderManager] = None
) -> Tuple[str, ProviderSource, str]:
    """Answers user queries grounded strictly in profiling metrics with failover."""
```
