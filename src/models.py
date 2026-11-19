"""
Domain models, Enums, and data contracts for DataGuard AI.
Completely decoupled from Streamlit and web frameworks.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InferredDataType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"

class IssueCategory(str, Enum):
    MISSING_VALUE = "missing_value"
    EXACT_DUPLICATE_ROW = "exact_duplicate_row"
    BUSINESS_KEY_DUPLICATE = "business_key_duplicate"
    KEY_DUPLICATE = "business_key_duplicate"
    NUMERIC_OUTLIER = "numeric_outlier"
    CASING_INCONSISTENCY = "casing_inconsistency"
    WHITESPACE_INCONSISTENCY = "whitespace_inconsistency"
    ALIAS_INCONSISTENCY = "alias_inconsistency"
    SPELLING_INCONSISTENCY = "spelling_inconsistency"
    FORMAT_ANOMALY = "format_anomaly"
    UNKNOWN_COLUMN = "unknown_column"

class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ADVISORY = "advisory"

class ConfidenceTier(str, Enum):
    COSMETIC = "cosmetic"
    HIGH_CONFIDENCE = "high"
    MEDIUM_CONFIDENCE = "medium"
    LOW_CONFIDENCE = "low"

class CleaningActionType(str, Enum):
    DROP_EXACT_DUPLICATES = "drop_exact_duplicates"
    RESOLVE_KEY_DUPLICATES = "resolve_key_duplicates"
    IMPUTE_MISSING = "impute_missing"
    DROP_MISSING_ROWS = "drop_missing_rows"
    TRIM_WHITESPACE = "trim_whitespace"
    STANDARDIZE_CASING = "standardize_casing"
    MAP_ALIASES = "map_aliases"
    FUZZY_STANDARDIZE = "fuzzy_standardize"
    CAP_OUTLIERS = "cap_outliers"

class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"

class ProviderSource(str, Enum):
    GEMINI = "Gemini"
    GROQ = "Groq"
    PYTHON_TEMPLATE = "Python Template Fallback"

class LLMErrorCategory(str, Enum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Profiling & Dataset Models
# ---------------------------------------------------------------------------

@dataclass
class DatasetMetadata:
    filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    ingestion_timestamp: str

@dataclass
class ColumnStatistics:
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    q1_val: Optional[float] = None
    q3_val: Optional[float] = None
    iqr_val: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    top_categories: Dict[str, int] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)

@dataclass
class ColumnProfile:
    column_name: str
    inferred_type: InferredDataType
    total_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    is_business_key: bool = False
    statistics: ColumnStatistics = field(default_factory=ColumnStatistics)

@dataclass
class DatasetProfile:
    metadata: DatasetMetadata
    columns: Dict[str, ColumnProfile]
    warnings: List[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Quality Issues & Scoring Models
# ---------------------------------------------------------------------------

@dataclass
class QualityIssue:
    issue_id: str
    category: IssueCategory
    column_name: Optional[str]
    severity: IssueSeverity
    affected_count: int
    affected_percentage: float
    sample_values: List[Any] = field(default_factory=list)
    sample_row_indices: List[int] = field(default_factory=list)
    explanation: str = ""
    confidence_tier: Optional[ConfidenceTier] = None
    is_advisory: bool = False
    suggested_action: Optional[CleaningActionType] = None

@dataclass
class DimensionScore:
    completeness: float     # 0.0 to 100.0
    validity: float         # 0.0 to 100.0
    consistency: float      # 0.0 to 100.0
    uniqueness: float       # 0.0 to 100.0
    active_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class QualityScorecard:
    overall_score: float                              # 0.0 to 100.0
    dataset_uniqueness_score: float                   # 0.0 to 100.0
    column_scores: Dict[str, float] = field(default_factory=dict)
    column_dimensions: Dict[str, DimensionScore] = field(default_factory=dict)
    deduction_reasons: List[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# AI Explanations & Provider Status Models
# ---------------------------------------------------------------------------

@dataclass
class ColumnSemanticInference:
    column_name: str
    suggested_meaning: str
    confidence: float
    candidate_rules: List[str]
    is_uncertain: bool

@dataclass
class ProviderStatusReport:
    active_provider: ProviderSource
    active_model: str
    gemini_healthy: bool
    groq_healthy: bool
    last_error_message: Optional[str] = None
    last_error_category: Optional[LLMErrorCategory] = None

@dataclass
class AIExplanation:
    issue_id: str
    plain_english_summary: str
    business_impact: str
    recommended_fix: str
    suggested_python_code: str
    provider_source: ProviderSource = ProviderSource.GEMINI
    model_name: str = "gemini-3.5-flash-lite"
    is_fallback: bool = False

# ---------------------------------------------------------------------------
# Cleaning & Value Mapping Models
# ---------------------------------------------------------------------------

@dataclass
class ValueMapping:
    original_value: str
    replacement_value: str
    confidence_tier: ConfidenceTier
    similarity_score: float
    frequency: int
    is_approved: bool = False

@dataclass
class CleaningProposal:
    proposal_id: str
    action_type: CleaningActionType
    column_name: Optional[str]
    parameters: Dict[str, Any]
    affected_rows: int
    status: ProposalStatus = ProposalStatus.PENDING
    value_mappings: List[ValueMapping] = field(default_factory=list)
