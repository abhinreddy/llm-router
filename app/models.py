"""
Pydantic request/response models for the llm-router API.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------

class RoutingPolicy(BaseModel):
    strategy: Literal["minimize_cost", "maximize_quality", "balanced", "minimize_latency"] = "balanced"
    quality_floor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable quality score (0.0–1.0).",
    )
    max_cost_per_token: Optional[float] = Field(
        default=None,
        description="Hard ceiling on cost per output token in USD. None means no ceiling.",
    )


# ---------------------------------------------------------------------------
# Classifier output (internal)
# ---------------------------------------------------------------------------

TaskType = Literal[
    "simple_qa",
    "summarization",
    "code_generation",
    "creative_writing",
    "math_reasoning",
    "analysis",
]


class ClassificationResult(BaseModel):
    task_type: TaskType
    complexity_score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user prompt to route.")
    policy: RoutingPolicy = Field(
        default_factory=RoutingPolicy,
        description="Routing policy. Defaults to balanced strategy.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum tokens to generate.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt prepended to the conversation.",
    )


class RouteResponse(BaseModel):
    response_text: str
    model_used: str
    task_type: TaskType
    complexity_score: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    routing_reasoning: str
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# /stats endpoint response
# ---------------------------------------------------------------------------

class ModelUsageStats(BaseModel):
    display_name: str
    request_count: int
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    pct_of_requests: float


class StatsResponse(BaseModel):
    total_requests: int
    api_calls: int
    cache_hits: int
    cache_hit_rate: float
    total_cost_usd: float
    hypothetical_opus_cost_usd: float
    cost_savings_usd: float
    cost_savings_pct: float
    average_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    cache_size: int
    requests_per_model: dict[str, ModelUsageStats]


# ---------------------------------------------------------------------------
# /models endpoint response
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    model_id: str
    display_name: str
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    quality_score: float
    avg_latency_ms: int
    context_window: int


class ModelsResponse(BaseModel):
    models: list[ModelSpec]


# ---------------------------------------------------------------------------
# /logs endpoint response
# ---------------------------------------------------------------------------

class LogRow(BaseModel):
    id: int
    timestamp: str
    prompt_snippet: str
    task_type: TaskType
    complexity_score: float
    classifier_used: str
    model_selected: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    routing_strategy: str
    quality_floor: float
    max_cost_per_token: Optional[float]


class LogsSummary(BaseModel):
    total_requests: int
    api_calls: int
    cache_hits: int
    cache_hit_rate: float
    total_cost_usd: float
    hypothetical_opus_cost_usd: float
    cost_savings_usd: float
    cost_savings_pct: float
    average_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    per_task_type: dict[str, Any]   # task_type  → {count, cost_usd}
    per_model: dict[str, Any]       # model_id   → {count, cost_usd}


class LogsResponse(BaseModel):
    summary: LogsSummary
    rows: list[LogRow]
    total_rows: int
    limit: int
    offset: int
