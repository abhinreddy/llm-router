"""
Model registry with pricing, quality scores, and configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    model_id: str
    display_name: str
    input_cost_per_million: float   # USD per 1M input tokens
    output_cost_per_million: float  # USD per 1M output tokens
    quality_score: float            # 0.0–1.0
    avg_latency_ms: int
    max_tokens: int
    context_window: int

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_million / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_million / 1_000_000

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_token
            + output_tokens * self.output_cost_per_token
        )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "claude-haiku-4-5-20251001": ModelConfig(
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        input_cost_per_million=0.25,
        output_cost_per_million=1.25,
        quality_score=0.65,
        avg_latency_ms=300,
        max_tokens=8192,
        context_window=200_000,
    ),
    "claude-sonnet-4-5-20250929": ModelConfig(
        model_id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        quality_score=0.85,
        avg_latency_ms=800,
        max_tokens=8192,
        context_window=200_000,
    ),
    "claude-opus-4-6": ModelConfig(
        model_id="claude-opus-4-6",
        display_name="Claude Opus 4.6",
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        quality_score=0.98,
        avg_latency_ms=1500,
        max_tokens=8192,
        context_window=200_000,
    ),
}

# Ordered from cheapest to most expensive — used for fallback chains
MODELS_BY_COST_ASC: list[str] = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-6",
]
