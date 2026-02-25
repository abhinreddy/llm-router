"""
Anthropic SDK wrapper.

Responsibilities:
  - Call the Anthropic messages API
  - Measure wall-clock latency
  - Extract token usage from the response
  - Calculate actual cost from config registry
"""

import logging
import time
from typing import Optional

import anthropic

from app.config import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Thin wrapper around the Anthropic Python SDK."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Send a completion request and return a structured result dict:
            {
                "response_text": str,
                "model_used":    str,
                "input_tokens":  int,
                "output_tokens": int,
                "cost_usd":      float,
                "latency_ms":    float,
            }

        Raises anthropic.APIError (or subclasses) on failure.
        """
        model_cfg = MODEL_REGISTRY.get(model_id)
        if model_cfg is None:
            raise ValueError(f"Unknown model id: {model_id!r}")

        messages = [{"role": "user", "content": prompt}]

        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        logger.info("Calling Anthropic API | model=%s max_tokens=%d", model_id, max_tokens)

        start = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = model_cfg.estimate_cost(input_tokens, output_tokens)

        response_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        logger.info(
            "Anthropic response | model=%s in_tok=%d out_tok=%d cost=$%.6f latency=%.0fms",
            model_id,
            input_tokens,
            output_tokens,
            cost_usd,
            latency_ms,
        )

        return {
            "response_text": response_text,
            "model_used": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": round(latency_ms, 1),
        }
