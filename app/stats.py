"""
Aggregate statistics tracker for the llm-router.

Tracks every completed request (whether served from cache or live API)
and exposes a snapshot via .snapshot() for the /stats endpoint.

Thread-safety: a single RLock guards all mutable state.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field

from app.config import MODEL_REGISTRY
from app.models import RouteResponse

logger = logging.getLogger(__name__)

# The "premium" baseline we compare against for savings calculation
_OPUS_ID = "claude-opus-4-6"


@dataclass
class _ModelBucket:
    request_count: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class StatsSnapshot:
    total_requests: int
    api_calls: int                      # requests that hit the API
    cache_hits: int                     # requests served from cache
    cache_hit_rate: float               # cache_hits / total_requests
    total_cost_usd: float               # money actually spent
    hypothetical_opus_cost_usd: float   # what it would have cost using Opus for every API call
    cost_savings_usd: float             # hypothetical_opus - actual
    cost_savings_pct: float             # savings as a percentage of opus cost
    average_latency_ms: float           # mean over API calls (cache hits excluded)
    total_input_tokens: int
    total_output_tokens: int
    requests_per_model: dict            # model_id → {request_count, total_cost_usd, pct_of_requests}


class StatsTracker:
    """Thread-safe aggregate stats accumulator."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._total_requests = 0
        self._cache_hits = 0
        self._total_cost_usd = 0.0
        self._hypothetical_opus_cost_usd = 0.0
        self._total_latency_ms = 0.0   # sum over API-only calls
        self._api_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._per_model: dict[str, _ModelBucket] = defaultdict(_ModelBucket)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, response: RouteResponse) -> None:
        """
        Accumulate stats from one completed request.

        For cache hits:  cost = 0, latency is not counted toward the API
                         latency average (it would skew it to near-zero).
        For API calls:   cost and latency are accumulated normally.
        """
        opus_cfg = MODEL_REGISTRY.get(_OPUS_ID)

        with self._lock:
            self._total_requests += 1

            if response.cache_hit:
                self._cache_hits += 1
                # No cost, no latency contribution, attribute to the model
                # that originally handled the prompt.
                bucket = self._per_model[response.model_used]
                bucket.request_count += 1
                logger.debug("Stats: cache hit recorded | model=%s", response.model_used)
                return

            # API call
            self._api_calls += 1
            self._total_cost_usd += response.cost_usd
            self._total_latency_ms += response.latency_ms
            self._total_input_tokens += response.input_tokens
            self._total_output_tokens += response.output_tokens

            # Hypothetical Opus cost for the same token counts
            if opus_cfg is not None:
                opus_cost = opus_cfg.estimate_cost(response.input_tokens, response.output_tokens)
                self._hypothetical_opus_cost_usd += opus_cost

            bucket = self._per_model[response.model_used]
            bucket.request_count += 1
            bucket.total_cost_usd += response.cost_usd
            bucket.total_input_tokens += response.input_tokens
            bucket.total_output_tokens += response.output_tokens

            logger.debug(
                "Stats: API call recorded | model=%s cost=$%.6f latency=%.0fms",
                response.model_used, response.cost_usd, response.latency_ms,
            )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            total = self._total_requests
            hits  = self._cache_hits
            api   = self._api_calls
            opus  = self._hypothetical_opus_cost_usd
            actual = self._total_cost_usd
            savings = opus - actual

            avg_latency = (self._total_latency_ms / api) if api else 0.0
            cache_hit_rate = (hits / total) if total else 0.0
            savings_pct = (savings / opus * 100) if opus > 0 else 0.0

            per_model_out = {}
            for model_id, bucket in self._per_model.items():
                cfg = MODEL_REGISTRY.get(model_id)
                per_model_out[model_id] = {
                    "display_name": cfg.display_name if cfg else model_id,
                    "request_count": bucket.request_count,
                    "total_cost_usd": round(bucket.total_cost_usd, 6),
                    "total_input_tokens": bucket.total_input_tokens,
                    "total_output_tokens": bucket.total_output_tokens,
                    "pct_of_requests": round(
                        bucket.request_count / total * 100 if total else 0.0, 1
                    ),
                }

            return StatsSnapshot(
                total_requests=total,
                api_calls=api,
                cache_hits=hits,
                cache_hit_rate=round(cache_hit_rate, 4),
                total_cost_usd=round(actual, 6),
                hypothetical_opus_cost_usd=round(opus, 6),
                cost_savings_usd=round(savings, 6),
                cost_savings_pct=round(savings_pct, 2),
                average_latency_ms=round(avg_latency, 1),
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                requests_per_model=per_model_out,
            )
