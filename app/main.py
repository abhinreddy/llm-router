"""
FastAPI application — LLM cost optimizer and intelligent router.

Endpoints
─────────
POST /route    Route a prompt to the optimal model and return the response.
GET  /models   List available models and their specs.
GET  /stats    Aggregate request/cost/latency statistics (DB-backed, persists across restarts).
GET  /logs     Paginated request log with per-request rows and a summary block.
GET  /health   Health check.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import anthropic as anthropic_sdk
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app import database
from app.cache import PromptCache
from app.classifier import HeuristicClassifier
from app.ml_classifier import MLClassifier
from app.config import MODEL_REGISTRY
from app.models import (
    LogsResponse,
    ModelSpec,
    ModelsResponse,
    ModelUsageStats,
    RouteRequest,
    RouteResponse,
    StatsResponse,
)
from app.providers.anthropic import AnthropicProvider
from app.router import RouterEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not set — /route calls will fail")

# ---------------------------------------------------------------------------
# Lifespan — DB initialisation
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()
    yield


# ---------------------------------------------------------------------------
# App + singletons
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Router",
    description="Cost-optimising intelligent router for Anthropic models.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_classifier        = MLClassifier()
heuristic_classifier = HeuristicClassifier()
router_engine        = RouterEngine()
provider             = AnthropicProvider(api_key=ANTHROPIC_API_KEY)
cache                = PromptCache(similarity_threshold=0.92, max_size=1_000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok", "api_key_configured": bool(ANTHROPIC_API_KEY)}


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """Return all available models with pricing and capability metadata."""
    specs = [
        ModelSpec(
            model_id=cfg.model_id,
            display_name=cfg.display_name,
            input_cost_per_million_tokens=cfg.input_cost_per_million,
            output_cost_per_million_tokens=cfg.output_cost_per_million,
            quality_score=cfg.quality_score,
            avg_latency_ms=cfg.avg_latency_ms,
            context_window=cfg.context_window,
        )
        for cfg in MODEL_REGISTRY.values()
    ]
    return ModelsResponse(models=specs)


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Aggregate statistics across all requests.  Data is read from the
    SQLite database so it persists across server restarts.
    """
    data = await database.get_stats()
    per_model = data.pop("requests_per_model")
    return StatsResponse(
        **data,
        cache_size=cache.size,
        requests_per_model={
            model_id: ModelUsageStats(**bucket)
            for model_id, bucket in per_model.items()
        },
    )


@app.get("/logs", response_model=LogsResponse)
async def get_logs(
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return."),
    offset: int = Query(default=0, ge=0, description="Row offset for pagination."),
    task_type: Optional[str] = Query(default=None, description="Filter by task type."),
    model: Optional[str] = Query(default=None, description="Filter by model_selected."),
    since: Optional[str] = Query(
        default=None,
        description="ISO-8601 datetime lower bound for timestamp (e.g. 2025-01-01T00:00:00Z).",
    ),
) -> LogsResponse:
    """
    Paginated request log.

    Returns a **summary** block aggregated over the filtered set plus the
    matching **rows** in reverse-chronological order.

    Query params:
      limit       Max rows to return (1–500, default 50).
      offset      Pagination offset (default 0).
      task_type   Filter to a specific task type.
      model       Filter to a specific model_selected value.
      since       Only include requests on or after this ISO-8601 datetime.
    """
    data = await database.get_logs(
        limit=limit,
        offset=offset,
        task_type=task_type,
        model=model,
        since=since,
    )
    return LogsResponse(**data)


@app.post("/route", response_model=RouteResponse)
async def route_prompt(
    request: RouteRequest,
    classifier: str = Query(
        default="ml",
        description="Classifier to use: 'ml' (default) or 'heuristic'.",
    ),
) -> RouteResponse:
    """
    Classify the prompt, select the optimal model per policy, call the model,
    and return the response with full cost/latency metadata.

    Cache: identical or very-similar prompts (Jaccard ≥ 0.92) with the same
    policy are served from the in-memory cache at zero cost.

    Fallback: on API failure the router escalates to the next higher-quality
    model until the request succeeds or all options are exhausted.

    Query params:
      classifier=ml         Use the trained ML classifier (default).
      classifier=heuristic  Use the rule-based heuristic classifier.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    active_classifier = (
        heuristic_classifier if classifier == "heuristic" else ml_classifier
    )

    logger.info(
        "Received /route request | classifier=%s strategy=%s quality_floor=%.2f prompt_len=%d",
        classifier,
        request.policy.strategy,
        request.policy.quality_floor,
        len(request.prompt),
    )

    # 1. Cache check — before any classification or routing work
    cached = cache.get(request.prompt, request.policy)
    if cached is not None:
        logger.info("Cache hit | model=%s task=%s", cached.model_used, cached.task_type)
        await database.log_request(
            prompt=request.prompt,
            task_type=cached.task_type,
            complexity_score=cached.complexity_score,
            classifier_used=classifier,
            model_selected=cached.model_used,
            input_tokens=cached.input_tokens,
            output_tokens=cached.output_tokens,
            cost_usd=0.0,
            latency_ms=cached.latency_ms,
            cache_hit=True,
            routing_strategy=request.policy.strategy,
            quality_floor=request.policy.quality_floor,
            max_cost_per_token=request.policy.max_cost_per_token,
        )
        return cached

    # 2. Classify  (fast in-memory, safe to call from async context)
    classification = active_classifier.classify(request.prompt)
    logger.info(
        "Classification | task=%s complexity=%.2f",
        classification.task_type,
        classification.complexity_score,
    )

    # 3. Route
    try:
        model_id, routing_reasoning = router_engine.route(classification, request.policy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 4. Call model — escalate through higher-quality fallbacks on failure
    fallback_chain = router_engine.get_fallback_chain(model_id)
    models_to_try = [model_id] + fallback_chain
    last_error: Exception | None = None

    for attempt_model in models_to_try:
        try:
            # provider.complete() is synchronous; run it in a thread pool so
            # it doesn't block the event loop during the API round-trip.
            result = await asyncio.to_thread(
                provider.complete,
                model_id=attempt_model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                system_prompt=request.system_prompt,
            )

            if attempt_model != model_id:
                routing_reasoning += (
                    f" (Escalated from {model_id} → {attempt_model} after API failure.)"
                )
                logger.warning("Fallback escalation | %s → %s", model_id, attempt_model)

            response = RouteResponse(
                response_text=result["response_text"],
                model_used=result["model_used"],
                task_type=classification.task_type,
                complexity_score=classification.complexity_score,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cost_usd=result["cost_usd"],
                latency_ms=result["latency_ms"],
                routing_reasoning=routing_reasoning,
                cache_hit=False,
            )

            # 5. Cache + log
            cache.put(request.prompt, request.policy, response)
            await database.log_request(
                prompt=request.prompt,
                task_type=response.task_type,
                complexity_score=response.complexity_score,
                classifier_used=classifier,
                model_selected=response.model_used,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                cache_hit=False,
                routing_strategy=request.policy.strategy,
                quality_floor=request.policy.quality_floor,
                max_cost_per_token=request.policy.max_cost_per_token,
            )
            return response

        except anthropic_sdk.AuthenticationError as exc:
            logger.error("Authentication failed — no fallback possible: %s", exc)
            raise HTTPException(status_code=401, detail="Invalid Anthropic API key.")

        except anthropic_sdk.RateLimitError as exc:
            logger.warning(
                "Rate limit on %s, escalating to next model: %s", attempt_model, exc
            )
            last_error = exc

        except anthropic_sdk.APIStatusError as exc:
            logger.warning(
                "API error %d on %s, escalating: %s",
                exc.status_code, attempt_model, exc.message,
            )
            last_error = exc

        except Exception as exc:
            logger.exception("Unexpected error calling %s", attempt_model)
            last_error = exc

    # All models exhausted
    logger.error("All models failed. Last error: %s", last_error)
    raise HTTPException(
        status_code=502,
        detail=f"All models failed. Last error: {last_error}",
    )
