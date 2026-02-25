"""
SQLite persistence layer for the llm-router.

All public functions are async (aiosqlite) so they compose naturally with
FastAPI's async request handlers.

DB file: data/router_logs.db
Table:   requests  — one row per routed request (cache hits included)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from app.config import MODEL_REGISTRY

logger = logging.getLogger(__name__)

_OPUS_ID = "claude-opus-4-6"
DB_PATH = Path(__file__).parent.parent / "data" / "router_logs.db"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS requests (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                   TEXT    NOT NULL,
    prompt_snippet              TEXT    NOT NULL,
    task_type                   TEXT    NOT NULL,
    complexity_score            REAL    NOT NULL,
    classifier_used             TEXT    NOT NULL,
    model_selected              TEXT    NOT NULL,
    input_tokens                INTEGER NOT NULL,
    output_tokens               INTEGER NOT NULL,
    cost_usd                    REAL    NOT NULL,
    hypothetical_opus_cost_usd  REAL    NOT NULL,
    latency_ms                  REAL    NOT NULL,
    cache_hit                   INTEGER NOT NULL,
    routing_strategy            TEXT    NOT NULL,
    quality_floor               REAL    NOT NULL,
    max_cost_per_token          REAL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_requests_timestamp  ON requests(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_requests_task_type  ON requests(task_type)",
    "CREATE INDEX IF NOT EXISTS idx_requests_model      ON requests(model_selected)",
]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create the database file, table, and indexes if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await db.execute(idx_sql)
        await db.commit()
    logger.info("Database ready at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def log_request(
    *,
    prompt: str,
    task_type: str,
    complexity_score: float,
    classifier_used: str,
    model_selected: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: float,
    cache_hit: bool,
    routing_strategy: str,
    quality_floor: float,
    max_cost_per_token: Optional[float],
) -> None:
    """Insert one request row.  Computes hypothetical_opus_cost_usd internally."""
    # Compute what this request would have cost at Opus prices.
    # Cache hits have cost_usd == 0 and don't represent an API call, so
    # their hypothetical cost is also 0 (nothing was actually spent or saved).
    if cache_hit:
        hypothetical_opus = 0.0
    else:
        opus_cfg = MODEL_REGISTRY.get(_OPUS_ID)
        hypothetical_opus = (
            opus_cfg.estimate_cost(input_tokens, output_tokens) if opus_cfg else 0.0
        )

    ts = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO requests (
                timestamp, prompt_snippet, task_type, complexity_score,
                classifier_used, model_selected,
                input_tokens, output_tokens, cost_usd, hypothetical_opus_cost_usd,
                latency_ms, cache_hit,
                routing_strategy, quality_floor, max_cost_per_token
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                prompt[:200],
                task_type,
                complexity_score,
                classifier_used,
                model_selected,
                input_tokens,
                output_tokens,
                cost_usd,
                hypothetical_opus,
                latency_ms,
                int(cache_hit),
                routing_strategy,
                quality_floor,
                max_cost_per_token,
            ),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Read — /stats aggregation
# ---------------------------------------------------------------------------

async def get_stats() -> dict[str, Any]:
    """
    Return a dict ready to unpack into StatsResponse (minus cache_size,
    which the caller supplies from the in-memory cache).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # --- Top-level aggregates ---
        row = await (
            await db.execute("""
                SELECT
                    COUNT(*)                                          AS total_requests,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END)   AS api_calls,
                    SUM(cache_hit)                                    AS cache_hits,
                    COALESCE(SUM(cost_usd), 0)                       AS total_cost_usd,
                    COALESCE(SUM(hypothetical_opus_cost_usd), 0)     AS hypothetical_opus_cost_usd,
                    COALESCE(AVG(CASE WHEN cache_hit = 0 THEN latency_ms END), 0) AS average_latency_ms,
                    COALESCE(SUM(input_tokens), 0)                   AS total_input_tokens,
                    COALESCE(SUM(output_tokens), 0)                  AS total_output_tokens
                FROM requests
            """)
        ).fetchone()

        total    = row["total_requests"] or 0
        api_calls = row["api_calls"] or 0
        hits     = row["cache_hits"] or 0
        actual   = float(row["total_cost_usd"])
        opus     = float(row["hypothetical_opus_cost_usd"])
        savings  = opus - actual

        cache_hit_rate = (hits / total)        if total > 0 else 0.0
        savings_pct    = (savings / opus * 100) if opus  > 0 else 0.0

        # --- Per-model breakdown ---
        model_rows = await (
            await db.execute("""
                SELECT
                    model_selected,
                    COUNT(*)                         AS request_count,
                    COALESCE(SUM(cost_usd), 0)       AS total_cost_usd,
                    COALESCE(SUM(input_tokens), 0)   AS total_input_tokens,
                    COALESCE(SUM(output_tokens), 0)  AS total_output_tokens
                FROM requests
                GROUP BY model_selected
            """)
        ).fetchall()

        requests_per_model: dict[str, dict] = {}
        for mr in model_rows:
            model_id = mr["model_selected"]
            cfg = MODEL_REGISTRY.get(model_id)
            requests_per_model[model_id] = {
                "display_name":       cfg.display_name if cfg else model_id,
                "request_count":      mr["request_count"],
                "total_cost_usd":     round(float(mr["total_cost_usd"]), 6),
                "total_input_tokens": mr["total_input_tokens"],
                "total_output_tokens": mr["total_output_tokens"],
                "pct_of_requests":    round(mr["request_count"] / total * 100 if total else 0.0, 1),
            }

    return {
        "total_requests":             total,
        "api_calls":                  api_calls,
        "cache_hits":                 hits,
        "cache_hit_rate":             round(cache_hit_rate, 4),
        "total_cost_usd":             round(actual, 6),
        "hypothetical_opus_cost_usd": round(opus, 6),
        "cost_savings_usd":           round(savings, 6),
        "cost_savings_pct":           round(savings_pct, 2),
        "average_latency_ms":         round(float(row["average_latency_ms"]), 1),
        "total_input_tokens":         int(row["total_input_tokens"]),
        "total_output_tokens":        int(row["total_output_tokens"]),
        "requests_per_model":         requests_per_model,
    }


# ---------------------------------------------------------------------------
# Read — /logs paginated rows + summary
# ---------------------------------------------------------------------------

async def get_logs(
    limit: int = 50,
    offset: int = 0,
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    since: Optional[str] = None,
) -> dict[str, Any]:
    """
    Return paginated request rows plus an aggregated summary over the
    same filtered set.

    Filters are applied consistently to both the rows query and the
    summary aggregation so the summary always matches what's shown.
    """
    # Build a reusable WHERE clause + params
    conditions: list[str] = []
    params: list[Any] = []

    if task_type:
        conditions.append("task_type = ?")
        params.append(task_type)
    if model:
        conditions.append("model_selected = ?")
        params.append(model)
    if since:
        conditions.append("timestamp >= ?")
        params.append(since)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # --- Total row count (for pagination metadata) ---
        count_row = await (
            await db.execute(f"SELECT COUNT(*) AS n FROM requests {where}", params)
        ).fetchone()
        total_rows = count_row["n"] or 0

        # --- Summary aggregation over filtered set ---
        summary_row = await (
            await db.execute(
                f"""
                SELECT
                    COUNT(*)                                          AS total_requests,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END)   AS api_calls,
                    SUM(cache_hit)                                    AS cache_hits,
                    COALESCE(SUM(cost_usd), 0)                       AS total_cost_usd,
                    COALESCE(SUM(hypothetical_opus_cost_usd), 0)     AS hypothetical_opus_cost_usd,
                    COALESCE(AVG(CASE WHEN cache_hit = 0 THEN latency_ms END), 0) AS average_latency_ms,
                    COALESCE(SUM(input_tokens), 0)                   AS total_input_tokens,
                    COALESCE(SUM(output_tokens), 0)                  AS total_output_tokens
                FROM requests {where}
                """,
                params,
            )
        ).fetchone()

        total    = summary_row["total_requests"] or 0
        hits     = summary_row["cache_hits"] or 0
        actual   = float(summary_row["total_cost_usd"])
        opus     = float(summary_row["hypothetical_opus_cost_usd"])
        savings  = opus - actual

        cache_hit_rate = (hits / total)        if total > 0 else 0.0
        savings_pct    = (savings / opus * 100) if opus  > 0 else 0.0

        # --- Per-task-type breakdown ---
        task_rows = await (
            await db.execute(
                f"""
                SELECT task_type,
                       COUNT(*)                   AS cnt,
                       COALESCE(SUM(cost_usd), 0) AS cost
                FROM requests {where}
                GROUP BY task_type
                """,
                params,
            )
        ).fetchall()
        per_task_type = {
            r["task_type"]: {"count": r["cnt"], "cost_usd": round(float(r["cost"]), 6)}
            for r in task_rows
        }

        # --- Per-model breakdown ---
        model_rows = await (
            await db.execute(
                f"""
                SELECT model_selected,
                       COUNT(*)                   AS cnt,
                       COALESCE(SUM(cost_usd), 0) AS cost
                FROM requests {where}
                GROUP BY model_selected
                """,
                params,
            )
        ).fetchall()
        per_model = {
            r["model_selected"]: {"count": r["cnt"], "cost_usd": round(float(r["cost"]), 6)}
            for r in model_rows
        }

        # --- Paginated rows (newest first) ---
        row_params = params + [limit, offset]
        raw_rows = await (
            await db.execute(
                f"""
                SELECT id, timestamp, prompt_snippet, task_type, complexity_score,
                       classifier_used, model_selected,
                       input_tokens, output_tokens, cost_usd, latency_ms, cache_hit,
                       routing_strategy, quality_floor, max_cost_per_token
                FROM requests {where}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                row_params,
            )
        ).fetchall()

    rows = [
        {
            "id":                  r["id"],
            "timestamp":           r["timestamp"],
            "prompt_snippet":      r["prompt_snippet"],
            "task_type":           r["task_type"],
            "complexity_score":    r["complexity_score"],
            "classifier_used":     r["classifier_used"],
            "model_selected":      r["model_selected"],
            "input_tokens":        r["input_tokens"],
            "output_tokens":       r["output_tokens"],
            "cost_usd":            r["cost_usd"],
            "latency_ms":          r["latency_ms"],
            "cache_hit":           bool(r["cache_hit"]),
            "routing_strategy":    r["routing_strategy"],
            "quality_floor":       r["quality_floor"],
            "max_cost_per_token":  r["max_cost_per_token"],
        }
        for r in raw_rows
    ]

    summary = {
        "total_requests":             total,
        "api_calls":                  summary_row["api_calls"] or 0,
        "cache_hits":                 hits,
        "cache_hit_rate":             round(cache_hit_rate, 4),
        "total_cost_usd":             round(actual, 6),
        "hypothetical_opus_cost_usd": round(opus, 6),
        "cost_savings_usd":           round(savings, 6),
        "cost_savings_pct":           round(savings_pct, 2),
        "average_latency_ms":         round(float(summary_row["average_latency_ms"]), 1),
        "total_input_tokens":         int(summary_row["total_input_tokens"]),
        "total_output_tokens":        int(summary_row["total_output_tokens"]),
        "per_task_type":              per_task_type,
        "per_model":                  per_model,
    }

    return {
        "summary":    summary,
        "rows":       rows,
        "total_rows": total_rows,
        "limit":      limit,
        "offset":     offset,
    }
