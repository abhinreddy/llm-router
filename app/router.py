"""
Router engine: maps classifier output + routing policy → model selection.

Selection logic per strategy
────────────────────────────
minimize_cost      → cheapest model that satisfies quality_floor
maximize_quality   → highest quality model that satisfies max_cost_per_token
balanced           → score = quality_score / cost_per_token_output (normalised)
minimize_latency   → lowest avg_latency model that satisfies quality_floor

All strategies also respect:
  - quality_floor  (hard lower bound on quality_score)
  - max_cost_per_token (hard upper bound on output cost per token)
"""

import logging
from typing import Optional

from app.config import MODEL_REGISTRY, MODELS_BY_COST_ASC, ModelConfig
from app.models import ClassificationResult, RoutingPolicy

logger = logging.getLogger(__name__)


class RouterEngine:
    """Selects the optimal model given a classification and policy."""

    def route(
        self,
        classification: ClassificationResult,
        policy: RoutingPolicy,
    ) -> tuple[str, str]:
        """
        Returns (model_id, routing_reasoning).

        Raises ValueError if no model satisfies the hard constraints.
        """
        candidates = self._filter_candidates(policy)

        if not candidates:
            raise ValueError(
                f"No model satisfies policy constraints: "
                f"quality_floor={policy.quality_floor}, "
                f"max_cost_per_token={policy.max_cost_per_token}"
            )

        selected, reasoning = self._apply_strategy(
            candidates, classification, policy
        )

        logger.info(
            "Routing decision | model=%s strategy=%s task=%s complexity=%.2f",
            selected.model_id,
            policy.strategy,
            classification.task_type,
            classification.complexity_score,
        )
        return selected.model_id, reasoning

    # ------------------------------------------------------------------
    # Candidate filtering (hard constraints)
    # ------------------------------------------------------------------

    def _filter_candidates(self, policy: RoutingPolicy) -> list[ModelConfig]:
        candidates = []
        for model_id in MODELS_BY_COST_ASC:
            cfg = MODEL_REGISTRY[model_id]
            if cfg.quality_score < policy.quality_floor:
                logger.debug(
                    "Skipping %s: quality %.2f < floor %.2f",
                    model_id, cfg.quality_score, policy.quality_floor,
                )
                continue
            if (
                policy.max_cost_per_token is not None
                and cfg.output_cost_per_token > policy.max_cost_per_token
            ):
                logger.debug(
                    "Skipping %s: cost/token %.6f > max %.6f",
                    model_id, cfg.output_cost_per_token, policy.max_cost_per_token,
                )
                continue
            candidates.append(cfg)
        return candidates

    # ------------------------------------------------------------------
    # Strategy application
    # ------------------------------------------------------------------

    def _apply_strategy(
        self,
        candidates: list[ModelConfig],
        classification: ClassificationResult,
        policy: RoutingPolicy,
    ) -> tuple[ModelConfig, str]:
        strategy = policy.strategy
        complexity = classification.complexity_score
        task_type = classification.task_type

        if strategy == "minimize_cost":
            selected = candidates[0]  # already sorted cheapest-first
            reasoning = (
                f"minimize_cost strategy: selected {selected.display_name} "
                f"(${selected.output_cost_per_million:.2f}/1M output tokens) "
                f"as the cheapest option above quality floor {policy.quality_floor}."
            )

        elif strategy == "maximize_quality":
            selected = max(candidates, key=lambda m: m.quality_score)
            reasoning = (
                f"maximize_quality strategy: selected {selected.display_name} "
                f"(quality={selected.quality_score}) as the highest-quality "
                f"available model."
            )

        elif strategy == "minimize_latency":
            selected = min(candidates, key=lambda m: m.avg_latency_ms)
            reasoning = (
                f"minimize_latency strategy: selected {selected.display_name} "
                f"(avg {selected.avg_latency_ms} ms) as the fastest model "
                f"above quality floor {policy.quality_floor}."
            )

        else:  # "balanced" — default
            selected, reasoning = self._balanced_selection(
                candidates, complexity, task_type
            )

        return selected, reasoning

    def _balanced_selection(
        self,
        candidates: list[ModelConfig],
        complexity: float,
        task_type: str,
    ) -> tuple[ModelConfig, str]:
        """
        Balanced heuristic:
          - complexity < 0.3  → prefer cheapest (haiku tier)
          - complexity 0.3–0.65 → prefer mid-tier (sonnet)
          - complexity > 0.65 → prefer highest quality (opus)

        Score each candidate, pick the best fit.
        """
        def _score(m: ModelConfig) -> float:
            # Normalised quality weight increases with complexity
            quality_weight = 0.3 + 0.7 * complexity
            # Normalised cost weight is the inverse
            cost_weight = 1.0 - quality_weight

            # Normalise quality over [0,1] range of candidates
            qualities = [c.quality_score for c in candidates]
            q_min, q_max = min(qualities), max(qualities)
            q_norm = (
                (m.quality_score - q_min) / (q_max - q_min)
                if q_max > q_min
                else 1.0
            )

            # Normalise cost (lower is better → invert)
            costs = [c.output_cost_per_million for c in candidates]
            c_min, c_max = min(costs), max(costs)
            c_norm = (
                1.0 - (m.output_cost_per_million - c_min) / (c_max - c_min)
                if c_max > c_min
                else 1.0
            )

            return quality_weight * q_norm + cost_weight * c_norm

        scored = sorted(candidates, key=_score, reverse=True)
        selected = scored[0]

        tier = (
            "high-complexity task" if complexity > 0.65
            else "medium-complexity task" if complexity > 0.3
            else "low-complexity task"
        )
        reasoning = (
            f"balanced strategy: {tier} (complexity={complexity:.2f}, "
            f"task={task_type}). Selected {selected.display_name} "
            f"(quality={selected.quality_score}, "
            f"${selected.output_cost_per_million:.2f}/1M out tokens) "
            f"as the best quality-cost trade-off."
        )
        return selected, reasoning

    # ------------------------------------------------------------------
    # Fallback chain (used by main.py on API error)
    # ------------------------------------------------------------------

    def get_fallback_chain(self, current_model_id: str) -> list[str]:
        """
        Returns models to try if current_model_id fails, ordered by
        ascending quality (cheapest → most capable).

        On failure we escalate to the next higher-quality model rather
        than degrading — a failed cheap model is more likely overloaded
        or rate-limited, and the user already accepted this policy's
        quality floor.  If the most capable model fails, there is no
        further fallback.
        """
        idx = (
            MODELS_BY_COST_ASC.index(current_model_id)
            if current_model_id in MODELS_BY_COST_ASC
            else -1
        )
        return MODELS_BY_COST_ASC[idx + 1:]
