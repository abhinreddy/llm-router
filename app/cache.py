"""
In-memory prompt cache with exact and fuzzy (Jaccard) matching.

Cache key = normalised(prompt) + policy fingerprint, so the same prompt
with different routing policies is cached independently.

Thread-safety: all public methods hold a lock so the cache is safe to use
from FastAPI's thread-pool workers.
"""

import logging
import threading
from collections import OrderedDict

from app.models import RouteResponse, RoutingPolicy

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().split())


def _policy_fingerprint(policy: RoutingPolicy) -> str:
    return f"{policy.strategy}:{policy.quality_floor}:{policy.max_cost_per_token}"


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two normalised strings."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class PromptCache:
    """
    LRU-bounded in-memory cache for RouteResponse objects.

    Lookup order:
      1. Exact match on normalised prompt + policy fingerprint  → O(1)
      2. Fuzzy match: Jaccard similarity ≥ similarity_threshold → O(n)

    Parameters
    ----------
    similarity_threshold : float
        Minimum Jaccard score to consider two prompts "the same".
        Defaults to 0.92 — high enough to avoid false positives on
        short prompts while catching typo / whitespace variants.
    max_size : int
        Maximum number of entries before the oldest is evicted (LRU).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 1_000,
    ) -> None:
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._lock = threading.Lock()

        # OrderedDict used as an LRU store:
        #   key   = "{normalised_prompt}|{policy_fingerprint}"
        #   value = (normalised_prompt, RouteResponse)
        self._store: OrderedDict[str, tuple[str, RouteResponse]] = OrderedDict()

        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prompt: str, policy: RoutingPolicy) -> RouteResponse | None:
        """
        Return a cached response or None.

        On a hit the entry is promoted to most-recently-used and the
        returned RouteResponse has cache_hit=True.
        """
        norm = _normalise(prompt)
        fp   = _policy_fingerprint(policy)
        exact_key = f"{norm}|{fp}"

        with self._lock:
            # 1. Exact match
            if exact_key in self._store:
                self._store.move_to_end(exact_key)
                self._hits += 1
                logger.debug("Cache exact-hit | key=%.60s…", exact_key)
                return self._store[exact_key][1].model_copy(update={"cache_hit": True})

            # 2. Fuzzy match (only within the same policy fingerprint)
            best_key: str | None = None
            best_sim = 0.0
            for key, (stored_norm, _) in self._store.items():
                if not key.endswith(f"|{fp}"):
                    continue
                sim = _jaccard(norm, stored_norm)
                if sim >= self._threshold and sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_key is not None:
                self._store.move_to_end(best_key)
                self._hits += 1
                logger.debug(
                    "Cache fuzzy-hit  | similarity=%.3f key=%.60s…", best_sim, best_key
                )
                return self._store[best_key][1].model_copy(update={"cache_hit": True})

            self._misses += 1
            return None

    def put(self, prompt: str, policy: RoutingPolicy, response: RouteResponse) -> None:
        """Insert or update a cache entry."""
        norm = _normalise(prompt)
        fp   = _policy_fingerprint(policy)
        key  = f"{norm}|{fp}"

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (norm, response)
                return

            if len(self._store) >= self._max_size:
                evicted = next(iter(self._store))
                del self._store[evicted]
                logger.debug("Cache evict (LRU) | key=%.60s…", evicted)

            self._store[key] = (norm, response)
            logger.debug("Cache put | key=%.60s…", key)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total else 0.0
