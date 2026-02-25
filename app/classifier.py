"""
Heuristic-based prompt classifier.

Scores prompts on two axes:
  - task_type: one of simple_qa | summarization | code_generation |
                          creative_writing | math_reasoning | analysis
  - complexity_score: float 0.0–1.0

All classification is done without any LLM call — pure rule-based heuristics
so the router overhead is near-zero.
"""

import logging
import re

from app.models import ClassificationResult, TaskType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists (lowercase)
# ---------------------------------------------------------------------------

_CODE_KEYWORDS = {
    "implement", "write code", "write a function", "write a script",
    "write a program", "refactor", "debug", "fix the bug", "unit test",
    "class ", "def ", "function ", "algorithm", "api endpoint",
    "data structure", "sql", "regex", "cli tool",
}

_SUMMARIZATION_KEYWORDS = {
    "summarize", "summarise", "tldr", "tl;dr", "brief summary",
    "key points", "main points", "in short", "condense", "overview",
    "abstract", "synopsis",
}

_MATH_KEYWORDS = {
    "prove", "proof", "calculate", "compute", "equation", "derivative",
    "integral", "matrix", "eigenvalue", "probability", "statistics",
    "theorem", "formula", "solve for", "differentiate", "integrate",
    "factorial", "prime number", "modulo", "arithmetic",
}

_CREATIVE_KEYWORDS = {
    "write a story", "write a poem", "write a song", "short story",
    "creative writing", "fictional", "narrative", "haiku", "sonnet",
    "limerick", "fantasy", "sci-fi", "world-build", "character",
    "plot twist", "dialogue",
}

_SIMPLE_QA_PATTERNS = [
    r"^(what|who|where|when|why|how|is|are|was|were|does|do|did|can|could|should|would)\b",
    r"\?$",
]

# Technical vocabulary that increases complexity score
_TECHNICAL_VOCAB = {
    "api", "algorithm", "architecture", "asynchronous", "authentication",
    "binary", "cache", "compiler", "concurrency", "database", "dependency",
    "deployment", "distributed", "encryption", "framework", "heuristic",
    "idempotent", "infrastructure", "latency", "microservice", "middleware",
    "neural", "oauth", "optimize", "parallelism", "recursion", "runtime",
    "scalability", "serialization", "throughput", "tokenization", "webhook",
    "abstract", "polymorphism", "inheritance", "encapsulation", "interface",
}

# Words that indicate multiple requirements / sub-questions
_CONSTRAINT_WORDS = {
    "must", "should", "ensure", "require", "need", "have to", "also",
    "additionally", "furthermore", "moreover", "as well as", "in addition",
    "and also", "make sure",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class HeuristicClassifier:
    """Classifies a prompt into a task type and complexity score."""

    def classify(self, prompt: str) -> ClassificationResult:
        task_type = self._detect_task_type(prompt)
        complexity_score = self._compute_complexity(prompt, task_type)

        logger.debug(
            "Classified prompt | task_type=%s complexity=%.2f len=%d",
            task_type,
            complexity_score,
            len(prompt),
        )
        return ClassificationResult(task_type=task_type, complexity_score=complexity_score)

    # ------------------------------------------------------------------
    # Task type detection
    # ------------------------------------------------------------------

    def _detect_task_type(self, prompt: str) -> TaskType:
        lower = prompt.lower()

        # Code generation — check for code blocks or explicit code keywords
        if "```" in prompt or any(kw in lower for kw in _CODE_KEYWORDS):
            return "code_generation"

        # Summarization
        if any(kw in lower for kw in _SUMMARIZATION_KEYWORDS):
            return "summarization"

        # Math / reasoning
        if any(kw in lower for kw in _MATH_KEYWORDS):
            return "math_reasoning"

        # Creative writing
        if any(kw in lower for kw in _CREATIVE_KEYWORDS):
            return "creative_writing"

        # Simple Q&A — short prompt that looks like a direct question
        word_count = len(prompt.split())
        if word_count <= 25 and any(
            re.search(pat, lower) for pat in _SIMPLE_QA_PATTERNS
        ):
            return "simple_qa"

        # Default
        return "analysis"

    # ------------------------------------------------------------------
    # Complexity scoring
    # ------------------------------------------------------------------

    def _compute_complexity(self, prompt: str, task_type: TaskType) -> float:
        """
        Combines four independent signals into a single 0–1 score:
          1. Prompt length (normalised)
          2. Number of constraints / requirements
          3. Technical vocabulary density
          4. Number of sub-questions / bullet points
        """
        score = 0.0

        # 1. Length signal (up to 0.35)
        words = prompt.split()
        word_count = len(words)
        # Saturates at ~500 words
        length_score = min(word_count / 500, 1.0) * 0.35
        score += length_score

        # 2. Constraint / requirements density (up to 0.25)
        lower = prompt.lower()
        constraint_hits = sum(1 for cw in _CONSTRAINT_WORDS if cw in lower)
        constraint_score = min(constraint_hits / 5, 1.0) * 0.25
        score += constraint_score

        # 3. Technical vocabulary density (up to 0.25)
        if word_count > 0:
            tech_hits = sum(1 for w in words if w.lower().strip(".,!?;:\"'") in _TECHNICAL_VOCAB)
            tech_density = tech_hits / word_count
            # Saturates at 15% density
            tech_score = min(tech_density / 0.15, 1.0) * 0.25
        else:
            tech_score = 0.0
        score += tech_score

        # 4. Sub-questions / enumerated requirements (up to 0.15)
        # Count question marks (beyond the first), numbered items, bullet points
        question_marks = prompt.count("?")
        numbered_items = len(re.findall(r"(?m)^\s*\d+[\.\)]\s", prompt))
        bullet_items = len(re.findall(r"(?m)^\s*[-*•]\s", prompt))
        sub_q_count = max(question_marks - 1, 0) + numbered_items + bullet_items
        sub_q_score = min(sub_q_count / 5, 1.0) * 0.15
        score += sub_q_score

        # Task-type adjustments: some task types skew complexity
        task_boosts: dict[TaskType, float] = {
            "math_reasoning": 0.10,
            "code_generation": 0.05,
            "simple_qa": -0.10,
            "summarization": -0.05,
            "creative_writing": 0.0,
            "analysis": 0.0,
        }
        score += task_boosts.get(task_type, 0.0)

        return round(max(0.0, min(score, 1.0)), 4)
