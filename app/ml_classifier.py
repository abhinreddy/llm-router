"""
ML-based prompt classifier.

Uses trained TF-IDF + sklearn pipelines saved by scripts/train_classifier.py.
Exposes the same interface as HeuristicClassifier so it can be used as a
drop-in replacement.

Falls back to HeuristicClassifier transparently if the model files are absent
(e.g. before training has been run).
"""

import logging
from pathlib import Path

from app.models import ClassificationResult

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"
_TASK_MODEL_PATH = _MODELS_DIR / "task_classifier.joblib"
_COMPLEXITY_MODEL_PATH = _MODELS_DIR / "complexity_model.joblib"


class MLClassifier:
    """
    Classifies prompts using trained TF-IDF + sklearn pipelines.

    Falls back to HeuristicClassifier when model files are not found or
    fail to load.
    """

    def __init__(self) -> None:
        self._task_model = None
        self._complexity_model = None
        self._fallback = None
        self._load_models()

    def _load_models(self) -> None:
        if not (_TASK_MODEL_PATH.exists() and _COMPLEXITY_MODEL_PATH.exists()):
            logger.warning(
                "ML model files not found in %s — using HeuristicClassifier fallback. "
                "Run `python scripts/train_classifier.py` to enable ML classification.",
                _MODELS_DIR,
            )
            self._init_fallback()
            return

        try:
            import joblib  # imported lazily so the app works without scikit-learn installed
            self._task_model       = joblib.load(_TASK_MODEL_PATH)
            self._complexity_model = joblib.load(_COMPLEXITY_MODEL_PATH)
            logger.info("MLClassifier loaded models from %s", _MODELS_DIR)
        except Exception as exc:
            logger.warning(
                "Failed to load ML models (%s) — using HeuristicClassifier fallback.",
                exc,
            )
            self._init_fallback()

    def _init_fallback(self) -> None:
        from app.classifier import HeuristicClassifier
        self._fallback = HeuristicClassifier()

    @property
    def using_ml(self) -> bool:
        """True when the trained ML models are active; False when on fallback."""
        return self._task_model is not None

    def classify(self, prompt: str) -> ClassificationResult:
        if self._fallback is not None:
            return self._fallback.classify(prompt)

        task_type = str(self._task_model.predict([prompt])[0])

        raw_complexity = float(self._complexity_model.predict([prompt])[0])
        complexity_score = round(max(0.0, min(1.0, raw_complexity)), 4)

        logger.debug(
            "ML classified | task_type=%s complexity=%.2f",
            task_type,
            complexity_score,
        )
        return ClassificationResult(task_type=task_type, complexity_score=complexity_score)
