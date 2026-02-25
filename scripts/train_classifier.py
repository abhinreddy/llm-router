#!/usr/bin/env python3
"""
Train ML classifiers for task_type and complexity.

Models trained:
  - Task type:   TF-IDF (1–2 grams) + LogisticRegression  (multi-class)
  - Complexity:  TF-IDF (1–2 grams) + Ridge               (regression)

Both models are wrapped in sklearn Pipelines so vectorizer + model are
saved together, making inference a single pipeline.predict() call.

Usage:
    python scripts/train_classifier.py

Prerequisites:
    python scripts/generate_dataset.py  (creates data/training_data.json)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "training_data.json"
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> tuple[list[str], list[str], list[float]]:
    with open(path) as f:
        data = json.load(f)

    prompts = [d["prompt"] for d in data]
    task_types = [d["task_type"] for d in data]
    complexities = [float(d["complexity"]) for d in data]
    return prompts, task_types, complexities


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_task_classifier(
    X_train: list[str],
    X_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> Pipeline:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print("Task Type Classifier  (TF-IDF + LogisticRegression)")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    return pipeline


def train_complexity_model(
    X_train: list[str],
    X_test: list[str],
    y_train: list[float],
    y_test: list[float],
) -> Pipeline:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("reg", Ridge(alpha=1.0)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred_raw = pipeline.predict(X_test)
    y_pred = np.clip(y_pred_raw, 0.0, 1.0)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print("Complexity Model  (TF-IDF + Ridge Regression)")
    print(f"{'='*60}")
    print(f"  MAE : {mae:.4f}  (mean absolute error, lower is better)")
    print(f"  R²  : {r2:.4f}  (1.0 = perfect, 0.0 = baseline mean)")

    # Bucket accuracy: how often does the prediction fall in the right
    # low/medium/high bucket?
    def bucket(v: float) -> str:
        if v < 0.33:
            return "low"
        elif v < 0.66:
            return "medium"
        return "high"

    buckets_true = [bucket(v) for v in y_test]
    buckets_pred = [bucket(v) for v in y_pred]
    bucket_acc = sum(t == p for t, p in zip(buckets_true, buckets_pred)) / len(y_test)
    print(f"  Bucket accuracy (low/medium/high): {bucket_acc:.4f}  ({bucket_acc * 100:.1f}%)")

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_PATH.exists():
        print(f"Error: dataset not found at {DATA_PATH}")
        print("Run `python scripts/generate_dataset.py` first.")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading dataset from {DATA_PATH} ...")
    prompts, task_types, complexities = load_dataset(DATA_PATH)
    print(f"Loaded {len(prompts)} samples")

    # --- Stratified 80/20 train/test split ---
    # Use index split so all three arrays stay aligned
    indices = list(range(len(prompts)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=task_types,
    )

    X_train = [prompts[i] for i in train_idx]
    X_test  = [prompts[i] for i in test_idx]
    yt_train = [task_types[i]   for i in train_idx]
    yt_test  = [task_types[i]   for i in test_idx]
    yc_train = [complexities[i] for i in train_idx]
    yc_test  = [complexities[i] for i in test_idx]

    print(f"Split: {len(X_train)} train / {len(X_test)} test")

    # --- Train ---
    task_model       = train_task_classifier(X_train, X_test, yt_train, yt_test)
    complexity_model = train_complexity_model(X_train, X_test, yc_train, yc_test)

    # --- Save ---
    task_path       = MODELS_DIR / "task_classifier.joblib"
    complexity_path = MODELS_DIR / "complexity_model.joblib"

    joblib.dump(task_model,       task_path)
    joblib.dump(complexity_model, complexity_path)

    print(f"\nSaved models:")
    print(f"  {task_path}")
    print(f"  {complexity_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
