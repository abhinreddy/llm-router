#!/usr/bin/env python3
"""
Generate a synthetic training dataset for the LLM router classifier.

Creates ~800 diverse prompts across 6 task types with low/medium/high complexity
using the Anthropic API (claude-haiku-4-5 for cost efficiency).

Output: data/training_data.json
  Fields: prompt (str), task_type (str), complexity (float 0.0–1.0)
"""

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_TYPES = [
    "simple_qa",
    "summarization",
    "code_generation",
    "creative_writing",
    "math_reasoning",
    "analysis",
]

# (label, center_value, jitter_range)
COMPLEXITY_LEVELS = [
    ("low",    0.15, 0.08),
    ("medium", 0.50, 0.10),
    ("high",   0.85, 0.08),
]

# 2 batches × 22 prompts × 6 types × 3 levels = 792 ≈ 800
BATCHES_PER_COMBO = 2
PROMPTS_PER_BATCH = 22

TASK_DESCRIPTIONS = {
    "simple_qa": (
        "direct, factual question-and-answer queries — what, who, where, when, why, how "
        "questions about facts, definitions, or straightforward explanations"
    ),
    "summarization": (
        "requests to summarize, condense, or extract key points from text, articles, "
        "documents, books, or concepts; includes TL;DR and overview requests"
    ),
    "code_generation": (
        "requests to write, implement, debug, refactor, review, or explain code in any "
        "programming language; includes API design, data structures, algorithms, scripts"
    ),
    "creative_writing": (
        "creative tasks like writing stories, poems, dialogues, scripts, song lyrics, "
        "world-building, character creation, or other imaginative/artistic content"
    ),
    "math_reasoning": (
        "mathematical problems, proofs, calculations, statistics, probability, algebra, "
        "calculus, geometry, number theory, or formal logical reasoning"
    ),
    "analysis": (
        "analytical tasks: comparing options, evaluating arguments, explaining complex "
        "concepts, reviewing work, diagnosing problems, or providing structured assessments"
    ),
}

COMPLEXITY_DESCRIPTIONS = {
    "low": (
        "simple, brief — typically 1 short sentence, single direct question or request, "
        "no multiple requirements, no multi-part structure, everyday vocabulary"
    ),
    "medium": (
        "moderately complex — 2–4 sentences, may have 2–3 requirements or sub-parts, "
        "some specificity, occasional domain terminology"
    ),
    "high": (
        "complex and detailed — long multi-part requests, many specific constraints, "
        "technical depth, multiple sub-questions, numbered requirements, or advanced topics"
    ),
}


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def build_generation_prompt(
    task_type: str,
    complexity_label: str,
    batch_num: int,
    count: int,
) -> str:
    task_desc = TASK_DESCRIPTIONS[task_type]
    complexity_desc = COMPLEXITY_DESCRIPTIONS[complexity_label]

    variety_hint = (
        "Use unusual topics, niche domains, edge cases, or uncommon phrasing."
        if batch_num == 2
        else "Use common, everyday topics with varied styles (formal, casual, technical, non-technical)."
    )

    return f"""Generate {count} diverse, realistic user prompts for this category:

Task type: **{task_type}**
Description: {task_desc}

Complexity level: **{complexity_label}**
Description: {complexity_desc}

Rules:
- Every prompt must clearly belong to the "{task_type}" category
- Every prompt must have **{complexity_label}** complexity (match the length/detail description exactly)
- {variety_hint}
- Vary topics across: technology, science, history, arts, business, everyday life, health, sports, etc.
- Do NOT include any meta-commentary, explanations, or labels — just the raw prompt text
- Make each prompt sound like something a real user would type

Return ONLY a valid JSON array of {count} strings, with no extra text before or after.
Example: ["prompt one", "prompt two", ...]"""


def generate_batch(
    client: anthropic.Anthropic,
    task_type: str,
    complexity_label: str,
    batch_num: int,
    retries: int = 3,
) -> list[str]:
    prompt_text = build_generation_prompt(task_type, complexity_label, batch_num, PROMPTS_PER_BATCH)

    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = response.content[0].text.strip()

            # Extract the JSON array from the response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON array in response: {text[:200]}")

            prompts = json.loads(text[start:end])
            valid = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
            return valid

        except Exception as exc:
            print(f"  Attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            if attempt < retries:
                time.sleep(2 ** attempt)

    return []  # Return empty on all failures rather than crashing the whole run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "training_data.json"

    dataset: list[dict] = []
    total_combos = len(TASK_TYPES) * len(COMPLEXITY_LEVELS) * BATCHES_PER_COMBO
    current = 0

    for task_type in TASK_TYPES:
        for complexity_label, center, jitter in COMPLEXITY_LEVELS:
            for batch_num in range(1, BATCHES_PER_COMBO + 1):
                current += 1
                print(
                    f"[{current:2d}/{total_combos}] {task_type:20s} / {complexity_label:6s} / batch {batch_num}",
                    end=" ... ",
                    flush=True,
                )

                prompts = generate_batch(client, task_type, complexity_label, batch_num)

                for prompt in prompts:
                    noise = random.uniform(-jitter, jitter)
                    complexity = round(max(0.05, min(0.95, center + noise)), 2)
                    dataset.append({
                        "prompt": prompt,
                        "task_type": task_type,
                        "complexity": complexity,
                    })

                print(f"got {len(prompts)} (total: {len(dataset)})", flush=True)

                # Brief pause to stay within rate limits
                time.sleep(0.4)

    random.shuffle(dataset)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(dataset)} prompts → {output_path}")

    # Distribution summary
    type_counts = Counter(d["task_type"] for d in dataset)
    print("\nTask-type distribution:")
    for task in TASK_TYPES:
        count = type_counts.get(task, 0)
        bar = "#" * (count // 5)
        print(f"  {task:22s} {count:4d}  {bar}")


if __name__ == "__main__":
    main()
