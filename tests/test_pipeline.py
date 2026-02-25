"""
Pipeline integration test — sends 5 diverse prompts through POST /route
and prints a formatted cost-comparison report.

Usage (server must be running):
    python tests/test_pipeline.py [--base-url http://localhost:8000]

The script can also be run stand-alone without a live server by calling the
FastAPI app directly via httpx's ASGI transport (default when --base-url is
omitted and the env var LLMROUTER_DIRECT=1 is set or the flag --direct is
passed).
"""

import argparse
import json
import pathlib
import sys
import textwrap
import time
from dataclasses import dataclass

# Ensure the project root is on sys.path so `app.*` imports work when the
# script is run directly (python tests/test_pipeline.py) rather than as a
# module (python -m tests.test_pipeline).
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

OPUS_ID = "claude-opus-4-6"

# Opus pricing (mirrors config.py — kept local so the script is self-contained)
OPUS_INPUT_PER_TOKEN  = 15.0  / 1_000_000
OPUS_OUTPUT_PER_TOKEN = 75.0  / 1_000_000


@dataclass
class TestCase:
    label: str
    prompt: str
    policy: dict
    max_tokens: int = 1024


TEST_CASES: list[TestCase] = [
    TestCase(
        label="1 · Simple Q&A",
        prompt="What is the capital of France?",
        policy={"strategy": "minimize_cost"},
    ),
    TestCase(
        label="2 · Code Generation",
        prompt=(
            "Write a Python function to merge two sorted arrays into a single "
            "sorted array. Include type hints, a docstring, and a brief example "
            "in the docstring showing input/output."
        ),
        policy={"strategy": "balanced"},
    ),
    TestCase(
        label="3 · Complex Math",
        prompt=(
            "Prove that the sum of 1/n² from n=1 to infinity converges to π²/6 "
            "(the Basel problem). Walk through the proof step by step, explaining "
            "the key ideas and any non-obvious steps."
        ),
        policy={"strategy": "maximize_quality"},
        max_tokens=2048,
    ),
    TestCase(
        label="4 · Summarization",
        prompt=(
            "Summarize the following passage in 3 bullet points:\n\n"
            "Large language models (LLMs) have transformed natural language processing "
            "by enabling a single pretrained model to perform well across a wide range "
            "of tasks with little or no task-specific training. Models like GPT-4 and "
            "Claude demonstrate emergent capabilities such as in-context learning, "
            "chain-of-thought reasoning, and instruction following. However, they also "
            "raise concerns around factual accuracy, bias, and the high computational "
            "cost of both training and inference. Researchers are actively exploring "
            "techniques such as RLHF, constitutional AI, and retrieval-augmented "
            "generation to address these shortcomings."
        ),
        policy={"strategy": "minimize_cost"},
    ),
    TestCase(
        label="5 · Creative Writing",
        prompt=(
            "Write a short story (around 200 words) about an astronaut who discovers "
            "an ancient library floating silently in deep space. Focus on atmosphere "
            "and the astronaut's sense of wonder."
        ),
        policy={"strategy": "balanced"},
    ),
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"
MAGENTA = "\033[35m"

MODEL_COLORS = {
    "claude-haiku-4-5-20251001":   GREEN,
    "claude-sonnet-4-5-20250929":  CYAN,
    "claude-opus-4-6":             MAGENTA,
}

def colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"

def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"

def bar(filled: int, total: int = 30, char: str = "█", empty: str = "░") -> str:
    n = round(filled / 100 * total)
    return char * n + empty * (total - n)

def model_label(model_id: str) -> str:
    color = MODEL_COLORS.get(model_id, "")
    short = {
        "claude-haiku-4-5-20251001":  "Haiku 4.5",
        "claude-sonnet-4-5-20250929": "Sonnet 4.5",
        "claude-opus-4-6":            "Opus 4.6",
    }.get(model_id, model_id)
    return colored(short, color)

def wrap(text: str, width: int = 72, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)

def separator(char: str = "─", width: int = 76) -> str:
    return dim(char * width)

# ---------------------------------------------------------------------------
# Request helper
# ---------------------------------------------------------------------------

def call_route(client: httpx.Client, tc: TestCase) -> dict:
    payload = {
        "prompt": tc.prompt,
        "policy": tc.policy,
        "max_tokens": tc.max_tokens,
    }
    t0 = time.perf_counter()
    resp = client.post("/route", json=payload, timeout=120)
    wall_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    data["_wall_ms"] = wall_ms
    return data

# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

def print_result(tc: TestCase, data: dict, opus_cost: float) -> None:
    savings = opus_cost - data["cost_usd"]
    savings_pct = (savings / opus_cost * 100) if opus_cost > 0 else 0.0
    complexity_pct = int(data["complexity_score"] * 100)

    print()
    print(separator("═"))
    print(bold(f"  {tc.label}"))
    print(separator())

    # Prompt (truncated to 2 lines)
    prompt_preview = tc.prompt.replace("\n", " ")
    if len(prompt_preview) > 90:
        prompt_preview = prompt_preview[:87] + "…"
    print(f"  {dim('Prompt:')} {prompt_preview}")
    print(f"  {dim('Policy:')} {tc.policy['strategy']}")
    print()

    # Classification row
    task_color = {
        "simple_qa":       GREEN,
        "summarization":   CYAN,
        "code_generation": BLUE,
        "creative_writing": MAGENTA,
        "math_reasoning":  YELLOW,
        "analysis":        "",
    }.get(data["task_type"], "")
    task_label = colored(data["task_type"], task_color)
    complexity_bar = bar(complexity_pct, total=20)
    print(f"  Task type  : {task_label}")
    print(f"  Complexity : [{complexity_bar}] {complexity_pct}%")
    print()

    # Model + tokens
    print(f"  Model used : {model_label(data['model_used'])}")
    print(f"  Tokens     : {data['input_tokens']:,} in  /  {data['output_tokens']:,} out")
    print(f"  Latency    : {data['latency_ms']:.0f} ms  (wall: {data['_wall_ms']:.0f} ms)")
    print()

    # Cost comparison
    actual_cost   = data["cost_usd"]
    opus_str      = colored(f"${opus_cost:.6f}", RED)
    actual_str    = colored(f"${actual_cost:.6f}", GREEN)
    savings_str   = colored(f"${savings:.6f}  ({savings_pct:.1f}% cheaper)", GREEN)
    print(f"  Cost (routed)  : {actual_str}")
    print(f"  Cost (opus)    : {opus_str}")
    print(f"  Savings        : {savings_str}")
    print()

    # Routing reasoning
    print(f"  {dim('Routing reasoning:')}")
    print(wrap(data["routing_reasoning"], indent="    "))
    print()

    # Response preview (first 300 chars)
    preview = data["response_text"].strip().replace("\n", " ")
    if len(preview) > 300:
        preview = preview[:297] + "…"
    print(f"  {dim('Response preview:')}")
    print(wrap(preview, indent="    "))


def print_summary(results: list[tuple[TestCase, dict, float]]) -> None:
    print()
    print(separator("═"))
    print(bold("  COST SUMMARY"))
    print(separator("═"))
    print()

    col = [34, 14, 14, 14, 14, 10]
    header = (
        f"  {'Prompt':<{col[0]}}"
        f"{'Model':>{col[1]}}"
        f"{'Tokens out':>{col[2]}}"
        f"{'Actual $':>{col[3]}}"
        f"{'Opus $':>{col[4]}}"
        f"{'Saved':>{col[5]}}"
    )
    print(bold(header))
    print("  " + separator("─", width=sum(col)))

    total_actual = 0.0
    total_opus   = 0.0

    for tc, data, opus_cost in results:
        actual = data["cost_usd"]
        savings = opus_cost - actual
        savings_pct = savings / opus_cost * 100 if opus_cost else 0

        # Truncate label for table
        label = tc.label[:col[0]-1].ljust(col[0]-1)
        mname = {
            "claude-haiku-4-5-20251001":  "Haiku",
            "claude-sonnet-4-5-20250929": "Sonnet",
            "claude-opus-4-6":            "Opus",
        }.get(data["model_used"], data["model_used"])
        mcolor = MODEL_COLORS.get(data["model_used"], "")

        print(
            f"  {label} "
            f"{colored(mname, mcolor):>{col[1] + len(mcolor) + len(RESET)}}"
            f"{data['output_tokens']:>{col[2]},}"
            f"  {colored(f'${actual:.6f}', GREEN):>{col[3] + len(GREEN) + len(RESET)}}"
            f"  {colored(f'${opus_cost:.6f}', RED):>{col[4] + len(RED) + len(RESET)}}"
            f"  {colored(f'{savings_pct:.0f}%', YELLOW):>{col[5] + len(YELLOW) + len(RESET)}}"
        )
        total_actual += actual
        total_opus   += opus_cost

    total_savings = total_opus - total_actual
    total_pct     = total_savings / total_opus * 100 if total_opus else 0

    print("  " + separator("─", width=sum(col)))
    print(
        f"  {'TOTAL':<{col[0]}} "
        f"{'':>{col[1]}}"
        f"{'':>{col[2]}}"
        f"  {colored(f'${total_actual:.6f}', GREEN):>{col[3] + len(GREEN) + len(RESET)}}"
        f"  {colored(f'${total_opus:.6f}', RED):>{col[4] + len(RED) + len(RESET)}}"
        f"  {colored(f'{total_pct:.0f}%', YELLOW):>{col[5] + len(YELLOW) + len(RESET)}}"
    )
    print()

    # Big savings callout
    savings_banner = (
        f"  Total saved vs always-Opus:  "
        f"{bold(colored(f'${total_savings:.6f}', GREEN))}  "
        f"({bold(colored(f'{total_pct:.1f}%', GREEN))} reduction)"
    )
    print(savings_banner)
    print()
    print(separator("═"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="llm-router pipeline test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running llm-router server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Call the FastAPI app in-process via ASGI transport (no server needed)",
    )
    args = parser.parse_args()

    # Transport: ephemeral uvicorn thread or real HTTP server
    if args.direct:
        import socket
        import threading
        import uvicorn
        from app.main import app as fastapi_app

        # Pick a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        base_url = f"http://127.0.0.1:{port}"
        cfg = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(cfg)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait until the server is ready (up to 5 s)
        for _ in range(50):
            time.sleep(0.1)
            try:
                httpx.get(f"{base_url}/health", timeout=1)
                break
            except Exception:
                pass
        else:
            print(colored("  ERROR: in-process server did not start in time", RED))
            sys.exit(1)

        print(dim(f"  [mode: in-process uvicorn → {base_url}]"))
    else:
        server = None
        base_url = args.base_url.rstrip("/")
        print(dim(f"  [mode: HTTP → {base_url}]"))

    print()
    print(bold("  llm-router · Pipeline Test"))
    print(bold("  5 prompts · 4 routing strategies · cost vs always-Opus"))
    print()

    client_kwargs = dict(base_url=base_url)

    results: list[tuple[TestCase, dict, float]] = []
    failed = 0

    with httpx.Client(**client_kwargs) as client:
        # Health check
        try:
            hc = client.get("/health", timeout=5)
            hc_data = hc.json()
            if hc_data.get("status") != "ok":
                print(colored("  WARNING: health check did not return ok", YELLOW))
            if not hc_data.get("api_key_configured"):
                print(colored("  ERROR: ANTHROPIC_API_KEY not configured on server", RED))
                sys.exit(1)
        except Exception as exc:
            print(colored(f"  ERROR: could not reach server at {base_url} — {exc}", RED))
            print(colored("  Start the server with:  uvicorn app.main:app --reload", YELLOW))
            sys.exit(1)

        for i, tc in enumerate(TEST_CASES, 1):
            print(f"  [{i}/{len(TEST_CASES)}] Running {tc.label} …", end="", flush=True)
            try:
                data = call_route(client, tc)

                # Compute what Opus would have cost for the same token counts
                opus_cost = (
                    data["input_tokens"]  * OPUS_INPUT_PER_TOKEN
                    + data["output_tokens"] * OPUS_OUTPUT_PER_TOKEN
                )

                results.append((tc, data, opus_cost))
                print(colored(" done", GREEN))
            except httpx.HTTPStatusError as exc:
                print(colored(f" FAILED ({exc.response.status_code})", RED))
                print(colored(f"    {exc.response.text[:200]}", RED))
                failed += 1
            except Exception as exc:
                print(colored(f" FAILED ({exc})", RED))
                failed += 1

    # Detailed results
    for tc, data, opus_cost in results:
        print_result(tc, data, opus_cost)

    # Summary table
    if results:
        print_summary(results)

    # Shut down the in-process server if we started one
    if args.direct and server is not None:
        server.should_exit = True

    if failed:
        print(colored(f"  {failed} test(s) failed.", RED))
        sys.exit(1)


if __name__ == "__main__":
    main()
