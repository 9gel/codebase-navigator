"""Automated evaluation harness for cn ask across multi-language test repositories."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

from codebase_navigator.ask import ask_codebase, load_llm_config

REPOS_DIR = Path(__file__).parent.parent / "eval" / "repos"

EVAL_TASKS = [
    {
        "repo": "flask",
        "question": "Where is the before_request hook dispatched during request handling in Flask?",
        "expected_keywords": [
            "preprocess_request",
            "full_dispatch_request",
            "before_request_funcs",
        ],
        "expected_files": ["src/flask/app.py", "flask/app.py", "app.py"],
    },
    {
        "repo": "httpx",
        "question": "How does httpx.Client determine which transport to dispatch a request to?",
        "expected_keywords": ["_transport_for_url", "HTTPTransport", "mounts"],
        "expected_files": ["httpx/_client.py", "httpx/_transports", "_client.py"],
    },
]


def run_eval(repo_name: str | None = None) -> bool:
    print("=" * 70)
    print("🎯 Running Codebase-Navigator Agent Evaluation Suite")
    print("=" * 70)

    if not REPOS_DIR.exists():
        print(f"⚠️  Evaluation repositories directory not found at: {REPOS_DIR}")
        return False

    all_passed = True
    for task in EVAL_TASKS:
        r_name = task["repo"]
        if repo_name and r_name != repo_name:
            continue

        repo_path = REPOS_DIR / r_name
        if not repo_path.is_dir():
            print(f"\n⏩ Skipping {r_name}: repo directory not found at {repo_path}")
            continue

        print(f"\n📁 Evaluating Repository: {r_name}")
        print(f"❓ Question: {task['question']}")

        config = load_llm_config(folder=repo_path)
        if not config.api_key:
            print(
                "⚠️  No API key found in environment (OPENROUTER_API_KEY/CN_API_KEY). Skipping live LLM call."
            )
            continue

        t0 = time.time()
        try:
            answer = ask_codebase(
                folder=repo_path,
                question=task["question"],
                config=config,
                verbose=True,
                output_stream=sys.stdout,
            )
            dt = time.time() - t0

            print(f"\n⏱️ Completed in {dt:.2f}s")
            print("-" * 50)
            print(answer[:500] + ("..." if len(answer) > 500 else ""))
            print("-" * 50)

            # Verification assertions
            kw_hits = [kw for kw in task["expected_keywords"] if kw.lower() in answer.lower()]
            file_hits = [f for f in task["expected_files"] if f.lower() in answer.lower()]

            print(f"✓ Keyword Hits: {kw_hits} / {task['expected_keywords']}")
            print(f"✓ File Reference Hits: {file_hits} / {task['expected_files']}")

            if len(kw_hits) > 0 and len(file_hits) > 0:
                print(f"✅ PASSED evaluation for {r_name}")
            else:
                print(f"❌ FAILED evaluation for {r_name} (insufficient grounding hits)")
                all_passed = False

        except Exception as e:
            print(f"❌ Error during evaluation for {r_name}: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(target)
