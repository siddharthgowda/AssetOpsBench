"""Run plan-execute scenarios and dump per-step timings.

Run::

    uv run python -m servers.battery.profiling.profile_scenario --scenarios 5
    uv run python -m servers.battery.profiling.profile_scenario --scenarios 1,2,4,5,6,7,8
    uv run python -m servers.battery.profiling.profile_scenario \\
        --question "Predict RUL for B0005, B0006, B0018"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_BATTERY_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BATTERY_DIR.parent.parent.parent
_DEFAULT_PROFILES_DIR = _BATTERY_DIR / "profiles"
_DEFAULT_MODEL_ID = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

load_dotenv(_REPO_ROOT / ".env")


async def run_scenario(question: str, model_id: str) -> dict:
    from agent.plan_execute.runner import PlanExecuteRunner
    from llm.litellm import LiteLLMBackend

    runner = PlanExecuteRunner(llm=LiteLLMBackend(model_id))
    r = await runner.run(question)
    return {
        "question": question,
        "total_s": round(r.total_duration_s, 3),
        "discovery_s": round(r.discovery_duration_s, 3),
        "planner_s": round(r.planning_duration_s, 3),
        "summary_s": round(r.summarization_duration_s, 3),
        "steps": [
            {
                "tool": h.tool or "",
                "step_s": round(h.duration_s or 0.0, 3),
                "tool_s": round(h.tool_call_duration_s or 0.0, 3),
            }
            for h in r.history
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scenarios",
        help="Comma-separated 0-indexed scenarios from "
        "src/scenarios/local/battery_utterances.json (e.g. '1,2,5').",
    )
    ap.add_argument("--question", help="Free-form query (mutually exclusive with --scenarios).")
    ap.add_argument("--model-id", default=_DEFAULT_MODEL_ID)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_PROFILES_DIR,
        help="Directory to write JSON outputs into (default: profiles/).",
    )
    args = ap.parse_args()

    if (args.scenarios is None) == (args.question is None):
        ap.error("provide exactly one of --scenarios or --question")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    if args.question:
        jobs = [(None, args.question)]
    else:
        sc_path = _REPO_ROOT / "src" / "scenarios" / "local" / "battery_utterances.json"
        scenarios = json.loads(sc_path.read_text(encoding="utf-8"))
        indices = [int(x.strip()) for x in args.scenarios.split(",") if x.strip()]
        jobs = [(n, scenarios[n]["query"]) for n in indices]

    run_dir = args.output_dir / f"scenarios_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[int | None, str]] = []
    for n, question in jobs:
        label = f"scenario {n}" if n is not None else "question"
        try:
            out = asyncio.run(run_scenario(question, args.model_id))
        except Exception as e:  # noqa: BLE001
            print(f"{label}: FAILED ({type(e).__name__}: {str(e)[:160]})")
            failures.append((n, f"{type(e).__name__}: {e}"))
            continue
        fname = f"scenario_{n}.json" if n is not None else "scenario.json"
        out_path = run_dir / fname
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"{label}: wrote {out_path}  total={out['total_s']}s  steps={len(out['steps'])}")

    if failures:
        print(f"\n{len(failures)} run(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
