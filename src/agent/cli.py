"""CLI entry point for the plan-execute runner.

Usage:
    plan-execute "What assets are available at site MAIN?"
    plan-execute --model-id watsonx/ibm/granite-3-3-8b-instruct --show-plan "List sensors"
    plan-execute --model-id litellm_proxy/GCP/claude-4-sonnet "What are the failure modes?"
    plan-execute --json "What is the current time?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from agent.models import OrchestratorResult

_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plan-execute",
        description="Run a question through the MCP plan-execute workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
model-id format:
  The provider is encoded in the model-id prefix:
    watsonx/<model>          IBM WatsonX  (e.g. watsonx/meta-llama/llama-3-3-70b-instruct)
    litellm_proxy/<model>    LiteLLM proxy (e.g. litellm_proxy/GCP/claude-4-sonnet)

environment variables:
  WATSONX_APIKEY        IBM WatsonX API key      (required for watsonx/* models)
  WATSONX_PROJECT_ID    IBM WatsonX project ID   (required for watsonx/* models)
  WATSONX_URL           IBM WatsonX endpoint     (optional, defaults to us-south)

  LITELLM_API_KEY       LiteLLM API key          (required for non-watsonx models)
  LITELLM_BASE_URL      LiteLLM base URL         (required for non-watsonx models)

  LOG_LEVEL             Log level for MCP servers (default: WARNING)

examples:
  plan-execute "What assets are at site MAIN?"
  plan-execute --model-id watsonx/ibm/granite-3-3-8b-instruct --show-plan "List sensors"
  plan-execute --model-id litellm_proxy/GCP/claude-4-sonnet "What are the failure modes?"
  plan-execute --verbose --show-history --json "How many IoT observations exist for CH-1?"
  plan-execute --scenarios -o results.txt
  plan-execute --scenarios custom_scenarios.json -o out.txt
""",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="The question to answer (not used with --scenarios).",
    )
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=f"litellm model string with provider prefix (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--server",
        action="append",
        metavar="NAME=PATH",
        dest="servers",
        default=[],
        help=(
            "Register an MCP server as NAME=PATH. "
            "Overrides the default servers. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated plan before execution.",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Print each step result after execution.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output the full result (answer, plan, history) as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO-level progress logs on stderr (default: WARNING+ only).",
    )
    parser.add_argument(
        "--scenarios",
        metavar="FILE",
        nargs="?",
        const="src/scenarios/local/battery_utterances.json",
        default=None,
        help=(
            "Run every scenario in a JSON array (objects with a 'query' field). "
            "Default FILE is src/scenarios/local/battery_utterances.json. "
            "Writes the combined report to --output."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.txt"),
        help="Output path for --scenarios (default: results.txt).",
    )
    return parser


def _setup_logging(verbose: bool) -> None:
    """Configure root logger to stderr; level depends on --verbose."""
    level = logging.INFO if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


def _build_llm(model_id: str):
    """Instantiate the LiteLLMBackend for the given model_id."""
    try:
        from llm.litellm import LiteLLMBackend
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        return LiteLLMBackend(model_id=model_id)
    except KeyError as exc:
        print(f"error: missing environment variable {exc}", file=sys.stderr)
        sys.exit(1)


def _parse_servers(entries: list[str]) -> dict[str, Path] | None:
    """Parse NAME=PATH pairs into a server_paths dict, or None if empty."""
    if not entries:
        return None
    result: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            print(
                f"error: --server requires NAME=PATH format, got: {entry!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        name, _, path = entry.partition("=")
        result[name.strip()] = Path(path.strip())
    return result


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _section_lines(title: str) -> list[str]:
    bar = "─" * 60
    return ["", bar, f"  {title}", bar]


def _render_run_text(
    result: OrchestratorResult,
    *,
    show_plan: bool,
    show_history: bool,
) -> str:
    """Human-readable plan / history / answer (same shape as CLI output)."""
    lines: list[str] = []
    if show_plan:
        lines.extend(_section_lines("Plan"))
        for step in result.plan.steps:
            deps = ", ".join(f"#{d}" for d in step.dependencies) or "none"
            lines.append(f"  [{step.step_number}] {step.server}: {step.task}")
            lines.append(f"       tool: {step.tool}  args: {step.tool_args}")
            lines.append(f"       deps={deps} | expected: {step.expected_output}")

    if show_history:
        lines.extend(_section_lines("Execution History"))
        for r in result.history:
            status = "OK " if r.success else "ERR"
            lines.append(f"  [{status}] Step {r.step_number} ({r.server}): {r.task}")
            if r.tool and r.tool.lower() not in ("none", "null", ""):
                lines.append(f"       tool: {r.tool}  args: {r.tool_args}")
            detail = r.response if r.success else f"Error: {r.error}"
            lines.append(f"        {detail}")

    lines.extend(_section_lines("Answer"))
    lines.append(result.answer)
    lines.append("")
    return "\n".join(lines)


def _load_scenarios(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit(f"error: {path} must be a JSON array of scenario objects")
    out: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"error: {path} item {i} must be an object")
        q = item.get("query")
        if not q or not isinstance(q, str):
            raise SystemExit(f"error: {path} item {i} missing string 'query'")
        out.append(item)
    return out


async def _run_scenarios(args: argparse.Namespace) -> None:
    from agent.plan_execute.runner import PlanExecuteRunner

    scenario_path = Path(args.scenarios).expanduser().resolve()
    if not scenario_path.is_file():
        raise SystemExit(f"error: scenarios file not found: {scenario_path}")

    scenarios = _load_scenarios(scenario_path)
    out_path: Path = args.output.expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    llm = _build_llm(args.model_id)
    server_paths = _parse_servers(args.servers)
    runner = PlanExecuteRunner(llm=llm, server_paths=server_paths)

    header_lines: list[str] = [
        f"Scenario batch run",
        f"Source: {scenario_path}",
        f"Scenarios: {len(scenarios)}",
        f"Model: {args.model_id}",
        "",
    ]

    # Write per-scenario: flush the file after every scenario completes (or errors).
    # If one scenario crashes, the previous ones remain on disk and the next starts fresh.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")

    completed = 0
    for item in scenarios:
        sid = item.get("id", "?")
        persona = item.get("persona", "")
        query = item["query"]
        banner = f"\n{'#' * 60}\nScenario {sid}: {persona}\n{'#' * 60}\n"

        print(f"[plan-execute] Scenario {sid}/{len(scenarios)} …", file=sys.stderr, flush=True)
        chunk: list[str] = [banner, "Query:", query, ""]
        try:
            result = await runner.run(query)
            chunk.append(
                _render_run_text(
                    result,
                    show_plan=True,
                    show_history=True,
                )
            )
            completed += 1
        except Exception as exc:  # noqa: BLE001
            # Record the failure and continue so one bad scenario does not lose the rest.
            chunk.append("---")
            chunk.append(f"SCENARIO {sid} FAILED: {type(exc).__name__}: {exc}")
            chunk.append("---")
            print(
                f"[plan-execute] Scenario {sid} crashed ({type(exc).__name__}); continuing",
                file=sys.stderr,
                flush=True,
            )

        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")

    print(
        f"Wrote {completed}/{len(scenarios)} scenario(s) to {out_path}",
        file=sys.stderr,
    )


async def _run(args: argparse.Namespace) -> None:
    from agent.plan_execute.runner import PlanExecuteRunner

    llm = _build_llm(args.model_id)
    server_paths = _parse_servers(args.servers)
    runner = PlanExecuteRunner(llm=llm, server_paths=server_paths)
    result = await runner.run(args.question)

    if args.output_json:
        output = {
            "question": result.question,
            "answer": result.answer,
            "plan": [
                {
                    "step": s.step_number,
                    "task": s.task,
                    "server": s.server,
                    "tool": s.tool,
                    "tool_args": s.tool_args,
                    "dependencies": s.dependencies,
                    "expected_output": s.expected_output,
                }
                for s in result.plan.steps
            ],
            "history": [
                {
                    "step": r.step_number,
                    "task": r.task,
                    "server": r.server,
                    "tool": r.tool,
                    "tool_args": r.tool_args,
                    "response": r.response,
                    "error": r.error,
                    "success": r.success,
                }
                for r in result.history
            ],
        }
        print(json.dumps(output, indent=2))
        return

    if args.show_plan:
        _print_section("Plan")
        for step in result.plan.steps:
            deps = ", ".join(f"#{d}" for d in step.dependencies) or "none"
            print(f"  [{step.step_number}] {step.server}: {step.task}")
            print(f"       tool: {step.tool}  args: {step.tool_args}")
            print(f"       deps={deps} | expected: {step.expected_output}")

    if args.show_history:
        _print_section("Execution History")
        for r in result.history:
            status = "OK " if r.success else "ERR"
            print(f"  [{status}] Step {r.step_number} ({r.server}): {r.task}")
            if r.tool and r.tool.lower() not in ("none", "null", ""):
                print(f"       tool: {r.tool}  args: {r.tool_args}")
            detail = r.response if r.success else f"Error: {r.error}"
            snippet = detail[:200] + ("..." if len(detail) > 200 else "")
            print(f"        {snippet}")

    _print_section("Answer")
    print(result.answer)
    print()

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = _build_parser().parse_args()
    _setup_logging(args.verbose)
    if args.scenarios:
        asyncio.run(_run_scenarios(args))
    elif args.question is None:
        _build_parser().error(
            "question is required unless --scenarios is set "
            "(e.g. plan-execute --scenarios -o results.txt)"
        )
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
