#!/usr/bin/env python3
"""Comprehensive scenario profiler — instruments LLM and MCP tool calls in-process.

Runs scenarios via PlanExecuteRunner directly (not as a subprocess) so we can
monkey-patch:
  - LiteLLMBackend.generate     → time every LLM API call
  - executor._call_tool          → time every MCP tool call

Plus background psutil thread sampling RSS every 250ms.

Outputs (under profiles/scenario_detailed_<timestamp>/):
  scenario_<n>.json   raw event data + summary
  scenario_<n>.md     readable per-scenario report
  combined.md         cross-scenario aggregate + bottleneck analysis

Usage:
    python scripts/profile_scenarios_detailed.py
    python scripts/profile_scenarios_detailed.py --scenarios 0,1,3 \\
        --cells B0005,B0006,B0018 --timeout 360
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import psutil  # type: ignore[import-untyped]
from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

_DEFAULT_SCENARIOS = "0,1,3"
_DEFAULT_CELLS = "B0005,B0006,B0018"
_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


# ─────────────────────────────────────────────────────────────────────────────
# Profiler state
# ─────────────────────────────────────────────────────────────────────────────


class ScenarioProfiler:
    """In-process profiler. Records every LLM call, MCP tool call, RSS sample."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.llm_events: list[dict[str, Any]] = []
        self.tool_events: list[dict[str, Any]] = []
        self.memory_samples: list[dict[str, Any]] = []
        self.phase_markers: list[dict[str, Any]] = []
        self._stop_sampling = threading.Event()
        self._sampler_thread: threading.Thread | None = None

    def reset(self) -> None:
        self.start_time = time.perf_counter()
        self.llm_events.clear()
        self.tool_events.clear()
        self.memory_samples.clear()
        self.phase_markers.clear()
        self._stop_sampling.clear()

    def now(self) -> float:
        return time.perf_counter() - self.start_time

    def mark_phase(self, name: str) -> None:
        self.phase_markers.append({"name": name, "t_s": round(self.now(), 4)})

    def start_memory_sampler(self) -> None:
        self._sampler_thread = threading.Thread(
            target=self._memory_loop, daemon=True
        )
        self._sampler_thread.start()

    def stop_memory_sampler(self) -> None:
        self._stop_sampling.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)

    def _memory_loop(self) -> None:
        proc = psutil.Process()
        while not self._stop_sampling.is_set():
            try:
                rss_mb = proc.memory_info().rss / 1e6
            except Exception:  # noqa: BLE001
                rss_mb = 0.0
            self.memory_samples.append(
                {"t_s": round(self.now(), 3), "rss_mb": round(rss_mb, 1)}
            )
            self._stop_sampling.wait(0.25)


_PROFILER = ScenarioProfiler()


# ─────────────────────────────────────────────────────────────────────────────
# Patches
# ─────────────────────────────────────────────────────────────────────────────


def install_patches() -> None:
    """Monkey-patch LLM backend + MCP _call_tool to record timing."""
    from llm.litellm import LiteLLMBackend
    from agent.plan_execute import executor as _exec_mod

    original_generate = LiteLLMBackend.generate

    def timed_generate(self: LiteLLMBackend, prompt: str, temperature: float = 0.0) -> str:
        t_start = _PROFILER.now()
        t0 = time.perf_counter()
        result = original_generate(self, prompt, temperature)
        wall_s = time.perf_counter() - t0
        _PROFILER.llm_events.append(
            {
                "t_start_s": round(t_start, 4),
                "wall_s": round(wall_s, 4),
                "prompt_chars": len(prompt),
                "response_chars": len(result),
                "temperature": temperature,
            }
        )
        return result

    LiteLLMBackend.generate = timed_generate

    original_call_tool = _exec_mod._call_tool

    async def timed_call_tool(server_path: Any, tool_name: str, args: dict) -> str:
        t_start = _PROFILER.now()
        t0 = time.perf_counter()
        args_repr = json.dumps(args, default=str)[:200]
        try:
            result = await original_call_tool(server_path, tool_name, args)
        except Exception as e:  # noqa: BLE001
            wall_s = time.perf_counter() - t0
            _PROFILER.tool_events.append(
                {
                    "t_start_s": round(t_start, 4),
                    "t_end_s": round(t_start + wall_s, 4),
                    "wall_s": round(wall_s, 4),
                    "server_path": str(server_path),
                    "tool_name": tool_name,
                    "args_preview": args_repr,
                    "args_chars": len(args_repr),
                    "response_chars": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            raise
        wall_s = time.perf_counter() - t0
        _PROFILER.tool_events.append(
            {
                "t_start_s": round(t_start, 4),
                "t_end_s": round(t_start + wall_s, 4),
                "wall_s": round(wall_s, 4),
                "server_path": str(server_path),
                "tool_name": tool_name,
                "args_preview": args_repr,
                "args_chars": len(args_repr),
                "response_chars": len(result),
            }
        )
        return result

    _exec_mod._call_tool = timed_call_tool


# ─────────────────────────────────────────────────────────────────────────────
# Phase assignment (post-hoc)
# ─────────────────────────────────────────────────────────────────────────────


def assign_phases(events: list[dict], markers: list[dict]) -> None:
    """Mutate events to add a 'phase' field by bucketing on phase_markers."""
    if not markers:
        for e in events:
            e["phase"] = "unknown"
        return
    # markers are ordered by t_s ascending. Find the latest marker with t_s <= event.t_start_s.
    sorted_markers = sorted(markers, key=lambda m: m["t_s"])
    for e in events:
        ts = e["t_start_s"]
        phase = sorted_markers[0]["name"]
        for m in sorted_markers:
            if m["t_s"] <= ts:
                phase = m["name"]
            else:
                break
        e["phase"] = phase


# ─────────────────────────────────────────────────────────────────────────────
# Run a single scenario
# ─────────────────────────────────────────────────────────────────────────────


async def run_one_scenario(
    runner: Any, question: str, scenario_label: str, timeout_s: float
) -> dict[str, Any]:
    _PROFILER.reset()
    _PROFILER.mark_phase("init")
    _PROFILER.start_memory_sampler()

    error: str | None = None
    runner_result: dict[str, Any] = {}

    try:
        # Hook into runner so we can mark phases between sub-stages.
        # We can't easily get inside runner.run() without forking it, but we can
        # use the OrchestratorResult fields after the fact for phase boundaries.
        _PROFILER.mark_phase("scenario_start")
        result = await asyncio.wait_for(runner.run(question), timeout=timeout_s)
        _PROFILER.mark_phase("scenario_end")
        runner_result = {
            "discovery_s": round(result.discovery_duration_s, 4),
            "planning_s": round(result.planning_duration_s, 4),
            "summarization_s": round(result.summarization_duration_s, 4),
            "total_s": round(result.total_duration_s, 4),
            "n_steps": len(result.history),
            "answer_chars": len(result.answer),
            "history": [
                {
                    "step_number": h.step_number,
                    "task": h.task,
                    "server": h.server,
                    "tool": h.tool,
                    "success": h.success,
                    "tool_call_duration_s": round(getattr(h, "tool_call_duration_s", 0.0) or 0.0, 4),
                    "response_chars": len(h.response or "") if h.success else 0,
                    "error": h.error if not h.success else None,
                }
                for h in result.history
            ],
        }
    except asyncio.TimeoutError:
        error = f"timed out after {timeout_s}s"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    _PROFILER.stop_memory_sampler()

    # Construct phase markers post-hoc using the runner timings.
    # Reset markers to derived ones.
    derived_markers: list[dict[str, Any]] = []
    if not error and runner_result:
        d = runner_result["discovery_s"]
        p = runner_result["planning_s"]
        derived_markers.append({"name": "discovery", "t_s": 0.0})
        derived_markers.append({"name": "planning", "t_s": round(d, 4)})
        # Step boundaries — approximate using cumulative tool call durations
        cursor = d + p
        for h in runner_result["history"]:
            tag = f"step_{h['step_number']}_{h['tool'] or 'utility'}"
            derived_markers.append({"name": tag, "t_s": round(cursor, 4)})
            cursor += h["tool_call_duration_s"]
        derived_markers.append({"name": "summarization", "t_s": round(cursor, 4)})

    if derived_markers:
        _PROFILER.phase_markers = derived_markers

    assign_phases(_PROFILER.llm_events, _PROFILER.phase_markers)
    assign_phases(_PROFILER.tool_events, _PROFILER.phase_markers)

    return {
        "label": scenario_label,
        "question": question,
        "error": error,
        "runner_result": runner_result,
        "phase_markers": list(_PROFILER.phase_markers),
        "llm_events": list(_PROFILER.llm_events),
        "tool_events": list(_PROFILER.tool_events),
        "memory_samples": list(_PROFILER.memory_samples),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def _wall_span(events: list[dict]) -> float:
    """Compute the parallel-aware wall span: union of all event intervals.

    For events that overlap (concurrent execution), the span counts the union
    of busy time. For sequential events, this matches the sum.
    """
    if not events:
        return 0.0
    intervals = sorted(
        ((e["t_start_s"], e.get("t_end_s", e["t_start_s"] + e["wall_s"])) for e in events),
        key=lambda x: x[0],
    )
    span = 0.0
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            span += cur_end - cur_start
            cur_start, cur_end = s, e
    span += cur_end - cur_start
    return span


def aggregate(scenario_data: dict[str, Any]) -> dict[str, Any]:
    """Compute per-scenario summary statistics."""
    llm_events = scenario_data["llm_events"]
    tool_events = scenario_data["tool_events"]
    llm_total_sum = sum(e["wall_s"] for e in llm_events)
    tool_total_sum = sum(e["wall_s"] for e in tool_events)
    llm_span = _wall_span(llm_events)
    tool_span = _wall_span(tool_events)
    runner_result = scenario_data.get("runner_result") or {}
    total = float(runner_result.get("total_s", 0.0))
    # "Other" using parallel-aware spans (more accurate when calls overlap)
    other = max(0.0, total - llm_span - tool_span)
    # Parallelism factor: how much of the apparent tool work was concurrent
    tool_parallelism = round(tool_total_sum / tool_span, 2) if tool_span > 0 else 1.0
    llm_total = llm_total_sum  # LLMs are typically sequential — sum == span
    tool_total = tool_total_sum  # keep sum for "total work"; report span separately

    rss_vals = [s["rss_mb"] for s in scenario_data["memory_samples"]]
    rss_peak = max(rss_vals) if rss_vals else 0.0
    rss_min = min(rss_vals) if rss_vals else 0.0
    rss_avg = (sum(rss_vals) / len(rss_vals)) if rss_vals else 0.0
    rss_growth = (rss_vals[-1] - rss_vals[0]) if len(rss_vals) >= 2 else 0.0

    # Per-phase aggregations
    phase_llm: dict[str, dict[str, float]] = {}
    phase_tool: dict[str, dict[str, float]] = {}
    for e in scenario_data["llm_events"]:
        d = phase_llm.setdefault(e["phase"], {"count": 0, "wall_s": 0.0})
        d["count"] += 1
        d["wall_s"] += e["wall_s"]
    for e in scenario_data["tool_events"]:
        d = phase_tool.setdefault(e["phase"], {"count": 0, "wall_s": 0.0})
        d["count"] += 1
        d["wall_s"] += e["wall_s"]

    return {
        "total_s": round(total, 4),
        "llm_total_s": round(llm_total, 4),         # sum of LLM call walls
        "llm_span_s": round(llm_span, 4),           # parallel-aware time
        "tool_total_s": round(tool_total, 4),       # sum of tool walls (incl. parallel)
        "tool_span_s": round(tool_span, 4),         # parallel-aware time
        "tool_parallelism": tool_parallelism,
        "other_s": round(other, 4),
        # Percentages use spans (parallel-aware), so they sum sensibly to ≤100%
        "llm_pct": round(100 * llm_span / total, 1) if total > 0 else 0.0,
        "tool_pct": round(100 * tool_span / total, 1) if total > 0 else 0.0,
        "other_pct": round(100 * other / total, 1) if total > 0 else 0.0,
        "n_llm_calls": len(scenario_data["llm_events"]),
        "n_tool_calls": len(scenario_data["tool_events"]),
        "rss_peak_mb": round(rss_peak, 1),
        "rss_min_mb": round(rss_min, 1),
        "rss_avg_mb": round(rss_avg, 1),
        "rss_growth_mb": round(rss_growth, 1),
        "phase_llm": {k: {"count": v["count"], "wall_s": round(v["wall_s"], 4)} for k, v in phase_llm.items()},
        "phase_tool": {k: {"count": v["count"], "wall_s": round(v["wall_s"], 4)} for k, v in phase_tool.items()},
    }


def render_md(scenario_data: dict[str, Any]) -> str:
    summary = aggregate(scenario_data)
    runner_result = scenario_data.get("runner_result") or {}
    lines: list[str] = []
    lines.append(f"# Scenario `{scenario_data['label']}`\n")
    if scenario_data.get("error"):
        lines.append(f"**ERROR:** `{scenario_data['error']}`\n")
        return "\n".join(lines)

    q = scenario_data["question"]
    lines.append(f"**Query:** {q[:200]}{'...' if len(q) > 200 else ''}")
    lines.append(f"**Total wall:** {summary['total_s']:.2f} s")
    lines.append(
        f"**Steps:** {runner_result.get('n_steps', 0)} | "
        f"**Answer chars:** {runner_result.get('answer_chars', 0)}\n"
    )

    lines.append("## Top-level breakdown (parallel-aware)\n")
    lines.append(
        "Spans use the union of event intervals — when N tool calls run "
        "concurrently, span = `max_end - min_start`, not `sum(walls)`.\n"
    )
    lines.append("| Bucket | Span (s) | Sum of walls (s) | % of scenario |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| LLM API calls | {summary['llm_span_s']:.2f} | "
        f"{summary['llm_total_s']:.2f} | {summary['llm_pct']:.1f}% |"
    )
    lines.append(
        f"| MCP tool calls | {summary['tool_span_s']:.2f} | "
        f"{summary['tool_total_s']:.2f} | {summary['tool_pct']:.1f}% |"
    )
    lines.append(
        f"| Other (LLM↔tool gap, dispatch, parsing) | {summary['other_s']:.2f} | — | {summary['other_pct']:.1f}% |"
    )
    lines.append(f"| **TOTAL scenario wall** | **{summary['total_s']:.2f}** | — | 100% |\n")
    lines.append(
        f"**Tool parallelism factor:** {summary['tool_parallelism']}× "
        f"({summary['tool_total_s']:.0f} s of work compressed into "
        f"{summary['tool_span_s']:.0f} s of wall via concurrency)\n"
    )

    if runner_result:
        lines.append("## Phase wall (from runner)\n")
        lines.append("| Phase | Wall (s) |")
        lines.append("|---|---|")
        lines.append(f"| Discovery | {runner_result.get('discovery_s', 0):.2f} |")
        lines.append(f"| Planning | {runner_result.get('planning_s', 0):.2f} |")
        for h in runner_result.get("history", []):
            tag = f"Step {h['step_number']}: {h['server']}/{h['tool'] or 'utility'}"
            ok = "✓" if h["success"] else "✗"
            lines.append(f"| {tag} {ok} | {h['tool_call_duration_s']:.2f} |")
        lines.append(f"| Summarization | {runner_result.get('summarization_s', 0):.2f} |")
        lines.append("")

    lines.append("## LLM calls\n")
    lines.append(
        f"- count: {summary['n_llm_calls']}, "
        f"total wall: {summary['llm_total_s']:.2f} s, "
        f"avg: {summary['llm_total_s']/summary['n_llm_calls']:.2f} s/call"
        if summary["n_llm_calls"]
        else "- count: 0"
    )
    lines.append("")
    if scenario_data["llm_events"]:
        lines.append("| # | t_start (s) | phase | wall (s) | prompt | response |")
        lines.append("|---|---|---|---|---|---|")
        for i, e in enumerate(scenario_data["llm_events"], 1):
            lines.append(
                f"| {i} | {e['t_start_s']:.2f} | {e.get('phase', '?')} | "
                f"{e['wall_s']:.2f} | {e['prompt_chars']} | {e['response_chars']} |"
            )
        lines.append("")
        lines.append("### Per-phase LLM totals\n")
        lines.append("| Phase | Count | Wall (s) |")
        lines.append("|---|---|---|")
        for phase, agg in sorted(summary["phase_llm"].items(), key=lambda kv: -kv[1]["wall_s"]):
            lines.append(f"| {phase} | {agg['count']} | {agg['wall_s']:.2f} |")
        lines.append("")

    lines.append("## MCP tool calls\n")
    lines.append(
        f"- count: {summary['n_tool_calls']}, "
        f"total wall: {summary['tool_total_s']:.2f} s, "
        f"avg: {summary['tool_total_s']/summary['n_tool_calls']:.2f} s/call"
        if summary["n_tool_calls"]
        else "- count: 0"
    )
    lines.append("")
    if scenario_data["tool_events"]:
        lines.append("| # | t_start (s) | t_end (s) | phase | tool | wall (s) | args preview | resp chars |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, e in enumerate(scenario_data["tool_events"], 1):
            err = " (ERR)" if e.get("error") else ""
            args_preview = e.get("args_preview", "")[:60]
            lines.append(
                f"| {i} | {e['t_start_s']:.2f} | {e.get('t_end_s', e['t_start_s']+e['wall_s']):.2f} | "
                f"{e.get('phase', '?')} | "
                f"{e['tool_name']}{err} | {e['wall_s']:.2f} | "
                f"`{args_preview}` | {e['response_chars']} |"
            )
        lines.append("")
        # Per-tool breakdown
        lines.append("### Per-tool aggregates\n")
        per_tool: dict[str, dict[str, float]] = {}
        for e in scenario_data["tool_events"]:
            d = per_tool.setdefault(e["tool_name"], {"count": 0, "sum_wall_s": 0.0, "max_wall_s": 0.0})
            d["count"] += 1
            d["sum_wall_s"] += e["wall_s"]
            d["max_wall_s"] = max(d["max_wall_s"], e["wall_s"])
        lines.append("| Tool | Count | Sum wall (s) | Max wall (s) | Avg wall (s) |")
        lines.append("|---|---|---|---|---|")
        for tool, agg in sorted(per_tool.items(), key=lambda kv: -kv[1]["sum_wall_s"]):
            avg = agg["sum_wall_s"] / agg["count"]
            lines.append(
                f"| {tool} | {int(agg['count'])} | {agg['sum_wall_s']:.1f} | "
                f"{agg['max_wall_s']:.1f} | {avg:.1f} |"
            )
        lines.append("")

    lines.append("## Memory profile\n")
    lines.append(f"- peak RSS: {summary['rss_peak_mb']:.1f} MB")
    lines.append(f"- min RSS:  {summary['rss_min_mb']:.1f} MB")
    lines.append(f"- avg RSS:  {summary['rss_avg_mb']:.1f} MB")
    lines.append(f"- growth (start→end): +{summary['rss_growth_mb']:.1f} MB")
    lines.append("")

    # Diagnosis
    biggest = max(
        ("LLM", summary["llm_total_s"]),
        ("Tool", summary["tool_total_s"]),
        ("Other", summary["other_s"]),
        key=lambda kv: kv[1],
    )
    lines.append("## Bottleneck diagnosis\n")
    lines.append(
        f"**Dominant cost:** {biggest[0]} calls — "
        f"{biggest[1]:.1f} s "
        f"({biggest[1] / summary['total_s'] * 100:.1f}% of total)\n"
    )
    if summary["n_llm_calls"] > 0:
        avg_llm = summary["llm_total_s"] / summary["n_llm_calls"]
        lines.append(
            f"- {summary['n_llm_calls']} LLM calls, avg {avg_llm:.2f} s each "
            f"(typical LLM API latency)"
        )
    if summary["n_tool_calls"] > 0:
        avg_tool = summary["tool_total_s"] / summary["n_tool_calls"]
        lines.append(
            f"- {summary['n_tool_calls']} tool calls, avg {avg_tool:.2f} s each"
        )
    return "\n".join(lines)


def render_combined(all_scenarios: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Combined scenario profile — bottleneck analysis\n")
    lines.append("| Scenario | Total (s) | LLM (s, %) | Tool (s, %) | Other (s, %) | LLM calls | Tool calls | Peak RSS (MB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sd in all_scenarios:
        agg = aggregate(sd)
        if sd.get("error"):
            lines.append(f"| {sd['label']} | ERROR | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {sd['label']} | {agg['total_s']:.1f} | "
            f"{agg['llm_total_s']:.1f} ({agg['llm_pct']:.0f}%) | "
            f"{agg['tool_total_s']:.1f} ({agg['tool_pct']:.0f}%) | "
            f"{agg['other_s']:.1f} ({agg['other_pct']:.0f}%) | "
            f"{agg['n_llm_calls']} | {agg['n_tool_calls']} | "
            f"{agg['rss_peak_mb']:.0f} |"
        )
    lines.append("")
    # Cross-scenario averages
    valid = [aggregate(sd) for sd in all_scenarios if not sd.get("error")]
    if valid:
        avg_llm_pct = sum(a["llm_pct"] for a in valid) / len(valid)
        avg_tool_pct = sum(a["tool_pct"] for a in valid) / len(valid)
        avg_other_pct = sum(a["other_pct"] for a in valid) / len(valid)
        lines.append(f"\n**Cross-scenario averages:** "
                     f"LLM {avg_llm_pct:.0f}%, Tool {avg_tool_pct:.0f}%, Other {avg_other_pct:.0f}%\n")
        if avg_llm_pct > 50:
            verdict = "**LLM API calls are the dominant cost.**"
        elif avg_tool_pct > 50:
            verdict = "**MCP tool calls are the dominant cost.**"
        else:
            verdict = "**Cost is split — no single layer dominates.**"
        lines.append(verdict)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _env_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }


async def _amain(args: argparse.Namespace) -> int:
    # Set BATTERY_BOOT_CELL_SUBSET so the battery server only loads the requested cells
    os.environ["BATTERY_BOOT_CELL_SUBSET"] = args.cells

    install_patches()

    from llm.litellm import LiteLLMBackend
    from agent.plan_execute.runner import PlanExecuteRunner

    llm = LiteLLMBackend(args.model_id)
    runner = PlanExecuteRunner(llm=llm)

    scenarios_path = _REPO_ROOT / "battery_scenarios.json"
    scenarios_data = json.loads(scenarios_path.read_text(encoding="utf-8"))
    indices = [int(x.strip()) for x in args.scenarios.split(",") if x.strip()]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = _REPO_ROOT / "profiles" / f"scenario_detailed_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir.relative_to(_REPO_ROOT)}")
    print(f"Cells:    {args.cells}")
    print(f"Model:    {args.model_id}")
    print(f"Scenarios: {indices}\n")

    results: list[dict[str, Any]] = []
    for i in indices:
        sc = scenarios_data[i]
        label = f"scenario_{i + 1}"
        persona = sc.get("persona", "?")
        question = sc.get("query", "")
        print(f"[{label}] {persona}")
        sd = await run_one_scenario(runner, question, label, args.timeout)
        sd["persona"] = persona
        sd["scenario_index"] = i

        json_path = out_dir / f"{label}.json"
        json_path.write_text(
            json.dumps(
                {
                    "git_sha": _git_sha(),
                    "env": _env_info(),
                    "model_id": args.model_id,
                    "cells": args.cells,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    **sd,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        md_path = out_dir / f"{label}.md"
        md_path.write_text(render_md(sd), encoding="utf-8")

        agg = aggregate(sd)
        if sd.get("error"):
            print(f"  ERROR: {sd['error']}\n")
        else:
            print(
                f"  total {agg['total_s']:.1f}s  "
                f"LLM {agg['llm_total_s']:.1f}s ({agg['llm_pct']:.0f}%)  "
                f"tool {agg['tool_total_s']:.1f}s ({agg['tool_pct']:.0f}%)  "
                f"other {agg['other_s']:.1f}s ({agg['other_pct']:.0f}%)  "
                f"calls={agg['n_llm_calls']}/{agg['n_tool_calls']}  "
                f"peak {agg['rss_peak_mb']:.0f}MB"
            )
        results.append(sd)

    combined = render_combined(results)
    (out_dir / "combined.md").write_text(combined, encoding="utf-8")
    print(f"\nCombined report: {(out_dir / 'combined.md').relative_to(_REPO_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=_DEFAULT_SCENARIOS,
                    help=f"Comma-separated 0-indexed scenario indices (default: {_DEFAULT_SCENARIOS})")
    ap.add_argument("--cells", default=_DEFAULT_CELLS,
                    help=f"BATTERY_BOOT_CELL_SUBSET (default: {_DEFAULT_CELLS})")
    ap.add_argument("--timeout", type=float, default=420.0,
                    help="Per-scenario timeout (s, default: 420)")
    ap.add_argument("--model-id", default=_DEFAULT_MODEL,
                    help=f"LLM model id (default: {_DEFAULT_MODEL})")
    args = ap.parse_args()

    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
