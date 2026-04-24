"""LLM-based plan generation for the plan-execute orchestrator.

Each plan step now includes the specific tool to call and its arguments,
so the executor needs no additional LLM calls — it calls the tool directly.
"""

from __future__ import annotations

import logging
import re

from llm import LLMBackend
from .models import Plan, PlanStep

_log = logging.getLogger(__name__)

_PLAN_PROMPT = """\
You are a planning assistant for industrial asset operations and maintenance.

Routing guide — for EACH step, match the terms that step concerns to the right
server below. A single question may span multiple asset types; freely mix servers
across steps when that happens (e.g. a query about both a chiller and a battery
should produce steps on both the iot server AND the battery server).

  - Lithium-ion / Li-ion / batteries / battery cells / packs / EV batteries /
    RUL / capacity fade / EOD / cycle life / 1.4 Ah / 2 Ah / SOH  → battery
  - Rotating machinery / motors / pumps / fans / bearings / vibration /
    FFT / envelope spectrum / ISO 10816                           → vibration
  - Chillers / HVAC / generic sensor history / asset discovery    → iot
  - Forecasting any time series / anomaly detection / sensitivity
    or correlation analysis on tabular data                        → tsfm
  - Work orders / maintenance records                              → wo
  - Failure modes / sensor-to-failure mapping                      → fmsr
  - Current date/time / JSON file reading                          → utilities

For lithium-ion RUL, end-of-discharge timing, voltage milestones/crossings, capacity
fade, impedance (Rct/Re), and fleet cell rankings, prefer battery server tools
(predict_rul, predict_voltage_curve, predict_voltage_milestones, get_battery_cycle_summary,
analyze_impedance_growth, detect_capacity_outliers, list_batteries, diagnose_battery)
over TSFM tools unless the user explicitly asks for time-series forecasting, model
fine-tuning, or generic tabular sensitivity — TSFM tools may be unavailable without
extra setup.

Do not plan a json_reader step for files that were not explicitly provided by the user
as an uploaded file or a named path. If the user describes data only in prose, treat it
as context for tool calls against live databases, not as a path on disk.

When calling list_batteries with no site context from the user, use site_name='MAIN'.

Decompose the question below into a sequence of subtasks. For each subtask,
assign a server and select the exact tool to call. Do NOT include tool arguments —
they will be resolved at execution time from the task description and prior results.

Available servers and tools:
{servers}

Output format — one block per step, exactly:

#Task1: <task description>
#Server1: <exact server name>
#Tool1: <exact tool name, or "none" if no tool call is needed>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

#Task2: <task description>
#Server2: <exact server name>
#Tool2: <exact tool name>
#Dependency2: #S1
#ExpectedOutput2: <what this step should produce>

Rules:
- Server and tool names must exactly match those listed above.
- Dependencies use #S<N> notation (e.g., #S1, #S2). Use "None" if none.
- Keep tasks specific and actionable.
- When a question applies the same tool to multiple specific assets (e.g. 'for each
  cell', 'top N cells', 'B0005 and B0006'), emit ONE STEP PER ASSET, each with the
  asset_id named explicitly in that step's task description. Do not cover multiple
  assets in a single step.
- Do not use tool "none" as a placeholder before real data retrieval: for each asset,
  call the actual battery (or other) MCP tool that returns data in the same step that
  names the asset_id. Reserve tool "none" only for final synthesis or formatting when
  all required numbers already appear in prior step results.
- The #Server field must always be one of the listed server names (iot, utilities,
  fmsr, tsfm, wo, vibration, battery). Never use the literal word "none" as #Server.
  For steps with #Tool none, set #Server to utilities (or battery if synthesizing
  only battery JSON results).
- Never invent tool names: each #Tool value must match a tool listed under one of
  the servers above, exactly as spelled.
- The same one-step-per-asset rule applies to get_battery_cycle_summary and
  analyze_impedance_growth: never use one vague step for "each cell" — one step per
  B0xxx asset_id in the task line.
- For MAE/RMSE or error metrics on battery voltage/EOD/RUL, derive them in a tool:none
  synthesis step from numeric outputs of battery tools — do not call TSFM evaluation
  or forecasting helpers unless that exact tool name appears in the server list.

Question: {question}

Plan:
"""

_TASK_RE = re.compile(r"#Task(\d+):\s*(.+)")
_SERVER_RE = re.compile(r"#Server(\d+):\s*(.+)")
_TOOL_RE = re.compile(r"#Tool(\d+):\s*(.+)")
_DEP_RE = re.compile(r"#Dependency(\d+):\s*(.+)")
_OUTPUT_RE = re.compile(r"#ExpectedOutput(\d+):\s*(.+)")
_DEP_NUM_RE = re.compile(r"#S(\d+)")


def parse_plan(raw: str) -> Plan:
    """Parse an LLM-generated plan string into a Plan object."""
    tasks = {int(m.group(1)): m.group(2).strip() for m in _TASK_RE.finditer(raw)}
    servers = {int(m.group(1)): m.group(2).strip() for m in _SERVER_RE.finditer(raw)}
    # Strip any trailing signature the LLM may copy from the server description
    # format "tool_name(param: type)" — only the bare name is needed.
    tools = {
        int(m.group(1)): m.group(2).strip().split("(")[0].strip()
        for m in _TOOL_RE.finditer(raw)
    }
    deps_raw = {int(m.group(1)): m.group(2).strip() for m in _DEP_RE.finditer(raw)}
    outputs = {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_RE.finditer(raw)}

    steps = []
    for n in sorted(tasks):
        raw_dep = deps_raw.get(n, "None").strip()

        if raw_dep.lower() == "none":
            dependencies = []
        else:
            dependencies = [int(x) for x in _DEP_NUM_RE.findall(raw_dep)]

            # Make sure dependency references only point to earlier valid steps.
            if not dependencies:
                raise ValueError(f"Invalid dependency format for step {n}: {raw_dep}")

            for dep in dependencies:
                if dep < 1 or dep >= n:
                    raise ValueError(
                        f"Invalid dependency reference for step {n}: #S{dep}"
                    )

        steps.append(
            PlanStep(
                step_number=n,
                task=tasks[n],
                server=servers.get(n, ""),
                tool=tools.get(n, ""),
                tool_args={},
                dependencies=dependencies,
                expected_output=outputs.get(n, ""),
            )
        )

    return Plan(steps=steps, raw=raw)


class Planner:
    """Decomposes a question into a structured execution plan using an LLM."""

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def generate_plan(
        self,
        question: str,
        server_descriptions: dict[str, str],
    ) -> Plan:
        """Generate a plan for a question given available servers and their tools.

        Args:
            question: The user question to answer.
            server_descriptions: Mapping of server_name -> formatted tool signatures.

        Returns:
            A Plan where each PlanStep includes the tool to call and its arguments.
        """
        servers_text = "\n\n".join(
            f"{name}:\n{desc}" for name, desc in server_descriptions.items()
        )
        prompt = _PLAN_PROMPT.format(servers=servers_text, question=question)
        raw = self._llm.generate(prompt)
        return parse_plan(raw)
