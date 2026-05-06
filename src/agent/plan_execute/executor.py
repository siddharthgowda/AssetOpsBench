"""MCP-based step executor for the plan-execute orchestrator.

The planner produces steps with no pre-filled arguments. For every step that
calls a tool the executor makes one LLM call to generate the concrete argument
dict from the task description, original question, and prior step results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from llm import LLMBackend
from .models import Plan, PlanStep, StepResult

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Maps agent names to either a uv entry-point name (str) or a script Path.
# Entry-point names are invoked as ``uv run <name>``; Paths fall back to
# ``python -m module.path`` (supports relative imports).
DEFAULT_SERVER_PATHS: dict[str, Path | str] = {
    "iot": "iot-mcp-server",
    "utilities": "utilities-mcp-server",
    "fmsr": "fmsr-mcp-server",
    "tsfm": "tsfm-mcp-server",
    "wo": "wo-mcp-server",
    "vibration": "vibration-mcp-server",
    "battery": "battery-mcp-server",
}

# One-line asset-type header shown to the planner before each server's tool list.
# Helps the planner route questions to the right server based on vocabulary in
# the query (e.g. "batteries", "cells", "RUL" → battery; "chiller", "HVAC" → iot).
# Planner is free to mix servers across steps — see planner.py prompt.
_SERVER_DESCRIPTIONS: dict[str, str] = {
    "iot": (
        "General industrial IoT sensor access — site/asset discovery, sensor listing, "
        "and time-series history for chillers, HVAC, and generic industrial assets."
    ),
    "utilities": "Generic helpers: current date/time, English time formatting, JSON file reading.",
    "fmsr": (
        "Failure-mode and sensor reasoning — list failure modes for an asset type and "
        "map sensors to failure modes."
    ),
    "tsfm": (
        "Generic time-series foundation-model forecasting, fine-tuning, anomaly detection, "
        "and quantitative sensitivity/correlation analysis on tabular time-series data."
    ),
    "wo": "Work-order / maintenance-record search.",
    "vibration": (
        "Rotating-machinery vibration analysis — FFT, envelope spectrum, bearing fault "
        "frequencies, ISO 10816 severity classification, and full diagnosis for motors, "
        "pumps, fans, and other rotating assets."
    ),
    "battery": (
        "Lithium-ion (Li-ion) battery cell analytics — RUL prediction, capacity-fade "
        "curves, end-of-discharge voltage timing, impedance (Rct/Re) growth, fleet "
        "outlier detection, and full Li-ion cell diagnosis. Use for anything about "
        "batteries, cells, packs, EV batteries, EOL, or capacity fade."
    ),
}

_PLACEHOLDER_RE = re.compile(r"\{step_(\d+)\}")
# Battery asset ID regex but this is a hack and should not be merged to
# the original code and this probably won't be needed if we are using better
# models
_ASSET_ID_RE = re.compile(r"\b(B\d{4})\b", re.IGNORECASE)

_ARG_RESOLUTION_PROMPT = """\
Generate the JSON arguments for the tool call below.

Question: {question}
Tool: {tool}
Tool parameters: {tool_schema}
Task: {task}

Prior step results:
{context}

YOUR RESPONSE MUST BE A SINGLE RAW JSON OBJECT AND NOTHING ELSE.
Do not write any explanation, reasoning, or prose — output only the JSON object.
Use EXACTLY the parameter names listed in "Tool parameters" above.
Use the task description and prior step results to determine the correct argument values.
If the task text names a specific asset_id (e.g. B0005), you MUST use exactly that ID.
Do not substitute a different cell from a fleet list or pick another list element.

JSON:"""

_NONE_STEP_SYNTHESIS_PROMPT = """\
You are synthesizing an intermediate answer for one plan step (no MCP tool call).

Original user question: {question}

Current step task: {task}

Expected output rubric: {expected_output}

Prior step results (use every step below; do not ignore earlier steps):
{context}

Instructions:
- Ground your answer in numeric values and structured fields from the prior results (JSON or text).
- When pairs of predicted vs actual (or baseline) numbers exist, compute MAE and/or RMSE explicitly.
- Rank or filter lists using the criteria from the question and show the ordering keys.
- If an upstream step failed or a metric cannot be computed, say so clearly — do not imply it was computed.
- Be concrete; avoid vague boilerplate.

Synthesis:"""


# Maximum characters per step's response included in the context shown to the
# synthesis / arg-resolution LLM. Prevents fan-out aggregates (14 cells × full
# per-cycle histories) from blowing past the model's context window.
_MAX_STEP_RESPONSE_CHARS = 6000


def _truncate_for_context(text: str, limit: int = _MAX_STEP_RESPONSE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars from step response]"


def _format_context_lines(context: dict[int, StepResult]) -> str:
    return "\n".join(
        f"Step {n}: "
        + (_truncate_for_context(r.response) if r.success else f"ERROR: {r.error}")
        for n, r in sorted(context.items())
    )

# This is a hack and is fine since we using very small llama models
# but this not be merged into the original repo (it's fine for our fork)
def _enrich_resolved_args(task: str, tool: str, args: dict | None) -> dict:
    """Fill common battery/tool args when the LLM omitted them but the task names an asset."""
    out = dict(args) if args else {}
    m = _ASSET_ID_RE.search(task)
    asset_id = m.group(1).upper() if m else None
    single_asset_tools = frozenset(
        {
            "predict_rul",
            "get_battery_cycle_summary",
            "predict_voltage_curve",
            "predict_voltage_milestones",
            "analyze_impedance_growth",
            "diagnose_battery",
        }
    )
    if tool == "predict_voltage_curve" and asset_id and not out.get("cycle_index"):
        out["cycle_index"] = 0
    if tool in single_asset_tools and asset_id and not out.get("asset_id"):
        out["asset_id"] = asset_id
    if tool == "list_batteries" and not out.get("site_name"):
        out["site_name"] = "MAIN"
    return _normalize_resolved_args(tool, out)


def _normalize_resolved_args(tool: str, args: dict) -> dict:
    """Coerce types so MCP JSON validation does not fail on LLM output."""
    out = dict(args)
    if tool == "predict_rul":
        fc = out.get("from_cycle")
        if isinstance(fc, str):
            try:
                out["from_cycle"] = int(float(fc.strip()))
            except ValueError:
                del out["from_cycle"]
        elif fc is not None and not isinstance(fc, int):
            try:
                out["from_cycle"] = int(fc)
            except (TypeError, ValueError):
                del out["from_cycle"]
    if tool == "predict_voltage_milestones":
        th = out.get("thresholds")
        if isinstance(th, str):
            cleaned = th.strip().strip("[]")
            try:
                out["thresholds"] = [
                    float(x.strip()) for x in cleaned.split(",") if x.strip()
                ]
            except ValueError:
                del out["thresholds"]
        elif isinstance(th, list):
            try:
                out["thresholds"] = [float(x) for x in th]
            except (TypeError, ValueError):
                del out["thresholds"]
    if tool == "predict_voltage_curve":
        ci = out.get("cycle_index")
        if isinstance(ci, str):
            try:
                out["cycle_index"] = int(float(ci.strip()))
            except ValueError:
                del out["cycle_index"]
        elif ci is not None and not isinstance(ci, int):
            try:
                out["cycle_index"] = int(ci)
            except (TypeError, ValueError):
                del out["cycle_index"]
    return out


def _omit_null_tool_args(args: dict) -> dict:
    """Strip null/empty placeholders before MCP call — Pydantic rejects None or '' for typed fields."""
    out: dict = {}
    for k, v in args.items():
        if v is None:
            continue
        if v == "":
            continue
        out[k] = v
    return out


_FOREACH_REF_RE = re.compile(r"#S(\d+)")


def _extract_foreach_items(response: str) -> list[dict]:
    """Extract a list of items from a prior step's response string.

    Tries, in order:
      1. Parse the whole response as JSON; look for common container keys.
      2. Find the last ``{...}`` block in the string and try JSON.
      3. Find the last ``[...]`` array and try JSON.
    Each item is normalized to a dict with at least ``asset_id`` when possible.
    Returns an empty list if nothing iterable can be found.
    """
    container_keys = ("items", "cells", "flagged_cells", "asset_ids", "rows", "docs")

    def _normalize(v: Any) -> list[dict]:
        if isinstance(v, list):
            out: list[dict] = []
            for el in v:
                if isinstance(el, dict):
                    out.append(el)
                elif isinstance(el, str):
                    out.append({"asset_id": el})
            return out
        return []

    def _probe(obj: Any) -> list[dict]:
        if isinstance(obj, list):
            return _normalize(obj)
        if isinstance(obj, dict):
            for k in container_keys:
                if k in obj and isinstance(obj[k], list):
                    return _normalize(obj[k])
            # Some tools return a dict of {asset_id: z_score, ...} — treat keys as assets.
            if obj and all(isinstance(k, str) for k in obj.keys()):
                return [{"asset_id": k, "value": v} for k, v in obj.items()]
        return []

    # 1. whole-string JSON
    try:
        items = _probe(json.loads(response))
        if items:
            return items
    except (ValueError, TypeError):
        pass
    # 2. last {...}
    try:
        start, end = response.rfind("{"), response.rfind("}") + 1
        if start != -1 and end > start:
            items = _probe(json.loads(response[start:end]))
            if items:
                return items
    except (ValueError, TypeError):
        pass
    # 3. last [...]
    try:
        start, end = response.rfind("["), response.rfind("]") + 1
        if start != -1 and end > start:
            items = _probe(json.loads(response[start:end]))
            if items:
                return items
    except (ValueError, TypeError):
        pass
    return []


def _foreach_item_to_args(tool: str, item: dict) -> dict:
    """Build the per-iteration tool args from one element of the source list."""
    out: dict = {}
    aid = item.get("asset_id")
    if isinstance(aid, str):
        out["asset_id"] = aid
    if tool == "list_batteries":
        out.setdefault("site_name", "MAIN")
    if tool == "predict_voltage_curve":
        out.setdefault("cycle_index", 0)
    return out


def _tool_unavailable_response(exc: BaseException) -> str | None:
    """Map missing optional deps (e.g. tsfm_public) to a structured JSON string."""
    msg = str(exc).lower()
    needles = (
        "tsfm_public",
        "modulenotfounderror",
        "no module named",
        "importerror",
    )
    if any(n in msg for n in needles):
        return json.dumps(
            {"status": "tool_unavailable", "reason": str(exc)},
            ensure_ascii=False,
        )
    return None


class Executor:
    """Executes plan steps by routing tool calls to MCP servers."""

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths = (
            DEFAULT_SERVER_PATHS if server_paths is None else server_paths
        )

    async def get_server_descriptions(self) -> dict[str, str]:
        """Query each registered MCP server and return formatted tool signatures.

        Each server's block is prefixed with a one-line asset-type header (from
        _SERVER_DESCRIPTIONS) so the planner can route query terms to the
        correct server on a per-step basis.
        """
        descriptions: dict[str, str] = {}
        for name, path in self._server_paths.items():
            try:
                tools = await _list_tools(path)
                lines = []
                header = _SERVER_DESCRIPTIONS.get(name)
                if header:
                    lines.append(f"  # {header}")
                for t in tools:
                    params = ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    lines.append(f"  - {t['name']}({params}): {t['description']}")
                descriptions[name] = "\n".join(lines)
            except Exception as exc:  # noqa: BLE001
                descriptions[name] = f"  (unavailable: {exc})"
        return descriptions

    async def execute_plan(self, plan: Plan, question: str) -> list[StepResult]:
        """Execute all plan steps in dependency order."""
        ordered = plan.resolved_order()
        total = len(ordered)

        # Pre-fetch tool schemas for all servers referenced in the plan so that
        # _resolve_args_with_llm can include exact parameter names in its prompt.
        server_names = {step.server for step in ordered}
        tool_schemas: dict[str, dict[str, str]] = {}  # server -> {tool_name -> sig}
        for name in server_names:
            path = self._server_paths.get(name)
            if path is None:
                continue
            try:
                tools = await _list_tools(path)
                tool_schemas[name] = {
                    t["name"]: ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    for t in tools
                }
            except Exception:  # noqa: BLE001
                tool_schemas[name] = {}

        context: dict[int, StepResult] = {}
        results: list[StepResult] = []
        for step in ordered:
            _log.info(
                "Step %d/%d [%s]: %s",
                step.step_number,
                total,
                step.server,
                step.task,
            )
            schema = tool_schemas.get(step.server, {}).get(step.tool, "")
            t_step = time.perf_counter()
            result = await self.execute_step(step, context, question, tool_schema=schema)
            result.duration_s = time.perf_counter() - t_step
            if result.success:
                _log.info("Step %d OK.", step.step_number)
            else:
                _log.warning("Step %d FAILED: %s", step.step_number, result.error)
            context[step.step_number] = result
            results.append(result)
        return results

    async def execute_step(
        self,
        step: PlanStep,
        context: dict[int, StepResult],
        question: str,
        tool_schema: str = "",
    ) -> StepResult:
        """Execute a single plan step.

        1. Resolve the MCP server assigned to this step.
        2. If no tool is specified, return expected_output directly.
        3. Call the LLM to generate tool arguments from the task and prior results.
        4. Call the tool and return its result.
        """
        server_path = self._server_paths.get(step.server)
        if server_path is None:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=(
                    f"Unknown server '{step.server}'. "
                    f"Registered servers: {list(self._server_paths)}"
                ),
            )

        # ── Fan-out branch: if the step carries a #Foreach directive, call the
        # same tool once per element of the referenced step's output and
        # aggregate results into a single {"items": [...]} JSON payload.
        if step.foreach and step.tool and step.tool.lower() not in ("none", "null"):
            m = _FOREACH_REF_RE.search(step.foreach)
            if not m:
                return StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response="",
                    error=f"Invalid #Foreach reference: {step.foreach!r}",
                    tool=step.tool,
                    tool_args=step.tool_args,
                )
            src_num = int(m.group(1))
            src = context.get(src_num)
            if src is None or not src.success:
                return StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response="",
                    error=f"Foreach source #S{src_num} unavailable or failed",
                    tool=step.tool,
                    tool_args=step.tool_args,
                )
            items = _extract_foreach_items(src.response)
            if not items:
                return StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response=json.dumps({"items": []}),
                    tool=step.tool,
                    tool_args={"foreach_source": step.foreach},
                )
            _log.info(
                "Step %d: fan-out over %d item(s) from %s",
                step.step_number,
                len(items),
                step.foreach,
            )

            async def _run_one(item: dict) -> dict:
                call_args = _foreach_item_to_args(step.tool, item)
                call_args = _omit_null_tool_args(
                    _normalize_resolved_args(step.tool, call_args)
                )
                try:
                    resp = await _call_tool(server_path, step.tool, call_args)
                    # Compact JSON whitespace — tool responses are often pretty-printed
                    # which wastes context budget (each 4-space indent × 100s of lines
                    # per cell × 14 cells can blow past the model's context window).
                    try:
                        resp = json.dumps(json.loads(resp), separators=(",", ":"), default=str)
                    except (ValueError, TypeError):
                        pass  # non-JSON response; leave as-is
                    return {"asset_id": call_args.get("asset_id"), "result": resp}
                except BaseException as tool_exc:  # noqa: BLE001
                    payload = _tool_unavailable_response(tool_exc)
                    if payload is not None:
                        return {"asset_id": call_args.get("asset_id"), "result": payload}
                    return {
                        "asset_id": call_args.get("asset_id"),
                        "error": str(tool_exc),
                    }

            t_tool = time.perf_counter()
            aggregated = await asyncio.gather(*[_run_one(it) for it in items])
            tool_call_duration_s = time.perf_counter() - t_tool
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=json.dumps({"items": aggregated}, default=str),
                tool=step.tool,
                tool_args={
                    "foreach_source": step.foreach,
                    "n_items": len(items),
                },
                tool_call_duration_s=tool_call_duration_s,
            )

        if not step.tool or step.tool.lower() in ("none", "null"):
            if not context:
                return StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response=step.expected_output,
                    tool=step.tool,
                    tool_args=step.tool_args,
                )
            ctx_text = _format_context_lines(context)
            syn_prompt = (
                _NONE_STEP_SYNTHESIS_PROMPT.replace("{question}", question)
                .replace("{task}", step.task)
                .replace("{expected_output}", step.expected_output or "(none)")
                .replace("{context}", ctx_text)
            )
            synthesized = self._llm.generate(syn_prompt)
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=synthesized,
                tool=step.tool,
                tool_args=step.tool_args,
            )

        resolved_args: dict = {}
        try:
            _log.info("Step %d: calling LLM to resolve args.", step.step_number)
            resolved_args = await _resolve_args_with_llm(
                question, step.task, step.tool, tool_schema, context, self._llm
            )
            resolved_args = _enrich_resolved_args(step.task, step.tool, resolved_args)

            t_tool = time.perf_counter()
            call_args = _omit_null_tool_args(resolved_args)
            try:
                response = await _call_tool(server_path, step.tool, call_args)
            except BaseException as tool_exc:  # noqa: BLE001
                payload = _tool_unavailable_response(tool_exc)
                if payload is not None:
                    response = payload
                else:
                    raise
            tool_call_duration_s = time.perf_counter() - t_tool
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=response,
                tool=step.tool,
                tool_args=call_args,
                tool_call_duration_s=tool_call_duration_s,
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=str(exc),
                tool=step.tool,
                tool_args=_omit_null_tool_args(resolved_args),
            )


# ── arg resolution ────────────────────────────────────────────────────────────


async def _resolve_args_with_llm(
    question: str,
    task: str,
    tool: str,
    tool_schema: str,
    context: dict[int, StepResult],
    llm: LLMBackend,
) -> dict:
    """Generate tool arguments from the task description and prior step results."""
    context_text = _format_context_lines(context)
    prompt = (
        _ARG_RESOLUTION_PROMPT
        .replace("{question}", question)
        .replace("{task}", task)
        .replace("{tool}", tool)
        .replace("{tool_schema}", tool_schema or "(unknown)")
        .replace("{context}", context_text or "(none)")
    )
    raw = llm.generate(prompt)
    resolved = _parse_json(raw)
    if resolved is None:
        _log.warning(
            "Tool '%s': arg resolution returned no parseable JSON (response: %r…)",
            tool, raw[:120],
        )
        return {}
    return resolved


def _parse_json(raw: str) -> dict | None:
    """Extract a JSON object from an LLM response, with markdown fence handling.

    Returns the parsed dict, or None if no JSON object could be extracted.
    An empty dict ``{}`` is a valid successful parse (e.g. for no-arg tools).
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).lstrip("json").strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    _log.debug("_parse_json: could not extract a JSON object from: %r…", raw[:120])
    return None


# ── MCP protocol helpers ──────────────────────────────────────────────────────


def _make_stdio_params(server: Path | str) -> "StdioServerParameters":
    """Build StdioServerParameters for a server spec.

    - str  → entry-point name; invoked as ``uv run <name>`` from the repo root.
    - Path → invoked as ``python -m module.path`` when under the repo root
             (supports relative imports), or directly otherwise.

    Servers whose deps live in an optional ``[dependency-groups]`` block need
    ``--group <name>`` passed to ``uv run``; otherwise ``uv`` resyncs the venv
    without those deps before spawning the subprocess.

    By default the MCP SDK only forwards a tiny whitelist of env vars to the
    spawned subprocess (HOME/PATH/SHELL/...). We pass ``os.environ`` explicitly
    so server-specific config (and ablation toggles) reaches the child.
    """
    from mcp import StdioServerParameters

    # Entry-point servers that need a specific uv dependency group.
    _SERVER_GROUPS = {
        "battery-mcp-server": "battery",
        "tsfm-mcp-server": "tsfm",
    }

    forwarded_env = {**os.environ}

    if isinstance(server, str):
        args = ["run"]
        group = _SERVER_GROUPS.get(server)
        if group:
            args.extend(["--group", group])
        args.append(server)
        return StdioServerParameters(
            command="uv",
            args=args,
            cwd=str(_REPO_ROOT),
            env=forwarded_env,
        )
    try:
        rel = server.relative_to(_REPO_ROOT)
        module = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
        return StdioServerParameters(
            command="python",
            args=["-m", module],
            cwd=str(_REPO_ROOT),
            env=forwarded_env,
        )
    except ValueError:
        return StdioServerParameters(
            command="python", args=[str(server)], env=forwarded_env
        )


async def _list_tools(server_path: Path | str) -> list[dict]:
    """Connect to an MCP server via stdio and list its tools with parameter info."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parameters = [
                    {
                        "name": k,
                        "type": v.get("type", "any"),
                        "required": k in required,
                    }
                    for k, v in props.items()
                ]
                tools.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": parameters,
                    }
                )
            return tools


async def _call_tool(server_path: Path | str, tool_name: str, args: dict) -> str:
    """Connect to an MCP server via stdio and call a tool."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return _extract_content(result.content)


def _extract_content(content: list[Any]) -> str:
    """Extract text from MCP tool call result content."""
    return "\n".join(getattr(item, "text", str(item)) for item in content)


def _resolve_args(args: dict, context: dict[int, StepResult]) -> dict:
    """Simple string substitution of {{step_N}} placeholders (kept for tests)."""
    resolved = {}
    for key, val in args.items():
        if isinstance(val, str):

            def _sub(m: re.Match) -> str:
                n = int(m.group(1))
                return context[n].response if n in context else m.group(0)

            resolved[key] = _PLACEHOLDER_RE.sub(_sub, val)
        else:
            resolved[key] = val
    return resolved


def _parse_tool_call(raw: str) -> dict:
    """Parse LLM output into a {tool, args} dict (utility, not used in main path)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"tool": None, "answer": text}
