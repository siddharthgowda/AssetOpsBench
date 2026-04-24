"""Data models for the plan-execute orchestration client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlanStep:
    """A single step in an execution plan.

    If ``foreach`` is set (e.g. ``"#S1"``), the executor resolves the referenced
    step's output into a list of items and calls ``tool`` once per item, in
    parallel. The aggregated result is stored as a JSON ``{"items": [...]}`` in
    the step's StepResult.response.
    """

    step_number: int
    task: str
    server: str
    tool: str
    tool_args: dict
    dependencies: list[int]
    expected_output: str
    foreach: Optional[str] = None  # e.g. "#S1" to fan out over step 1's output


@dataclass
class Plan:
    """An execution plan composed of ordered steps."""

    steps: list[PlanStep]
    raw: str  # Raw LLM output, preserved for debugging

    def get_step(self, number: int) -> Optional[PlanStep]:
        return next((s for s in self.steps if s.step_number == number), None)

    def resolved_order(self) -> list[PlanStep]:
        """Return steps in topological order (dependencies before dependents)."""
        seen: set[int] = set()
        ordered: list[PlanStep] = []

        def visit(n: int) -> None:
            if n in seen:
                return
            step = self.get_step(n)
            if step is None:
                return
            for dep in step.dependencies:
                visit(dep)
            seen.add(n)
            ordered.append(step)

        for step in self.steps:
            visit(step.step_number)
        return ordered


@dataclass
class StepResult:
    """Result of executing a single plan step."""

    step_number: int
    task: str
    server: str
    response: str
    error: Optional[str] = None
    tool: str = ""
    tool_args: dict = field(default_factory=dict)
    duration_s: float = 0.0
    tool_call_duration_s: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None
