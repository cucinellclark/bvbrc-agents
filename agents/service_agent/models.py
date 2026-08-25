"""Pydantic models for the BV-BRC Service Agent v2.

Three-phase workflow construction: Decompose -> Build -> Compose.

Key models:
  - AgentConfig: LLM and API configuration
  - StepPlan / WorkflowPlan: Phase 1 output (abstract DAG)
  - ValidatedStep: Phase 2 output (concrete params per step)
  - InformationRequest: Structured user question
  - AgentState: Serializable state for pause/resume across phases
  - AgentResult: Final output returned to the orchestrator
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, Field

from shared.models import (
    ToolCall,  # noqa: F401 -- re-export for backward compat
    ToolExecution,  # noqa: F401
    BaseAgentConfig,
    BaseAgentState,
    BaseAgentResult,
)
from shared.config import LLM_DEFAULTS


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


class AgentConfig(BaseAgentConfig):
    """Configuration for the service agent.

    Adds service-specific fields on top of BaseAgentConfig.
    """

    # Optional lightweight model for intent classification.
    classifier_model: str | None = LLM_DEFAULTS.get("classifier_model")

    # User preference for auto-submitting planned workflows.
    auto_submit_preference: str | None = None

    # session_id and workspace_path are inherited from BaseAgentConfig.


# ---------------------------------------------------------------------------
# Phase 1 models: WorkflowPlan (abstract DAG)
# ---------------------------------------------------------------------------


class StepPlan(BaseModel):
    """A single step in the abstract workflow plan (Phase 1 output)."""

    step_id: str  # Unique identifier within workflow
    service_name: str  # BV-BRC service name
    intent: str  # What this step accomplishes
    depends_on: list[str] = Field(default_factory=list)
    input_sources: dict[str, Any] = Field(default_factory=dict)


class WorkflowPlan(BaseModel):
    """Phase 1 output: abstract DAG of service steps with dependencies."""

    workflow_name: str
    description: str
    steps: list[StepPlan]

    # Computed after creation
    topological_order: list[str] = Field(default_factory=list)
    independent_subgraphs: list[list[str]] = Field(default_factory=list)

    def get_step(self, step_id: str) -> StepPlan:
        """Look up a step by its ID. Raises ValueError if not found."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise ValueError(f"Step '{step_id}' not found in workflow plan")

    def compute_topological_order(self) -> list[str]:
        """Compute topological order using Kahn's algorithm. Returns step_ids."""
        in_degree: dict[str, int] = {s.step_id: 0 for s in self.steps}
        adjacency: dict[str, list[str]] = {s.step_id: [] for s in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                adjacency[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.steps):
            visited = set(order)
            cycle_nodes = [s.step_id for s in self.steps if s.step_id not in visited]
            raise ValueError(
                f"Cycle detected in workflow plan. "
                f"Nodes involved in cycle: {cycle_nodes}"
            )

        self.topological_order = order
        return order

    def compute_independent_subgraphs(self) -> list[list[str]]:
        """Find disconnected components in the workflow DAG using union-find."""
        parent: dict[str, str] = {s.step_id: s.step_id for s in self.steps}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for step in self.steps:
            for dep in step.depends_on:
                union(step.step_id, dep)

        groups: dict[str, list[str]] = {}
        for step in self.steps:
            root = find(step.step_id)
            groups.setdefault(root, []).append(step.step_id)

        if self.topological_order:
            topo_idx = {sid: i for i, sid in enumerate(self.topological_order)}
            for root in groups:
                groups[root].sort(key=lambda s: topo_idx.get(s, 0))

        self.independent_subgraphs = list(groups.values())
        return self.independent_subgraphs


# ---------------------------------------------------------------------------
# Phase 2 models: ValidatedStep (concrete params)
# ---------------------------------------------------------------------------


class ValidatedStep(BaseModel):
    """Phase 2 output: a fully validated service step with concrete parameters."""

    step_id: str
    service_name: str
    api_name: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    output_patterns: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    auto_corrections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Intent classification (pre-dispatch before 3-phase pipeline)
# ---------------------------------------------------------------------------


class Intent(BaseModel):
    """Result of the lightweight intent classifier."""

    action: Literal["plan", "submit", "status", "cancel", "modify", "unknown"] = "plan"
    workflow_id: str | None = None
    confidence: float = 1.0
    reasoning: str = ""
    submit_after_plan: bool = False


# ---------------------------------------------------------------------------
# Information request (structured question for the user)
# ---------------------------------------------------------------------------


class InformationRequest(BaseModel):
    """Structured request for user input. Returned when the agent needs more info."""

    status: str = "needs_input"
    question: str
    context: str = ""
    options: list[str] | None = None
    partial_state: dict | None = None


# ---------------------------------------------------------------------------
# Agent state (serializable for pause/resume)
# ---------------------------------------------------------------------------


class AgentState(BaseAgentState):
    """Tracks the full state of a three-phase agent execution.

    Fully serializable so the orchestrator can pause after Phase 1 or
    mid-Phase 2, store the state, and resume after getting user input.
    """

    # Phase tracking
    current_phase: Literal["decompose", "build", "compose", "done"] = "decompose"

    # Phase 1 output
    workflow_plan: WorkflowPlan | None = None

    # Phase 2 progress
    completed_steps: dict[str, ValidatedStep] = Field(default_factory=dict)
    current_step_id: str | None = None
    current_step_messages: list[dict[str, Any]] = Field(default_factory=list)

    # Phase 3 output
    manifest: dict | None = None

    # Workflow engine persistence
    workflow_id: str | None = None
    persisted: bool = False

    # GoWe-specific state
    submission_id: str | None = None
    submission_ids: list[str] = Field(default_factory=list)
    cwl_document: dict | None = None
    submission_inputs: dict | None = None

    # Agent result status (override base with service-specific literals)
    status: Literal["in_progress", "needs_input", "completed", "error"] = "in_progress"  # type: ignore[assignment]
    question: str | None = None
    error_message: str | None = None

    # Lifecycle operation output (non-planning paths)
    operation_message: str | None = None
    auto_submitted: bool = False

    # Intent from classifier
    classified_intent: Intent | None = None

    # -----------------------------------------------------------------------
    # Phase 2 step management
    # -----------------------------------------------------------------------

    def remaining_steps(self) -> list[str]:
        """Step IDs not yet built, in topological order."""
        if not self.workflow_plan:
            return []
        return [
            s
            for s in self.workflow_plan.topological_order
            if s not in self.completed_steps
        ]

    def next_buildable_batches(self) -> list[list[str]]:
        """Yield batches of step_ids that can be built in parallel."""
        if not self.workflow_plan:
            return []

        remaining = set(self.remaining_steps())
        batches: list[list[str]] = []
        completed = set(self.completed_steps.keys())

        while remaining:
            batch = [
                s
                for s in remaining
                if all(
                    d in completed for d in self.workflow_plan.get_step(s).depends_on
                )
            ]
            if not batch:
                break
            batches.append(batch)
            for s in batch:
                remaining.discard(s)
                completed.add(s)

        return batches

    def get_upstream_outputs(self, step_id: str) -> dict[str, dict[str, str]]:
        """Get output patterns from all dependencies of step_id."""
        if not self.workflow_plan:
            return {}
        step = self.workflow_plan.get_step(step_id)
        return {
            dep: self.completed_steps[dep].output_patterns
            for dep in step.depends_on
            if dep in self.completed_steps
        }

    def mark_step_complete(self, step_id: str, validated_step: ValidatedStep) -> None:
        """Record a step as successfully built."""
        self.completed_steps[step_id] = validated_step
        if step_id == self.current_step_id:
            self.current_step_id = None
            self.current_step_messages = []

    # -----------------------------------------------------------------------
    # Result conversion
    # -----------------------------------------------------------------------

    def to_result(self) -> "AgentResult":
        """Convert current state to an AgentResult for the caller."""
        elapsed = time.time() - self.start_time

        # Collect unique services from completed steps
        sources: list[str] = []
        for step in self.completed_steps.values():
            if step.service_name not in sources:
                sources.append(step.service_name)

        return AgentResult(
            status=self.status,
            manifest=self.manifest,
            workflow_plan=(
                self.workflow_plan.model_dump() if self.workflow_plan else None
            ),
            completed_steps={
                sid: vs.model_dump() for sid, vs in self.completed_steps.items()
            },
            question=self.question,
            error_message=self.error_message,
            sources=sources,
            tool_trace=self.tool_executions,
            elapsed_seconds=round(elapsed, 2),
            workflow_id=self.workflow_id,
            persisted=self.persisted,
            operation_message=self.operation_message,
            auto_submitted=self.auto_submitted,
            submission_id=self.submission_id,
            submission_ids=self.submission_ids,
            cwl_document=self.cwl_document,
        )


# ---------------------------------------------------------------------------
# Submission result (returned after workflow engine submission)
# ---------------------------------------------------------------------------


class SubmissionResult(BaseModel):
    """Tracks the outcome of submitting a manifest to the workflow engine."""

    workflow_id: str
    submission_id: str | None = None
    status: str
    engine_url: str
    status_url: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Agent result (returned to caller / orchestrator)
# ---------------------------------------------------------------------------


class AgentResult(BaseAgentResult):
    """Returned by run_agent(). Clean interface for consumers.

    Status values:
      - "completed": Full workflow manifest ready in `manifest`
      - "needs_input": Agent needs user input; question in `question`
      - "error": Unrecoverable error; details in `error_message`
    """

    manifest: dict | None = None
    workflow_plan: dict | None = None
    completed_steps: dict[str, dict] = Field(default_factory=dict)
    question: str | None = None
    error_message: str | None = None
    submission: SubmissionResult | None = None
    workflow_id: str | None = None
    persisted: bool = False

    # GoWe-specific fields
    submission_id: str | None = None
    submission_ids: list[str] = Field(default_factory=list)
    cwl_document: dict | None = None

    # Lifecycle operation output
    operation_message: str | None = None
    auto_submitted: bool = False

    def pretty(self) -> str:
        """Human-readable summary for CLI output."""
        import json

        # Lifecycle operations (submit/status/cancel) return a concise message
        if self.operation_message:
            lines = [self.operation_message]
            if self.auto_submitted:
                lines.append("(auto-submitted)")
            return "\n".join(lines)

        lines = [
            f"Status: {self.status}",
            f"Elapsed: {self.elapsed_seconds}s",
        ]

        if self.sources:
            lines.append(f"Services: {', '.join(self.sources)}")

        if self.question:
            lines.extend(["", "--- QUESTION FOR USER ---", self.question])

        if self.error_message:
            lines.extend(["", "--- ERROR ---", self.error_message])

        if self.workflow_plan:
            lines.extend(["", "--- WORKFLOW PLAN ---"])
            plan = self.workflow_plan
            lines.append(f"  Name: {plan.get('workflow_name', 'unnamed')}")
            lines.append(f"  Description: {plan.get('description', '')}")
            steps = plan.get("steps", [])
            lines.append(f"  Steps ({len(steps)}):")
            for s in steps:
                deps = s.get("depends_on", [])
                dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                lines.append(
                    f"    - {s.get('step_id')}: {s.get('service_name')}"
                    f" -- {s.get('intent')}{dep_str}"
                )
            topo = plan.get("topological_order", [])
            if topo:
                lines.append(f"  Build order: {' -> '.join(topo)}")

        if self.completed_steps:
            lines.extend(["", "--- VALIDATED STEPS ---"])
            for sid, step_data in self.completed_steps.items():
                lines.append(f"\n  [{sid}] {step_data.get('service_name', '?')}")
                if step_data.get("auto_corrections"):
                    lines.append(
                        f"    Auto-corrections: "
                        f"{', '.join(step_data['auto_corrections'])}"
                    )
                if step_data.get("warnings"):
                    lines.append(f"    Warnings: {', '.join(step_data['warnings'])}")
                params = step_data.get("params", {})
                params_str = json.dumps(params, indent=6, default=str)
                if len(params_str) > 500:
                    params_str = params_str[:500] + "\n      ... [truncated]"
                lines.append(f"    Params: {params_str}")

        if self.manifest:
            lines.extend(["", "--- WORKFLOW MANIFEST ---"])
            manifest_str = json.dumps(self.manifest, indent=2, default=str)
            if len(manifest_str) > 2000:
                manifest_str = manifest_str[:2000] + "\n... [truncated]"
            lines.append(manifest_str)

        if self.workflow_id:
            lines.extend(["", "--- ENGINE REGISTRATION ---"])
            lines.append(f"  Workflow ID: {self.workflow_id}")
            lines.append(f"  Persisted: {self.persisted}")
            if not self.persisted:
                lines.append(
                    "  WARNING: Planning did not register with the engine; "
                    "please re-plan."
                )

        if self.submission:
            lines.extend(["", "--- SUBMISSION ---"])
            lines.append(f"  Workflow ID: {self.submission.workflow_id}")
            lines.append(f"  Status: {self.submission.status}")
            lines.append(f"  Status URL: {self.submission.status_url}")
            if self.submission.error:
                lines.append(f"  Error: {self.submission.error}")

        if self.tool_trace:
            lines.extend(
                [
                    "",
                    f"--- TOOL EXECUTIONS ({len(self.tool_trace)}) ---",
                ]
            )
            for i, ex in enumerate(self.tool_trace, 1):
                tc = ex.tool_call
                duration = f" ({ex.duration_ms:.0f}ms)" if ex.duration_ms else ""
                lines.append(f"\n  {i}. {tc.name}{duration}")
                args_str = json.dumps(tc.arguments, default=str)
                if len(args_str) > 120:
                    args_str = json.dumps(tc.arguments, indent=4, default=str)
                lines.append(f"     Args: {args_str}")
                if ex.error:
                    lines.append(f"     ERROR: {ex.error}")

        return "\n".join(lines)
