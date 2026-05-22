"""Plan execution engine.

Takes a Plan and executes steps respecting their dependency graph.
Independent steps run concurrently via asyncio. Dependent steps wait
for their upstream results, which are threaded into the downstream
agent's context.

Supports single-step plans (agent routing) and multi-step plans
(pipeline routing) with the same code path.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from orchestrator.events.events import Event, EventType, error_event
from orchestrator.executor.agent_executor import execute_agent_step
from orchestrator.models import OrchestratorRequest
from orchestrator.registry.agent_registry import AgentRegistry
from orchestrator.router.models import Plan, Step

logger = logging.getLogger(__name__)


async def execute_plan(
    plan: Plan,
    registry: AgentRegistry,
    request: OrchestratorRequest,
) -> AsyncGenerator[Event, None]:
    """Execute all steps in a plan, respecting dependencies.

    Steps with no unmet dependencies are launched concurrently.
    When a step completes, any steps that were waiting solely on it
    become eligible to run. Upstream results are passed to downstream
    steps via the ``upstream_results`` context field.

    If a step fails, dependent steps are skipped (not the whole
    pipeline). Independent steps continue to run.

    Args:
        plan: The execution plan from the router.
        registry: Agent registry for looking up agent handles.
        request: The original orchestrator request.

    Yields:
        Event objects from each step's execution.
    """
    num_steps = len(plan.steps)

    # Fast path: single-step plan (no DAG overhead)
    if num_steps == 1:
        async for event in _run_single_step(plan.steps[0], 0, registry, request):
            yield event
        return

    # --- Multi-step DAG execution ---

    yield Event(
        type=EventType.AGENT_PROGRESS,
        data={
            "message": f"Executing pipeline with {num_steps} steps",
            "step_count": num_steps,
        },
    )

    # Per-step state
    results: dict[int, dict[str, Any]] = {}   # step_index -> result dict
    failed: set[int] = set()                   # step indices that failed
    completed: set[int] = set()                # step indices that finished (success or fail)
    events_by_step: dict[int, list[Event]] = {i: [] for i in range(num_steps)}

    # Build reverse dependency map: step_index -> set of steps waiting on it
    dependents: dict[int, set[int]] = {i: set() for i in range(num_steps)}
    for i, step in enumerate(plan.steps):
        for dep in step.depends_on:
            dependents[dep].add(i)

    def _ready(step_index: int) -> bool:
        """Check if a step's dependencies are all satisfied."""
        step = plan.steps[step_index]
        for dep in step.depends_on:
            if dep not in completed:
                return False
        return True

    def _blocked(step_index: int) -> bool:
        """Check if a step has a failed dependency."""
        step = plan.steps[step_index]
        for dep in step.depends_on:
            if dep in failed:
                return True
        return False

    async def _run_step(step_index: int) -> list[Event]:
        """Run a single step, collecting its events into a list."""
        step = plan.steps[step_index]
        collected: list[Event] = []

        # Gather upstream results for context threading
        upstream: dict[str, Any] = {}
        for dep in step.depends_on:
            if dep in results:
                dep_agent = plan.steps[dep].agent_key
                upstream[f"step_{dep}_{dep_agent}"] = results[dep]

        async for event in _run_single_step(
            step, step_index, registry, request, upstream_results=upstream,
        ):
            collected.append(event)

        return collected

    # Seed the ready queue with steps that have no dependencies
    pending = set(range(num_steps))
    running: dict[int, asyncio.Task] = {}

    while pending or running:
        # Launch any steps that are now ready
        newly_ready = [i for i in list(pending) if _ready(i) and not _blocked(i)]
        for i in newly_ready:
            pending.discard(i)
            running[i] = asyncio.create_task(_run_step(i))

        # Skip steps that are blocked by failed dependencies
        newly_blocked = [i for i in list(pending) if _blocked(i)]
        for i in newly_blocked:
            pending.discard(i)
            failed.add(i)
            completed.add(i)
            skip_event = Event(
                type=EventType.AGENT_PROGRESS,
                step_index=i,
                agent_name=plan.steps[i].agent_key,
                data={
                    "agent": plan.steps[i].agent_key,
                    "message": (
                        f"Skipping step {i} ({plan.steps[i].agent_key}) — "
                        f"upstream dependency failed"
                    ),
                    "skipped": True,
                },
            )
            yield skip_event

        if not running:
            break

        # Wait for at least one running task to complete
        done, _ = await asyncio.wait(
            running.values(), return_when=asyncio.FIRST_COMPLETED,
        )

        # Process completed tasks
        for task in done:
            # Find which step this task belongs to
            step_index = next(i for i, t in running.items() if t is task)
            del running[step_index]
            completed.add(step_index)

            step_events = task.result()
            events_by_step[step_index] = step_events

            # Check if this step succeeded or failed
            step_failed = False
            for event in step_events:
                if event.type == EventType.ORCHESTRATOR_ERROR:
                    step_failed = True
                if event.type == EventType.AGENT_RESULT:
                    results[step_index] = event.data.get("result_for_ui", {})

            if step_failed:
                failed.add(step_index)

            # Yield all events from this step
            for event in step_events:
                yield event


async def _run_single_step(
    step: Step,
    step_index: int,
    registry: AgentRegistry,
    request: OrchestratorRequest,
    upstream_results: dict[str, Any] | None = None,
) -> AsyncGenerator[Event, None]:
    """Execute a single step, handling agent lookup and health checks.

    Args:
        step: The step to execute.
        step_index: Index of this step in the plan.
        registry: Agent registry for looking up agent handles.
        request: The original orchestrator request.
        upstream_results: Results from upstream steps to thread into context.

    Yields:
        Event objects from the step's execution.
    """
    # Look up the agent
    try:
        agent = registry.get(step.agent_key)
    except KeyError:
        yield error_event(
            message=(
                f"Agent '{step.agent_key}' not found in registry. "
                f"Available: {', '.join(registry.agent_keys)}"
            ),
        )
        return

    # Check agent health
    if not agent.is_healthy:
        yield Event(
            type=EventType.AGENT_PROGRESS,
            agent_name=agent.key,
            step_index=step_index,
            data={
                "agent": agent.key,
                "warning": f"Agent '{agent.key}' is unhealthy, attempting anyway",
            },
        )

    # Execute the step
    async for event in execute_agent_step(
        step=step,
        agent=agent,
        request=request,
        step_index=step_index,
        upstream_results=upstream_results,
    ):
        yield event
