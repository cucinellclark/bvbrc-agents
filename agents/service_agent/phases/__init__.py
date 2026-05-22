"""Three-phase workflow construction for the Service Agent v2.

Phase 1: Decompose (request -> WorkflowPlan)
Phase 2: Build (WorkflowPlan -> ValidatedStep[])
Phase 3: Compose (ValidatedStep[] -> manifest)
"""

from service_agent.phases.decompose import decompose
from service_agent.phases.build import build_step
from service_agent.phases.compose import compose_manifest

__all__ = ["decompose", "build_step", "compose_manifest"]
