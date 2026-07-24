"""Service agent workflow phases.

Active phase: populate (GoWe-first discover/select/populate/submit).
"""

from service_agent.phases.populate import populate_and_submit

__all__ = ["populate_and_submit"]
