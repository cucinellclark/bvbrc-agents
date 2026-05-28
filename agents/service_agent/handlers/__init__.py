"""Lifecycle operation handlers for the BV-BRC Service Agent v2.

Each handler performs a direct workflow engine operation without running
the 3-phase planning pipeline.  Handlers return AgentResult objects that
are indistinguishable from planning results to upstream consumers.
"""

from service_agent.handlers.submit import handle_submit
from service_agent.handlers.status import handle_status
from service_agent.handlers.cancel import handle_cancel

__all__ = ["handle_submit", "handle_status", "handle_cancel"]
