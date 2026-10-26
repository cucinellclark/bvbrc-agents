"""Moved to ``bvbrc_mcp.tools.helpdesk`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import helpdesk_agent.tools.helpdesk`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.helpdesk as _module

sys.modules[__name__] = _module
