"""Moved to ``bvbrc_mcp.tools.solr_tools`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import data_agent.mcp_tools.agent_data_tools`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.solr_tools as _module

sys.modules[__name__] = _module
