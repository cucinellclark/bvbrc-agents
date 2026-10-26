"""Moved to ``bvbrc_mcp.tools.output_knowledge`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import analysis_agent.output_knowledge`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.output_knowledge as _module

sys.modules[__name__] = _module
