"""Moved to ``bvbrc_mcp.tools.analysis`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import analysis_agent.tools`` returns the package module itself.
The analysis agent dispatches through the shared tool registry; the local
dispatch table that used to live here was unused.
"""

import sys

import bvbrc_mcp.tools.analysis as _module

sys.modules[__name__] = _module
