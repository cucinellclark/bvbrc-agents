"""Moved to ``bvbrc_mcp.tools.field_registry`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import data_agent.prompts.field_registry`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.field_registry as _module

sys.modules[__name__] = _module
