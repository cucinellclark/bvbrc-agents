"""Moved to ``bvbrc_mcp.tools.collection_reference`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import data_agent.prompts.collection_reference`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.collection_reference as _module

sys.modules[__name__] = _module
