"""Moved to ``bvbrc_mcp.tools.workspace`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import shared.tools.workspace`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.workspace as _module

sys.modules[__name__] = _module
