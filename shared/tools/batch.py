"""Moved to ``bvbrc_mcp.tools.batch`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import shared.tools.batch`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.batch as _module

sys.modules[__name__] = _module
