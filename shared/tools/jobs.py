"""Moved to ``bvbrc_mcp.tools.jobs`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import shared.tools.jobs`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.jobs as _module

sys.modules[__name__] = _module
