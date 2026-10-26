"""Moved to ``bvbrc_mcp.tools._clients`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import shared.tools._mcp_imports`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools._clients as _module

sys.modules[__name__] = _module
