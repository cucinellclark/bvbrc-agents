"""Moved to ``bvbrc_mcp.tools.url_utils`` (the bvbrc-mcp package).

This module name is an alias for it, so existing imports and patches keep
working: ``import shared.tools.url_utils`` returns the package module itself.
"""

import sys

import bvbrc_mcp.tools.url_utils as _module

sys.modules[__name__] = _module
