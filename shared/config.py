"""Shared LLM configuration loader for all agents.

Provides a single import point so agents don't need to manipulate sys.path
to reach ``config/llm_config.py``.

Usage (from any agent):
    from shared.config import load_llm_defaults, LLM_DEFAULTS
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the config directory is on sys.path so ``llm_config`` is importable.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

from llm_config import (  # noqa: E402, F401  -- re-export
    load_llm_defaults,
    get_excluded_params,
    uses_max_completion_tokens,
    get_temperature_override,
)

# Eagerly loaded defaults -- agents can use this directly instead of
# calling ``load_llm_defaults()`` at module scope in every models.py.
LLM_DEFAULTS = load_llm_defaults()
