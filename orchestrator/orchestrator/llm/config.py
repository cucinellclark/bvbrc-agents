"""LLM configuration for the orchestrator's routing and synthesis calls.

Loads defaults from the shared Agents/config/llm.yaml so that the model
endpoint is configured in one place for the entire system. Environment
variables (LLM_BASE_URL, LLM_MODEL, etc.) override the YAML values.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

# Make the shared config loader importable
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "config")
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

from llm_config import load_llm_defaults  # noqa: E402

_DEFAULTS = load_llm_defaults()


class LLMConfig(BaseModel):
    """Configuration for the orchestrator's LLM calls.

    Defaults are loaded from Agents/config/llm.yaml. Override via
    constructor kwargs, environment variables, or YAML edits.
    """

    # Endpoint (OpenAI-compatible)
    base_url: str = _DEFAULTS["base_url"]
    api_key: str = _DEFAULTS["api_key"]
    model: str = _DEFAULTS["model"]

    # Generation settings
    temperature: float = _DEFAULTS["temperature"]
    max_tokens: int = _DEFAULTS["max_tokens"]

    # Timeouts
    timeout_seconds: int = _DEFAULTS["timeout_seconds"]
