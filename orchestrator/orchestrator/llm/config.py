"""LLM configuration for the orchestrator's routing and synthesis calls.

Loads structural defaults (temperature, max_tokens, timeout_seconds) from
the shared Agents/config/llm.yaml.  Model / URL / API-key are always
supplied per-request via ``llm_override`` — the class defaults here are
placeholders that allow the class definition to load without error.
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

    Callers (routing_config, override_config in server.py) always pass
    base_url, api_key, and model explicitly.  The placeholder defaults
    below exist only so this class can be imported without a KeyError
    when llm.yaml omits endpoint fields (they arrive per-request).
    """

    # Endpoint (OpenAI-compatible) — always overridden by callers
    base_url: str = _DEFAULTS.get("base_url", "")
    api_key: str = _DEFAULTS.get("api_key", "")
    model: str = _DEFAULTS.get("model", "")

    # Generation settings
    temperature: float = _DEFAULTS.get("temperature", 0.0)
    max_tokens: int = _DEFAULTS.get("max_tokens", 16384)

    # Timeouts
    timeout_seconds: int = _DEFAULTS.get("timeout_seconds", 180)
