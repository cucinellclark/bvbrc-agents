"""Shared LLM configuration loader for the orchestrator and all agents.

Reads LLM settings from a central YAML file (Agents/config/llm.yaml) with
environment variable overrides. This module is imported by each agent's
AgentConfig and the orchestrator's LLMConfig to provide consistent defaults.

Resolution order (highest priority wins):
  1. Environment variables: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
     LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS
  2. YAML config file (default: Agents/config/llm.yaml, or set LLM_CONFIG_PATH)
  3. Hardcoded fallback defaults

Usage from any agent:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
    from llm_config import load_llm_defaults
    defaults = load_llm_defaults()
    # defaults is a dict: {"base_url": ..., "api_key": ..., "model": ..., ...}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Default config file: Agents/config/llm.yaml (next to this module)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "llm.yaml"

# Hardcoded fallback values (used only if YAML is missing AND no env vars set)
_FALLBACK_DEFAULTS: dict[str, Any] = {
    "base_url": "http://mango.cels.anl.gov:8004/v1",
    "api_key": "not-needed",
    "model": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "temperature": 0.0,
    "max_tokens": 4096,
    "timeout_seconds": 60,
}

# Mapping from environment variable names to config keys
_ENV_VAR_MAP: dict[str, str] = {
    "LLM_BASE_URL": "base_url",
    "LLM_API_KEY": "api_key",
    "LLM_MODEL": "model",
    "LLM_TEMPERATURE": "temperature",
    "LLM_MAX_TOKENS": "max_tokens",
    "LLM_TIMEOUT_SECONDS": "timeout_seconds",
}

# Type coercion for numeric fields
_TYPE_MAP: dict[str, type] = {
    "temperature": float,
    "max_tokens": int,
    "timeout_seconds": int,
}


# ---------------------------------------------------------------------------
# Model-specific parameter handling
#
# The Argo Gateway API has per-model parameter rules. Some models reject
# "max_tokens" but accept "max_completion_tokens"; some reject "temperature".
#
# Matching is case-insensitive and uses substring containment, so "gpt5"
# will match model names like "gpt5", "gpt5mini", "gpt51", etc.
# ---------------------------------------------------------------------------

# Parameters to exclude entirely from the API call
MODEL_PARAM_EXCLUSIONS: dict[str, set[str]] = {
    "gpt5": {"max_tokens", "temperature"},
    "o3": {"temperature", "max_tokens"},
    "o4-mini": {"temperature", "max_tokens"},
}

# Models that accept "max_completion_tokens" instead of "max_tokens"
MODEL_USE_MAX_COMPLETION_TOKENS: set[str] = {"gpt5", "o3", "o4-mini", "gpt41"}

# Models that require a fixed temperature value (applied only if temperature
# is not excluded entirely). More-specific patterns are checked first.
MODEL_TEMPERATURE_OVERRIDE: dict[str, float] = {
    "gpt5": 1.0,
}


def get_excluded_params(model: str) -> set[str]:
    """Return the set of parameter names to exclude for a given model.

    Args:
        model: The model name string (e.g. "gpt5", "gpt41").

    Returns:
        Set of parameter names to omit (e.g. {"temperature", "max_tokens"}).
        Empty set if no exclusions apply.
    """
    model_lower = model.lower()
    excluded: set[str] = set()
    for pattern, params in MODEL_PARAM_EXCLUSIONS.items():
        if pattern in model_lower:
            excluded |= params
    return excluded


def uses_max_completion_tokens(model: str) -> bool:
    """Return True if this model requires max_completion_tokens instead of max_tokens."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in MODEL_USE_MAX_COMPLETION_TOKENS)


def get_temperature_override(model: str) -> float | None:
    """Return a forced temperature value for models that require it, or None."""
    model_lower = model.lower()
    for pattern, value in MODEL_TEMPERATURE_OVERRIDE.items():
        if pattern in model_lower:
            return value
    return None


def load_llm_defaults(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load LLM defaults from YAML + environment variables.

    Args:
        config_path: Explicit path to the YAML config file.
            Falls back to LLM_CONFIG_PATH env var, then Agents/config/llm.yaml.

    Returns:
        Dict with keys: base_url, api_key, model, temperature, max_tokens,
        timeout_seconds.
    """
    # Start with hardcoded fallbacks
    defaults = dict(_FALLBACK_DEFAULTS)

    # Layer 1: YAML file
    path = Path(
        config_path
        or os.environ.get("LLM_CONFIG_PATH", "")
        or _DEFAULT_CONFIG_PATH
    )
    if path.exists():
        try:
            import yaml
        except ImportError:
            # PyYAML not available -- skip file loading, rely on env vars
            yaml = None  # type: ignore[assignment]

        if yaml is not None:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            llm_section = raw.get("llm", {})
            if isinstance(llm_section, dict):
                for key in _FALLBACK_DEFAULTS:
                    if key in llm_section:
                        defaults[key] = llm_section[key]

    # Layer 2: Environment variable overrides (always win)
    for env_var, config_key in _ENV_VAR_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Coerce numeric types
            if config_key in _TYPE_MAP:
                try:
                    value = _TYPE_MAP[config_key](value)  # type: ignore[assignment]
                except (ValueError, TypeError):
                    pass  # Keep string value; Pydantic will catch the error
            defaults[config_key] = value

    return defaults
