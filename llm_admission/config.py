"""Configuration for the LLM admission proxy.

All settings come from environment variables with sensible defaults.
No YAML file for v1 — keep it simple.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionConfig:
    """Immutable proxy configuration, resolved once at startup."""

    # Real vLLM endpoint (single upstream for v1).
    upstream_url: str = os.environ.get(
        "LLM_UPSTREAM_URL", "http://mango.cels.anl.gov:8004/v1"
    )

    # Max concurrent requests forwarded to upstream.
    max_in_flight: int = int(os.environ.get("ADMISSION_MAX_IN_FLIGHT", "4"))

    # Max requests waiting in the semaphore queue.
    # Over this, the proxy returns 429 immediately.
    max_waiting: int = int(os.environ.get("ADMISSION_MAX_WAITING", "16"))

    # Proxy listen port on holly.
    port: int = int(os.environ.get("ADMISSION_PORT", "8005"))

    # Bind host.
    host: str = os.environ.get("ADMISSION_HOST", "0.0.0.0")

    def __post_init__(self) -> None:
        if self.max_in_flight < 1:
            raise ValueError(f"max_in_flight must be >= 1, got {self.max_in_flight}")
        if self.max_waiting < 0:
            raise ValueError(f"max_waiting must be >= 0, got {self.max_waiting}")
        if not self.upstream_url:
            raise ValueError("upstream_url must not be empty")


def load_config() -> AdmissionConfig:
    """Create config from current environment."""
    return AdmissionConfig()
