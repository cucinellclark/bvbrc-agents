"""Configuration for the LLM admission proxy.

All settings come from environment variables with sensible defaults.
No YAML file for v1 — keep it simple.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_upstream_urls() -> tuple[str, ...]:
    """Build the upstream URL list from environment variables.

    Priority:
      1. ``LLM_UPSTREAM_URLS`` — comma-separated list of upstream base URLs.
         Example: ``http://mango:8004/v1,http://mango:8005/v1``
      2. ``LLM_UPSTREAM_URL`` — single upstream (backward compatible).
      3. Default: ``http://mango.cels.anl.gov:8004/v1``
    """
    multi = os.environ.get("LLM_UPSTREAM_URLS", "").strip()
    if multi:
        urls = tuple(u.strip() for u in multi.split(",") if u.strip())
        if urls:
            return urls

    single = os.environ.get(
        "LLM_UPSTREAM_URL", "http://mango.cels.anl.gov:8004/v1"
    ).strip()
    return (single,)


@dataclass(frozen=True)
class AdmissionConfig:
    """Immutable proxy configuration, resolved once at startup."""

    # Upstream vLLM endpoints.  Requests are distributed round-robin.
    upstream_urls: tuple[str, ...] = field(default_factory=_parse_upstream_urls)

    # Max concurrent requests forwarded across ALL upstreams (shared budget).
    max_in_flight: int = int(os.environ.get("ADMISSION_MAX_IN_FLIGHT", "4"))

    # Max requests waiting in the semaphore queue.
    # Over this, the proxy returns 429 immediately.
    max_waiting: int = int(os.environ.get("ADMISSION_MAX_WAITING", "16"))

    # Proxy listen port on holly.
    port: int = int(os.environ.get("ADMISSION_PORT", "8005"))

    # Bind host.
    host: str = os.environ.get("ADMISSION_HOST", "0.0.0.0")

    # ── Backward-compat property ──────────────────────────────────────
    @property
    def upstream_url(self) -> str:
        """Return the first upstream URL (backward compat for logs, etc.)."""
        return self.upstream_urls[0]

    def __post_init__(self) -> None:
        if self.max_in_flight < 1:
            raise ValueError(f"max_in_flight must be >= 1, got {self.max_in_flight}")
        if self.max_waiting < 0:
            raise ValueError(f"max_waiting must be >= 0, got {self.max_waiting}")
        if not self.upstream_urls:
            raise ValueError("At least one upstream URL is required")
        for url in self.upstream_urls:
            if not url:
                raise ValueError("upstream URL must not be empty")


def load_config() -> AdmissionConfig:
    """Create config from current environment."""
    return AdmissionConfig()
