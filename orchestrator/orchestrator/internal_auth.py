"""Gateway-to-orchestrator authentication and LLM endpoint allowlisting.

The orchestrator listens on a routable interface (the gateway runs on a
different host), so it cannot rely on binding to localhost. Two checks keep
it from being used directly:

* **Internal key.** Every request except ``GET /health`` must carry the
  shared secret in the ``X-Copilot-Internal-Key`` header. The gateway reads
  the same secret from its own copy of the key file. The key comes from
  ``COPILOT_INTERNAL_KEY`` if set, otherwise from the file named by
  ``COPILOT_INTERNAL_KEY_FILE`` (default ``~/.copilot_internal_key``).
  The orchestrator refuses to start without one.

* **LLM host allowlist.** ``llm_override.base_url`` arrives in the request
  body. Without a check, a caller could point the agents at a server it
  controls, which would then choose the tool calls. Only hosts listed in
  ``orchestrator.llm_allowed_hosts`` (agents.yaml) are accepted.
"""

from __future__ import annotations

import hmac
import logging
import os
import stat
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

INTERNAL_KEY_HEADER = "X-Copilot-Internal-Key"
DEFAULT_KEY_FILE = Path.home() / ".copilot_internal_key"

# Paths reachable without the internal key. Health carries no user data
# and is used by the gateway's readiness probe.
UNAUTHENTICATED_PATHS = frozenset({"/health"})


def load_internal_key() -> str | None:
    """Return the shared gateway/orchestrator secret, or None if unset."""
    key = os.environ.get("COPILOT_INTERNAL_KEY", "").strip()
    if key:
        return key

    key_file = Path(
        os.environ.get("COPILOT_INTERNAL_KEY_FILE", "") or DEFAULT_KEY_FILE
    ).expanduser()
    if not key_file.exists():
        return None

    mode = key_file.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(
            f"Internal key file {key_file} is readable by other users; "
            f"run: chmod 600 {key_file}"
        )
    return key_file.read_text().strip() or None


def internal_key_matches(expected: str | None, supplied: str | None) -> bool:
    """Constant-time comparison; False when either side is missing."""
    if not expected or not supplied:
        return False
    return hmac.compare_digest(expected.encode(), supplied.encode())


def llm_host_allowed(base_url: str, allowed_hosts: list[str]) -> bool:
    """True if ``base_url`` is http(s) on a host in ``allowed_hosts``.

    Hosts are compared by hostname only (case-insensitive), so any port on
    an allowed host is accepted — mango serves models on several ports.
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    allowed = {h.strip().lower() for h in allowed_hosts if h and h.strip()}
    return parsed.hostname.lower() in allowed
