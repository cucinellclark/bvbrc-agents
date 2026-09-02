"""OpenAI-compatible admission proxy for BV-BRC Copilot.

Sits on holly between the orchestrator/agents and vLLM on mango.
Gates ``/v1/chat/completions`` through a bounded semaphore so vLLM
never sees more than ``max_in_flight`` concurrent requests.

All other paths (``/v1/models``, etc.) pass through ungated.

Run via::

    uvicorn llm_admission.proxy:app --host 0.0.0.0 --port 8005

Or via ``start_admission_proxy.sh``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_admission.config import AdmissionConfig, load_config

logger = logging.getLogger("llm_admission")

# ---------------------------------------------------------------------------
# Module-level state (set during lifespan)
# ---------------------------------------------------------------------------

_config: AdmissionConfig | None = None
_semaphore: asyncio.Semaphore | None = None
_http_client: httpx.AsyncClient | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown — create the semaphore and httpx client."""
    global _config, _semaphore, _http_client

    _config = load_config()
    _semaphore = asyncio.Semaphore(_config.max_in_flight)

    # Long-lived connection pool to the upstream vLLM.
    # max_connections slightly above max_in_flight so we never block on
    # the pool after acquiring the semaphore.
    _http_client = httpx.AsyncClient(
        base_url=_config.upstream_url,
        timeout=httpx.Timeout(
            connect=10.0,
            read=300.0,   # vLLM streams can be long (thinking models)
            write=10.0,
            pool=30.0,    # wait for a pool connection
        ),
        limits=httpx.Limits(
            max_connections=_config.max_in_flight + 2,
            max_keepalive_connections=_config.max_in_flight + 2,
        ),
        follow_redirects=False,
        http2=False,
    )

    logger.info(
        "Admission proxy started: upstream=%s  max_in_flight=%d  max_waiting=%d  port=%d",
        _config.upstream_url,
        _config.max_in_flight,
        _config.max_waiting,
        _config.port,
    )

    yield

    # Shutdown
    await _http_client.aclose()
    logger.info("Admission proxy stopped")


app = FastAPI(title="LLM Admission Proxy", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _waiters_count() -> int:
    """Return the number of coroutines waiting on the semaphore.

    In CPython 3.11+, ``Semaphore._waiters`` starts as ``None`` and is
    lazily initialized to a ``deque`` when the first waiter blocks.
    """
    assert _semaphore is not None
    waiters = _semaphore._waiters
    return len(waiters) if waiters is not None else 0


# ---------------------------------------------------------------------------
# /health — metrics endpoint (manual curl, no Prometheus for v1)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Report admission metrics."""
    assert _config is not None and _semaphore is not None
    in_flight = _config.max_in_flight - _semaphore._value
    waiting = _waiters_count()
    return {
        "status": "ok",
        "in_flight": in_flight,
        "waiting": waiting,
        "max_in_flight": _config.max_in_flight,
        "max_waiting": _config.max_waiting,
        "upstream": _config.upstream_url,
    }


# ---------------------------------------------------------------------------
# /v1/chat/completions — gated reverse proxy
# ---------------------------------------------------------------------------


def _reject_queue_full() -> JSONResponse:
    """429 response when the wait queue is full."""
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "10"},
        content={
            "error": {
                "message": "Model is busy, please retry shortly",
                "type": "admission_queue_full",
                "code": 429,
            }
        },
    )


@app.api_route(
    "/v1/chat/completions",
    methods=["POST"],
    # Disable default validation — we forward the raw body.
    include_in_schema=False,
)
async def chat_completions(request: Request):
    """Admission-gated reverse proxy for /v1/chat/completions."""
    assert _config is not None and _semaphore is not None and _http_client is not None

    # ── Check bounded wait queue ──────────────────────────────────────
    # Safe without locking: single asyncio event loop, no preemption
    # between this check and the acquire() call below.
    current_waiting = _waiters_count()
    if current_waiting >= _config.max_waiting:
        logger.warning(
            "Rejecting request: queue full (waiting=%d, max_waiting=%d)",
            current_waiting,
            _config.max_waiting,
        )
        return _reject_queue_full()

    # ── Wait for a slot ───────────────────────────────────────────────
    # If the caller disconnects while waiting, asyncio raises
    # CancelledError which automatically removes us from the
    # semaphore's internal waiters deque.
    wait_start = time.monotonic()
    try:
        await _semaphore.acquire()
    except asyncio.CancelledError:
        logger.info("Client disconnected while waiting in queue")
        raise

    wait_secs = time.monotonic() - wait_start
    if wait_secs > 1.0:
        logger.info("Slot acquired after %.1fs wait", wait_secs)

    # ── Forward to upstream ───────────────────────────────────────────
    try:
        return await _forward_to_upstream(request)
    finally:
        _semaphore.release()


async def _forward_to_upstream(request: Request) -> StreamingResponse | JSONResponse:
    """Stream the upstream response back to the caller.

    If the caller disconnects mid-stream, the httpx request is cancelled
    which signals vLLM's abort-on-disconnect.
    """
    assert _http_client is not None

    body = await request.body()

    # Forward relevant headers (Authorization, Content-Type).
    forward_headers: dict[str, str] = {}
    if auth := request.headers.get("authorization"):
        forward_headers["Authorization"] = auth
    forward_headers["Content-Type"] = request.headers.get(
        "content-type", "application/json"
    )
    # Forward Accept header — important for SSE streaming detection.
    if accept := request.headers.get("accept"):
        forward_headers["Accept"] = accept

    try:
        upstream_req = _http_client.build_request(
            method="POST",
            url="/chat/completions",
            content=body,
            headers=forward_headers,
        )
        upstream_resp = await _http_client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        logger.error("Upstream timeout")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Upstream LLM timed out",
                    "type": "upstream_timeout",
                    "code": 504,
                }
            },
        )
    except httpx.ConnectError as exc:
        logger.error("Upstream connection error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Cannot reach upstream LLM: {exc}",
                    "type": "upstream_unreachable",
                    "code": 502,
                }
            },
        )
    except asyncio.CancelledError:
        logger.info("Client disconnected during upstream request")
        raise

    # ── Stream the response back ──────────────────────────────────────
    # We stream regardless of whether upstream returns SSE or a single
    # JSON response. This keeps the proxy simple and avoids buffering
    # large responses.

    response_headers: dict[str, str] = {}
    content_type = upstream_resp.headers.get("content-type", "")
    if content_type:
        response_headers["Content-Type"] = content_type

    async def _stream_chunks() -> AsyncIterator[bytes]:
        """Yield chunks from the upstream response.

        If the downstream client disconnects, the generator is closed
        by Starlette, which cancels the asyncio task. httpx detects the
        cancellation and closes the upstream connection, signaling
        abort-on-disconnect to vLLM.
        """
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        except asyncio.CancelledError:
            logger.info("Client disconnected during streaming")
            raise
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        content=_stream_chunks(),
        status_code=upstream_resp.status_code,
        headers=response_headers,
    )


# ---------------------------------------------------------------------------
# Catch-all — ungated passthrough for /v1/models, etc.
# ---------------------------------------------------------------------------


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def passthrough(request: Request, path: str):
    """Ungated passthrough for all non-completions endpoints."""
    assert _http_client is not None

    body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None

    # Forward all headers except Host.
    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() != "host"
    }

    # Strip the leading "v1/" prefix from the path — the httpx client's
    # base_url already includes "/v1", so forwarding "v1/models" as-is
    # would produce "/v1/v1/models".
    upstream_path = path
    if upstream_path.startswith("v1/"):
        upstream_path = upstream_path[3:]  # "v1/models" -> "models"

    try:
        upstream_req = _http_client.build_request(
            method=request.method,
            url=f"/{upstream_path}",
            content=body,
            headers=forward_headers,
        )
        upstream_resp = await _http_client.send(upstream_req, stream=False)
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Upstream timeout", "type": "upstream_timeout"}},
        )
    except httpx.ConnectError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Cannot reach upstream: {exc}", "type": "upstream_unreachable"}},
        )

    return JSONResponse(
        status_code=upstream_resp.status_code,
        content=upstream_resp.json() if upstream_resp.headers.get("content-type", "").startswith("application/json") else {"raw": upstream_resp.text},
        headers={k: v for k, v in upstream_resp.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")},
    )


# ---------------------------------------------------------------------------
# __main__ support
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "llm_admission.proxy:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )
