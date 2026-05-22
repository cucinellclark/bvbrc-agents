"""Async stream utilities for composing event generators."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TypeVar

from orchestrator.events.events import Event

T = TypeVar("T")


async def merge_streams(
    *streams: AsyncGenerator[Event, None],
) -> AsyncGenerator[Event, None]:
    """Merge multiple async event streams into a single ordered stream.

    Events are yielded in arrival order (whichever stream produces first).
    All streams are consumed concurrently.
    """
    queue: asyncio.Queue[Event | None] = asyncio.Queue()
    active = len(streams)

    async def _consume(stream: AsyncGenerator[Event, None]) -> None:
        nonlocal active
        try:
            async for event in stream:
                await queue.put(event)
        finally:
            active -= 1
            if active == 0:
                await queue.put(None)  # Sentinel: all streams exhausted

    tasks = [asyncio.create_task(_consume(s)) for s in streams]

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def collect_events(
    stream: AsyncGenerator[Event, None],
) -> list[Event]:
    """Consume an entire event stream and return all events as a list."""
    events: list[Event] = []
    async for event in stream:
        events.append(event)
    return events
