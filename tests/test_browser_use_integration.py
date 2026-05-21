"""Integration tests against the real browser-use cloud.

Skipped unless BROWSER_USE_API_KEY is in the environment.

Run:
    BROWSER_USE_API_KEY=bu_... uv run pytest tests/test_browser_use_integration.py -v
"""

import asyncio
import os
from pathlib import Path

import pytest

from harness_agent.browser_use import (
    BrowserSessionPollHandler,
    BrowserSessionPump,
    BrowserSessionResultWaiter,
    BrowserUseService,
    HttpxBrowserUseClient,
    SQLiteBrowserProfileStore,
    SQLiteBrowserSessionStore,
)
from harness_agent.bus import EventBus
from harness_agent.events import (
    BrowserSessionCompleted,
    BrowserSessionFailed,
    BrowserSessionPollDue,
    BrowserSessionStopped,
)
from harness_agent.store import SQLiteEventStore
from harness_agent.tools import BrowserRunInput


API_KEY = os.environ.get("BROWSER_USE_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not API_KEY, reason="BROWSER_USE_API_KEY not set"),
]


def _make_real_stack(tmp_path: Path):
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    profile_store = SQLiteBrowserProfileStore(tmp_path / "profiles.sqlite3")
    session_store = SQLiteBrowserSessionStore(tmp_path / "sessions.sqlite3")
    bus = EventBus(event_store)
    assert API_KEY is not None
    client = HttpxBrowserUseClient(api_key=API_KEY, timeout_seconds=60.0)
    waiter = BrowserSessionResultWaiter()
    bus.subscribe(BrowserSessionCompleted, waiter.handle_completed)
    bus.subscribe(BrowserSessionFailed, waiter.handle_failed)
    bus.subscribe(BrowserSessionStopped, waiter.handle_stopped)
    handler = BrowserSessionPollHandler(client=client, session_store=session_store)
    bus.subscribe(BrowserSessionPollDue, handler.handle_poll_due)
    service = BrowserUseService(
        bus=bus,
        client=client,
        profile_store=profile_store,
        session_store=session_store,
        result_waiter=waiter,
        profile_cap=5,
        default_model="bu-mini",
        default_run_timeout_seconds=600.0,
    )
    pump = BrowserSessionPump(session_store=session_store, bus=bus)
    return service, client, pump


async def _run_with_pump(pump, coro, *, timeout: float = 300.0, interval: float = 1.0):
    stop = asyncio.Event()

    async def ticker():
        while not stop.is_set():
            await pump.tick()
            await asyncio.sleep(interval)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    finally:
        stop.set()
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_real_browser_run_visits_a_simple_page(tmp_path: Path) -> None:
    service, client, pump = _make_real_stack(tmp_path)
    try:
        record = await _run_with_pump(
            pump,
            service.run(
                user_id="integration-test",
                conversation_id="ci-1",
                generation=1,
                parent_call_id="call-1",
                input=BrowserRunInput(
                    task=(
                        "Open https://example.com and return the exact text of the "
                        "<h1> element on the page."
                    ),
                    timeout_seconds=300.0,
                ),
            ),
        )
    finally:
        await client.aclose()

    assert record.status == "completed", f"expected completed, got {record.status} (error={record.error})"
    assert record.output is not None
    assert "example" in record.output.lower()
