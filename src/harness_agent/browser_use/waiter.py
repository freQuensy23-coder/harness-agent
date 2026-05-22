"""Event-mediated completion waiter for browser.run.

BrowserUseService.run() registers expect(session_id), kicks off the
cloud session, then awaits wait(session_id). One of the three
terminal events (Completed / Failed / Stopped) resolves the future."""

import asyncio

from harness_agent.events import (
    BrowserSessionCompleted,
    BrowserSessionFailed,
    BrowserSessionStopped,
    EventBase,
)


TerminalEvent = (
    BrowserSessionCompleted
    | BrowserSessionFailed
    | BrowserSessionStopped
)

EventTuple = tuple[EventBase, ...]


class BrowserSessionResultWaiter:
    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future[TerminalEvent]] = {}

    def expect(self, *, session_id: str) -> None:
        self._futures[session_id] = asyncio.get_running_loop().create_future()

    async def wait(self, *, session_id: str) -> TerminalEvent:
        return await self._futures[session_id]

    def forget(self, *, session_id: str) -> None:
        self._futures.pop(session_id, None)

    async def handle_completed(self, event: BrowserSessionCompleted) -> EventTuple:
        self._resolve(event.session_id, event)
        return ()

    async def handle_failed(self, event: BrowserSessionFailed) -> EventTuple:
        self._resolve(event.session_id, event)
        return ()

    async def handle_stopped(self, event: BrowserSessionStopped) -> EventTuple:
        self._resolve(event.session_id, event)
        return ()

    def _resolve(self, session_id: str, event: TerminalEvent) -> None:
        future = self._futures.get(session_id)
        if future is not None and not future.done():
            future.set_result(event)
