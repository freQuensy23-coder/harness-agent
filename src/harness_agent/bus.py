import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

from harness_agent.events import AgentEvent, EventBase
from harness_agent.store import SQLiteEventStore


HandlerResult = Iterable[EventBase]
EventHandler = Callable[[Any], Awaitable[HandlerResult]]


class EventBus:
    def __init__(self, store: SQLiteEventStore) -> None:
        self._store = store
        self._handlers: dict[type[EventBase], list[EventHandler]] = {}

    def subscribe(
        self,
        event_class: type[EventBase],
        handler: EventHandler,
    ) -> None:
        if event_class not in self._handlers:
            self._handlers[event_class] = []
        self._handlers[event_class].append(handler)

    def send(self, event: AgentEvent) -> asyncio.Task[None]:
        return asyncio.create_task(self.publish(event))

    async def publish(
        self,
        event: AgentEvent,
        *,
        idempotent_replay: bool = False,
    ) -> None:
        """Persist `event` and dispatch handlers.

        `idempotent_replay=True` opts the caller into outbox-style retry
        semantics: if the event id is already in the store (a previous
        attempt crashed between append and handler dispatch), the append
        is treated as a no-op and handlers are dispatched anyway. The
        flag propagates into recursively-published follow-ups, so
        downstream events with deterministic ids stay idempotent too.
        Callers using this path must ensure handler effects are
        idempotent (typically via deterministic event ids).
        """
        try:
            await self._store.append(event)
        except sqlite3.IntegrityError:
            if not idempotent_replay:
                raise
        pending: list[EventBase] = []
        event_class = event.__class__
        if event_class in self._handlers:
            for handler in self._handlers[event_class]:
                result = await handler(event)
                pending.extend(result)
        for next_event in pending:
            await self.publish(
                cast(AgentEvent, next_event),
                idempotent_replay=idempotent_replay,
            )
