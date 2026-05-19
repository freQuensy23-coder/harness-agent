from collections.abc import Awaitable, Callable, Iterable

from harness_agent.events import AgentEvent, EventBase
from harness_agent.store import SQLiteEventStore


HandlerResult = Iterable[EventBase]
EventHandler = Callable[[AgentEvent], Awaitable[HandlerResult]]


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

    async def publish(self, event: AgentEvent) -> None:
        await self._store.append(event)
        pending: list[EventBase] = []
        event_class = event.__class__
        if event_class in self._handlers:
            for handler in self._handlers[event_class]:
                result = await handler(event)
                pending.extend(result)
        for next_event in pending:
            await self.publish(next_event)
