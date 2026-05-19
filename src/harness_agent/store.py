import json
from pathlib import Path

import aiosqlite
from pydantic import TypeAdapter

from harness_agent.events import AgentEvent, EventBase


class SQLiteEventStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)

    async def append(self, event: EventBase) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into events (id, type, occurred_at, payload)
                values (?, ?, ?, ?)
                """,
                (
                    event.id,
                    getattr(event, "type"),
                    event.occurred_at.isoformat(),
                    event.model_dump_json(),
                ),
            )
            await db.commit()

    async def list_events(self) -> list[AgentEvent]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                "select payload from events order by sequence asc"
            )
        return [self._adapter.validate_python(json.loads(row[0])) for row in rows]

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists events (
                    sequence integer primary key autoincrement,
                    id text not null unique,
                    type text not null,
                    occurred_at text not null,
                    payload text not null
                )
                """
            )
            await db.commit()
