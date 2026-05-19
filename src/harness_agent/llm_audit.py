import json
from pathlib import Path

import aiosqlite
from pydantic import BaseModel

from harness_agent.llm import LlmClient, LlmRequest, LlmResponse


class LlmAuditRecord(BaseModel):
    sequence: int
    user_id: str
    conversation_id: str
    generation: int
    message_json: list[str]
    tool_names: list[str]


class SQLiteLlmAuditStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def append_request(self, request: LlmRequest) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into llm_requests (
                    user_id,
                    conversation_id,
                    generation,
                    system,
                    message_json,
                    tool_names_json
                )
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.user_id,
                    request.conversation_id,
                    request.generation,
                    request.system,
                    json.dumps(
                        [message.model_dump_json() for message in request.messages],
                        ensure_ascii=False,
                    ),
                    json.dumps([tool.name for tool in request.tools], ensure_ascii=False),
                ),
            )
            await db.commit()

    async def list_requests(self, conversation_id: str) -> list[LlmAuditRecord]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                """
                select
                    sequence,
                    user_id,
                    conversation_id,
                    generation,
                    message_json,
                    tool_names_json
                from llm_requests
                where conversation_id = ?
                order by sequence asc
                """,
                (conversation_id,),
            )
        return [
            LlmAuditRecord(
                sequence=row[0],
                user_id=row[1],
                conversation_id=row[2],
                generation=row[3],
                message_json=json.loads(row[4]),
                tool_names=json.loads(row[5]),
            )
            for row in rows
        ]

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists llm_requests (
                    sequence integer primary key autoincrement,
                    user_id text not null,
                    conversation_id text not null,
                    generation integer not null,
                    system text not null,
                    message_json text not null,
                    tool_names_json text not null
                )
                """
            )
            await db.commit()


class AuditedLlmClient(LlmClient):
    def __init__(self, *, inner: LlmClient, store: SQLiteLlmAuditStore) -> None:
        self._inner = inner
        self._store = store

    async def respond(self, request: LlmRequest) -> LlmResponse:
        await self._store.append_request(request)
        return await self._inner.respond(request)
