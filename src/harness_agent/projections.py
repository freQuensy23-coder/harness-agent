import json
from pathlib import Path
from typing import Literal

import aiosqlite
from pydantic import TypeAdapter

from harness_agent.content import ContentRef
from harness_agent.llm import (
    AssistantMessage,
    AssistantToolCallMessage,
    LlmMessage,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.runtime import RuntimeToolResult
from harness_agent.tools import ToolInput


MessageRole = Literal["user", "assistant"]


class SQLiteConversationProjection:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._message_adapter: TypeAdapter[LlmMessage] = TypeAdapter(LlmMessage)

    async def append_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        role: MessageRole,
        text: str,
        generation: int | None = None,
    ) -> None:
        if role == "user":
            await self.append_user_message(
                user_id=user_id,
                conversation_id=conversation_id,
                text=text,
            )
            return
        if role == "assistant":
            await self.append_assistant_message(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=0 if generation is None else generation,
                text=text,
            )
            return
        raise ValueError(f"unsupported message role: {role}")

    async def append_user_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        text: str,
        attachments: list[ContentRef] | None = None,
    ) -> None:
        message = UserMessage(text=text, attachments=[] if attachments is None else attachments)
        await self._append_item(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=None,
            item_kind="user",
            text=text,
            tool_call_id=None,
            tool_name=None,
            payload_json=None,
            message_json=message.model_dump_json(),
        )

    async def append_assistant_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        text: str,
    ) -> None:
        message = AssistantMessage(text=text)
        await self._append_item(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            item_kind="assistant",
            text=text,
            tool_call_id=None,
            tool_name=None,
            payload_json=None,
            message_json=message.model_dump_json(),
        )

    async def append_tool_exchange(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        call_id: str,
        tool_name: str,
        input: ToolInput,
        result: RuntimeToolResult,
        attachments: list[ContentRef] | None = None,
    ) -> None:
        await self._ensure_schema()
        result_text = result.render_for_llm(tool_name)
        tool_attachments = [] if attachments is None else attachments
        tool_call_message = AssistantToolCallMessage(
            call_id=call_id,
            name=tool_name,
            arguments=input.model_dump(mode="json"),
        )
        tool_result_message = ToolResultMessage(
            call_id=call_id,
            name=tool_name,
            content=result_text,
        )
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into conversation_items (
                    user_id,
                    conversation_id,
                    generation,
                    item_kind,
                    text,
                    tool_call_id,
                    tool_name,
                    payload_json,
                    message_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    conversation_id,
                    generation,
                    "assistant_tool_call",
                    None,
                    call_id,
                    tool_name,
                    json.dumps(input.model_dump(mode="json")),
                    tool_call_message.model_dump_json(),
                ),
            )
            await db.execute(
                """
                insert into conversation_items (
                    user_id,
                    conversation_id,
                    generation,
                    item_kind,
                    text,
                    tool_call_id,
                    tool_name,
                    payload_json,
                    message_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    conversation_id,
                    generation,
                    "tool_result",
                    result_text,
                    call_id,
                    tool_name,
                    result.model_dump_json(),
                    tool_result_message.model_dump_json(),
                ),
            )
            for attachment in tool_attachments:
                image_context_message = UserMessage(
                    text=f"Opened image file {attachment.workspace_path}",
                    attachments=[attachment],
                )
                await db.execute(
                    """
                    insert into conversation_items (
                        user_id,
                        conversation_id,
                        generation,
                        item_kind,
                        text,
                        tool_call_id,
                        tool_name,
                        payload_json,
                        message_json
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        conversation_id,
                        generation,
                        "tool_context",
                        image_context_message.text,
                        call_id,
                        tool_name,
                        None,
                        image_context_message.model_dump_json(),
                    ),
                )
            await db.commit()

    async def list_llm_messages(self, conversation_id: str) -> list[LlmMessage]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                """
                select message_json
                from conversation_items
                where conversation_id = ?
                order by sequence asc
                """,
                (conversation_id,),
            )
        return [self._message_adapter.validate_json(row[0]) for row in rows]

    async def list_tool_history_json(self, conversation_id: str) -> list[str]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                """
                select message_json
                from conversation_items
                where conversation_id = ?
                  and item_kind in ('assistant_tool_call', 'tool_result')
                order by sequence asc
                """,
                (conversation_id,),
            )
        return [row[0] for row in rows]

    async def list_messages(self, conversation_id: str) -> list[tuple[str, str]]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                """
                select item_kind, text
                from conversation_items
                where conversation_id = ?
                  and item_kind in ('user', 'assistant')
                order by sequence asc
                """,
                (conversation_id,),
            )
        return [(row[0], row[1]) for row in rows]

    async def _append_item(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int | None,
        item_kind: str,
        text: str | None,
        tool_call_id: str | None,
        tool_name: str | None,
        payload_json: str | None,
        message_json: str,
    ) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into conversation_items (
                    user_id,
                    conversation_id,
                    generation,
                    item_kind,
                    text,
                    tool_call_id,
                    tool_name,
                    payload_json,
                    message_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    conversation_id,
                    generation,
                    item_kind,
                    text,
                    tool_call_id,
                    tool_name,
                    payload_json,
                    message_json,
                ),
            )
            await db.commit()

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists conversation_items (
                    sequence integer primary key autoincrement,
                    user_id text not null,
                    conversation_id text not null,
                    generation integer,
                    item_kind text not null,
                    text text,
                    tool_call_id text,
                    tool_name text,
                    payload_json text,
                    message_json text not null
                )
                """
            )
            await db.commit()
