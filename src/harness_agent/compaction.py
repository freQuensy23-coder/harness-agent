import json
from uuid import uuid4

from harness_agent.bus import EventBus
from harness_agent.events import AgentTurnRequested, ContextCompacting
from harness_agent.llm import (
    LlmClient,
    LlmRequest,
    UserMessage,
    estimate_request_tokens,
)
from harness_agent.projections import ConversationItemRecord, SQLiteConversationProjection
from harness_agent.runtime import UserRuntime
from harness_agent.tools import FileWriteInput


SUMMARY_SYSTEM_PROMPT = (
    "Summarize previous conversation context for a future model request. "
    "Preserve decisions, open tasks, user preferences, assistant conclusions, "
    "tool calls, tool results, and file or image context. Be concise and factual."
)


class SessionArchiveWriter:
    def __init__(self, runtime: UserRuntime) -> None:
        self._runtime = runtime

    async def write(
        self,
        *,
        user_id: str,
        archive_path: str,
        records: list[ConversationItemRecord],
    ) -> None:
        content = "".join(
            f"{json.dumps(record.to_archive_dict(), sort_keys=True)}\n"
            for record in records
        )
        result = await self._runtime.file_write(
            user_id,
            FileWriteInput(path=archive_path, content=content),
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)


class ContextCompactor:
    def __init__(
        self,
        *,
        bus: EventBus,
        llm: LlmClient,
        projection: SQLiteConversationProjection,
        archive_writer: SessionArchiveWriter,
        max_tokens_per_model: int,
        reserved_tokens: int = 15_000,
        keep_last_messages: int = 2,
    ) -> None:
        self._bus = bus
        self._llm = llm
        self._projection = projection
        self._archive_writer = archive_writer
        self._threshold_tokens = max(0, max_tokens_per_model - reserved_tokens)
        self._keep_last_messages = keep_last_messages

    async def compact_if_needed(
        self,
        *,
        request: LlmRequest,
        event: AgentTurnRequested,
    ) -> LlmRequest:
        records = await self._projection.list_item_records(event.conversation_id)
        tail_start = self._tail_start(records)
        compacted_records = records[:tail_start]
        if not compacted_records:
            return request

        estimated_tokens = estimate_request_tokens(request)
        if estimated_tokens < self._threshold_tokens:
            return request

        archive_path = f"/workspace/.old-sessions/{uuid4().hex}.jsonl"
        await self._bus.publish(
            ContextCompacting(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=event.generation,
                estimated_tokens=estimated_tokens,
                threshold_tokens=self._threshold_tokens,
                archive_path=archive_path,
            )
        )
        await self._archive_writer.write(
            user_id=event.user_id,
            archive_path=archive_path,
            records=records,
        )
        summary_text = await self._summarize(
            event=event,
            records=compacted_records,
        )
        summary_message = UserMessage(
            text=(
                "Previous conversation summary:\n"
                f"{summary_text}\n\n"
                f"Archived raw session: {archive_path}"
            )
        )
        await self._projection.append_context_summary(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            summary_message=summary_message,
            archive_path=archive_path,
            tail_records=records[tail_start:],
        )
        return request.model_copy(
            update={
                "messages": await self._projection.list_llm_messages(
                    event.conversation_id
                )
            }
        )

    async def _summarize(
        self,
        *,
        event: AgentTurnRequested,
        records: list[ConversationItemRecord],
    ) -> str:
        jsonl = "".join(
            f"{json.dumps(record.to_archive_dict(), sort_keys=True)}\n"
            for record in records
        )
        response = await self._llm.respond(
            LlmRequest(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=event.generation,
                system=SUMMARY_SYSTEM_PROMPT,
                messages=[
                    UserMessage(
                        text=(
                            "Summarize these JSONL conversation items for future context:\n"
                            f"{jsonl}"
                        )
                    )
                ],
                tools=[],
            )
        )
        if response.kind != "assistant_text":
            raise RuntimeError("context compaction summary returned a tool call")
        return response.text

    def _tail_start(self, records: list[ConversationItemRecord]) -> int:
        tail_start = max(0, len(records) - self._keep_last_messages)
        if tail_start <= 0 or tail_start >= len(records):
            return tail_start
        tail_first = records[tail_start]
        previous = records[tail_start - 1]
        if (
            tail_first.item_kind == "tool_result"
            and previous.item_kind == "assistant_tool_call"
            and previous.tool_call_id == tail_first.tool_call_id
        ):
            return tail_start - 1
        return tail_start
