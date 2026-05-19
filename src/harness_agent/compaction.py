import json
import math
from uuid import uuid4

from harness_agent.llm import (
    AssistantToolCallMessage,
    AssistantText,
    LlmClient,
    LlmMessage,
    LlmRequest,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.projections import (
    ProjectedConversationMessage,
    SQLiteConversationProjection,
    summary_to_llm_message,
)
from harness_agent.runtime import UserRuntime
from harness_agent.tools import FileWriteInput, ToolSpec


DEFAULT_COMPACTION_RESERVED_TOKENS = 15_000
DEFAULT_COMPACTION_KEEP_LAST_MESSAGES = 2


class ContextCompactor:
    def __init__(
        self,
        *,
        llm: LlmClient,
        projection: SQLiteConversationProjection,
        runtime: UserRuntime,
        max_tokens_per_model: int,
        reserved_tokens: int = DEFAULT_COMPACTION_RESERVED_TOKENS,
        keep_last_messages: int = DEFAULT_COMPACTION_KEEP_LAST_MESSAGES,
    ) -> None:
        self._llm = llm
        self._projection = projection
        self._runtime = runtime
        self._max_tokens_per_model = max_tokens_per_model
        self._reserved_tokens = max(0, reserved_tokens)
        self._keep_last_messages = max(0, keep_last_messages)

    async def compact_if_needed(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        system: str,
        messages: list[LlmMessage],
        tools: list[ToolSpec],
    ) -> list[LlmMessage]:
        if not self._should_compact(system=system, messages=messages, tools=tools):
            return messages

        window = await self._projection.list_context_window(conversation_id)
        if len(window.rows) <= self._keep_last_messages:
            return messages

        compacted_rows, retained_rows = self._split_rows_for_retention(window.rows)
        if not compacted_rows:
            return messages
        source_messages: list[LlmMessage] = []
        if window.summary is not None:
            source_messages.append(summary_to_llm_message(window.summary))
        source_messages.extend(row.message for row in compacted_rows)
        if not source_messages:
            return messages

        archive_path = f"/workspace/.old-sessions/{uuid4().hex}.jsonl"
        await self._archive_messages(
            user_id=user_id,
            path=archive_path,
            messages=source_messages,
        )
        summary = await self._summarize(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            messages=source_messages,
        )
        await self._projection.record_compaction(
            user_id=user_id,
            conversation_id=conversation_id,
            compacted_through_sequence=compacted_rows[-1].sequence,
            source_message_count=len(source_messages),
            archive_path=archive_path,
            summary=summary,
        )
        return [
            summary_to_llm_message(summary),
            *[row.message for row in retained_rows],
        ]

    def _should_compact(
        self,
        *,
        system: str,
        messages: list[LlmMessage],
        tools: list[ToolSpec],
    ) -> bool:
        threshold = max(1, self._max_tokens_per_model - self._reserved_tokens)
        return estimate_request_tokens(system=system, messages=messages, tools=tools) >= threshold

    def _split_rows_for_retention(
        self,
        rows: list[ProjectedConversationMessage],
    ) -> tuple[list[ProjectedConversationMessage], list[ProjectedConversationMessage]]:
        if self._keep_last_messages == 0:
            return rows, []

        split_index = max(0, len(rows) - self._keep_last_messages)
        while split_index > 0:
            retained = rows[split_index:]
            retained_tool_calls = {
                row.message.call_id
                for row in retained
                if isinstance(row.message, AssistantToolCallMessage)
            }
            missing_tool_calls = [
                row.message.call_id
                for row in retained
                if isinstance(row.message, ToolResultMessage)
                and row.message.call_id not in retained_tool_calls
            ]
            if not missing_tool_calls:
                break
            split_index -= 1
        return rows[:split_index], rows[split_index:]

    async def _archive_messages(
        self,
        *,
        user_id: str,
        path: str,
        messages: list[LlmMessage],
    ) -> None:
        content = "\n".join(message.model_dump_json() for message in messages)
        if content:
            content = f"{content}\n"
        await self._runtime.file_write(
            user_id,
            FileWriteInput(path=path, content=content),
        )

    async def _summarize(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        messages: list[LlmMessage],
    ) -> str:
        jsonl = "\n".join(message.model_dump_json() for message in messages)
        response = await self._llm.respond(
            LlmRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=generation,
                system=(
                    "You compact conversation context for an autonomous coding agent. "
                    "Summarize every relevant instruction, decision, file path, tool call, "
                    "tool result, error, and unresolved task. Preserve exact names, ids, "
                    "commands, and outcomes. Do not invent facts."
                ),
                messages=[
                    UserMessage(
                        text=(
                            "Summarize these prior conversation items. They are JSONL "
                            "records using their original message kinds, including tool "
                            "calls and tool results when present.\n\n"
                            f"{jsonl}"
                        )
                    )
                ],
                tools=[],
            )
        )
        if not isinstance(response, AssistantText):
            raise RuntimeError("Context compaction expected assistant text summary")
        return response.text.strip()


def estimate_request_tokens(
    *,
    system: str,
    messages: list[LlmMessage],
    tools: list[ToolSpec],
) -> int:
    char_count = len(system)
    char_count += sum(len(message.model_dump_json()) for message in messages)
    for tool in tools:
        char_count += len(tool.name)
        char_count += len(tool.description)
        char_count += len(json.dumps(tool.parameters_schema(), sort_keys=True))
    return max(1, math.ceil(char_count / 4))
