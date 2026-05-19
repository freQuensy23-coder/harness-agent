import json
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from harness_agent.llm import (
    LlmClient,
    LlmMessage,
    LlmRequest,
    UserMessage,
    estimate_request_tokens,
)
from harness_agent.projections import ConversationItemRecord, SQLiteConversationProjection
from harness_agent.runtime import UserRuntime
from harness_agent.tools import FileWriteInput


SUMMARY_SYSTEM_PROMPT = "\n".join(
    [
        "You summarize prior conversation context for a coding agent.",
        "Preserve decisions, constraints, file paths, tool calls, tool results,",
        "and unresolved work. Do not add facts that are not present.",
    ]
)


TokenEstimator = Callable[[LlmRequest], int]


@dataclass(frozen=True)
class ContextCompactionConfig:
    max_tokens_per_model: int
    reserve_tokens: int = 15_000
    keep_last_messages: int = 2

    @property
    def threshold(self) -> int:
        return self.max_tokens_per_model - self.reserve_tokens


@dataclass(frozen=True)
class CompactionSnapshot:
    active_records: list[ConversationItemRecord]
    compacted_records: list[ConversationItemRecord]
    tail_records: list[ConversationItemRecord]
    snapshot_max_sequence: int
    jsonl: str


@dataclass(frozen=True)
class CompactionDraft:
    snapshot: CompactionSnapshot
    summary: str
    archive_path: str


class ContextCompactor:
    def __init__(
        self,
        *,
        projection: SQLiteConversationProjection,
        runtime: UserRuntime,
        llm: LlmClient,
        config: ContextCompactionConfig,
        estimate_tokens: TokenEstimator = estimate_request_tokens,
    ) -> None:
        self._projection = projection
        self._runtime = runtime
        self._llm = llm
        self._config = config
        self._estimate_tokens = estimate_tokens

    @property
    def threshold(self) -> int:
        return self._config.threshold

    @property
    def keep_last_messages(self) -> int:
        return self._config.keep_last_messages

    def should_compact(self, request: LlmRequest) -> tuple[bool, int]:
        token_estimate = self._estimate_tokens(request)
        return token_estimate >= self.threshold, token_estimate

    async def create_snapshot(
        self,
        *,
        conversation_id: str,
    ) -> CompactionSnapshot | None:
        active_records = await self._projection.list_active_context_items(conversation_id)
        if len(active_records) <= self.keep_last_messages:
            return None

        compacted_records = active_records[: -self.keep_last_messages]
        tail_records = active_records[-self.keep_last_messages :]
        return CompactionSnapshot(
            active_records=active_records,
            compacted_records=compacted_records,
            tail_records=tail_records,
            snapshot_max_sequence=max(record.sequence for record in active_records),
            jsonl=_records_to_jsonl(compacted_records),
        )

    async def create_draft(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        snapshot: CompactionSnapshot,
    ) -> CompactionDraft:
        summary = await self._summarize(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            jsonl=snapshot.jsonl,
        )
        archive_path = f"/workspace/.old-sessions/{uuid4().hex}.jsonl"
        return CompactionDraft(snapshot=snapshot, summary=summary, archive_path=archive_path)

    async def commit_draft(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        draft: CompactionDraft,
    ) -> list[LlmMessage]:
        await self._runtime.file_write(
            user_id,
            FileWriteInput(path=draft.archive_path, content=draft.snapshot.jsonl),
        )
        committed = await self._projection.append_compacted_context_if_unchanged(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            summary=draft.summary,
            archive_path=draft.archive_path,
            compacted_records=draft.snapshot.compacted_records,
            tail_records=draft.snapshot.tail_records,
            snapshot_max_sequence=draft.snapshot.snapshot_max_sequence,
        )
        if not committed:
            return await self._projection.list_llm_messages(conversation_id)
        return await self._projection.list_llm_messages(conversation_id)

    async def _summarize(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        jsonl: str,
    ) -> str:
        response = await self._llm.respond(
            LlmRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=generation,
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
        return _assistant_text(response)


def _assistant_text(response) -> str:
    if response.kind == "assistant_text":
        return response.text
    raise RuntimeError("context compaction summary response was not text")


def _records_to_jsonl(records: list[ConversationItemRecord]) -> str:
    return "\n".join(
        json.dumps(
            {
                "sequence": record.sequence,
                "item_kind": record.item_kind,
                "generation": record.generation,
                "text": record.text,
                "tool_call_id": record.tool_call_id,
                "tool_name": record.tool_name,
                "payload": _json_or_none(record.payload_json),
                "message": json.loads(record.message_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in records
    )


def _json_or_none(value: str | None):
    if value is None:
        return None
    return json.loads(value)
