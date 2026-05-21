import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

from harness_agent.llm import (
    LlmClient,
    LlmMessage,
    LlmRequest,
    estimate_request_tokens,
)
from harness_agent.projections import ConversationItemRecord, SQLiteConversationProjection
from harness_agent.runtime import UserRuntime
from harness_agent.tools import FileWriteInput


# Verbatim Claude Code /compact prompt — do not edit.
SUMMARY_SYSTEM_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
   - Note any security-relevant instructions or constraints the user stated (e.g., sensitive files or data to avoid, operations that must not be performed, credential or secret handling rules). These MUST be preserved verbatim in the summary so they continue to apply after compaction.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent. Preserve any security-relevant instructions or constraints verbatim so they remain in effect after compaction.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response."""


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
        boundary_index = _tail_boundary(
            active_records, self._config.keep_last_messages
        )
        if boundary_index is None:
            return None

        compacted_records = active_records[:boundary_index]
        tail_records = active_records[boundary_index:]
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
            records=snapshot.compacted_records,
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
        if committed:
            await self._write_archive_best_effort(user_id, draft)
        return await self._projection.list_llm_messages(conversation_id)

    async def _write_archive_best_effort(self, user_id: str, draft: CompactionDraft) -> None:
        for _ in range(_ARCHIVE_WRITE_ATTEMPTS):
            try:
                result = await self._runtime.file_write(
                    user_id,
                    FileWriteInput(path=draft.archive_path, content=draft.snapshot.jsonl),
                )
            except Exception:
                continue
            if result.exit_code == 0:
                return
        logger.warning(
            "compaction archive write failed after {} attempts: {}",
            _ARCHIVE_WRITE_ATTEMPTS,
            draft.archive_path,
        )

    async def _summarize(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        records: list[ConversationItemRecord],
    ) -> str:
        request = LlmRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[record.message for record in records],
            tools=[],
        )
        last_text = ""
        for _ in range(_SUMMARY_ATTEMPTS):
            last_text = _assistant_text(await self._llm.respond(request))
            extracted = _summary_block(last_text)
            if extracted:
                return extracted
        stripped = _strip_analysis(last_text)
        return stripped or last_text.strip()


_SUMMARY_ATTEMPTS = 2
_ARCHIVE_WRITE_ATTEMPTS = 3


def _assistant_text(response) -> str:
    if response.kind == "assistant_text":
        return response.text
    raise RuntimeError("context compaction summary response was not text")


def _summary_block(text: str) -> str | None:
    cleaned = _strip_analysis(text)
    match = re.search(r"<summary>(.*)</summary>", cleaned, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _strip_analysis(text: str) -> str:
    return re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL).strip()


def _tail_boundary(
    records: list[ConversationItemRecord],
    keep_last_user_messages: int,
) -> int | None:
    user_boundary = _nth_user_message_from_end(records, keep_last_user_messages)
    if user_boundary is None:
        return _latest_safe_boundary(records)
    if user_boundary == 0:
        return None
    return user_boundary


def _nth_user_message_from_end(
    records: list[ConversationItemRecord],
    n: int,
) -> int | None:
    if n <= 0:
        return None
    user_count = 0
    for index in range(len(records) - 1, -1, -1):
        if records[index].item_kind != "user":
            continue
        user_count += 1
        if user_count == n:
            return index
    return None


def _latest_safe_boundary(records: list[ConversationItemRecord]) -> int | None:
    for index in range(len(records) - 1, 0, -1):
        if records[index].item_kind in ("tool_result", "tool_context"):
            continue
        return index
    return None


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
