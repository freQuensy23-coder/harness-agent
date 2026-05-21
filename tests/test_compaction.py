import json
from pathlib import Path

import aiosqlite
import pytest

from harness_agent.compaction import (
    CompactionConfig,
    CompactionService,
    _strip_analysis,
    _summary_block,
    _tail_boundary,
)
from harness_agent.config import LlmConfig, load_config
from harness_agent.events import (
    CompactionCommitted,
    CompactionConflicted,
    CompactionRequested,
    CompactionSkipped,
    CompactionSnapshotReady,
    CompactionSummaryReady,
)
from harness_agent.llm import (
    AssistantMessage,
    AssistantText,
    FakeLlmClient,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    UserMessage,
    estimate_request_tokens,
)
from harness_agent.projections import (
    CONTEXT_SUMMARY_KIND,
    ConversationItemRecord,
    SQLiteConversationProjection,
)
from harness_agent.tools import default_tool_registry


def _record(*, sequence: int, item_kind: str, text: str = "") -> ConversationItemRecord:
    message: LlmMessage
    if item_kind == "user":
        message = UserMessage(text=text)
    else:
        message = AssistantMessage(text=text)
    return ConversationItemRecord(
        sequence=sequence,
        user_id="u:1",
        conversation_id="c",
        generation=1,
        item_kind=item_kind,
        text=text,
        tool_call_id=None,
        tool_name=None,
        payload_json=None,
        message_json=message.model_dump_json(),
        message=message,
    )


async def _seed_conversation(
    projection: SQLiteConversationProjection,
    *,
    conversation_id: str,
    texts: list[tuple[str, str]],
) -> None:
    for role, text in texts:
        if role == "user":
            await projection.append_user_message(
                user_id="u:1",
                conversation_id=conversation_id,
                text=text,
            )
        elif role == "assistant":
            await projection.append_assistant_message(
                user_id="u:1",
                conversation_id=conversation_id,
                generation=1,
                text=text,
            )
        else:
            raise AssertionError(f"unexpected role: {role}")


@pytest.mark.asyncio
async def test_list_active_returns_all_when_no_summary(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "hello"),
            ("assistant", "hi"),
            ("user", "more"),
        ],
    )

    records = await projection.list_active_context_items("c")

    assert [r.item_kind for r in records] == ["user", "assistant", "user"]
    assert [r.message for r in records] == [
        UserMessage(text="hello"),
        AssistantMessage(text="hi"),
        UserMessage(text="more"),
    ]
    assert all(isinstance(r, ConversationItemRecord) for r in records)


@pytest.mark.asyncio
async def test_list_active_returns_summary_plus_tail_after_summary(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "old-user"),
            ("assistant", "old-assistant"),
            ("user", "tail-user"),
        ],
    )
    all_before = await projection.list_all_context_items("c")
    compacted = [all_before[0].sequence, all_before[1].sequence]
    tail = [all_before[2].sequence]
    snapshot_max = all_before[-1].sequence

    committed = await projection.append_compacted_context_if_unchanged(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        summary="recap",
        archive_path="/workspace/.old-sessions/c.jsonl",
        compacted_sequences=compacted,
        tail_sequences=tail,
        snapshot_max_sequence=snapshot_max,
    )
    assert committed is True

    # Append a newer non-summary record after the summary commit.
    await projection.append_assistant_message(
        user_id="u:1",
        conversation_id="c",
        generation=2,
        text="after-summary",
    )

    records = await projection.list_active_context_items("c")
    assert [r.item_kind for r in records] == [
        CONTEXT_SUMMARY_KIND,
        "user",
        "assistant",
    ]
    summary_record = records[0]
    assert summary_record.message == UserMessage(text="Previous conversation summary:\nrecap")
    assert records[1].message == UserMessage(text="tail-user")
    assert records[2].message == AssistantMessage(text="after-summary")


@pytest.mark.asyncio
async def test_list_llm_messages_reflects_summary_after_compaction_commit(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "old-user"),
            ("assistant", "old-assistant"),
            ("user", "tail-user"),
        ],
    )
    all_before = await projection.list_all_context_items("c")
    compacted = [all_before[0].sequence, all_before[1].sequence]
    tail = [all_before[2].sequence]
    snapshot_max = all_before[-1].sequence

    committed = await projection.append_compacted_context_if_unchanged(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        summary="recap",
        archive_path="/workspace/.old-sessions/c.jsonl",
        compacted_sequences=compacted,
        tail_sequences=tail,
        snapshot_max_sequence=snapshot_max,
    )
    assert committed is True

    await projection.append_assistant_message(
        user_id="u:1",
        conversation_id="c",
        generation=2,
        text="after-summary",
    )

    # list_llm_messages must reflect the active projection: summary +
    # tail + newer; the compacted "old-user" / "old-assistant" rows are
    # excluded even though they remain in the table.
    assert await projection.list_llm_messages("c") == [
        UserMessage(text="Previous conversation summary:\nrecap"),
        UserMessage(text="tail-user"),
        AssistantMessage(text="after-summary"),
    ]


@pytest.mark.asyncio
async def test_list_active_drops_compacted_records_outside_payload_tail(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "scratch-1"),
            ("assistant", "scratch-2"),
            ("user", "scratch-3"),
            ("assistant", "scratch-4"),
            ("user", "tail-only"),
        ],
    )
    all_before = await projection.list_all_context_items("c")
    # Compact the first four, keep the last as the tail.
    compacted = [r.sequence for r in all_before[:4]]
    tail = [all_before[4].sequence]
    snapshot_max = all_before[-1].sequence

    committed = await projection.append_compacted_context_if_unchanged(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        summary="recap",
        archive_path="/workspace/.old-sessions/c.jsonl",
        compacted_sequences=compacted,
        tail_sequences=tail,
        snapshot_max_sequence=snapshot_max,
    )
    assert committed is True

    records = await projection.list_active_context_items("c")
    # Only the summary + the single tail record survive; the four scratch
    # rows that pre-dated the summary are filtered out even though they
    # remain in the table.
    assert [r.item_kind for r in records] == [CONTEXT_SUMMARY_KIND, "user"]
    assert records[1].message == UserMessage(text="tail-only")

    # Confirm the dropped rows are still persisted (not deleted).
    assert [r.item_kind for r in await projection.list_all_context_items("c")] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        CONTEXT_SUMMARY_KIND,
    ]


@pytest.mark.asyncio
async def test_append_compacted_writes_summary_row_and_returns_true(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
        ],
    )
    all_before = await projection.list_all_context_items("c")
    compacted = [all_before[0].sequence, all_before[1].sequence]
    tail = [all_before[2].sequence]
    snapshot_max = all_before[-1].sequence

    committed = await projection.append_compacted_context_if_unchanged(
        user_id="u:1",
        conversation_id="c",
        generation=3,
        summary="the recap",
        archive_path="/workspace/.old-sessions/c.jsonl",
        compacted_sequences=compacted,
        tail_sequences=tail,
        snapshot_max_sequence=snapshot_max,
    )
    assert committed is True

    rows = await projection.list_all_context_items("c")
    summary_row = rows[-1]
    assert summary_row.item_kind == CONTEXT_SUMMARY_KIND
    assert summary_row.text == "the recap"
    assert summary_row.generation == 3
    assert summary_row.user_id == "u:1"
    assert summary_row.conversation_id == "c"
    assert summary_row.message == UserMessage(text="Previous conversation summary:\nthe recap")
    assert summary_row.payload_json is not None
    payload = json.loads(summary_row.payload_json)
    assert payload == {
        "archive_path": "/workspace/.old-sessions/c.jsonl",
        "compacted_sequences": compacted,
        "tail_sequences": tail,
    }


@pytest.mark.asyncio
async def test_append_compacted_returns_false_when_max_sequence_changed(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
        ],
    )
    all_before = await projection.list_all_context_items("c")
    snapshot_max = all_before[-1].sequence
    compacted = [all_before[0].sequence, all_before[1].sequence]
    tail = [all_before[2].sequence]

    # A racer extends the conversation after we observed snapshot_max.
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="c",
        text="racer",
    )

    committed = await projection.append_compacted_context_if_unchanged(
        user_id="u:1",
        conversation_id="c",
        generation=4,
        summary="should-not-commit",
        archive_path="/workspace/.old-sessions/c.jsonl",
        compacted_sequences=compacted,
        tail_sequences=tail,
        snapshot_max_sequence=snapshot_max,
    )
    assert committed is False

    rows = await projection.list_all_context_items("c")
    assert [r.item_kind for r in rows] == ["user", "assistant", "user", "user"]
    assert all(r.item_kind != CONTEXT_SUMMARY_KIND for r in rows)

    # Confirm no stray rows leaked from a half-written transaction.
    async with aiosqlite.connect(tmp_path / "messages.sqlite3") as db:
        cursor = await db.execute(
            "select count(*) from conversation_items where item_kind = ?",
            (CONTEXT_SUMMARY_KIND,),
        )
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 0


def test_estimate_request_tokens_counts_system_messages_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_models: list[str] = []
    encoded_parts: list[str] = []

    class RecordingTokenizer:
        def encode(self, value: str) -> list[int]:
            encoded_parts.append(value)
            return list(range(len(value.split())))

    def encoding_for_model(model: str) -> RecordingTokenizer:
        requested_models.append(model)
        return RecordingTokenizer()

    monkeypatch.setattr(
        "harness_agent.llm.tiktoken.encoding_for_model",
        encoding_for_model,
    )

    shell_exec_tool = default_tool_registry().by_name("shell.exec")
    request = LlmRequest(
        user_id="u:1",
        conversation_id="cli:1",
        generation=1,
        system="system prompt",
        messages=[UserMessage(text="hello user"), AssistantMessage(text="hi there")],
        tools=[shell_exec_tool],
    )

    tokens = estimate_request_tokens(request)

    assert requested_models == ["gpt-5"]
    # System prompt, each message (json-dumped), each tool (json-dumped) all tokenized.
    assert "system prompt" in encoded_parts
    assert any("hello user" in part for part in encoded_parts)
    assert any("hi there" in part for part in encoded_parts)
    assert any("shell__exec" in part for part in encoded_parts)
    # 1 system + 2 messages + 1 tool = 4 parts encoded.
    assert len(encoded_parts) == 4
    assert tokens > 0


def test_config_loads_compaction_knobs_with_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text("llm:\n  api_key: test-key\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.llm.max_tokens_per_model == 128_000
    assert config.llm.compaction_reserve_tokens == 15_000
    assert config.llm.compaction_keep_last_user_messages == 2

    override_path = tmp_path / "harness-override.yaml"
    override_path.write_text(
        """
llm:
  api_key: test-key
  max_tokens_per_model: 50000
  compaction_reserve_tokens: 5000
  compaction_keep_last_user_messages: 5
""",
        encoding="utf-8",
    )

    override = load_config(override_path)
    assert override.llm.max_tokens_per_model == 50_000
    assert override.llm.compaction_reserve_tokens == 5_000
    assert override.llm.compaction_keep_last_user_messages == 5

    # Direct construction also exposes the defaults.
    defaults = LlmConfig()
    assert defaults.max_tokens_per_model == 128_000
    assert defaults.compaction_reserve_tokens == 15_000
    assert defaults.compaction_keep_last_user_messages == 2


def test_tail_boundary_finds_nth_user_message_from_end() -> None:
    records = [
        _record(sequence=1, item_kind="user", text="u1"),
        _record(sequence=2, item_kind="assistant", text="a1"),
        _record(sequence=3, item_kind="user", text="u2"),
        _record(sequence=4, item_kind="assistant", text="a2"),
        _record(sequence=5, item_kind="user", text="u3"),
        _record(sequence=6, item_kind="assistant", text="a3"),
        _record(sequence=7, item_kind="user", text="u4"),
    ]

    # With n=2, boundary is the index of the 2nd-from-last user message.
    # User messages are at indexes 0, 2, 4, 6 -> 2nd from end is index 4.
    assert _tail_boundary(records, keep_last_user_messages=2) == 4


def test_tail_boundary_returns_none_when_n_users_with_first_at_index_zero() -> None:
    # Only two user messages, with the first at index 0; n=2 picks index 0 ->
    # _tail_boundary maps that to None (nothing to compact).
    records = [
        _record(sequence=1, item_kind="user", text="u1"),
        _record(sequence=2, item_kind="assistant", text="a1"),
        _record(sequence=3, item_kind="user", text="u2"),
    ]

    assert _tail_boundary(records, keep_last_user_messages=2) is None


def test_tail_boundary_falls_back_to_latest_safe_when_fewer_users() -> None:
    # Only one user message but n=2 requested; _nth_user_message_from_end
    # returns None, so we fall back to _latest_safe_boundary, which is the
    # last index that is not tool_result / tool_context. Here the last
    # record is assistant -> boundary should be that index.
    records = [
        _record(sequence=1, item_kind="user", text="u1"),
        _record(sequence=2, item_kind="assistant", text="a1"),
        _record(sequence=3, item_kind="assistant", text="a2"),
    ]

    assert _tail_boundary(records, keep_last_user_messages=2) == 2


def test_tail_boundary_safe_fallback_skips_tool_result_and_tool_context() -> None:
    # No user messages -> nth_user returns None -> falls back to safe boundary.
    # The trailing tool_result and tool_context rows must be skipped so the
    # boundary lands on the latest non-tool-context-or-result index.
    records = [
        _record(sequence=1, item_kind="assistant", text="a1"),
        _record(sequence=2, item_kind="assistant", text="a2"),
        _record(sequence=3, item_kind="tool_result", text="r1"),
        _record(sequence=4, item_kind="tool_context", text="ctx1"),
    ]

    # Latest safe boundary walks from end skipping tool_result/tool_context.
    # Index 3 (tool_context) skipped, index 2 (tool_result) skipped,
    # index 1 (assistant) accepted.
    assert _tail_boundary(records, keep_last_user_messages=1) == 1


def test_summary_block_extracts_trailing_summary_when_analysis_quotes_a_tag() -> None:
    # The analysis section quotes a literal "<summary>" tag string, but the
    # real summary follows. After _strip_analysis removes the analysis block,
    # only the real <summary>...</summary> remains and is extracted.
    text = (
        "<analysis>I will close with a <summary> tag soon.</analysis>"
        "<summary>real summary body</summary>"
    )

    assert _summary_block(text) == "real summary body"


def test_summary_block_preserves_literal_summary_tag_inside_body() -> None:
    # The regex is greedy across DOTALL: when the body itself contains a
    # literal </summary> token, the greedy match keeps reading until the
    # outermost </summary> at the end. Inner text including the embedded
    # </summary> sequence must be preserved.
    text = (
        "<summary>line one</summary> mention of </summary> still in body"
        "</summary>"
    )

    extracted = _summary_block(text)
    assert extracted is not None
    assert extracted.startswith("line one</summary>")
    assert "still in body" in extracted


def test_strip_analysis_removes_analysis_block_nondestructively() -> None:
    text = (
        "prefix text<analysis>throw away this analysis block</analysis>"
        "suffix <summary>kept</summary>"
    )

    stripped = _strip_analysis(text)
    assert "throw away" not in stripped
    assert "prefix text" in stripped
    assert "suffix" in stripped
    assert "<summary>kept</summary>" in stripped


def _make_config() -> CompactionConfig:
    return CompactionConfig(max_tokens_per_model=128_000)


def _make_requested(
    *,
    compaction_id: str = "cid:test",
    conversation_id: str = "c",
    user_id: str = "u:1",
    generation: int = 1,
) -> CompactionRequested:
    return CompactionRequested(
        compaction_id=compaction_id,
        user_id=user_id,
        conversation_id=conversation_id,
        generation=generation,
    )


def _make_snapshot(
    *,
    compaction_id: str = "cid:test",
    conversation_id: str = "c",
    user_id: str = "u:1",
    generation: int = 1,
    compacted_sequences: list[int],
    tail_sequences: list[int],
    snapshot_max_sequence: int,
    archive_path: str = "/workspace/.old-sessions/cid:test.jsonl",
) -> CompactionSnapshotReady:
    return CompactionSnapshotReady(
        compaction_id=compaction_id,
        user_id=user_id,
        conversation_id=conversation_id,
        generation=generation,
        compacted_sequences=compacted_sequences,
        tail_sequences=tail_sequences,
        snapshot_max_sequence=snapshot_max_sequence,
        archive_path=archive_path,
    )


def _make_summary_ready(
    *,
    snapshot: CompactionSnapshotReady,
    summary: str,
) -> CompactionSummaryReady:
    return CompactionSummaryReady(
        compaction_id=snapshot.compaction_id,
        user_id=snapshot.user_id,
        conversation_id=snapshot.conversation_id,
        generation=snapshot.generation,
        compacted_sequences=list(snapshot.compacted_sequences),
        tail_sequences=list(snapshot.tail_sequences),
        snapshot_max_sequence=snapshot.snapshot_max_sequence,
        archive_path=snapshot.archive_path,
        summary=summary,
    )


@pytest.mark.asyncio
async def test_handle_requested_emits_snapshot_ready_on_happy_path(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
            ("user", "u3"),
        ],
    )
    all_records = await projection.list_all_context_items("c")
    assert [r.item_kind for r in all_records] == ["user", "assistant", "user", "assistant", "user"]
    # With keep_last_user_messages=2, boundary is at the 2nd-from-last user message
    # (index 2 — i.e. "u2"). So compacted = records[:2], tail = records[2:].
    expected_compacted = [r.sequence for r in all_records[:2]]
    expected_tail = [r.sequence for r in all_records[2:]]
    snapshot_max = all_records[-1].sequence

    config = CompactionConfig(max_tokens_per_model=128_000)
    llm = FakeLlmClient([])
    service = CompactionService(projection=projection, llm=llm, config=config)

    event = _make_requested(compaction_id="cid:happy")
    batch = await service.handle_requested(event)

    assert len(batch) == 1
    snapshot = batch[0]
    assert isinstance(snapshot, CompactionSnapshotReady)
    assert snapshot.compaction_id == "cid:happy"
    assert snapshot.conversation_id == "c"
    assert snapshot.user_id == "u:1"
    assert snapshot.generation == 1
    assert snapshot.compacted_sequences == expected_compacted
    assert snapshot.tail_sequences == expected_tail
    assert snapshot.snapshot_max_sequence == snapshot_max
    assert snapshot.archive_path == "/workspace/.old-sessions/cid:happy.jsonl"


@pytest.mark.asyncio
async def test_handle_requested_emits_skipped_when_no_boundary(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "only-user"),
        ],
    )

    config = CompactionConfig(max_tokens_per_model=128_000)
    llm = FakeLlmClient([])
    service = CompactionService(projection=projection, llm=llm, config=config)

    event = _make_requested(compaction_id="cid:skip")
    batch = await service.handle_requested(event)

    assert len(batch) == 1
    skipped = batch[0]
    assert isinstance(skipped, CompactionSkipped)
    assert skipped.compaction_id == "cid:skip"
    assert skipped.conversation_id == "c"
    assert skipped.user_id == "u:1"
    assert skipped.generation == 1
    assert skipped.reason == "no_boundary"


@pytest.mark.asyncio
async def test_handle_snapshot_ready_calls_llm_once_when_summary_tag_present(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
        ],
    )
    all_records = await projection.list_all_context_items("c")
    compacted_seq = [r.sequence for r in all_records[:2]]
    tail_seq = [all_records[2].sequence]
    snapshot_max = all_records[-1].sequence

    config = CompactionConfig(max_tokens_per_model=128_000)
    responses: list[LlmResponse] = [AssistantText(text="<summary>X</summary>")]
    llm = FakeLlmClient(responses)
    service = CompactionService(projection=projection, llm=llm, config=config)

    snapshot = _make_snapshot(
        compaction_id="cid:once",
        compacted_sequences=compacted_seq,
        tail_sequences=tail_seq,
        snapshot_max_sequence=snapshot_max,
        archive_path="/workspace/.old-sessions/cid:once.jsonl",
    )
    batch = await service.handle_snapshot_ready(snapshot)

    assert len(llm.requests) == 1
    assert len(batch) == 1
    ready = batch[0]
    assert isinstance(ready, CompactionSummaryReady)
    assert ready.summary == "X"
    assert ready.compaction_id == "cid:once"
    assert ready.compacted_sequences == compacted_seq
    assert ready.tail_sequences == tail_seq
    assert ready.snapshot_max_sequence == snapshot_max
    assert ready.archive_path == "/workspace/.old-sessions/cid:once.jsonl"


@pytest.mark.asyncio
async def test_handle_snapshot_ready_retries_once_when_summary_tag_missing(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
        ],
    )
    all_records = await projection.list_all_context_items("c")
    compacted_seq = [r.sequence for r in all_records[:2]]
    tail_seq = [all_records[2].sequence]
    snapshot_max = all_records[-1].sequence

    config = CompactionConfig(max_tokens_per_model=128_000)
    responses: list[LlmResponse] = [
        AssistantText(text="no tag here"),
        AssistantText(text="<summary>second-try</summary>"),
    ]
    llm = FakeLlmClient(responses)
    service = CompactionService(projection=projection, llm=llm, config=config)

    snapshot = _make_snapshot(
        compaction_id="cid:retry",
        compacted_sequences=compacted_seq,
        tail_sequences=tail_seq,
        snapshot_max_sequence=snapshot_max,
    )
    batch = await service.handle_snapshot_ready(snapshot)

    assert len(llm.requests) == 2
    assert len(batch) == 1
    ready = batch[0]
    assert isinstance(ready, CompactionSummaryReady)
    assert ready.summary == "second-try"


@pytest.mark.asyncio
async def test_handle_snapshot_ready_falls_back_to_analysis_stripped_when_retries_exhausted(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
        ],
    )
    all_records = await projection.list_all_context_items("c")
    compacted_seq = [r.sequence for r in all_records[:2]]
    tail_seq = [all_records[2].sequence]
    snapshot_max = all_records[-1].sequence

    config = CompactionConfig(max_tokens_per_model=128_000)
    responses: list[LlmResponse] = [
        AssistantText(text="<analysis>noise 1</analysis>still no tag"),
        AssistantText(
            text="<analysis>noise 2</analysis>raw fallback body"
        ),
    ]
    llm = FakeLlmClient(responses)
    service = CompactionService(projection=projection, llm=llm, config=config)

    snapshot = _make_snapshot(
        compaction_id="cid:fallback",
        compacted_sequences=compacted_seq,
        tail_sequences=tail_seq,
        snapshot_max_sequence=snapshot_max,
    )
    batch = await service.handle_snapshot_ready(snapshot)

    assert len(llm.requests) == 2  # _SUMMARY_ATTEMPTS = 2
    assert len(batch) == 1
    ready = batch[0]
    assert isinstance(ready, CompactionSummaryReady)
    # _strip_analysis removes the <analysis>...</analysis> block on the last
    # response and falls back to the stripped text.
    assert ready.summary == "raw fallback body"


@pytest.mark.asyncio
async def test_handle_summary_ready_emits_committed_when_cas_succeeds(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
        ],
    )
    all_records = await projection.list_all_context_items("c")
    compacted_seq = [r.sequence for r in all_records[:2]]
    tail_seq = [all_records[2].sequence]
    snapshot_max = all_records[-1].sequence

    config = CompactionConfig(max_tokens_per_model=128_000)
    llm = FakeLlmClient([])
    service = CompactionService(projection=projection, llm=llm, config=config)

    snapshot = _make_snapshot(
        compaction_id="cid:commit",
        compacted_sequences=compacted_seq,
        tail_sequences=tail_seq,
        snapshot_max_sequence=snapshot_max,
        archive_path="/workspace/.old-sessions/cid:commit.jsonl",
    )
    summary_ready = _make_summary_ready(snapshot=snapshot, summary="the recap")
    batch = await service.handle_summary_ready(summary_ready)

    assert len(batch) == 1
    committed = batch[0]
    assert isinstance(committed, CompactionCommitted)
    assert committed.compaction_id == "cid:commit"
    assert committed.archive_path == "/workspace/.old-sessions/cid:commit.jsonl"
    assert committed.compacted_sequences == compacted_seq
    assert committed.user_id == "u:1"
    assert committed.conversation_id == "c"
    assert committed.generation == 1

    # The projection should now contain a context_summary row.
    rows = await projection.list_all_context_items("c")
    assert rows[-1].item_kind == CONTEXT_SUMMARY_KIND
    assert rows[-1].text == "the recap"


@pytest.mark.asyncio
async def test_handle_summary_ready_emits_conflicted_when_max_sequence_advanced(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await _seed_conversation(
        projection,
        conversation_id="c",
        texts=[
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
        ],
    )
    all_records = await projection.list_all_context_items("c")
    compacted_seq = [r.sequence for r in all_records[:2]]
    tail_seq = [all_records[2].sequence]
    snapshot_max = all_records[-1].sequence

    # A racer appends a new user message between snapshot and summary.
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="c",
        text="racer",
    )

    config = CompactionConfig(max_tokens_per_model=128_000)
    llm = FakeLlmClient([])
    service = CompactionService(projection=projection, llm=llm, config=config)

    snapshot = _make_snapshot(
        compaction_id="cid:conflict",
        compacted_sequences=compacted_seq,
        tail_sequences=tail_seq,
        snapshot_max_sequence=snapshot_max,
    )
    summary_ready = _make_summary_ready(snapshot=snapshot, summary="stale")
    batch = await service.handle_summary_ready(summary_ready)

    assert len(batch) == 1
    conflicted = batch[0]
    assert isinstance(conflicted, CompactionConflicted)
    assert conflicted.compaction_id == "cid:conflict"
    assert conflicted.reason == "cas_lost"
    assert conflicted.user_id == "u:1"
    assert conflicted.conversation_id == "c"
    assert conflicted.generation == 1

    # No context_summary row should have been written.
    rows = await projection.list_all_context_items("c")
    assert all(r.item_kind != CONTEXT_SUMMARY_KIND for r in rows)
