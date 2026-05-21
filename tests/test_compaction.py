import json
from pathlib import Path

import aiosqlite
import pytest

from harness_agent.config import LlmConfig, load_config
from harness_agent.llm import (
    AssistantMessage,
    LlmRequest,
    UserMessage,
    estimate_request_tokens,
)
from harness_agent.projections import (
    CONTEXT_SUMMARY_KIND,
    ConversationItemRecord,
    SQLiteConversationProjection,
)
from harness_agent.tools import default_tool_registry


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
