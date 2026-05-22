import json
from pathlib import Path

import pytest

from harness_agent.bus import EventBus
from harness_agent.events import (
    AssistantTextProduced,
    SessionLogAppendFailed,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.llm import AssistantText, FakeLlmClient
from harness_agent.runtime import RuntimeToolResult
from harness_agent.runtime.fake import FakeUserRuntime
from harness_agent.session_log import SessionLogWriter, safe_conversation_id_part
from harness_agent.session_search_service import SessionSearchService
from harness_agent.store import SQLiteEventStore
from harness_agent.tool_executor import ToolCallExecutor
from harness_agent.tools import SessionSearchInput, ShellExecInput
from harness_agent.turns import ConversationTurnCoordinator


async def _coord_at(conversation_id: str, generation: int) -> ConversationTurnCoordinator:
    coord = ConversationTurnCoordinator()
    for _ in range(generation):
        await coord.request_generation(conversation_id)
    return coord


@pytest.mark.asyncio
async def test_session_log_writer_appends_events(tmp_path: Path) -> None:
    runtime = FakeUserRuntime()
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    coordinator = await _coord_at("conv-1", 1)
    writer = SessionLogWriter(runtime=runtime, turn_coordinator=coordinator)
    bus.subscribe(UserTextReceived, writer.handle_user_text)
    bus.subscribe(AssistantTextProduced, writer.handle_assistant_text)
    bus.subscribe(ToolCallCompleted, writer.handle_tool_call_completed)

    await bus.publish(
        UserTextReceived(
            user_id="alex",
            conversation_id="conv-1",
            source="cli",
            text="Where did we deploy?",
        )
    )
    await bus.publish(
        AssistantTextProduced(
            user_id="alex",
            conversation_id="conv-1",
            generation=1,
            text="We deployed to fly.io.",
        )
    )
    await bus.publish(
        ToolCallCompleted(
            user_id="alex",
            conversation_id="conv-1",
            generation=1,
            call_id="t-1",
            tool_name="shell.exec",
            input=ShellExecInput(command="ls"),
            result=RuntimeToolResult(stdout="files", exit_code=0),
        )
    )

    raw = await runtime.read_session_log("alex", "conv-1")
    lines = [json.loads(line) for line in raw.splitlines()]
    assert [entry["role"] for entry in lines] == ["user", "assistant", "tool"]
    assert lines[2]["tool_name"] == "shell.exec"
    assert lines[2]["stdout"] == "files"


@pytest.mark.asyncio
async def test_session_search_returns_top_summarised_matches() -> None:
    runtime = FakeUserRuntime()
    # Seed three prior sessions: two relevant to "deploy", one off-topic.
    for conv_id, records in (
        (
            "conv-old-1",
            [
                {"role": "user", "text": "How do I deploy to fly.io?"},
                {"role": "assistant", "text": "Deploy with flyctl deploy."},
            ],
        ),
        (
            "conv-old-2",
            [
                {"role": "user", "text": "Configure docker for deploy."},
                {"role": "assistant", "text": "Use Dockerfile in repo root."},
            ],
        ),
        (
            "conv-off-topic",
            [
                {"role": "user", "text": "What's the weather today?"},
                {"role": "assistant", "text": "I cannot check weather."},
            ],
        ),
    ):
        for record in records:
            await runtime.append_session_log(
                "alex", conv_id, json.dumps(record)
            )
    # Also write the current conversation; it must be excluded from results.
    await runtime.append_session_log(
        "alex",
        "conv-current",
        json.dumps({"role": "user", "text": "deploy now"}),
    )

    llm = FakeLlmClient(
        [
            AssistantText(text="Earlier session covered deploy to fly.io."),
            AssistantText(text="Earlier session covered docker deploy config."),
        ]
    )
    service = SessionSearchService(runtime=runtime, llm=llm)
    result = await service.execute(
        user_id="alex",
        current_conversation_id="conv-current",
        input=SessionSearchInput(query="deploy", limit=3),
    )
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    returned_ids = [entry["conversation_id"] for entry in payload["results"]]
    assert "conv-current" not in returned_ids
    assert "conv-off-topic" not in returned_ids
    assert set(returned_ids) == {"conv-old-1", "conv-old-2"}
    assert all("Earlier session" in entry["summary"] for entry in payload["results"])


@pytest.mark.asyncio
async def test_session_search_excludes_current_conversation_with_colon_id() -> None:
    """Conversation IDs may carry source prefixes like `tg:456`. The
    runtime percent-encodes the colon into `tg%3A456` on disk and decodes
    back to the raw `tg:456` in `list_session_logs`. The search must
    therefore compare against the raw current_conversation_id (no
    encoding) so distinct ids `tg:456` and `tg-456` route to distinct
    files and the current conversation is correctly excluded.
    """
    runtime = FakeUserRuntime()
    await runtime.append_session_log(
        "alex",
        "tg:456",
        json.dumps({"role": "user", "text": "earlier deploy discussion"}),
    )
    assert safe_conversation_id_part("tg:456") == "tg%3A456"
    assert safe_conversation_id_part("tg-456") == "tg-456"
    llm = FakeLlmClient([AssistantText(text="never used")])
    service = SessionSearchService(runtime=runtime, llm=llm)
    result = await service.execute(
        user_id="alex",
        current_conversation_id="tg:456",
        input=SessionSearchInput(query="deploy"),
    )
    payload = json.loads(result.stdout)
    assert payload["results"] == []
    assert payload["message"] == "No prior sessions available to search."


@pytest.mark.asyncio
async def test_session_search_summary_sees_tool_input(tmp_path: Path) -> None:
    """A query that matches only the tool's input (command / path /
    arguments) must reach the summariser. Without input in the
    transcript, the FTS would 'hit' on the raw JSONL but the summary
    LLM would not know what was matched."""
    runtime = FakeUserRuntime()
    await runtime.append_session_log(
        "alex",
        "conv-old",
        json.dumps(
            {
                "role": "tool",
                "tool_name": "shell.exec",
                "input": {"command": "kubectl apply -f manifests/prod.yaml"},
                "stdout": "deployment.apps/api configured",
                "stderr": "",
                "exit_code": 0,
            }
        ),
    )
    captured: list[str] = []

    class CapturingLlm:
        async def respond(self, request):  # type: ignore[no-untyped-def]
            captured.append(request.messages[-1].text)
            return AssistantText(text="Earlier session ran kubectl apply.")

    service = SessionSearchService(runtime=runtime, llm=CapturingLlm())  # type: ignore[arg-type]
    result = await service.execute(
        user_id="alex",
        current_conversation_id="conv-current",
        input=SessionSearchInput(query="kubectl apply -f manifests/prod.yaml"),
    )
    payload = json.loads(result.stdout)
    assert len(payload["results"]) == 1
    transcript = captured[0]
    assert "kubectl apply -f manifests/prod.yaml" in transcript


@pytest.mark.asyncio
async def test_session_log_writer_publishes_failure_event_on_runtime_error(
    tmp_path: Path,
) -> None:
    """If the runtime's append fails, the writer surfaces a typed
    SessionLogAppendFailed event so observers can react instead of
    finding silent gaps in the JSONL log."""

    class BoomRuntime(FakeUserRuntime):
        async def append_session_log(self, user_id, conversation_id, line):  # type: ignore[no-untyped-def, override]
            raise RuntimeError("disk full")

    runtime = BoomRuntime()
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    coordinator = await _coord_at("conv-1", 1)
    writer = SessionLogWriter(runtime=runtime, turn_coordinator=coordinator)
    bus.subscribe(UserTextReceived, writer.handle_user_text)

    await bus.publish(
        UserTextReceived(
            user_id="alex",
            conversation_id="conv-1",
            source="cli",
            text="hello",
        )
    )

    failures = [
        event
        for event in await store.list_events()
        if isinstance(event, SessionLogAppendFailed)
    ]
    assert len(failures) == 1
    assert failures[0].role == "user"
    assert failures[0].conversation_id == "conv-1"
    assert "disk full" in failures[0].error


@pytest.mark.asyncio
async def test_session_search_reads_prior_session_with_colon_id() -> None:
    """Regression for the double-encoding bug: a prior session written
    with conversation_id `tg:456` must be discoverable from a different
    current conversation. Previously `list_session_logs` returned
    `tg%3A456` and `read_session_log` would re-encode that to
    `tg%253A456` and miss the file."""
    runtime = FakeUserRuntime()
    await runtime.append_session_log(
        "alex",
        "tg:456",
        json.dumps({"role": "user", "text": "deploy worker yesterday"}),
    )
    llm = FakeLlmClient([AssistantText(text="Earlier session covered deploy.")])
    service = SessionSearchService(runtime=runtime, llm=llm)
    result = await service.execute(
        user_id="alex",
        current_conversation_id="cli-abc",
        input=SessionSearchInput(query="deploy"),
    )
    payload = json.loads(result.stdout)
    assert len(payload["results"]) == 1
    assert payload["results"][0]["conversation_id"] == "tg:456"
    assert "Earlier session" in payload["results"][0]["summary"]


@pytest.mark.asyncio
async def test_session_search_summary_sees_tool_stdout(tmp_path: Path) -> None:
    """A ToolCallCompleted appended via SessionLogWriter carries stdout in the
    JSONL record. The transcript formatter must surface that stdout to the
    summariser so search hits on tool output land in a useful summary,
    not a blank `[TOOL:name]` line.
    """
    runtime = FakeUserRuntime()
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    coordinator = await _coord_at("conv-old", 1)
    writer = SessionLogWriter(runtime=runtime, turn_coordinator=coordinator)
    bus.subscribe(ToolCallCompleted, writer.handle_tool_call_completed)
    await bus.publish(
        ToolCallCompleted(
            user_id="alex",
            conversation_id="conv-old",
            generation=1,
            call_id="t-7",
            tool_name="shell.exec",
            input=ShellExecInput(command="cat /etc/secret"),
            result=RuntimeToolResult(stdout="deploy-credentials xyz123", exit_code=0),
        )
    )

    captured_transcripts: list[str] = []

    class CapturingLlm:
        async def respond(self, request):  # type: ignore[no-untyped-def]
            text = request.messages[-1].text
            captured_transcripts.append(text)
            return AssistantText(text="Captured.")

    service = SessionSearchService(runtime=runtime, llm=CapturingLlm())  # type: ignore[arg-type]
    result = await service.execute(
        user_id="alex",
        current_conversation_id="conv-current",
        input=SessionSearchInput(query="deploy-credentials"),
    )
    payload = json.loads(result.stdout)
    assert len(payload["results"]) == 1
    assert "deploy-credentials xyz123" in captured_transcripts[0]


@pytest.mark.asyncio
async def test_session_search_ranks_by_hit_count_and_caps_at_limit() -> None:
    """Five matching sessions, three distinct hit counts, limit=3. Must
    return the top 3 by hit count, in descending order, with
    match_score reporting the count."""
    runtime = FakeUserRuntime()
    seeds = [
        ("conv-many", "deploy deploy deploy deploy deploy"),  # 5 hits
        ("conv-some", "deploy deploy deploy"),  # 3 hits
        ("conv-few", "deploy"),  # 1 hit
        ("conv-also-few", "deploy"),  # 1 hit (tie)
        ("conv-mid", "deploy deploy"),  # 2 hits
    ]
    for conv_id, text in seeds:
        await runtime.append_session_log(
            "alex", conv_id, json.dumps({"role": "user", "text": text})
        )

    summaries = iter(
        [
            AssistantText(text="top match summary"),
            AssistantText(text="second summary"),
            AssistantText(text="third summary"),
        ]
    )

    class ScriptedLlm:
        async def respond(self, request):  # type: ignore[no-untyped-def]
            return next(summaries)

    service = SessionSearchService(runtime=runtime, llm=ScriptedLlm())  # type: ignore[arg-type]
    result = await service.execute(
        user_id="alex",
        current_conversation_id="conv-current",
        input=SessionSearchInput(query="deploy", limit=3),
    )
    payload = json.loads(result.stdout)
    assert payload["count"] == 3
    returned = [(entry["conversation_id"], int(entry["match_score"])) for entry in payload["results"]]
    # Top 3 by descending hit count.
    assert returned[0] == ("conv-many", 5)
    assert returned[1] == ("conv-some", 3)
    assert returned[2] == ("conv-mid", 2)
    # The two 1-hit sessions are truncated below the limit.
    returned_ids = {entry["conversation_id"] for entry in payload["results"]}
    assert "conv-few" not in returned_ids
    assert "conv-also-few" not in returned_ids


@pytest.mark.asyncio
async def test_session_search_reports_no_matches_when_empty() -> None:
    runtime = FakeUserRuntime()
    llm = FakeLlmClient([])
    service = SessionSearchService(runtime=runtime, llm=llm)
    result = await service.execute(
        user_id="alex",
        current_conversation_id="conv-current",
        input=SessionSearchInput(query="deploy"),
    )
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["results"] == []


@pytest.mark.asyncio
async def test_session_search_summary_failure_does_not_leak_transcript() -> None:
    """If the auxiliary summariser raises, the contract is to return a
    sanitized 'summary unavailable' marker — never the raw transcript,
    which may carry tool stdout containing secrets the user did not
    intend for the model to see verbatim."""
    runtime = FakeUserRuntime()
    await runtime.append_session_log(
        "alex",
        "conv-old",
        json.dumps(
            {
                "role": "tool",
                "tool_name": "shell.exec",
                "stdout": "deploy-credentials sk-SECRET-VALUE",
            }
        ),
    )

    class FailingLlm:
        async def respond(self, request):  # type: ignore[no-untyped-def]
            raise TimeoutError("summariser down")

    service = SessionSearchService(runtime=runtime, llm=FailingLlm())  # type: ignore[arg-type]
    result = await service.execute(
        user_id="alex",
        current_conversation_id="conv-current",
        input=SessionSearchInput(query="deploy-credentials"),
    )
    payload = json.loads(result.stdout)
    assert len(payload["results"]) == 1
    summary = payload["results"][0]["summary"]
    assert "summary unavailable" in summary
    assert "sk-SECRET-VALUE" not in summary


@pytest.mark.asyncio
async def test_session_log_writer_skips_stale_assistant_text(tmp_path: Path) -> None:
    """SessionLogWriter must skip AssistantTextProduced events for a
    superseded generation, mirroring ConversationProjector. Without this
    a stale older generation's wrap-up text would land in the JSONL log
    after a newer turn already advanced the conversation."""
    runtime = FakeUserRuntime()
    coordinator = ConversationTurnCoordinator()
    writer = SessionLogWriter(runtime=runtime, turn_coordinator=coordinator)
    # Open the conversation and immediately request a newer generation;
    # generation 1 is now stale relative to current=2.
    gen_1 = await coordinator.request_generation("conv-1")
    gen_2 = await coordinator.request_generation("conv-1")
    assert gen_1 == 1 and gen_2 == 2
    await writer.handle_assistant_text(
        AssistantTextProduced(
            user_id="alex",
            conversation_id="conv-1",
            generation=1,
            text="stale answer",
        )
    )
    await writer.handle_assistant_text(
        AssistantTextProduced(
            user_id="alex",
            conversation_id="conv-1",
            generation=2,
            text="fresh answer",
        )
    )
    log = await runtime.read_session_log("alex", "conv-1")
    lines = [json.loads(line) for line in log.splitlines() if line.strip()]
    assert [entry["text"] for entry in lines] == ["fresh answer"]


@pytest.mark.asyncio
async def test_session_log_writer_skips_stale_tool_completed(tmp_path: Path) -> None:
    runtime = FakeUserRuntime()
    coordinator = ConversationTurnCoordinator()
    writer = SessionLogWriter(runtime=runtime, turn_coordinator=coordinator)
    await coordinator.request_generation("conv-1")  # gen 1
    await coordinator.request_generation("conv-1")  # gen 2
    await writer.handle_tool_call_completed(
        ToolCallCompleted(
            user_id="alex",
            conversation_id="conv-1",
            generation=1,
            call_id="t-stale",
            tool_name="shell.exec",
            input=ShellExecInput(command="echo stale"),
            result=RuntimeToolResult(stdout="stale", exit_code=0),
        )
    )
    log = await runtime.read_session_log("alex", "conv-1")
    assert log == ""


@pytest.mark.asyncio
async def test_session_search_through_tool_call_executor(tmp_path: Path) -> None:
    """End-to-end: the session.search tool, invoked via ToolCallRequested
    on the event bus, executes through ToolCallExecutor and produces a
    ToolCallCompleted with the search payload as stdout."""
    runtime = FakeUserRuntime()
    await runtime.append_session_log(
        "alex",
        "conv-old",
        json.dumps({"role": "user", "text": "how do we deploy the worker?"}),
    )
    llm = FakeLlmClient([AssistantText(text="Earlier the user asked about deploy.")])
    service = SessionSearchService(runtime=runtime, llm=llm)
    executor = ToolCallExecutor(runtime=runtime, session_search=service)
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    bus.subscribe(ToolCallRequested, executor.handle_tool_call_requested)
    completed: list[ToolCallCompleted] = []

    async def _capture(event: ToolCallCompleted):
        completed.append(event)
        return ()

    bus.subscribe(ToolCallCompleted, _capture)

    await bus.publish(
        ToolCallRequested(
            user_id="alex",
            conversation_id="conv-current",
            generation=1,
            call_id="ss-1",
            tool_name="session.search",
            input=SessionSearchInput(query="deploy"),
        )
    )
    assert len(completed) == 1
    payload = json.loads(completed[0].result.stdout)
    assert payload["success"] is True
    assert payload["results"][0]["conversation_id"] == "conv-old"
    assert "Earlier the user asked" in payload["results"][0]["summary"]
