import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from harness_agent.bus import EventBus
from harness_agent.content import ContentRef
from harness_agent.context import AgentFileSet, ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    AssistantTextProduced,
    ImageJobCompleted,
    ImageJobFailed,
    ImageJobRequested,
    ImageJobStarted,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import AgentTurnHandler, ConversationProjector
from harness_agent.image_generate import (
    GeminiImageGenerator,
    GeneratedImage,
    ImageGenerationError,
    ImageGenerator,
)
from harness_agent.image_jobs import (
    ImageDeliveryHandler,
    ImageJobService,
    SQLiteImageJobStore,
)
from harness_agent.llm import AssistantText, FakeLlmClient, LlmToolCall
from harness_agent.memory_service import MemoryService
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime, RuntimeToolResult
from harness_agent.session_search_service import SessionSearchService
from harness_agent.store import SQLiteEventStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import (
    ImageGenerateInput,
    ImageStatusInput,
    default_tool_registry,
)


def _wire_image_service(
    *,
    tmp_path: Path,
    generator: ImageGenerator,
    runtime: FakeUserRuntime,
) -> tuple[ImageJobService, EventBus, SQLiteEventStore]:
    """Create a fully wired ImageJobService for unit tests.

    Without subscribers, start() would hang waiting on ImageJobStarted.
    """
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(event_store)
    job_store = SQLiteImageJobStore(tmp_path / "image_jobs.sqlite3")
    service = ImageJobService(
        bus=bus, store=job_store, generator=generator, runtime=runtime
    )
    bus.subscribe(ImageJobRequested, service.handle_requested)
    bus.subscribe(ImageJobStarted, service.handle_started)
    bus.subscribe(ImageJobCompleted, service.handle_completed)
    bus.subscribe(ImageJobFailed, service.handle_failed)
    return service, bus, event_store


def _tool_executor_for_test(
    *,
    runtime: FakeUserRuntime,
    memory_service: MemoryService | None = None,
    session_search: SessionSearchService | None = None,
    session_search_llm: FakeLlmClient | None = None,
    **kwargs: object,
) -> ToolCallExecutor:
    return ToolCallExecutor(
        runtime=runtime,
        memory_service=memory_service or MemoryService(runtime=runtime),
        session_search=session_search
        or SessionSearchService(
            runtime=runtime,
            llm=session_search_llm or FakeLlmClient([]),
        ),
        **kwargs,
    )


class _CapturingHandler:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content.decode("utf-8")))
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "here you go"},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"PNGBYTES").decode("ascii"),
                                }
                            },
                        ]
                    }
                }
            ]
        }
        return httpx.Response(200, json=body)


@pytest.mark.asyncio
async def test_gemini_image_generator_sends_flex_tier_and_aspect_ratio() -> None:
    handler = _CapturingHandler()
    transport = httpx.MockTransport(handler)
    generator = GeminiImageGenerator(
        api_key="test-key",
        model="gemini-2.5-flash-image",
        service_tier="flex",
        transport=transport,
    )

    image = await generator.generate(
        ImageGenerateInput(
            prompt="a cat in a hat",
            output_path="/workspace/content/cat.png",
            aspect_ratio="16:9",
        )
    )

    assert image == GeneratedImage(
        mime_type="image/png",
        data=b"PNGBYTES",
        text="here you go",
    )
    assert len(handler.requests) == 1
    sent = handler.requests[0]
    assert sent.method == "POST"
    assert sent.url.path.endswith("/models/gemini-2.5-flash-image:generateContent")
    assert sent.headers["x-goog-api-key"] == "test-key"
    body = handler.bodies[0]
    assert body["serviceTier"] == "flex"
    generation_config = body["generationConfig"]
    assert isinstance(generation_config, dict)
    assert generation_config["imageConfig"] == {"aspectRatio": "16:9"}
    assert generation_config["responseModalities"] == ["IMAGE"]
    contents = body["contents"]
    assert isinstance(contents, list)
    assert contents[0]["parts"] == [{"text": "a cat in a hat"}]


@pytest.mark.asyncio
async def test_gemini_image_generator_refuses_placeholder_api_key() -> None:
    generator = GeminiImageGenerator(api_key="replace-me")
    with pytest.raises(ImageGenerationError):
        await generator.generate(
            ImageGenerateInput(prompt="x", output_path="/workspace/x.png")
        )


@pytest.mark.asyncio
async def test_gemini_image_generator_raises_when_no_image_part() -> None:
    body = {
        "candidates": [
            {"content": {"parts": [{"text": "blocked: safety"}]}}
        ]
    }
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json=body))
    generator = GeminiImageGenerator(api_key="k", transport=transport)
    with pytest.raises(ImageGenerationError, match="no image part"):
        await generator.generate(
            ImageGenerateInput(prompt="x", output_path="/workspace/x.png")
        )


class _BlockingGenerator(ImageGenerator):
    """Generator that holds an event until released, then returns bytes."""

    def __init__(self, *, data: bytes = b"BYTES") -> None:
        self.release_event = asyncio.Event()
        self.start_event = asyncio.Event()
        self.generate_calls: list[ImageGenerateInput] = []
        self._data = data

    async def generate(self, input: ImageGenerateInput) -> GeneratedImage:
        self.generate_calls.append(input)
        self.start_event.set()
        await self.release_event.wait()
        return GeneratedImage(mime_type="image/png", data=self._data)


class _FailingGenerator(ImageGenerator):
    async def generate(self, input: ImageGenerateInput) -> GeneratedImage:
        raise ImageGenerationError("safety block")


@pytest.mark.asyncio
async def test_image_generate_returns_immediately_while_generation_runs(
    tmp_path: Path,
) -> None:
    runtime = FakeUserRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    generator = _BlockingGenerator()
    service, _bus, _store = _wire_image_service(
        tmp_path=tmp_path, generator=generator, runtime=runtime
    )

    record = await service.start(
        user_id="u:1",
        conversation_id="cli:1",
        parent_call_id="call-1",
        input=ImageGenerateInput(prompt="kitty", output_path="/workspace/content/k.png"),
    )

    # Job is persisted as running before the generator finishes.
    assert record.status == "running"
    persisted = await service.get(
        job_id=record.id, user_id="u:1", conversation_id="cli:1"
    )
    assert persisted is not None
    assert persisted.status == "running"
    await generator.start_event.wait()

    # Release the generator, then the job completes and writes to the workspace.
    generator.release_event.set()
    for _ in range(50):
        await asyncio.sleep(0)
        finished = await service.get(
            job_id=record.id, user_id="u:1", conversation_id="cli:1"
        )
        if finished is not None and finished.status == "completed":
            break
    assert finished is not None
    assert finished.status == "completed"
    assert finished.mime_type == "image/png"
    assert finished.size_bytes == len(b"BYTES")
    assert runtime.content_write_calls == [("/workspace/content/k.png", b"BYTES")]


@pytest.mark.asyncio
async def test_image_status_attaches_image_when_completed(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T")
    )
    job_store = SQLiteImageJobStore(tmp_path / "image_jobs.sqlite3")
    generator = _BlockingGenerator(data=b"\x89PNG\r\n\x1a\nfake-bytes")
    bus = EventBus(store)
    image_jobs = ImageJobService(
        bus=bus, store=job_store, generator=generator, runtime=runtime
    )
    bus.subscribe(ImageJobRequested, image_jobs.handle_requested)
    bus.subscribe(ImageJobStarted, image_jobs.handle_started)
    bus.subscribe(ImageJobCompleted, image_jobs.handle_completed)
    bus.subscribe(ImageJobFailed, image_jobs.handle_failed)

    # Seed the LLM with a generate+status sequence.
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="gen",
                name="image.generate",
                input=ImageGenerateInput(
                    prompt="kitty",
                    output_path="/workspace/content/k.png",
                ),
            ),
            LlmToolCall(
                call_id="stat",
                name="image.status",
                input=ImageStatusInput(image_id="filled-in-later"),
            ),
            AssistantText(text="done"),
        ]
    )
    tool_results = ToolCallResultWaiter()
    coordinator = ConversationTurnCoordinator()
    handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    tool_executor = _tool_executor_for_test(runtime=runtime, image_jobs=image_jobs)
    bus.subscribe(
        UserTextReceived,
        ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text,
    )
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(
        ToolCallCompleted,
        ConversationProjector(
            projection,
            turn_coordinator=coordinator,
        ).handle_tool_call_completed,
    )
    bus.subscribe(ToolCallCompleted, handler.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, handler.handle_agent_turn)

    # When the generator hits start_event, release it so by the time the second
    # tool call (image.status) runs, the job is already completed.
    async def release_after_start() -> None:
        await generator.start_event.wait()
        generator.release_event.set()

    release_task = asyncio.create_task(release_after_start())

    # Patch the second LlmToolCall to use the actual image_id minted during the run.
    real_respond = llm.respond

    async def respond_with_real_id(request: object) -> object:
        response = await real_respond(request)  # type: ignore[arg-type]
        if (
            isinstance(response, LlmToolCall)
            and response.name == "image.status"
            and isinstance(response.input, ImageStatusInput)
            and response.input.image_id == "filled-in-later"
        ):
            jobs = await image_jobs.list_for_conversation(
                user_id="u:1", conversation_id="cli:1"
            )
            for _ in range(50):
                if jobs and jobs[0].status == "completed":
                    break
                await asyncio.sleep(0)
                jobs = await image_jobs.list_for_conversation(
                    user_id="u:1", conversation_id="cli:1"
                )
            assert jobs, "expected a job to exist"
            response.input.image_id = jobs[0].id
        return response

    llm.respond = respond_with_real_id  # type: ignore[method-assign]

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:1",
            source="cli",
            text="render me a kitty",
        )
    )
    await release_task

    events = await store.list_events()
    status_events = [
        e for e in events if e.type == "tool.call.completed" and e.tool_name == "image.status"
    ]
    assert len(status_events) == 1
    completed_event = status_events[0]
    assert completed_event.result.exit_code == 0
    assert completed_event.attachments
    attachment = completed_event.attachments[0]
    assert isinstance(attachment, ContentRef)
    assert attachment.kind == "image"
    assert attachment.mime_type == "image/png"
    assert attachment.workspace_path == "/workspace/content/k.png"


@pytest.mark.asyncio
async def test_image_status_reports_running_then_failed(tmp_path: Path) -> None:
    runtime = FakeUserRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    service, _bus, _store = _wire_image_service(
        tmp_path=tmp_path, generator=_FailingGenerator(), runtime=runtime
    )

    record = await service.start(
        user_id="u:1",
        conversation_id="cli:1",
        parent_call_id="call-1",
        input=ImageGenerateInput(prompt="x", output_path="/workspace/x.png"),
    )

    latest = record
    for _ in range(50):
        await asyncio.sleep(0)
        fetched = await service.get(
            job_id=record.id, user_id="u:1", conversation_id="cli:1"
        )
        if fetched is None:
            continue
        latest = fetched
        if latest.status != "running":
            break
    assert latest.status == "failed"
    assert latest.error == "safety block"


@pytest.mark.asyncio
async def test_image_job_runtime_raising_marks_failed_and_cleans_up(
    tmp_path: Path,
) -> None:
    class _RaisingRuntime(FakeUserRuntime):
        async def write_content_file(
            self, user_id: str, path: str, content: bytes
        ) -> RuntimeToolResult:
            raise RuntimeError("disk on fire")

    runtime = _RaisingRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    service, _bus, _store = _wire_image_service(
        tmp_path=tmp_path,
        generator=_BlockingGenerator(),  # immediate when released
        runtime=runtime,
    )
    record = await service.start(
        user_id="u:1",
        conversation_id="cli:1",
        parent_call_id="call-1",
        input=ImageGenerateInput(prompt="x", output_path="/workspace/x.png"),
    )
    # release the generator so the runtime path executes and raises
    generator = service._generator  # type: ignore[attr-defined]
    assert isinstance(generator, _BlockingGenerator)
    generator.release_event.set()

    latest = record
    for _ in range(50):
        await asyncio.sleep(0)
        fetched = await service.get(
            job_id=record.id, user_id="u:1", conversation_id="cli:1"
        )
        if fetched is None:
            continue
        latest = fetched
        if latest.status != "running":
            break
    assert latest.status == "failed"
    assert latest.error is not None and "disk on fire" in latest.error
    # background task must not be leaked
    assert record.id not in service._tasks  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_image_job_write_failure_marks_failed(tmp_path: Path) -> None:
    class _BadWriteRuntime(FakeUserRuntime):
        async def write_content_file(
            self, user_id: str, path: str, content: bytes
        ) -> RuntimeToolResult:
            return RuntimeToolResult(stderr="quota exceeded", exit_code=1)

    runtime = _BadWriteRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    generator = _BlockingGenerator()
    service, _bus, _store = _wire_image_service(
        tmp_path=tmp_path, generator=generator, runtime=runtime
    )
    record = await service.start(
        user_id="u:1",
        conversation_id="cli:1",
        parent_call_id="call-1",
        input=ImageGenerateInput(prompt="x", output_path="/workspace/x.png"),
    )
    generator.release_event.set()
    latest = record
    for _ in range(50):
        await asyncio.sleep(0)
        fetched = await service.get(
            job_id=record.id, user_id="u:1", conversation_id="cli:1"
        )
        if fetched is None:
            continue
        latest = fetched
        if latest.status != "running":
            break
    assert latest.status == "failed"
    assert latest.error == "quota exceeded"
    assert record.id not in service._tasks  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_image_status_unknown_id_returns_error_result(tmp_path: Path) -> None:
    runtime = FakeUserRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    image_jobs, _bus, _store = _wire_image_service(
        tmp_path=tmp_path, generator=_BlockingGenerator(), runtime=runtime
    )
    tool_executor = _tool_executor_for_test(runtime=runtime, image_jobs=image_jobs)
    event = ToolCallRequested(
        user_id="u:1",
        conversation_id="cli:1",
        generation=1,
        call_id="probe",
        tool_name="image.status",
        input=ImageStatusInput(image_id="does-not-exist"),
    )
    batch = await tool_executor.handle_tool_call_requested(event)
    completed = next(e for e in batch if isinstance(e, ToolCallCompleted))
    assert completed.result.exit_code == 1
    assert "Unknown image job" in completed.result.stderr
    assert completed.attachments == []


@pytest.mark.asyncio
async def test_image_generate_does_not_block_other_conversations(tmp_path: Path) -> None:
    """An in-flight image.generate must not block tool calls in another conversation."""

    runtime = FakeUserRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    blocker = _BlockingGenerator()
    image_jobs, _bus, _store = _wire_image_service(
        tmp_path=tmp_path, generator=blocker, runtime=runtime
    )
    tool_executor = _tool_executor_for_test(runtime=runtime, image_jobs=image_jobs)

    # Conversation A starts an image generation that will block until released.
    start_event = ToolCallRequested(
        user_id="u:1",
        conversation_id="cli:a",
        generation=1,
        call_id="gen-a",
        tool_name="image.generate",
        input=ImageGenerateInput(prompt="x", output_path="/workspace/a.png"),
    )
    batch_a = await tool_executor.handle_tool_call_requested(start_event)
    completed_a = next(e for e in batch_a if isinstance(e, ToolCallCompleted))
    assert completed_a.result.exit_code == 0
    # Give the background task one event-loop tick to enter generate().
    for _ in range(20):
        if blocker.start_event.is_set():
            break
        await asyncio.sleep(0)
    # image.generate returned even though the generator is still blocked.
    assert not blocker.release_event.is_set()
    assert blocker.start_event.is_set()

    # Meanwhile conversation B can run its own status probe.
    other_event = ToolCallRequested(
        user_id="u:2",
        conversation_id="cli:b",
        generation=1,
        call_id="probe-b",
        tool_name="image.status",
        input=ImageStatusInput(image_id="missing"),
    )
    batch_b = await tool_executor.handle_tool_call_requested(other_event)
    completed_b = next(e for e in batch_b if isinstance(e, ToolCallCompleted))
    assert "Unknown image job" in completed_b.result.stderr

    # Tidy up the dangling generator task.
    blocker.release_event.set()
    job_id = str(json.loads(completed_a.result.stdout)["id"])
    latest = await image_jobs.get(job_id=job_id, user_id="u:1", conversation_id="cli:a")
    for _ in range(50):
        await asyncio.sleep(0)
        latest = await image_jobs.get(
            job_id=job_id, user_id="u:1", conversation_id="cli:a"
        )
        if latest is not None and latest.status == "completed":
            break
    assert latest is not None and latest.status == "completed"


@pytest.mark.asyncio
async def test_image_job_lifecycle_emits_event_chain(tmp_path: Path) -> None:
    """Verify the event-driven contract: Requested -> Started -> Completed."""
    runtime = FakeUserRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    generator = _BlockingGenerator()
    service, _bus, event_store = _wire_image_service(
        tmp_path=tmp_path, generator=generator, runtime=runtime
    )

    record = await service.start(
        user_id="u:1",
        conversation_id="cli:1",
        parent_call_id="call-1",
        input=ImageGenerateInput(prompt="hat", output_path="/workspace/hat.png"),
    )
    # Started must be in the audit log before start() returned.
    types_so_far = [e.type for e in await event_store.list_events()]
    assert "image.job.requested" in types_so_far
    assert "image.job.started" in types_so_far

    generator.release_event.set()
    for _ in range(80):
        await asyncio.sleep(0)
        if any(e.type == "image.job.completed" for e in await event_store.list_events()):
            break
    types_after = [e.type for e in await event_store.list_events()]
    # Lifecycle is fully persisted as events, in order.
    requested_idx = types_after.index("image.job.requested")
    started_idx = types_after.index("image.job.started")
    completed_idx = types_after.index("image.job.completed")
    assert requested_idx < started_idx < completed_idx
    assert "image.job.failed" not in types_after

    # And the completion event carries enough to push the image without re-querying.
    completed_events = [
        e for e in await event_store.list_events() if e.type == "image.job.completed"
    ]
    assert len(completed_events) == 1
    payload = completed_events[0]
    assert payload.job_id == record.id
    assert payload.output_path == "/workspace/hat.png"
    assert payload.mime_type == "image/png"
    assert payload.size_bytes == len(b"BYTES")


@pytest.mark.asyncio
async def test_image_delivery_handler_appends_image_to_projection(
    tmp_path: Path,
) -> None:
    """ImageJobCompleted must surface the image into the conversation projection
    as a multimodal user message so the next agent turn sees it without polling.
    """
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        files={"/workspace/cat.png": b"\x89PNG\r\n\x1a\nfake-bytes"},
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    delivery = ImageDeliveryHandler(runtime=runtime, projection=projection)
    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    bus.subscribe(ImageJobCompleted, delivery.handle_completed)

    await bus.publish(
        ImageJobCompleted(
            job_id="job-1",
            user_id="u:1",
            conversation_id="cli:1",
            output_path="/workspace/cat.png",
            mime_type="image/png",
            size_bytes=len(b"\x89PNG\r\n\x1a\nfake-bytes"),
        )
    )

    messages = await projection.list_llm_messages("cli:1")
    user_messages = [m for m in messages if m.kind == "user"]
    assert len(user_messages) == 1
    delivered = user_messages[0]
    assert "image.generate" in delivered.text
    assert "job-1" in delivered.text
    assert len(delivered.attachments) == 1
    attachment = delivered.attachments[0]
    assert attachment.kind == "image"
    assert attachment.mime_type == "image/png"
    assert attachment.workspace_path == "/workspace/cat.png"


@pytest.mark.asyncio
async def test_recover_interrupted_jobs_emits_failed_for_stuck_running(
    tmp_path: Path,
) -> None:
    """Records left in 'running' from a previous process must be transitioned
    to failed on next startup, via the normal ImageJobFailed event path.
    """
    runtime = FakeUserRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    # Pre-populate the store with a stuck 'running' record (simulates a crash).
    job_store = SQLiteImageJobStore(tmp_path / "image_jobs.sqlite3")
    await job_store.insert(
        job_id="ghost",
        user_id="u:1",
        conversation_id="cli:1",
        parent_call_id="call-ghost",
        prompt="ghost",
        output_path="/workspace/ghost.png",
        aspect_ratio="1:1",
    )
    # Wire a fresh service over the same store.
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(event_store)
    service = ImageJobService(
        bus=bus,
        store=job_store,
        generator=_BlockingGenerator(),
        runtime=runtime,
    )
    bus.subscribe(ImageJobRequested, service.handle_requested)
    bus.subscribe(ImageJobStarted, service.handle_started)
    bus.subscribe(ImageJobCompleted, service.handle_completed)
    bus.subscribe(ImageJobFailed, service.handle_failed)

    recovered = await service.recover_interrupted_jobs()
    assert [r.id for r in recovered] == ["ghost"]

    # Event log must show the failure transition through the normal channel.
    types = [e.type for e in await event_store.list_events()]
    assert "image.job.failed" in types
    final = await service.get(
        job_id="ghost", user_id="u:1", conversation_id="cli:1"
    )
    assert final is not None
    assert final.status == "failed"
    assert final.error == "interrupted by process restart"


@pytest.mark.asyncio
async def test_full_event_driven_image_flow_lands_image_in_next_turn_context(
    tmp_path: Path,
) -> None:
    """End-to-end through the real event-driven pipeline.

    Wires: ToolCallExecutor + ImageJobService + ImageDeliveryHandler +
    ConversationProjector + AgentTurnHandler. Asserts that after the
    background image generation finishes, the second LLM turn for the same
    conversation receives the rendered image as a user-message attachment
    in its LlmRequest — i.e. the model can see the image without ever
    calling image.status.
    """
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T")
    )
    bus = EventBus(event_store)

    image_data = b"\x89PNG\r\n\x1a\nFAKE-PNG-PAYLOAD"
    generator = _BlockingGenerator(data=image_data)
    image_jobs = ImageJobService(
        bus=bus,
        store=SQLiteImageJobStore(tmp_path / "image_jobs.sqlite3"),
        generator=generator,
        runtime=runtime,
    )
    delivery = ImageDeliveryHandler(runtime=runtime, projection=projection)
    bus.subscribe(ImageJobRequested, image_jobs.handle_requested)
    bus.subscribe(ImageJobStarted, image_jobs.handle_started)
    bus.subscribe(ImageJobCompleted, image_jobs.handle_completed)
    bus.subscribe(ImageJobFailed, image_jobs.handle_failed)
    bus.subscribe(ImageJobCompleted, delivery.handle_completed)

    # LLM script: turn 1 emits image.generate, then finishes with text.
    # Turn 2 will run after the user sends a follow-up; we capture its
    # request to inspect what the model actually sees.
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="gen",
                name="image.generate",
                input=ImageGenerateInput(
                    prompt="a cat in a hat",
                    output_path="/workspace/content/cat.png",
                ),
            ),
            AssistantText(text="working on it"),
            AssistantText(text="here is your image"),
        ]
    )
    tool_results = ToolCallResultWaiter()
    coordinator = ConversationTurnCoordinator()
    handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    tool_executor = _tool_executor_for_test(runtime=runtime, image_jobs=image_jobs)
    bus.subscribe(
        UserTextReceived,
        ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text,
    )
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(
        ToolCallCompleted,
        ConversationProjector(
            projection,
            turn_coordinator=coordinator,
        ).handle_tool_call_completed,
    )
    bus.subscribe(ToolCallCompleted, handler.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, handler.handle_agent_turn)
    bus.subscribe(
        AssistantTextProduced,
        ConversationProjector(
            projection,
            turn_coordinator=coordinator,
        ).handle_assistant_text,
    )

    # Turn 1: user asks for an image; agent fires image.generate; tool
    # returns instantly with a running job; agent says "working on it".
    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:e2e",
            source="cli",
            text="draw me a cat",
        )
    )

    # Sanity: image.generate already published Requested + Started, but the
    # generator is still blocked, so no Completed yet.
    types_after_turn_1 = [e.type for e in await event_store.list_events()]
    assert "image.job.requested" in types_after_turn_1
    assert "image.job.started" in types_after_turn_1
    assert "image.job.completed" not in types_after_turn_1

    # Now let the (fake) generation finish.
    generator.release_event.set()
    for _ in range(80):
        await asyncio.sleep(0)
        if any(
            e.type == "image.job.completed"
            for e in await event_store.list_events()
        ):
            break

    types_after_completion = [e.type for e in await event_store.list_events()]
    assert "image.job.completed" in types_after_completion
    # Projection now carries a synthetic user message with the image attached.
    history = await projection.list_llm_messages("cli:e2e")
    delivered = [m for m in history if m.kind == "user" and m.attachments]
    assert len(delivered) == 1
    delivered_attachment = delivered[0].attachments[0]
    assert delivered_attachment.kind == "image"
    assert delivered_attachment.workspace_path == "/workspace/content/cat.png"
    assert delivered_attachment.mime_type == "image/png"

    # Turn 2: user follows up; agent's LlmRequest must carry the image in
    # its messages list — that's what "the model sees the image in context"
    # means in practice.
    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:e2e",
            source="cli",
            text="what do you think?",
        )
    )

    assert len(llm.requests) == 3, "expected three LLM calls (gen, mid, follow-up)"
    follow_up_request = llm.requests[-1]
    user_messages_with_attachments = [
        m
        for m in follow_up_request.messages
        if m.kind == "user" and m.attachments
    ]
    assert user_messages_with_attachments, (
        "follow-up turn must see the generated image in its message history"
    )
    seen_attachment = user_messages_with_attachments[0].attachments[0]
    assert seen_attachment.workspace_path == "/workspace/content/cat.png"
    assert seen_attachment.mime_type == "image/png"


@pytest.mark.asyncio
async def test_image_delivery_handler_skips_when_file_unreadable(
    tmp_path: Path,
) -> None:
    """If the saved file vanished between ImageJobCompleted and delivery
    (e.g. workspace cleanup), the handler must not append a broken message.
    """
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    # No files seeded -> read_file_bytes returns exit_code=1.
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T")
    )
    delivery = ImageDeliveryHandler(runtime=runtime, projection=projection)
    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    bus.subscribe(ImageJobCompleted, delivery.handle_completed)

    await bus.publish(
        ImageJobCompleted(
            job_id="vanished",
            user_id="u:1",
            conversation_id="cli:1",
            output_path="/workspace/missing.png",
            mime_type="image/png",
            size_bytes=42,
        )
    )

    history = await projection.list_llm_messages("cli:1")
    # Nothing should land in the projection on an unreadable file.
    assert history == []
