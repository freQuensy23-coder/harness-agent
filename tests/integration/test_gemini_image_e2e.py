"""Live end-to-end tests against Gemini Nano Banana (gemini-2.5-flash-image).

Skipped unless `GEMINI_API_KEY` is set, so they never run in default CI.
Run explicitly with:

    GEMINI_API_KEY=... uv run pytest tests/integration/test_gemini_image_e2e.py -s

The flex service tier is cheap but can take several minutes per call; tests
allow up to 5 minutes per generation.
"""

import asyncio
import os
from pathlib import Path

import pytest

from harness_agent.content import ContentRef
from harness_agent.context import AgentFileSet
from harness_agent.events import ToolCallCompleted, ToolCallRequested
from harness_agent.image_generate import GeminiImageGenerator
from harness_agent.image_jobs import ImageJobService, SQLiteImageJobStore
from harness_agent.runtime import FakeUserRuntime
from harness_agent.tool_executor import ToolCallExecutor
from harness_agent.tools import ImageGenerateInput, ImageStatusInput


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_WAIT_SECONDS = 300.0

requires_gemini = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY not set; export it to run live Gemini Nano Banana tests.",
)


def _gemini_generator() -> GeminiImageGenerator:
    assert GEMINI_API_KEY is not None
    return GeminiImageGenerator(
        api_key=GEMINI_API_KEY,
        service_tier="flex",
        timeout_seconds=MAX_WAIT_SECONDS,
    )


@requires_gemini
@pytest.mark.asyncio
async def test_gemini_generator_returns_real_png() -> None:
    generator = _gemini_generator()
    image = await generator.generate(
        ImageGenerateInput(
            prompt=(
                "A simple flat illustration of an orange tabby cat wearing a red bowtie, "
                "centered on a plain white background. Cartoon style."
            ),
            output_path="/workspace/cat.png",
        )
    )
    assert image.mime_type.startswith("image/")
    # PNG magic header or JPEG SOI — both are valid images.
    assert image.data[:8] == b"\x89PNG\r\n\x1a\n" or image.data[:3] == b"\xff\xd8\xff"
    assert len(image.data) > 1024, f"image too small: {len(image.data)} bytes"


@requires_gemini
@pytest.mark.asyncio
async def test_image_job_lifecycle_with_real_gemini(tmp_path: Path) -> None:
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T")
    )
    store = SQLiteImageJobStore(tmp_path / "image_jobs.sqlite3")
    service = ImageJobService(store=store, generator=_gemini_generator(), runtime=runtime)

    record = await service.start(
        user_id="u:e2e",
        conversation_id="cli:e2e",
        parent_call_id="probe",
        input=ImageGenerateInput(
            prompt="A small grayscale icon of a sailboat on calm water. Minimalist.",
            output_path="/workspace/content/boat.png",
            aspect_ratio="1:1",
        ),
    )

    # service.start() must return immediately, before the API call completes.
    assert record.status == "running"
    assert record.mime_type is None
    assert record.size_bytes is None

    deadline = asyncio.get_running_loop().time() + MAX_WAIT_SECONDS
    latest = record
    while asyncio.get_running_loop().time() < deadline:
        fetched = await service.get(
            job_id=record.id, user_id="u:e2e", conversation_id="cli:e2e"
        )
        assert fetched is not None
        latest = fetched
        if latest.status != "running":
            break
        await asyncio.sleep(2)

    assert latest.status == "completed", (
        f"job did not complete in {MAX_WAIT_SECONDS}s — last record: {latest}"
    )
    assert latest.mime_type is not None and latest.mime_type.startswith("image/")
    assert latest.size_bytes is not None and latest.size_bytes > 1024
    assert runtime.content_write_calls
    written_path, written_bytes = runtime.content_write_calls[-1]
    assert written_path == "/workspace/content/boat.png"
    assert len(written_bytes) == latest.size_bytes
    assert record.id not in service._tasks  # type: ignore[attr-defined]


@requires_gemini
@pytest.mark.asyncio
async def test_image_status_attaches_real_png_via_tool_executor(tmp_path: Path) -> None:
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T")
    )
    store = SQLiteImageJobStore(tmp_path / "image_jobs.sqlite3")
    service = ImageJobService(store=store, generator=_gemini_generator(), runtime=runtime)
    executor = ToolCallExecutor(runtime=runtime, image_jobs=service)

    # 1. image.generate -- returns immediately with a job id.
    start_event = ToolCallRequested(
        user_id="u:e2e",
        conversation_id="cli:e2e",
        generation=1,
        call_id="gen",
        tool_name="image.generate",
        input=ImageGenerateInput(
            prompt="A small pixel-art mushroom on a transparent background. Bright colors.",
            output_path="/workspace/content/mushroom.png",
        ),
    )
    start_batch = await executor.handle_tool_call_requested(start_event)
    start_completed = next(e for e in start_batch if isinstance(e, ToolCallCompleted))
    assert start_completed.result.exit_code == 0
    # The stdout is the job record JSON; pull out the id.
    import json as _json

    job_payload = _json.loads(start_completed.result.stdout)
    job_id = str(job_payload["id"])
    assert job_payload["status"] == "running"

    # 2. Poll image.status through the executor until completed.
    deadline = asyncio.get_running_loop().time() + MAX_WAIT_SECONDS
    final_completed: ToolCallCompleted | None = None
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(2)
        status_event = ToolCallRequested(
            user_id="u:e2e",
            conversation_id="cli:e2e",
            generation=1,
            call_id=f"status-{int(asyncio.get_running_loop().time())}",
            tool_name="image.status",
            input=ImageStatusInput(image_id=job_id),
        )
        batch = await executor.handle_tool_call_requested(status_event)
        completed = next(e for e in batch if isinstance(e, ToolCallCompleted))
        payload = _json.loads(completed.result.stdout)
        if payload["status"] != "running":
            final_completed = completed
            break

    assert final_completed is not None, "image.status never reported a terminal state"
    final_payload = _json.loads(final_completed.result.stdout)
    assert final_payload["status"] == "completed", (
        f"job ended in non-completed state: {final_payload}"
    )
    assert final_completed.attachments, "expected the saved image to be attached"
    attachment = final_completed.attachments[0]
    assert isinstance(attachment, ContentRef)
    assert attachment.kind == "image"
    assert attachment.mime_type.startswith("image/")
    assert attachment.workspace_path == "/workspace/content/mushroom.png"
    assert attachment.size_bytes > 1024
