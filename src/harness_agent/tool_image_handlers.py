"""Image tool handlers for `image.generate` and `image.status` calls."""

from harness_agent.content import content_ref_from_bytes
from harness_agent.events import ToolCallRequested
from harness_agent.image_jobs import ImageJobService, render_image_job_record
from harness_agent.runtime import RuntimeToolResult, UserRuntime
from harness_agent.tool_call_results import ToolExecutionResult
from harness_agent.tools import ImageGenerateInput, ImageStatusInput


class ImageToolHandlers:
    """Handlers backed by `ImageJobService` plus runtime file reads."""

    def __init__(self, *, runtime: UserRuntime, image_jobs: ImageJobService) -> None:
        self._runtime = runtime
        self._image_jobs = image_jobs

    async def generate(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        record = await self._image_jobs.start(
            user_id=user_id,
            conversation_id=event.conversation_id,
            parent_call_id=event.call_id,
            input=ImageGenerateInput.model_validate(event.input),
        )
        return RuntimeToolResult(stdout=render_image_job_record(record))

    async def status(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> ToolExecutionResult:
        input = ImageStatusInput.model_validate(event.input)
        record = await self._image_jobs.get(
            job_id=input.image_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
        )
        if record is None:
            return ToolExecutionResult(
                result=RuntimeToolResult(
                    stderr=f"Unknown image job: {input.image_id}\n",
                    exit_code=1,
                )
            )
        if record.status != "completed":
            return ToolExecutionResult(
                result=RuntimeToolResult(stdout=render_image_job_record(record))
            )
        if record.mime_type is None:
            raise RuntimeError(
                f"completed image job {record.id} is missing mime_type"
            )
        read = await self._runtime.read_file_bytes(user_id, record.output_path, None)
        if read.result.exit_code != 0:
            return ToolExecutionResult(result=read.result)
        if read.file is None:
            raise RuntimeError("image file read succeeded without file content")
        content_ref = content_ref_from_bytes(
            kind="image",
            file_name=_basename(record.output_path),
            mime_type=record.mime_type,
            workspace_path=record.output_path,
            content=read.file.content,
        )
        return ToolExecutionResult(
            result=RuntimeToolResult(stdout=render_image_job_record(record)),
            attachments=[content_ref],
        )


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] or path
