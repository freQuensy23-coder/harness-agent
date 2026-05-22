from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from harness_agent.content import ContentRef
from harness_agent.runtime import RuntimeToolResult
from harness_agent.tools import ToolInput, parse_stored_tool_input


class EventBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TelegramReplyTarget(BaseModel):
    kind: Literal["telegram"] = "telegram"
    chat_id: int


class CliReplyTarget(BaseModel):
    kind: Literal["cli"] = "cli"
    request_id: str


ReplyTarget = Annotated[
    TelegramReplyTarget | CliReplyTarget,
    Field(discriminator="kind"),
]


class InboundAttachment(BaseModel):
    kind: Literal["image", "file"]
    file_name: str
    mime_type: str | None = None
    size_bytes: int
    workspace_path: str
    content_base64: str
    source_id: str | None = None


class TelegramTextReceived(EventBase):
    type: Literal["telegram.text.received"] = "telegram.text.received"
    telegram_user_id: int
    telegram_chat_id: int
    telegram_message_id: int
    text: str
    attachments: list[InboundAttachment] = Field(default_factory=list[InboundAttachment])


class CliTextReceived(EventBase):
    type: Literal["cli.text.received"] = "cli.text.received"
    cli_user_id: str
    conversation_id: str
    request_id: str
    text: str


class UserTextReceived(EventBase):
    type: Literal["user.text.received"] = "user.text.received"
    user_id: str
    conversation_id: str
    source: Literal["telegram", "cli", "api", "scheduler", "subagent"]
    text: str
    attachments: list[InboundAttachment] = Field(default_factory=list[InboundAttachment])
    reply_target: ReplyTarget | None = None


class AgentTurnRequested(EventBase):
    type: Literal["agent.turn.requested"] = "agent.turn.requested"
    user_id: str
    conversation_id: str
    generation: int
    input_event_id: str
    reply_target: ReplyTarget | None = None


class AgentTurnSuperseded(EventBase):
    type: Literal["agent.turn.superseded"] = "agent.turn.superseded"
    user_id: str
    conversation_id: str
    generation: int
    superseded_by: int
    reason: Literal["newer_user_message"] = "newer_user_message"


class AgentGenerationStarted(EventBase):
    type: Literal["agent.generation.started"] = "agent.generation.started"
    user_id: str
    conversation_id: str
    generation: int
    reply_target: ReplyTarget | None = None


class CompactionRequested(EventBase):
    type: Literal["compaction.requested"] = "compaction.requested"
    compaction_id: str
    user_id: str
    conversation_id: str
    generation: int


class CompactionSnapshotReady(EventBase):
    type: Literal["compaction.snapshot.ready"] = "compaction.snapshot.ready"
    compaction_id: str
    user_id: str
    conversation_id: str
    generation: int
    compacted_sequences: list[int]
    tail_sequences: list[int]
    snapshot_max_sequence: int
    archive_path: str


class CompactionSummaryReady(EventBase):
    type: Literal["compaction.summary.ready"] = "compaction.summary.ready"
    compaction_id: str
    user_id: str
    conversation_id: str
    generation: int
    compacted_sequences: list[int]
    tail_sequences: list[int]
    snapshot_max_sequence: int
    archive_path: str
    summary: str


class CompactionCommitted(EventBase):
    type: Literal["compaction.committed"] = "compaction.committed"
    compaction_id: str
    user_id: str
    conversation_id: str
    generation: int
    archive_path: str
    compacted_sequences: list[int]


class CompactionConflicted(EventBase):
    type: Literal["compaction.conflicted"] = "compaction.conflicted"
    compaction_id: str
    user_id: str
    conversation_id: str
    generation: int
    reason: Literal["cas_lost"]


class CompactionSkipped(EventBase):
    type: Literal["compaction.skipped"] = "compaction.skipped"
    compaction_id: str
    user_id: str
    conversation_id: str
    generation: int
    reason: Literal["no_boundary"]


class ToolCallRequested(EventBase):
    type: Literal["tool.call.requested"] = "tool.call.requested"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    reply_target: ReplyTarget | None = None

    @field_validator("input", mode="before")
    @classmethod
    def parse_input_for_tool(cls, value: Any, info: ValidationInfo) -> Any:
        tool_name = info.data.get("tool_name")
        if tool_name is None:
            return value
        return parse_stored_tool_input(tool_name, value)


class ToolCallError(EventBase):
    type: Literal["tool.call.error"] = "tool.call.error"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    error: str
    reply_target: ReplyTarget | None = None

    @field_validator("input", mode="before")
    @classmethod
    def parse_input_for_tool(cls, value: Any, info: ValidationInfo) -> Any:
        tool_name = info.data.get("tool_name")
        if tool_name is None:
            return value
        return parse_stored_tool_input(tool_name, value)


class ToolCallCompleted(EventBase):
    type: Literal["tool.call.completed"] = "tool.call.completed"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    result: RuntimeToolResult
    attachments: list[ContentRef] = Field(default_factory=list[ContentRef])
    reply_target: ReplyTarget | None = None

    @field_validator("input", mode="before")
    @classmethod
    def parse_input_for_tool(cls, value: Any, info: ValidationInfo) -> Any:
        tool_name = info.data.get("tool_name")
        if tool_name is None:
            return value
        return parse_stored_tool_input(tool_name, value)


class AssistantTextProduced(EventBase):
    type: Literal["assistant.text.produced"] = "assistant.text.produced"
    user_id: str
    conversation_id: str
    generation: int
    text: str
    reply_target: ReplyTarget | None = None


class ScheduledMessageDue(EventBase):
    type: Literal["scheduled.message.due"] = "scheduled.message.due"
    schedule_id: str
    user_id: str
    conversation_id: str
    text: str
    reply_target: ReplyTarget | None = None


class SubAgentRequested(EventBase):
    type: Literal["subagent.requested"] = "subagent.requested"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    parent_call_id: str
    name: str
    prompt: str
    timeout_seconds: float


class SubAgentStarted(EventBase):
    type: Literal["subagent.started"] = "subagent.started"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    parent_call_id: str
    name: str
    prompt: str = ""
    timeout_seconds: float = 0.0


class SubAgentTimedOut(EventBase):
    type: Literal["subagent.timed_out"] = "subagent.timed_out"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str


class SubAgentCompleted(EventBase):
    type: Literal["subagent.completed"] = "subagent.completed"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    result: str


class SubAgentFailed(EventBase):
    type: Literal["subagent.failed"] = "subagent.failed"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    error: str


class SubAgentCancelled(EventBase):
    type: Literal["subagent.cancelled"] = "subagent.cancelled"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str


class ImageJobRequested(EventBase):
    type: Literal["image.job.requested"] = "image.job.requested"
    job_id: str
    user_id: str
    conversation_id: str
    parent_call_id: str
    prompt: str
    output_path: str
    aspect_ratio: str


class ImageJobStarted(EventBase):
    type: Literal["image.job.started"] = "image.job.started"
    job_id: str
    user_id: str
    conversation_id: str
    parent_call_id: str
    prompt: str
    output_path: str
    aspect_ratio: str


class ImageJobCompleted(EventBase):
    type: Literal["image.job.completed"] = "image.job.completed"
    job_id: str
    user_id: str
    conversation_id: str
    output_path: str
    mime_type: str
    size_bytes: int


class ImageJobFailed(EventBase):
    type: Literal["image.job.failed"] = "image.job.failed"
    job_id: str
    user_id: str
    conversation_id: str
    error: str


class MemoryReviewCompleted(EventBase):
    type: Literal["memory.review.completed"] = "memory.review.completed"
    user_id: str
    conversation_id: str
    actions: list[str] = Field(default_factory=list[str])
    note: str | None = None


class SessionLogAppendFailed(EventBase):
    type: Literal["session.log.append.failed"] = "session.log.append.failed"
    user_id: str
    conversation_id: str
    role: Literal["user", "assistant", "tool"]
    error: str


AgentEvent = Annotated[
    TelegramTextReceived
    | CliTextReceived
    | UserTextReceived
    | AgentTurnRequested
    | AgentTurnSuperseded
    | AgentGenerationStarted
    | CompactionRequested
    | CompactionSnapshotReady
    | CompactionSummaryReady
    | CompactionCommitted
    | CompactionConflicted
    | CompactionSkipped
    | ToolCallRequested
    | ToolCallError
    | ToolCallCompleted
    | AssistantTextProduced
    | ScheduledMessageDue
    | SubAgentRequested
    | SubAgentStarted
    | SubAgentTimedOut
    | SubAgentCompleted
    | SubAgentFailed
    | SubAgentCancelled
    | ImageJobRequested
    | ImageJobStarted
    | ImageJobCompleted
    | ImageJobFailed
    | MemoryReviewCompleted
    | SessionLogAppendFailed,
    Field(discriminator="type"),
]
