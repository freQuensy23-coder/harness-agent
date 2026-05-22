import base64
from uuid import uuid4

from harness_agent.bus import EventBus
from harness_agent.compaction import CompactionConfig
from harness_agent.content import content_ref_from_bytes
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentGenerationStarted,
    AgentTurnRequested,
    AgentTurnSuperseded,
    AssistantTextProduced,
    CliReplyTarget,
    CliTextReceived,
    CompactionRequested,
    EventBase,
    TelegramReplyTarget,
    TelegramTextReceived,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.identity import StaticIdentityResolver
from harness_agent.llm import (
    LlmRequest,
    LlmClient,
    estimate_request_tokens,
)
from harness_agent.mcp import McpManager
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import UserRuntime
from harness_agent.subagents import SubAgentLookup, SubAgentRecord
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import (
    ToolRegistry,
    ToolSpec,
)


EventBatch = tuple[EventBase, ...]


class IdentityHandler:
    def __init__(self, resolver: StaticIdentityResolver) -> None:
        self._resolver = resolver

    async def handle_telegram_text(self, event: TelegramTextReceived) -> EventBatch:
        identity = await self._resolver.resolve_telegram(
            telegram_user_id=event.telegram_user_id,
            telegram_chat_id=event.telegram_chat_id,
        )
        return (
            UserTextReceived(
                user_id=identity.user_id,
                conversation_id=identity.conversation_id,
                source="telegram",
                text=event.text,
                attachments=event.attachments,
                reply_target=TelegramReplyTarget(chat_id=event.telegram_chat_id),
            ),
        )

    async def handle_cli_text(self, event: CliTextReceived) -> EventBatch:
        identity = await self._resolver.resolve_cli(
            cli_user_id=event.cli_user_id,
            conversation_id=event.conversation_id,
        )
        return (
            UserTextReceived(
                user_id=identity.user_id,
                conversation_id=identity.conversation_id,
                source="cli",
                text=event.text,
                reply_target=CliReplyTarget(request_id=event.request_id),
            ),
        )


class ConversationProjector:
    def __init__(
        self,
        projection: SQLiteConversationProjection,
        *,
        turn_coordinator: ConversationTurnCoordinator | None = None,
    ) -> None:
        self._projection = projection
        self._turn_coordinator = turn_coordinator

    async def handle_user_text(self, event: UserTextReceived) -> EventBatch:
        await self._projection.append_user_message(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            text=event.text,
            attachments=[
                content_ref_from_bytes(
                    kind=attachment.kind,
                    file_name=attachment.file_name,
                    mime_type=attachment.mime_type,
                    workspace_path=attachment.workspace_path,
                    content=base64.b64decode(attachment.content_base64),
                )
                for attachment in event.attachments
            ],
        )
        return ()

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        if not await self._can_project_generation(event.conversation_id, event.generation):
            return ()
        await self._projection.append_assistant_message(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            text=event.text,
        )
        return ()

    async def handle_tool_call_completed(self, event: ToolCallCompleted) -> EventBatch:
        if not await self._can_project_generation(event.conversation_id, event.generation):
            return ()
        await self._projection.append_tool_exchange(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            call_id=event.call_id,
            tool_name=event.tool_name,
            input=event.input,
            result=event.result,
            attachments=event.attachments,
        )
        return ()

    async def _can_project_generation(self, conversation_id: str, generation: int) -> bool:
        if self._turn_coordinator is None:
            return True
        return await self._turn_coordinator.is_current(conversation_id, generation)


class ContentIngestionHandler:
    def __init__(self, *, runtime: UserRuntime) -> None:
        self._runtime = runtime

    async def handle_user_text(self, event: UserTextReceived) -> EventBatch:
        for attachment in event.attachments:
            await self._runtime.write_content_file(
                event.user_id,
                attachment.workspace_path,
                base64.b64decode(attachment.content_base64),
            )
        return ()


class AgentTurnHandler:
    def __init__(
        self,
        *,
        bus: EventBus,
        context_builder: ContextBuilder,
        llm: LlmClient,
        tool_registry: ToolRegistry,
        projection: SQLiteConversationProjection,
        sub_agent_lookup: SubAgentLookup,
        mcp_manager: McpManager | None = None,
        turn_coordinator: ConversationTurnCoordinator | None = None,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        self._bus = bus
        self._context_builder = context_builder
        self._llm = llm
        self._tool_registry = tool_registry
        self._projection = projection
        self._mcp_manager = mcp_manager
        if turn_coordinator is None:
            turn_coordinator = ConversationTurnCoordinator()
        self._turn_coordinator = turn_coordinator
        self._sub_agent_lookup = sub_agent_lookup
        self._compaction_config = compaction_config

    async def handle_user_text(self, event: UserTextReceived) -> EventBatch:
        generation = await self._turn_coordinator.request_generation(event.conversation_id)
        return (
            AgentTurnRequested(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=generation,
                input_event_id=event.id,
                reply_target=event.reply_target,
            ),
        )

    async def handle_agent_turn(self, event: AgentTurnRequested) -> EventBatch:
        return await self._run_step(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            reply_target=event.reply_target,
        )

    async def handle_tool_call_completed(self, event: ToolCallCompleted) -> EventBatch:
        return await self._run_step(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            reply_target=event.reply_target,
        )

    async def _run_step(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        reply_target: TelegramReplyTarget | CliReplyTarget | None,
    ) -> EventBatch:
        if await self._stop_if_superseded(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
        ):
            return ()
        context = await self._context_builder.build(user_id)
        if await self._stop_if_superseded(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
        ):
            return ()
        messages = await self._projection.list_llm_messages(conversation_id)
        sub_agent_record = await self._sub_agent_for_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        tools = await self._tools_for_user(user_id, sub_agent_record)
        system_prompt = _system_prompt_for_turn(context.system, sub_agent_record)

        request = LlmRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )
        if (
            self._compaction_config is not None
            and estimate_request_tokens(request)
            >= self._compaction_config.threshold
        ):
            await self._bus.publish(
                CompactionRequested(
                    compaction_id=uuid4().hex,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    generation=generation,
                )
            )
            messages = await self._projection.list_llm_messages(conversation_id)
            request = LlmRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=generation,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

        await self._bus.publish(
            AgentGenerationStarted(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=generation,
                reply_target=reply_target,
            )
        )
        response = await self._llm.respond(request)
        if await self._stop_if_superseded(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
        ):
            return ()
        if response.kind == "assistant_text":
            return (
                AssistantTextProduced(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    generation=generation,
                    text=response.text,
                    reply_target=reply_target,
                ),
            )
        if response.kind == "tool_call":
            return (
                ToolCallRequested(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    generation=generation,
                    call_id=response.call_id,
                    tool_name=response.name,
                    input=response.input,
                    reply_target=reply_target,
                ),
            )
        raise ValueError(f"unsupported LLM response kind: {response.kind}")

    async def _stop_if_superseded(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
    ) -> bool:
        current_generation = await self._turn_coordinator.current_generation(
            conversation_id
        )
        if current_generation == generation:
            return False
        await self._bus.publish(
            AgentTurnSuperseded(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=generation,
                superseded_by=current_generation,
            )
        )
        return True

    async def _tools_for_user(
        self,
        user_id: str,
        sub_agent_record: SubAgentRecord | None,
    ) -> list[ToolSpec]:
        tools = list(self._tool_registry.list_for_model())
        if self._mcp_manager is not None:
            tools.extend(await self._mcp_manager.list_tool_specs(user_id))
        if sub_agent_record is not None:
            tools = [tool for tool in tools if not tool.name.startswith("agent.")]
        return tools

    async def _sub_agent_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> SubAgentRecord | None:
        return await self._sub_agent_lookup.get_by_child_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )


class TelegramReplyHandler:
    def __init__(self, replies: list[tuple[int, str]]) -> None:
        self._replies = replies

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        target = event.reply_target
        if not isinstance(target, TelegramReplyTarget):
            return ()
        self._replies.append((target.chat_id, event.text))
        return ()


SUB_AGENT_SYSTEM_PREFIX = (
    "You are a sub-agent named '{name}'. A parent agent delegated a single "
    "task to you and will read only your final assistant message as the "
    "result. Complete the task, then place the result in your last message. "
    "You cannot spawn further sub-agents."
)
SUB_AGENT_TASK_PREFIX = "Task delegated by the parent agent:"


def _system_prompt_for_turn(
    base_system: str,
    sub_agent_record: SubAgentRecord | None,
) -> str:
    if sub_agent_record is None:
        return base_system
    header = SUB_AGENT_SYSTEM_PREFIX.format(name=sub_agent_record.name)
    task_block = f"{SUB_AGENT_TASK_PREFIX}\n{sub_agent_record.prompt}"
    return "\n\n".join(block for block in (header, task_block, base_system) if block)
