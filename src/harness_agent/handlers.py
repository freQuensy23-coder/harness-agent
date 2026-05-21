import base64

from harness_agent.bus import EventBus
from harness_agent.compaction import ContextCompactor
from harness_agent.content import content_ref_from_bytes
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    AssistantTextProduced,
    CliReplyTarget,
    CliTextReceived,
    EventBase,
    TelegramReplyTarget,
    TelegramTextReceived,
    ToolCallCompleted,
    UserTextReceived,
)
from harness_agent.identity import StaticIdentityResolver
from harness_agent.llm import LlmClient
from harness_agent.mcp import McpManager
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import UserRuntime
from harness_agent.subagents import SubAgentLookup
from harness_agent.tool_executor import ToolCallResultWaiter
from harness_agent.turn_runner import AgentTurnRunner
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import ToolRegistry


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
        mcp_manager: McpManager | None = None,
        turn_coordinator: ConversationTurnCoordinator | None = None,
        tool_results: ToolCallResultWaiter | None = None,
        compactor: ContextCompactor | None = None,
        sub_agent_lookup: SubAgentLookup | None = None,
    ) -> None:
        if turn_coordinator is None:
            turn_coordinator = ConversationTurnCoordinator()
        self._turn_coordinator = turn_coordinator
        if tool_results is None:
            tool_results = ToolCallResultWaiter()
        self._runner = AgentTurnRunner(
            bus=bus,
            context_builder=context_builder,
            llm=llm,
            tool_registry=tool_registry,
            projection=projection,
            mcp_manager=mcp_manager,
            turn_coordinator=turn_coordinator,
            tool_results=tool_results,
            compactor=compactor,
            sub_agent_lookup=sub_agent_lookup,
        )

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
        await self._runner.run(event)
        return ()


class TelegramReplyHandler:
    def __init__(self, replies: list[tuple[int, str]]) -> None:
        self._replies = replies

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        target = event.reply_target
        if not isinstance(target, TelegramReplyTarget):
            return ()
        self._replies.append((target.chat_id, event.text))
        return ()
