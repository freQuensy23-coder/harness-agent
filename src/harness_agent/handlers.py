import base64

from harness_agent.bus import EventBus
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    AgentTurnSuperseded,
    AssistantTextProduced,
    CliReplyTarget,
    CliTextReceived,
    EventBase,
    TelegramReplyTarget,
    TelegramTextReceived,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.identity import StaticIdentityResolver
from harness_agent.llm import (
    AssistantToolCallMessage,
    LlmRequest,
    LlmClient,
    ToolResultMessage,
    UserMessage,
    UserMessageAttachment,
)
from harness_agent.mcp import McpManager
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import UserRuntime
from harness_agent.tool_executor import ToolCallResultWaiter
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import (
    ToolRegistry,
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
                UserMessageAttachment(
                    kind=attachment.kind,
                    file_name=attachment.file_name,
                    mime_type=attachment.mime_type,
                    size_bytes=attachment.size_bytes,
                    workspace_path=attachment.workspace_path,
                    content_base64=(
                        attachment.content_base64 if attachment.kind == "image" else None
                    ),
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
        if tool_results is None:
            tool_results = ToolCallResultWaiter()
        self._tool_results = tool_results

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
        await self._run_turn(event)
        return ()

    async def _run_turn(self, event: AgentTurnRequested) -> None:
        async with self._turn_coordinator.run_slot(event.conversation_id):
            if await self._stop_if_superseded(event):
                return
            context = await self._context_builder.build(event.user_id)
            if await self._stop_if_superseded(event):
                return
            messages = await self._projection.list_llm_messages(event.conversation_id)
            tools = await self._tools_for_user(event.user_id)

            while True:
                if await self._stop_if_superseded(event):
                    return
                response = await self._llm.respond(
                    LlmRequest(
                        user_id=event.user_id,
                        conversation_id=event.conversation_id,
                        generation=event.generation,
                        system=context.system,
                        messages=messages,
                        tools=tools,
                    )
                )
                if await self._stop_if_superseded(event):
                    return
                if response.kind == "assistant_text":
                    await self._bus.publish(
                        AssistantTextProduced(
                            user_id=event.user_id,
                            conversation_id=event.conversation_id,
                            generation=event.generation,
                            text=response.text,
                            reply_target=event.reply_target,
                        )
                    )
                    return
                if response.kind == "tool_call":
                    requested = ToolCallRequested(
                        user_id=event.user_id,
                        conversation_id=event.conversation_id,
                        generation=event.generation,
                        call_id=response.call_id,
                        tool_name=response.name,
                        input=response.input,
                        reply_target=event.reply_target,
                    )
                    self._tool_results.expect(requested)
                    await self._bus.publish(requested)
                    completed = await self._tool_results.wait(requested)
                    if await self._stop_if_superseded(event):
                        return
                    result = completed.result
                    messages.extend(
                        [
                            AssistantToolCallMessage(
                                call_id=response.call_id,
                                name=response.name,
                                arguments=response.input.model_dump(mode="json"),
                            ),
                            ToolResultMessage(
                                call_id=response.call_id,
                                name=response.name,
                                content=result.render_for_llm(response.name),
                            ),
                        ]
                    )
                    for attachment in result.attachments:
                        messages.append(
                            UserMessage(
                                text=f"Opened image file {attachment.workspace_path}",
                                attachments=[attachment],
                            )
                        )

    async def _stop_if_superseded(self, event: AgentTurnRequested) -> bool:
        current_generation = await self._turn_coordinator.current_generation(
            event.conversation_id
        )
        if current_generation == event.generation:
            return False
        await self._bus.publish(
            AgentTurnSuperseded(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=event.generation,
                superseded_by=current_generation,
            )
        )
        return True

    async def _tools_for_user(self, user_id: str):
        tools = list(self._tool_registry.list_for_model())
        if self._mcp_manager is not None:
            tools.extend(await self._mcp_manager.list_tool_specs(user_id))
        return tools


class TelegramReplyHandler:
    def __init__(self, replies: list[tuple[int, str]]) -> None:
        self._replies = replies

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        if event.reply_target is None:
            return ()
        self._replies.append((event.reply_target.chat_id, event.text))
        return ()
