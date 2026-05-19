from dataclasses import dataclass

from harness_agent.bus import EventBus
from harness_agent.compaction import ContextCompactor
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    AgentTurnSuperseded,
    AssistantTextProduced,
    ToolCallRequested,
)
from harness_agent.llm import (
    AssistantText,
    AssistantToolCallMessage,
    LlmClient,
    LlmMessage,
    LlmRequest,
    LlmToolCall,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.mcp import McpManager
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.tool_executor import ToolCallResultWaiter
from harness_agent.tools import ToolRegistry
from harness_agent.turns import ConversationTurnCoordinator


@dataclass(frozen=True)
class BuildLlmRequestCommand:
    event: AgentTurnRequested


@dataclass(frozen=True)
class PublishAssistantTextCommand:
    event: AgentTurnRequested
    response: AssistantText


@dataclass(frozen=True)
class RunToolCallCommand:
    event: AgentTurnRequested
    response: LlmToolCall


class TurnSuperseded(Exception):
    pass


class TurnSupersessionGuard:
    def __init__(
        self,
        *,
        bus: EventBus,
        turn_coordinator: ConversationTurnCoordinator,
    ) -> None:
        self._bus = bus
        self._turn_coordinator = turn_coordinator

    async def ensure_current(self, event: AgentTurnRequested) -> None:
        current_generation = await self._turn_coordinator.current_generation(
            event.conversation_id
        )
        if current_generation == event.generation:
            return
        await self._bus.publish(
            AgentTurnSuperseded(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=event.generation,
                superseded_by=current_generation,
            )
        )
        raise TurnSuperseded


class TurnRequestBuilder:
    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
        projection: SQLiteConversationProjection,
        mcp_manager: McpManager | None = None,
        compactor: ContextCompactor | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._tool_registry = tool_registry
        self._projection = projection
        self._mcp_manager = mcp_manager
        self._compactor = compactor

    async def build(self, command: BuildLlmRequestCommand) -> LlmRequest:
        event = command.event
        context = await self._context_builder.build(event.user_id)
        tools = list(self._tool_registry.list_for_model())
        if self._mcp_manager is not None:
            tools.extend(await self._mcp_manager.list_tool_specs(event.user_id))
        request = LlmRequest(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            system=context.system,
            messages=await self._projection.list_llm_messages(event.conversation_id),
            tools=tools,
        )
        return request

    async def compact_if_needed(
        self,
        *,
        request: LlmRequest,
        event: AgentTurnRequested,
    ) -> LlmRequest:
        if self._compactor is None:
            return request
        return await self._compactor.compact_if_needed(request=request, event=event)


class TurnEventPublisher:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def publish_assistant_text(self, command: PublishAssistantTextCommand) -> None:
        await self._bus.publish(
            AssistantTextProduced(
                user_id=command.event.user_id,
                conversation_id=command.event.conversation_id,
                generation=command.event.generation,
                text=command.response.text,
                reply_target=command.event.reply_target,
            )
        )

    async def publish_tool_call(self, event: ToolCallRequested) -> None:
        await self._bus.publish(event)


class ToolCallCommandRunner:
    def __init__(
        self,
        *,
        publisher: TurnEventPublisher,
        tool_results: ToolCallResultWaiter,
    ) -> None:
        self._publisher = publisher
        self._tool_results = tool_results

    async def run(self, command: RunToolCallCommand) -> list[LlmMessage]:
        response = command.response
        requested = ToolCallRequested(
            user_id=command.event.user_id,
            conversation_id=command.event.conversation_id,
            generation=command.event.generation,
            call_id=response.call_id,
            tool_name=response.name,
            input=response.input,
            reply_target=command.event.reply_target,
        )
        self._tool_results.expect(requested)
        await self._publisher.publish_tool_call(requested)
        completed = await self._tool_results.wait(requested)
        messages: list[LlmMessage] = [
            AssistantToolCallMessage(
                call_id=response.call_id,
                name=response.name,
                arguments=response.input.model_dump(mode="json"),
            ),
            ToolResultMessage(
                call_id=response.call_id,
                name=response.name,
                content=completed.result.render_for_llm(response.name),
            ),
        ]
        for attachment in completed.attachments:
            messages.append(
                UserMessage(
                    text=f"Opened image file {attachment.workspace_path}",
                    attachments=[attachment],
                )
            )
        return messages


class AgentTurnRunner:
    def __init__(
        self,
        *,
        llm: LlmClient,
        request_builder: TurnRequestBuilder,
        publisher: TurnEventPublisher,
        tool_runner: ToolCallCommandRunner,
        guard: TurnSupersessionGuard,
    ) -> None:
        self._llm = llm
        self._request_builder = request_builder
        self._publisher = publisher
        self._tool_runner = tool_runner
        self._guard = guard

    async def run(self, event: AgentTurnRequested) -> None:
        await self._guard.ensure_current(event)
        request = await self._request_builder.build(BuildLlmRequestCommand(event=event))
        await self._guard.ensure_current(event)
        while True:
            request = await self._request_builder.compact_if_needed(
                request=request,
                event=event,
            )
            await self._guard.ensure_current(event)
            response = await self._llm.respond(request)
            await self._guard.ensure_current(event)
            if response.kind == "assistant_text":
                await self._publisher.publish_assistant_text(
                    PublishAssistantTextCommand(event=event, response=response)
                )
                return
            if response.kind == "tool_call":
                request.messages.extend(
                    await self._tool_runner.run(
                        RunToolCallCommand(event=event, response=response)
                    )
                )
                await self._guard.ensure_current(event)
                continue
            raise RuntimeError(f"unsupported LLM response: {response}")
