from harness_agent.bus import EventBus
from harness_agent.compaction import ContextCompactor
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    AgentTurnSuperseded,
    AssistantTextProduced,
    ContextCompacting,
    ToolCallRequested,
)
from harness_agent.llm import (
    AssistantToolCallMessage,
    LlmClient,
    LlmMessage,
    LlmRequest,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.mcp import McpManager
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.tool_executor import ToolCallResultWaiter
from harness_agent.tools import ToolRegistry
from harness_agent.turns import ConversationTurnCoordinator


class AgentTurnRunner:
    def __init__(
        self,
        *,
        bus: EventBus,
        context_builder: ContextBuilder,
        llm: LlmClient,
        tool_registry: ToolRegistry,
        projection: SQLiteConversationProjection,
        turn_coordinator: ConversationTurnCoordinator,
        tool_results: ToolCallResultWaiter,
        mcp_manager: McpManager | None = None,
        compactor: ContextCompactor | None = None,
    ) -> None:
        self._bus = bus
        self._context_builder = context_builder
        self._llm = llm
        self._tool_registry = tool_registry
        self._projection = projection
        self._mcp_manager = mcp_manager
        self._turn_coordinator = turn_coordinator
        self._tool_results = tool_results
        self._compactor = compactor

    async def run(self, event: AgentTurnRequested) -> None:
        async with self._turn_coordinator.run_slot(event.conversation_id):
            if await self._stop_if_superseded(event):
                return
            context = await self._context_builder.build(event.user_id)
            if await self._stop_if_superseded(event):
                return
            messages = await self._projection.list_llm_messages(event.conversation_id)
            tools = await self._tools_for_user(event.user_id)

            while True:
                messages = await self._prepare_messages(event, context.system, messages, tools)
                if messages is None:
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
                    messages = await self._run_tool(event, response, messages)

    async def _prepare_messages(
        self,
        event: AgentTurnRequested,
        system: str,
        messages: list[LlmMessage],
        tools,
    ) -> list[LlmMessage] | None:
        if await self._stop_if_superseded(event):
            return None
        if self._compactor is None:
            return messages

        request = LlmRequest(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            system=system,
            messages=messages,
            tools=tools,
        )
        should_compact, token_estimate = self._compactor.should_compact(request)
        if not should_compact:
            return messages

        snapshot = await self._compactor.create_snapshot(
            conversation_id=event.conversation_id,
        )
        if snapshot is None:
            return messages
        if await self._stop_if_superseded(event):
            return None

        await self._bus.publish(
            ContextCompacting(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=event.generation,
                token_estimate=token_estimate,
                threshold=self._compactor.threshold,
                keep_last_messages=self._compactor.keep_last_messages,
            )
        )
        if await self._stop_if_superseded(event):
            return None
        draft = await self._compactor.create_draft(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            snapshot=snapshot,
        )
        if await self._stop_if_superseded(event):
            return None
        compacted = await self._compactor.commit_draft(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            draft=draft,
        )
        if await self._stop_if_superseded(event):
            return None
        return compacted

    async def _run_tool(self, event, response, messages: list[LlmMessage]) -> list[LlmMessage]:
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
            return messages
        result = completed.result
        next_messages = [
            *messages,
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
        for attachment in completed.attachments:
            next_messages.append(
                UserMessage(
                    text=f"Opened image file {attachment.workspace_path}",
                    attachments=[attachment],
                )
            )
        return next_messages

    async def _tools_for_user(self, user_id: str):
        tools = list(self._tool_registry.list_for_model())
        if self._mcp_manager is not None:
            tools.extend(await self._mcp_manager.list_tool_specs(user_id))
        return tools

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
