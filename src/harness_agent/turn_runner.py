from harness_agent.bus import EventBus
from harness_agent.compaction import ContextCompactor
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentGenerationStarted,
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
    LlmToolCall,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.mcp import McpManager
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.subagents import SubAgentLookup, SubAgentRecord
from harness_agent.tool_executor import ToolCallResultWaiter
from harness_agent.tools import ToolRegistry, ToolSpec
from harness_agent.turns import ConversationTurnCoordinator


SUB_AGENT_SYSTEM_PREFIX = (
    "You are a sub-agent named '{name}'. A parent agent delegated a single "
    "task to you and will read only your final assistant message as the "
    "result. Complete the task, then place the result in your last message. "
    "You cannot spawn further sub-agents."
)
SUB_AGENT_TASK_PREFIX = "Task delegated by the parent agent:"


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
        sub_agent_lookup: SubAgentLookup | None = None,
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
        self._sub_agent_lookup = sub_agent_lookup

    async def run(self, event: AgentTurnRequested) -> None:
        async with self._turn_coordinator.run_slot(event.conversation_id):
            if await self._stop_if_superseded(event):
                return
            context = await self._context_builder.build(event.user_id)
            if await self._stop_if_superseded(event):
                return
            messages = await self._projection.list_llm_messages(event.conversation_id)
            sub_agent_record = await self._sub_agent_for_conversation(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
            )
            tools = await self._tools_for_user(event.user_id, sub_agent_record)
            system_prompt = _system_prompt_for_turn(context.system, sub_agent_record)

            while True:
                messages = await self._prepare_messages(event, system_prompt, messages, tools)
                if messages is None:
                    return
                await self._bus.publish(
                    AgentGenerationStarted(
                        user_id=event.user_id,
                        conversation_id=event.conversation_id,
                        generation=event.generation,
                        reply_target=event.reply_target,
                    )
                )
                response = await self._llm.respond(
                    LlmRequest(
                        user_id=event.user_id,
                        conversation_id=event.conversation_id,
                        generation=event.generation,
                        system=system_prompt,
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
        tools: list[ToolSpec],
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

    async def _run_tool(
        self,
        event: AgentTurnRequested,
        response: LlmToolCall,
        messages: list[LlmMessage],
    ) -> list[LlmMessage]:
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

    async def _tools_for_user(
        self,
        user_id: str,
        sub_agent_record: SubAgentRecord | None,
    ) -> list[ToolSpec]:
        tools: list[ToolSpec] = list(self._tool_registry.list_for_model())
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
        if self._sub_agent_lookup is None:
            return None
        return await self._sub_agent_lookup.get_by_child_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
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


def _system_prompt_for_turn(
    base_system: str,
    sub_agent_record: SubAgentRecord | None,
) -> str:
    if sub_agent_record is None:
        return base_system
    header = SUB_AGENT_SYSTEM_PREFIX.format(name=sub_agent_record.name)
    task_block = f"{SUB_AGENT_TASK_PREFIX}\n{sub_agent_record.prompt}"
    return "\n\n".join(block for block in (header, task_block, base_system) if block)
