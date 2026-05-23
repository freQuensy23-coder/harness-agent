"""Sub-agent tool handlers for `agent.*` calls."""

from harness_agent.events import ToolCallRequested
from harness_agent.runtime import RuntimeToolResult
from harness_agent.subagents import (
    SubAgentService,
    render_sub_agent_record,
    render_sub_agent_records,
)
from harness_agent.tools import (
    AgentCancelInput,
    AgentListInput,
    AgentResultInput,
    AgentRunInput,
    AgentSpawnInput,
)


class SubAgentToolHandlers:
    """Handlers backed by `SubAgentService` for the `agent.*` tool family."""

    def __init__(self, *, sub_agents: SubAgentService) -> None:
        self._sub_agents = sub_agents

    async def run(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentRunInput.model_validate(event.input)
        record = await self._sub_agents.run(
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
            parent_call_id=event.call_id,
            input=input,
        )
        exit_code = 0 if record.status == "completed" else 1
        return RuntimeToolResult(
            stdout=render_sub_agent_record(record),
            stderr="" if record.error is None else record.error,
            exit_code=exit_code,
        )

    async def spawn(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentSpawnInput.model_validate(event.input)
        record = await self._sub_agents.spawn(
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
            parent_call_id=event.call_id,
            input=input,
        )
        return RuntimeToolResult(stdout=render_sub_agent_record(record))

    async def result(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentResultInput.model_validate(event.input)
        record = await self._sub_agents.result(
            agent_id=input.agent_id,
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
        )
        if record is None:
            return RuntimeToolResult(stderr=f"Unknown sub-agent: {input.agent_id}\n", exit_code=1)
        return RuntimeToolResult(stdout=render_sub_agent_record(record))

    async def list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentListInput.model_validate(event.input)
        records = await self._sub_agents.list_for_parent(
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
            include_completed=input.include_completed,
        )
        return RuntimeToolResult(stdout=render_sub_agent_records(records))

    async def cancel(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentCancelInput.model_validate(event.input)
        record = await self._sub_agents.cancel(
            agent_id=input.agent_id,
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
        )
        if record is None:
            return RuntimeToolResult(stderr=f"Unknown sub-agent: {input.agent_id}\n", exit_code=1)
        return RuntimeToolResult(stdout=render_sub_agent_record(record))
