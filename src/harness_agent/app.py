from pathlib import Path
from asyncio import Queue

from loguru import logger

from harness_agent.adapters.cli import event_from_cli_send
from harness_agent.adapters.telegram import AiogramTelegramAdapter
from harness_agent.bus import EventBus
from harness_agent.config import HarnessConfig
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentEvent,
    AgentTurnRequested,
    AssistantTextProduced,
    CliTextReceived,
    ScheduledMessageDue,
    SubAgentCancelled,
    SubAgentCompleted,
    SubAgentFailed,
    SubAgentRequested,
    SubAgentStarted,
    SubAgentTimedOut,
    TelegramTextReceived,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import (
    AgentTurnHandler,
    ContentIngestionHandler,
    ConversationProjector,
    EventBatch,
    IdentityHandler,
)
from harness_agent.identity import StaticIdentityResolver
from harness_agent.llm import OpenAIResponsesClient
from harness_agent.llm_audit import AuditedLlmClient, SQLiteLlmAuditStore
from harness_agent.mcp import McpManager
from harness_agent.memory_review import MemoryReviewService
from harness_agent.memory_service import MemoryService
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import DockerUserRuntime, SQLiteSpawnedProcessStore
from harness_agent.session_log import SessionLogWriter
from harness_agent.session_search_service import SessionSearchService
from harness_agent.scheduler import (
    SchedulerDueHandler,
    SchedulerPump,
    SchedulerService,
    SQLiteScheduleStore,
)
from harness_agent.store import SQLiteEventStore
from harness_agent.subagents import SQLiteSubAgentStore, SubAgentService
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import default_tool_registry
from harness_agent.web_fetch import HttpxWebFetcher


class HarnessApp:
    def __init__(self, *, config: HarnessConfig) -> None:
        self._config = config
        db_path = config.database.path
        events_path = _derived_db_path(db_path, "events")
        llm_path = _derived_db_path(db_path, "llm")
        messages_path = _derived_db_path(db_path, "messages")
        runtime_path = _derived_db_path(db_path, "runtime")
        schedules_path = _derived_db_path(db_path, "schedules")
        sub_agents_path = _derived_db_path(db_path, "subagents")
        tasks_path = _derived_db_path(db_path, "tasks")

        self.event_store = SQLiteEventStore(events_path)
        self.llm_audit_store = SQLiteLlmAuditStore(llm_path)
        self.projection = SQLiteConversationProjection(messages_path)
        self.schedule_store = SQLiteScheduleStore(schedules_path)
        self.sub_agent_store = SQLiteSubAgentStore(sub_agents_path)
        self.task_store = SQLiteTaskStore(tasks_path)
        self.bus = EventBus(self.event_store)
        self.turn_coordinator = ConversationTurnCoordinator()
        self.tool_results = ToolCallResultWaiter()
        self.runtime = DockerUserRuntime(
            image=config.runtime.docker.image,
            container_prefix=config.runtime.docker.container_prefix,
            network=config.runtime.docker.network,
            memory=config.runtime.docker.memory,
            cpus=config.runtime.docker.cpus,
            ensure_container=True,
            spawned_process_store=SQLiteSpawnedProcessStore(runtime_path),
        )
        self.mcp_manager = McpManager(
            runtime=self.runtime,
            global_servers=config.mcp.servers,
        )
        self.llm = AuditedLlmClient(
            inner=OpenAIResponsesClient(
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
                model=config.llm.model,
            ),
            store=self.llm_audit_store,
        )
        self.sub_agents = SubAgentService(
            bus=self.bus,
            store=self.sub_agent_store,
        )
        self.telegram: AiogramTelegramAdapter | None = None
        self.scheduler_service: SchedulerService | None = None
        self._cli_replies: dict[str, Queue[str]] = {}
        self._wire()

    async def publish(self, event: AgentEvent) -> None:
        await self.bus.publish(event)

    async def send_cli(
        self,
        *,
        text: str,
        user_id: str,
        conversation_id: str | None,
    ) -> str:
        event = event_from_cli_send(
            text=text,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self._cli_replies[event.request_id] = Queue(maxsize=1)
        await self.bus.publish(event)
        reply = await self._cli_replies[event.request_id].get()
        del self._cli_replies[event.request_id]
        return reply

    async def run_telegram(self) -> None:
        if not self._config.telegram.enabled:
            raise RuntimeError("telegram.enabled is false")
        if self._config.telegram.bot_token is None:
            raise RuntimeError("telegram.bot_token is required when telegram is enabled")
        self.telegram = AiogramTelegramAdapter(
            token=self._config.telegram.bot_token,
            bus=self.bus,
        )
        self.telegram.register_outbound_handlers()
        self.scheduler_service = SchedulerService(
            pump=SchedulerPump(store=self.schedule_store, bus=self.bus),
            poll_seconds=self._config.scheduler.poll_seconds,
        )
        await self.scheduler_service.start()
        logger.info("Starting Telegram polling")
        try:
            await self.telegram.start_polling()
        finally:
            await self.scheduler_service.stop()

    def _wire(self) -> None:
        identity_handler = IdentityHandler(StaticIdentityResolver())
        content_ingestion_handler = ContentIngestionHandler(runtime=self.runtime)
        conversation_projector = ConversationProjector(
            self.projection,
            turn_coordinator=self.turn_coordinator,
        )
        agent_turn_handler = AgentTurnHandler(
            bus=self.bus,
            context_builder=ContextBuilder(runtime=self.runtime),
            llm=self.llm,
            tool_registry=default_tool_registry(),
            projection=self.projection,
            mcp_manager=self.mcp_manager,
            turn_coordinator=self.turn_coordinator,
            tool_results=self.tool_results,
            sub_agent_lookup=self.sub_agents,
        )
        self.memory_service = MemoryService(runtime=self.runtime)
        self.session_search_service = SessionSearchService(
            runtime=self.runtime,
            llm=self.llm,
        )
        tool_call_executor = ToolCallExecutor(
            runtime=self.runtime,
            task_store=self.task_store,
            schedule_store=self.schedule_store,
            web_fetcher=HttpxWebFetcher(llm=self.llm),
            mcp_manager=self.mcp_manager,
            sub_agents=self.sub_agents,
            memory_service=self.memory_service,
            session_search=self.session_search_service,
        )
        scheduler_due_handler = SchedulerDueHandler()
        session_log_writer = SessionLogWriter(
            runtime=self.runtime,
            turn_coordinator=self.turn_coordinator,
        )
        self.memory_review: MemoryReviewService | None = None
        if self._config.memory.enabled:
            self.memory_review = MemoryReviewService(
                bus=self.bus,
                llm=self.llm,
                tool_results=self.tool_results,
                projection=self.projection,
                tool_registry=default_tool_registry(),
                turn_coordinator=self.turn_coordinator,
                nudge_interval=self._config.memory.nudge_interval,
                max_iterations=self._config.memory.review_max_iterations,
            )
        self.bus.subscribe(TelegramTextReceived, identity_handler.handle_telegram_text)
        self.bus.subscribe(CliTextReceived, identity_handler.handle_cli_text)
        self.bus.subscribe(ScheduledMessageDue, scheduler_due_handler.handle_due)
        self.bus.subscribe(UserTextReceived, content_ingestion_handler.handle_user_text)
        self.bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
        self.bus.subscribe(UserTextReceived, session_log_writer.handle_user_text)
        self.bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
        self.bus.subscribe(AssistantTextProduced, session_log_writer.handle_assistant_text)
        self.bus.subscribe(ToolCallCompleted, session_log_writer.handle_tool_call_completed)
        self.bus.subscribe(AssistantTextProduced, self.sub_agents.handle_assistant_text)
        self.bus.subscribe(SubAgentRequested, self.sub_agents.handle_requested)
        self.bus.subscribe(SubAgentStarted, self.sub_agents.handle_started)
        self.bus.subscribe(SubAgentTimedOut, self.sub_agents.handle_timed_out)
        self.bus.subscribe(SubAgentCompleted, self.sub_agents.handle_completed)
        self.bus.subscribe(SubAgentFailed, self.sub_agents.handle_failed)
        self.bus.subscribe(SubAgentCancelled, self.sub_agents.handle_cancelled)
        self.bus.subscribe(ToolCallRequested, tool_call_executor.handle_tool_call_requested)
        self.bus.subscribe(ToolCallCompleted, self.tool_results.handle_tool_call_completed)
        self.bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
        self.bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
        self.bus.subscribe(
            AgentTurnRequested,
            agent_turn_handler.handle_agent_turn,
        )
        self.bus.subscribe(AssistantTextProduced, self._send_cli_reply)
        if self.memory_review is not None:
            self.bus.subscribe(
                AssistantTextProduced,
                self.memory_review.handle_assistant_text,
            )
            self.bus.subscribe(
                ToolCallCompleted,
                self.memory_review.handle_tool_call_completed,
            )

    async def _send_cli_reply(self, event: AssistantTextProduced) -> EventBatch:
        if event.reply_target is None:
            return ()
        if event.reply_target.kind != "cli":
            return ()
        if not await self.turn_coordinator.is_current(
            event.conversation_id,
            event.generation,
        ):
            return ()
        self._cli_replies[event.reply_target.request_id].put_nowait(event.text)
        return ()


def _derived_db_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}.{suffix}{path.suffix}")
