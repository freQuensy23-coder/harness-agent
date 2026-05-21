import asyncio
from asyncio import Queue
from pathlib import Path

from loguru import logger

from harness_agent.adapters.cli import event_from_cli_send
from harness_agent.adapters.telegram import AiogramTelegramAdapter
from harness_agent.browser_use import (
    BrowserSessionPollHandler,
    BrowserSessionPump,
    BrowserSessionPumpService,
    BrowserSessionResultWaiter,
    BrowserUseService,
    HttpxBrowserUseClient,
    SQLiteBrowserProfileStore,
    SQLiteBrowserSessionStore,
)
from harness_agent.bus import EventBus
from harness_agent.config import HarnessConfig
from harness_agent.context import ContextBuilder
from harness_agent.events import (
    AgentEvent,
    AgentTurnRequested,
    AssistantTextProduced,
    BrowserSessionCompleted,
    BrowserSessionFailed,
    BrowserSessionPollDue,
    BrowserSessionStopped,
    CliTextReceived,
    ScheduledMessageDue,
    TelegramTextReceived,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import (
    AgentTurnHandler,
    ContentIngestionHandler,
    ConversationProjector,
    IdentityHandler,
)
from harness_agent.identity import StaticIdentityResolver
from harness_agent.llm import OpenAIResponsesClient
from harness_agent.llm_audit import AuditedLlmClient, SQLiteLlmAuditStore
from harness_agent.mcp import McpManager
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import DockerUserRuntime
from harness_agent.scheduler import (
    SchedulerDueHandler,
    SchedulerPump,
    SchedulerService,
    SQLiteScheduleStore,
)
from harness_agent.store import SQLiteEventStore
from harness_agent.subagents import SQLiteSubAgentStore, SubAgentResultWaiter, SubAgentService
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
        schedules_path = _derived_db_path(db_path, "schedules")
        sub_agents_path = _derived_db_path(db_path, "subagents")
        tasks_path = _derived_db_path(db_path, "tasks")
        browser_profiles_path = _derived_db_path(db_path, "browser_profiles")
        browser_sessions_path = _derived_db_path(db_path, "browser_sessions")

        self.event_store = SQLiteEventStore(events_path)
        self.llm_audit_store = SQLiteLlmAuditStore(llm_path)
        self.projection = SQLiteConversationProjection(messages_path)
        self.schedule_store = SQLiteScheduleStore(schedules_path)
        self.sub_agent_store = SQLiteSubAgentStore(sub_agents_path)
        self.task_store = SQLiteTaskStore(tasks_path)
        self.browser_profile_store = SQLiteBrowserProfileStore(browser_profiles_path)
        self.browser_session_store = SQLiteBrowserSessionStore(browser_sessions_path)
        self.bus = EventBus(self.event_store)
        self.turn_coordinator = ConversationTurnCoordinator()
        self.tool_results = ToolCallResultWaiter()
        self.sub_agent_results = SubAgentResultWaiter()
        self.runtime = DockerUserRuntime(
            image=config.runtime.docker.image,
            container_prefix=config.runtime.docker.container_prefix,
            network=config.runtime.docker.network,
            memory=config.runtime.docker.memory,
            cpus=config.runtime.docker.cpus,
            ensure_container=True,
        )
        self.mcp_manager = McpManager(
            runtime=self.runtime,
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
            result_waiter=self.sub_agent_results,
        )
        self.browser_use_results = BrowserSessionResultWaiter()
        self.browser_use_client = HttpxBrowserUseClient(
            api_key=config.browser_use.api_key,
            base_url=config.browser_use.base_url,
            timeout_seconds=config.browser_use.request_timeout_seconds,
        )
        self.browser_use_service = BrowserUseService(
            bus=self.bus,
            client=self.browser_use_client,
            profile_store=self.browser_profile_store,
            session_store=self.browser_session_store,
            result_waiter=self.browser_use_results,
            profile_cap=config.browser_use.profile_cap,
            default_model=config.browser_use.default_model,
            default_run_timeout_seconds=config.browser_use.run_timeout_seconds,
        )
        self.browser_use_poll_handler = BrowserSessionPollHandler(
            client=self.browser_use_client,
            session_store=self.browser_session_store,
        )
        self.browser_use_pump_service = BrowserSessionPumpService(
            pump=BrowserSessionPump(
                session_store=self.browser_session_store,
                bus=self.bus,
            ),
            poll_seconds=config.browser_use.poll_interval_seconds,
        )
        self.scheduler_service = SchedulerService(
            pump=SchedulerPump(store=self.schedule_store, bus=self.bus),
            poll_seconds=self._config.scheduler.poll_seconds,
        )
        self.telegram: AiogramTelegramAdapter | None = None
        self._cli_replies: dict[str, Queue[str]] = {}
        self._background_uses = 0
        self._background_lock = asyncio.Lock()
        self._wire()

    async def publish(self, event: AgentEvent) -> None:
        await self.bus.publish(event)

    async def start_background_services(self) -> None:
        async with self._background_lock:
            if self._background_uses == 0:
                await self.scheduler_service.start()
                await self.browser_use_pump_service.start()
            self._background_uses += 1

    async def stop_background_services(self) -> None:
        async with self._background_lock:
            self._background_uses -= 1
            if self._background_uses > 0:
                return
            await self.browser_use_pump_service.stop()
            await self.scheduler_service.stop()
            await self.browser_use_client.aclose()

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
        await self.start_background_services()
        try:
            await self.bus.publish(event)
            reply = await self._cli_replies[event.request_id].get()
        finally:
            del self._cli_replies[event.request_id]
            await self.stop_background_services()
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
        await self.start_background_services()
        logger.info("Starting Telegram polling")
        try:
            await self.telegram.start_polling()
        finally:
            await self.stop_background_services()

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
        )
        tool_call_executor = ToolCallExecutor(
            runtime=self.runtime,
            task_store=self.task_store,
            schedule_store=self.schedule_store,
            web_fetcher=HttpxWebFetcher(),
            mcp_manager=self.mcp_manager,
            sub_agents=self.sub_agents,
            browser_use_service=self.browser_use_service,
        )
        scheduler_due_handler = SchedulerDueHandler()
        self.bus.subscribe(TelegramTextReceived, identity_handler.handle_telegram_text)
        self.bus.subscribe(CliTextReceived, identity_handler.handle_cli_text)
        self.bus.subscribe(ScheduledMessageDue, scheduler_due_handler.handle_due)
        self.bus.subscribe(UserTextReceived, content_ingestion_handler.handle_user_text)
        self.bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
        self.bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
        self.bus.subscribe(AssistantTextProduced, self.sub_agent_results.handle_assistant_text)
        self.bus.subscribe(ToolCallRequested, tool_call_executor.handle_tool_call_requested)
        self.bus.subscribe(ToolCallCompleted, self.tool_results.handle_tool_call_completed)
        self.bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
        self.bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
        self.bus.subscribe(
            AgentTurnRequested,
            agent_turn_handler.handle_agent_turn,
        )
        self.bus.subscribe(AssistantTextProduced, self._send_telegram_reply)
        self.bus.subscribe(AssistantTextProduced, self._send_cli_reply)
        self.bus.subscribe(
            BrowserSessionPollDue,
            self.browser_use_poll_handler.handle_poll_due,
        )
        self.bus.subscribe(
            BrowserSessionCompleted,
            self.browser_use_results.handle_completed,
        )
        self.bus.subscribe(
            BrowserSessionFailed,
            self.browser_use_results.handle_failed,
        )
        self.bus.subscribe(
            BrowserSessionStopped,
            self.browser_use_results.handle_stopped,
        )

    async def _send_telegram_reply(self, event: AssistantTextProduced) -> tuple:
        if event.reply_target is None:
            return ()
        if event.reply_target.kind != "telegram":
            return ()
        if not await self.turn_coordinator.is_current(
            event.conversation_id,
            event.generation,
        ):
            return ()
        if self.telegram is None:
            raise RuntimeError("telegram adapter is not running")
        await self.telegram.send_assistant_text(event)
        return ()

    async def _send_cli_reply(self, event: AssistantTextProduced) -> tuple:
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
