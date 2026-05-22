from pathlib import Path
from typing import Literal

import pytest

from harness_agent.browser_use import (
    BrowserSessionResultWaiter,
    BrowserUseService,
    CloudMessagesPage,
    CloudProfile,
    CloudSessionState,
    SQLiteBrowserProfileStore,
    SQLiteBrowserSessionStore,
)
from harness_agent.bus import EventBus
from harness_agent.scheduler import SQLiteScheduleStore
from harness_agent.store import SQLiteEventStore
from harness_agent.subagents import SQLiteSubAgentStore, SubAgentService
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.web_fetch import WebFetchExtractionWaiter


class _UnusedCloudClient:
    """Cloud client stub for tests that construct ToolCallExecutor but never
    exercise browser.* tools. Any call surfaces as a loud failure."""

    async def create_profile(self, *, internal_user_id: str) -> CloudProfile:
        raise AssertionError("browser-use cloud should not be touched in this test")

    async def delete_profile(self, *, cloud_profile_id: str) -> None:
        raise AssertionError("browser-use cloud should not be touched in this test")

    async def create_session(
        self,
        *,
        task: str,
        cloud_profile_id: str,
        model: str,
        keep_alive: bool,
        proxy_country_code: str | None = None,
    ) -> CloudSessionState:
        raise AssertionError("browser-use cloud should not be touched in this test")

    async def send_task(
        self,
        *,
        cloud_session_id: str,
        task: str,
        model: str,
    ) -> CloudSessionState:
        raise AssertionError("browser-use cloud should not be touched in this test")

    async def get_session(self, *, cloud_session_id: str) -> CloudSessionState:
        raise AssertionError("browser-use cloud should not be touched in this test")

    async def stop_session(
        self,
        *,
        cloud_session_id: str,
        strategy: Literal["task", "session"],
    ) -> CloudSessionState:
        raise AssertionError("browser-use cloud should not be touched in this test")

    async def list_messages(
        self,
        *,
        cloud_session_id: str,
        after: str | None = None,
        limit: int = 50,
    ) -> CloudMessagesPage:
        raise AssertionError("browser-use cloud should not be touched in this test")


@pytest.fixture
def task_store(tmp_path: Path) -> SQLiteTaskStore:
    return SQLiteTaskStore(tmp_path / "_tasks.sqlite3")


@pytest.fixture
def schedule_store(tmp_path: Path) -> SQLiteScheduleStore:
    return SQLiteScheduleStore(tmp_path / "_schedules.sqlite3")


@pytest.fixture
def sub_agents(tmp_path: Path) -> SubAgentService:
    bus = EventBus(SQLiteEventStore(tmp_path / "_subagents_events.sqlite3"))
    return SubAgentService(
        bus=bus,
        store=SQLiteSubAgentStore(tmp_path / "_subagents.sqlite3"),
    )


@pytest.fixture
def web_fetch_waiter() -> WebFetchExtractionWaiter:
    """Bare waiter for tests that construct ToolCallExecutor but never
    exercise web.fetch. Tests that DO exercise web.fetch can ignore this
    fixture and wire their own waiter + handler."""
    return WebFetchExtractionWaiter()


@pytest.fixture
def browser_use_service(tmp_path: Path) -> BrowserUseService:
    bus = EventBus(SQLiteEventStore(tmp_path / "_browser_use_events.sqlite3"))
    return BrowserUseService(
        bus=bus,
        client=_UnusedCloudClient(),
        profile_store=SQLiteBrowserProfileStore(tmp_path / "_browser_use_profiles.sqlite3"),
        session_store=SQLiteBrowserSessionStore(tmp_path / "_browser_use_sessions.sqlite3"),
        result_waiter=BrowserSessionResultWaiter(),
        profile_cap=5,
        default_model="bu-mini",
    )
