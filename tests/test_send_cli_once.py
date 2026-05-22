from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import pytest

from harness_agent import send_cli_once


class _RecordingApp:
    """Records lifecycle entry/exit + send_cli order so the helper's
    contract (background_services held around send, released only after)
    can be asserted without booting a real HarnessApp."""

    def __init__(self, *, raise_on_send: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._raise_on_send = raise_on_send

    @asynccontextmanager
    async def background_services(self) -> AsyncIterator[None]:
        self.calls.append("enter")
        try:
            yield
        finally:
            self.calls.append("exit")

    async def send_cli(
        self,
        *,
        text: str,
        user_id: str,
        conversation_id: str | None,
    ) -> str:
        self.calls.append(f"send:{text}/{user_id}/{conversation_id}")
        if self._raise_on_send is not None:
            raise self._raise_on_send
        return "reply"


@pytest.mark.asyncio
async def test_send_cli_once_holds_services_around_send_on_success() -> None:
    app = _RecordingApp()

    reply = await send_cli_once(
        app,  # type: ignore[arg-type]
        text="hi",
        user_id="u:1",
        conversation_id="cli:1",
    )

    assert reply == "reply"
    assert app.calls == ["enter", "send:hi/u:1/cli:1", "exit"]


@pytest.mark.asyncio
async def test_send_cli_once_releases_services_even_when_send_raises() -> None:
    boom = RuntimeError("send blew up")
    app = _RecordingApp(raise_on_send=boom)

    with pytest.raises(RuntimeError, match="send blew up"):
        await send_cli_once(
            app,  # type: ignore[arg-type]
            text="hi",
            user_id="u:1",
            conversation_id=None,
        )

    assert app.calls == ["enter", "send:hi/u:1/None", "exit"]
