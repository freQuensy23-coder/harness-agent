"""browser-use cloud REST API client and Protocol.

The Protocol pins the surface BrowserUseService depends on; tests
substitute a FakeCloudClient. HttpxBrowserUseClient is the production
implementation against https://api.browser-use.com/api/v3."""

import asyncio
from typing import Any, Literal, Protocol

import httpx

from harness_agent.browser_use.cloud_dtos import (
    CloudMessagesPage,
    CloudProfile,
    CloudSessionState,
)


class BrowserUseCloudClient(Protocol):
    async def create_profile(self, *, internal_user_id: str) -> CloudProfile: ...

    async def delete_profile(self, *, cloud_profile_id: str) -> None: ...

    async def create_session(
        self,
        *,
        task: str,
        cloud_profile_id: str,
        model: str,
        keep_alive: bool,
        proxy_country_code: str | None = None,
    ) -> CloudSessionState: ...

    async def send_task(
        self,
        *,
        cloud_session_id: str,
        task: str,
        model: str,
    ) -> CloudSessionState: ...

    async def get_session(self, *, cloud_session_id: str) -> CloudSessionState: ...

    async def stop_session(
        self,
        *,
        cloud_session_id: str,
        strategy: Literal["task", "session"],
    ) -> CloudSessionState: ...

    async def list_messages(
        self,
        *,
        cloud_session_id: str,
        after: str | None = None,
        limit: int = 50,
    ) -> CloudMessagesPage: ...


class HttpxBrowserUseClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.browser-use.com/api/v3",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _http(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    headers={"X-Browser-Use-API-Key": self._api_key},
                    timeout=self._timeout_seconds,
                )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def create_profile(self, *, internal_user_id: str) -> CloudProfile:
        http = await self._http()
        response = await http.post(
            "/profiles",
            json={"name": f"harness-{internal_user_id}", "userId": internal_user_id},
        )
        response.raise_for_status()
        return CloudProfile.model_validate(response.json())

    async def delete_profile(self, *, cloud_profile_id: str) -> None:
        http = await self._http()
        response = await http.delete(f"/profiles/{cloud_profile_id}")
        response.raise_for_status()

    async def create_session(
        self,
        *,
        task: str,
        cloud_profile_id: str,
        model: str,
        keep_alive: bool,
        proxy_country_code: str | None = None,
    ) -> CloudSessionState:
        http = await self._http()
        body: dict[str, Any] = {
            "task": task,
            "model": model,
            "profileId": cloud_profile_id,
            "keepAlive": keep_alive,
        }
        if proxy_country_code is not None:
            body["proxyCountryCode"] = proxy_country_code
        response = await http.post("/sessions", json=body)
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def send_task(
        self,
        *,
        cloud_session_id: str,
        task: str,
        model: str,
    ) -> CloudSessionState:
        http = await self._http()
        body = {
            "task": task,
            "model": model,
            "sessionId": cloud_session_id,
            "keepAlive": True,
        }
        response = await http.post("/sessions", json=body)
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def get_session(self, *, cloud_session_id: str) -> CloudSessionState:
        http = await self._http()
        response = await http.get(f"/sessions/{cloud_session_id}")
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def stop_session(
        self,
        *,
        cloud_session_id: str,
        strategy: Literal["task", "session"],
    ) -> CloudSessionState:
        http = await self._http()
        response = await http.post(
            f"/sessions/{cloud_session_id}/stop",
            json={"strategy": strategy},
        )
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def list_messages(
        self,
        *,
        cloud_session_id: str,
        after: str | None = None,
        limit: int = 50,
    ) -> CloudMessagesPage:
        http = await self._http()
        params: dict[str, str | int] = {"limit": limit}
        if after is not None:
            params["after"] = after
        response = await http.get(
            f"/sessions/{cloud_session_id}/messages",
            params=params,
        )
        response.raise_for_status()
        return CloudMessagesPage.model_validate(response.json())
