import httpx

from harness_agent.runtime import RuntimeToolResult
from harness_agent.tools import WebFetchInput


class WebFetcher:
    async def fetch(self, input: WebFetchInput) -> RuntimeToolResult:
        raise NotImplementedError


class HttpxWebFetcher(WebFetcher):
    async def fetch(self, input: WebFetchInput) -> RuntimeToolResult:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            response = await client.get(input.url)
        text = response.text[: input.max_bytes]
        stderr = "" if response.is_success else f"HTTP {response.status_code}"
        return RuntimeToolResult(stdout=text, stderr=stderr, exit_code=0 if response.is_success else 1)
