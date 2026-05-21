import asyncio
import json
from typing import Any, cast

from pydantic import BaseModel, Field

from harness_agent.mcp_models import McpServerConfig
from harness_agent.runtime import DockerUserRuntime, RuntimeToolResult
from harness_agent.tools import ToolSpec


class McpToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class _McpToolPayload(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(alias="inputSchema")


class _McpToolListResponse(BaseModel):
    tools: list[_McpToolPayload]


class _McpContentItem(BaseModel):
    type: str
    text: str | None = None


class _McpCallToolResponse(BaseModel):
    content: list[_McpContentItem]


class McpStdioSession:
    def __init__(
        self,
        *,
        process: asyncio.subprocess.Process,
        server: McpServerConfig,
    ) -> None:
        self._process = process
        self._server = server
        self._next_id = 1

    async def initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "harness-agent", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> list[McpToolDefinition]:
        response = await self._request("tools/list", {})
        parsed = _McpToolListResponse.model_validate(response)
        return [
            McpToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in parsed.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> RuntimeToolResult:
        response = await self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        parsed = _McpCallToolResponse.model_validate(response)
        text_parts = [item.text for item in parsed.content if item.type == "text" and item.text is not None]
        return RuntimeToolResult(stdout="\n".join(text_parts))

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        await self._write_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        response = await self._read_json()
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write_json(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    async def _write_json(self, payload: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise RuntimeError(f"MCP server {self._server.name} stdin is closed")
        self._process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
        await self._process.stdin.drain()

    async def _read_json(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError(f"MCP server {self._server.name} stdout is closed")
        line = await self._process.stdout.readline()
        if not line:
            stderr = ""
            if self._process.stderr is not None:
                stderr = (await self._process.stderr.read()).decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP server {self._server.name} closed stdout: {stderr}")
        return cast(dict[str, Any], json.loads(line.decode("utf-8")))


class McpManager:
    def __init__(
        self,
        *,
        runtime: DockerUserRuntime,
        global_servers: list[McpServerConfig] | None = None,
    ) -> None:
        self._runtime = runtime
        self._global_servers = [] if global_servers is None else list(global_servers)
        self._sessions: dict[tuple[str, str], McpStdioSession] = {}
        self._tool_cache: dict[tuple[str, str], list[McpToolDefinition]] = {}
        self._server_cache: dict[tuple[str, str], McpServerConfig] = {}

    async def list_tool_specs(self, user_id: str) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for server in await self._servers_for_user(user_id):
            for tool in await self._list_server_tools(user_id, server.name):
                specs.append(
                    ToolSpec(
                        name=f"mcp.{server.name}.{tool.name}",
                        description=f"MCP {server.name}/{tool.name}: {tool.description}",
                        input_schema=tool.input_schema,
                    )
                )
        return specs

    async def call_tool(
        self,
        *,
        user_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> RuntimeToolResult:
        _, server_name, mcp_tool_name = tool_name.split(".", 2)
        session = await self._session(user_id, server_name)
        return await session.call_tool(mcp_tool_name, arguments)

    async def _list_server_tools(
        self,
        user_id: str,
        server_name: str,
    ) -> list[McpToolDefinition]:
        key = (user_id, server_name)
        if key not in self._tool_cache:
            self._tool_cache[key] = await (await self._session(user_id, server_name)).list_tools()
        return self._tool_cache[key]

    async def _session(
        self,
        user_id: str,
        server_name: str,
    ) -> McpStdioSession:
        key = (user_id, server_name)
        if key not in self._sessions:
            server = await self._server_for_user(user_id, server_name)
            process = await self._runtime.open_stdio(
                user_id=user_id,
                argv=server.command,
                cwd=server.cwd,
            )
            session = McpStdioSession(process=process, server=server)
            await session.initialize()
            self._sessions[key] = session
        return self._sessions[key]

    async def _servers_for_user(self, user_id: str) -> list[McpServerConfig]:
        user_servers = await self._runtime.list_mcp_servers(user_id)
        global_names = {server.name for server in self._global_servers}
        merged: list[McpServerConfig] = list(self._global_servers)
        for server in user_servers:
            if server.name in global_names:
                continue
            merged.append(server)
        for server in merged:
            self._server_cache[(user_id, server.name)] = server
        return merged

    async def _server_for_user(
        self,
        user_id: str,
        server_name: str,
    ) -> McpServerConfig:
        key = (user_id, server_name)
        if key not in self._server_cache:
            await self._servers_for_user(user_id)
        if key not in self._server_cache:
            raise KeyError(f"MCP server is not configured for user: {server_name}")
        return self._server_cache[key]
