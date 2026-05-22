"""Unit tests for McpStdioSession protocol parsing.

We don't spin up an actual MCP server. Instead we plug a fake
`asyncio.subprocess.Process` whose stdin/stdout are in-memory streams,
so we can:
  - assert exact JSON-RPC framing the session writes to stdin, and
  - feed scripted responses to stdout and observe how list_tools /
    call_tool parse them.

This pins the contract for inputSchema alias handling and text-content
filtering without needing Docker.
"""

import json
from typing import Any

import pytest

from harness_agent.mcp import McpStdioSession
from harness_agent.mcp_models import McpServerConfig


def _server() -> McpServerConfig:
    return McpServerConfig(name="local", command=["python", "/u/m.py"])


class _FakeProcess:
    """Implements only the bits of asyncio.subprocess.Process the session
    actually touches: .stdin.write/.drain and .stdout.readline."""

    def __init__(self, scripted_responses: list[dict[str, Any]]) -> None:
        self.stdin_buffer: list[bytes] = []
        self._scripted = list(scripted_responses)
        self.stdin = _FakeStdin(self.stdin_buffer)
        self.stdout = _FakeStdout(self._scripted)
        self.stderr = None


class _FakeStdin:
    def __init__(self, buffer: list[bytes]) -> None:
        self._buffer = buffer

    def write(self, data: bytes) -> None:
        self._buffer.append(data)

    async def drain(self) -> None:
        return None


class _FakeStdout:
    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self._scripted = scripted

    async def readline(self) -> bytes:
        if not self._scripted:
            return b""
        payload = self._scripted.pop(0)
        return (json.dumps(payload) + "\n").encode("utf-8")


def _request_payloads(buffer: list[bytes]) -> list[dict[str, Any]]:
    return [json.loads(chunk.decode("utf-8")) for chunk in buffer]


@pytest.mark.asyncio
async def test_list_tools_parses_input_schema_via_alias() -> None:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo input text.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                        "required": ["msg"],
                    },
                },
                {
                    "name": "noop",
                    "description": "Does nothing.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        },
    }
    process = _FakeProcess(scripted_responses=[response])
    session = McpStdioSession(process=process, server=_server())  # type: ignore[arg-type]
    tools = await session.list_tools()

    assert [t.name for t in tools] == ["echo", "noop"]
    assert tools[0].input_schema == {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }
    sent = _request_payloads(process.stdin_buffer)
    assert sent == [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    ]


@pytest.mark.asyncio
async def test_call_tool_joins_text_content_and_ignores_non_text() -> None:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {"type": "text", "text": "first line"},
                {"type": "image", "url": "data:..."},
                {"type": "text"},
                {"type": "text", "text": "second line"},
            ]
        },
    }
    process = _FakeProcess(scripted_responses=[response])
    session = McpStdioSession(process=process, server=_server())  # type: ignore[arg-type]

    result = await session.call_tool("echo", {"msg": "hi"})

    assert result.stdout == "first line\nsecond line"
    assert result.exit_code == 0
    sent = _request_payloads(process.stdin_buffer)
    assert sent == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"msg": "hi"}},
        }
    ]


@pytest.mark.asyncio
async def test_call_tool_returns_empty_stdout_when_all_content_is_non_text() -> None:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {"type": "image", "url": "data:..."},
                {"type": "blob", "data": "..."},
            ]
        },
    }
    process = _FakeProcess(scripted_responses=[response])
    session = McpStdioSession(process=process, server=_server())  # type: ignore[arg-type]

    result = await session.call_tool("noop", {})

    assert result.stdout == ""


@pytest.mark.asyncio
async def test_request_raises_when_server_returns_error_object() -> None:
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32000, "message": "tool not found"},
    }
    process = _FakeProcess(scripted_responses=[response])
    session = McpStdioSession(process=process, server=_server())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        await session.call_tool("missing", {})


@pytest.mark.asyncio
async def test_read_json_raises_when_server_closes_stdout() -> None:
    process = _FakeProcess(scripted_responses=[])
    session = McpStdioSession(process=process, server=_server())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="closed stdout"):
        await session.call_tool("echo", {"msg": "hi"})


@pytest.mark.asyncio
async def test_initialize_emits_initialize_then_initialized_notification() -> None:
    init_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
    }
    process = _FakeProcess(scripted_responses=[init_response])
    session = McpStdioSession(process=process, server=_server())  # type: ignore[arg-type]
    await session.initialize()

    sent = _request_payloads(process.stdin_buffer)
    assert sent[0]["method"] == "initialize"
    assert sent[0]["params"]["protocolVersion"] == "2024-11-05"
    assert sent[1] == {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    assert "id" not in sent[1]


@pytest.mark.asyncio
async def test_list_tools_rejects_response_missing_input_schema_field() -> None:
    bad_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {"name": "broken", "description": "no inputSchema field"},
            ]
        },
    }
    process = _FakeProcess(scripted_responses=[bad_response])
    session = McpStdioSession(process=process, server=_server())  # type: ignore[arg-type]

    with pytest.raises(Exception):
        await session.list_tools()
