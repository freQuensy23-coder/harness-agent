import asyncio
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from uuid import uuid4

import pytest

from harness_agent.context import ContextBuilder
from harness_agent.mcp import McpManager
from harness_agent.runtime import DockerUserRuntime, SQLiteSpawnedProcessStore
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tools import (
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileEditOperation,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
    ShellKillInput,
    ShellReadInput,
    ShellSpawnInput,
    TaskCreateInput,
    TaskGetInput,
    TaskListInput,
    TaskStopInput,
    TaskUpdateInput,
    WebFetchInput,
)
from harness_agent.web_fetch import HttpxWebFetcher


@pytest.fixture
def docker_runtime(tmp_path):
    prefix = f"harness-e2e-{uuid4().hex[:8]}"
    user_id = "u:e2e"
    runtime = DockerUserRuntime(
        image="python:3.14-slim",
        container_prefix=prefix,
        ensure_container=True,
        network="bridge",
        memory="1g",
        cpus="1",
        spawned_process_store=SQLiteSpawnedProcessStore(tmp_path / "runtime.sqlite3"),
    )
    yield runtime, user_id, f"{prefix}-u-e2e"
    subprocess.run(["docker", "rm", "-f", f"{prefix}-u-e2e"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "volume", "rm", f"{prefix}-u-e2e-workspace"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def local_http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"web-fetch-ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    thread.join(timeout=5)


@pytest.mark.asyncio
async def test_all_container_tools_skills_soul_and_mcp_work_in_docker(docker_runtime):
    runtime, user_id, container_name = docker_runtime
    await runtime.ensure_user_container(user_id)

    await runtime.file_write(user_id, FileWriteInput(path="/workspace/agent/SOUL.md", content="SOUL_MARKER"))
    await runtime.file_write(user_id, FileWriteInput(path="/workspace/agent/AGENTS.md", content="AGENTS_MARKER"))
    await runtime.file_write(user_id, FileWriteInput(path="/workspace/agent/USER.md", content="USER_MARKER"))
    await runtime.file_write(user_id, FileWriteInput(path="/workspace/agent/TOOLS.md", content="TOOLS_MARKER"))
    await runtime.file_write(
        user_id,
        FileWriteInput(
            path="/workspace/skills/demo/SKILL.md",
            content="---\nname: demo\ndescription: Demo skill.\n---\nSKILL_MARKER\n",
        ),
    )

    context = await ContextBuilder(runtime=runtime).build(user_id)
    assert "SOUL_MARKER" in context.system
    assert "AGENTS_MARKER" in context.system
    assert "USER_MARKER" in context.system
    assert "TOOLS_MARKER" in context.system
    assert "SKILL_MARKER" in context.system
    assert await runtime.list_skills(user_id)

    await runtime.file_write(
        user_id,
        FileWriteInput(path="/workspace/files/a.txt", content="alpha\nbeta\n"),
    )
    assert "alpha\nbeta\n" == (await runtime.file_read(user_id, FileReadInput(path="/workspace/files/a.txt"))).stdout
    await runtime.file_edit(
        user_id,
        FileEditInput(path="/workspace/files/a.txt", old="beta", new="gamma"),
    )
    await runtime.file_multi_edit(
        user_id,
        FileMultiEditInput(
            path="/workspace/files/a.txt",
            edits=[
                FileEditOperation(old="alpha", new="one"),
                FileEditOperation(old="gamma", new="two"),
            ],
        ),
    )
    assert "one\ntwo\n" == (await runtime.file_read(user_id, FileReadInput(path="/workspace/files/a.txt"))).stdout
    assert "files/a.txt" in (await runtime.file_glob(user_id, FileGlobInput(pattern="files/*.txt"))).stdout
    assert "two" in (await runtime.file_grep(user_id, FileGrepInput(pattern="two", path="/workspace/files"))).stdout
    assert "/workspace/files/a.txt" in (await runtime.file_list(user_id, FileListInput(path="/workspace/files"))).stdout
    assert "one\ntwo\n" == (await runtime.shell_exec(user_id, ShellExecInput(command="cat files/a.txt"))).stdout

    spawned = await runtime.shell_spawn(
        user_id,
        ShellSpawnInput(command="while true; do echo alive; sleep 1; done"),
    )
    await asyncio.sleep(1.2)
    assert "alive" in (await runtime.shell_read(user_id, ShellReadInput(process_id=spawned.stdout))).stdout
    assert "killed" in (await runtime.shell_kill(user_id, ShellKillInput(process_id=spawned.stdout))).stdout

    await runtime.file_write(
        user_id,
        FileWriteInput(path="/workspace/mcp_echo.py", content=MCP_ECHO_SERVER),
    )
    await runtime.file_write(
        user_id,
        FileWriteInput(
            path="/workspace/mcp/local.yaml",
            content="name: local\ncommand: [python, /workspace/mcp_echo.py]\ncwd: /workspace\n",
        ),
    )
    mcp = McpManager(runtime=runtime)
    mcp_specs = await mcp.list_tool_specs(user_id)
    assert [spec.name for spec in mcp_specs] == ["mcp.local.echo"]
    assert (
        await mcp.call_tool(
            user_id=user_id,
            tool_name="mcp.local.echo",
            arguments={"text": f"{container_name}:mcp-ok"},
        )
    ).stdout == f"{container_name}:mcp-ok"


@pytest.mark.asyncio
async def test_web_fetch_and_task_tools_work(local_http_server, tmp_path):
    fetched = await HttpxWebFetcher().fetch(WebFetchInput(url=local_http_server))
    assert fetched.stdout == "web-fetch-ok"

    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    task = await store.create(
        user_id="u:1",
        conversation_id="cli:tasks",
        title=TaskCreateInput(title="first").title,
        status="pending",
    )
    assert (await store.get(task_id=TaskGetInput(task_id=task.id).task_id, user_id="u:1", conversation_id="cli:tasks")).title == "first"
    assert len(await store.list(user_id="u:1", conversation_id="cli:tasks", include_stopped=TaskListInput(include_stopped=True).include_stopped)) == 1
    updated = await store.update(
        task_id=TaskUpdateInput(task_id=task.id, title="second", status="in_progress").task_id,
        user_id="u:1",
        conversation_id="cli:tasks",
        title="second",
        status="in_progress",
    )
    assert updated.title == "second"
    stopped = await store.update(
        task_id=TaskStopInput(task_id=task.id).task_id,
        user_id="u:1",
        conversation_id="cli:tasks",
        title=None,
        status="stopped",
    )
    assert stopped.status == "stopped"


MCP_ECHO_SERVER = """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "echo", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo text.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
    elif method == "tools/call":
        result = {
            "content": [
                {
                    "type": "text",
                    "text": request["params"]["arguments"]["text"],
                }
            ]
        }
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}) + "\\n")
    sys.stdout.flush()
"""
