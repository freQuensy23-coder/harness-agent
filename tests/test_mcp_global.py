from pathlib import Path

import pytest

from harness_agent.config import load_config
from harness_agent.mcp import McpManager
from harness_agent.mcp_models import McpServerConfig, parse_mcp_server_yaml
from harness_agent.runtime import DockerProcessResult, DockerUserRuntime, InMemorySpawnedProcessStore


class _FakeRuntime:
    def __init__(self, user_servers: list[McpServerConfig]) -> None:
        self._user_servers = list(user_servers)
        self.list_calls: list[str] = []

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        self.list_calls.append(user_id)
        return list(self._user_servers)


def test_yaml_config_loads_global_mcp_servers(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        """
llm:
  api_key: test-key
browser_use:
  api_key: test-bu-key
mcp:
  servers:
    - name: notion
      command: ["python", "/workspace/notion_mcp.py"]
      cwd: /workspace
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert len(config.mcp.servers) == 1
    assert config.mcp.servers[0].name == "notion"
    assert config.mcp.servers[0].command == ["python", "/workspace/notion_mcp.py"]


def test_yaml_config_defaults_to_no_global_mcp_servers(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        "llm:\n  api_key: test-key\nbrowser_use:\n  api_key: test-bu-key\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.mcp.servers == []


@pytest.mark.asyncio
async def test_global_servers_are_returned_when_user_has_none() -> None:
    runtime = _FakeRuntime(user_servers=[])
    manager = McpManager(
        runtime=runtime,  # type: ignore[arg-type]
        global_servers=[
            McpServerConfig(name="notion", command=["python", "/g/notion.py"]),
        ],
    )

    servers = await manager._servers_for_user("u:1")

    assert [s.name for s in servers] == ["notion"]


@pytest.mark.asyncio
async def test_user_servers_merge_with_global_servers() -> None:
    runtime = _FakeRuntime(
        user_servers=[McpServerConfig(name="local", command=["python", "/u/local.py"])],
    )
    manager = McpManager(
        runtime=runtime,  # type: ignore[arg-type]
        global_servers=[
            McpServerConfig(name="notion", command=["python", "/g/notion.py"]),
        ],
    )

    servers = await manager._servers_for_user("u:1")

    assert [s.name for s in servers] == ["notion", "local"]


@pytest.mark.asyncio
async def test_global_server_wins_on_name_conflict_with_user_server() -> None:
    runtime = _FakeRuntime(
        user_servers=[
            McpServerConfig(name="notion", command=["python", "/u/user_override.py"]),
            McpServerConfig(name="local", command=["python", "/u/local.py"]),
        ],
    )
    global_notion = McpServerConfig(name="notion", command=["python", "/g/notion.py"])
    manager = McpManager(
        runtime=runtime,  # type: ignore[arg-type]
        global_servers=[global_notion],
    )

    servers = await manager._servers_for_user("u:1")

    assert [s.name for s in servers] == ["notion", "local"]
    notion = next(s for s in servers if s.name == "notion")
    assert notion.command == ["python", "/g/notion.py"]


@pytest.mark.asyncio
async def test_manager_without_globals_behaves_like_before() -> None:
    runtime = _FakeRuntime(
        user_servers=[McpServerConfig(name="local", command=["python", "/u/local.py"])],
    )
    manager = McpManager(runtime=runtime)  # type: ignore[arg-type]

    servers = await manager._servers_for_user("u:1")

    assert [s.name for s in servers] == ["local"]


def test_parse_mcp_server_yaml_returns_config_for_valid_input() -> None:
    server = parse_mcp_server_yaml(
        "name: local\ncommand: [python, /workspace/mcp.py]\n",
        path="/workspace/mcp/local.yaml",
    )

    assert server is not None
    assert server.name == "local"
    assert server.command == ["python", "/workspace/mcp.py"]


def test_parse_mcp_server_yaml_skips_invalid_yaml() -> None:
    # Unclosed bracket -- yaml.safe_load raises YAMLError.
    server = parse_mcp_server_yaml(
        "name: local\ncommand: [python, /workspace/mcp.py\n",
        path="/workspace/mcp/broken.yaml",
    )

    assert server is None


def test_parse_mcp_server_yaml_skips_schema_violation() -> None:
    # Missing required `command` field.
    server = parse_mcp_server_yaml(
        "name: local\n",
        path="/workspace/mcp/missing_command.yaml",
    )

    assert server is None


def test_parse_mcp_server_yaml_skips_unknown_field() -> None:
    # ConfigDict(extra="forbid") rejects unknown fields.
    server = parse_mcp_server_yaml(
        "name: local\ncommand: [python]\nbogus: true\n",
        path="/workspace/mcp/bad.yaml",
    )

    assert server is None


class _ScriptedDockerRunner:
    def __init__(self, results: list[DockerProcessResult]) -> None:
        self._results = results
        self.calls: list[list[str]] = []

    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        self.calls.append(argv)
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_list_mcp_servers_skips_malformed_yaml_files() -> None:
    # `list_mcp_servers` first runs `find`, then `cat` for each path. The
    # second path's content is malformed YAML, the third is schema-invalid;
    # both must be skipped while the first is returned.
    find_stdout = (
        "/workspace/mcp/good.yaml\n"
        "/workspace/mcp/broken.yaml\n"
        "/workspace/mcp/missing_field.yaml\n"
    )
    runner = _ScriptedDockerRunner(
        [
            DockerProcessResult(stdout=find_stdout, stderr="", exit_code=0),
            DockerProcessResult(
                stdout="name: good\ncommand: [python, /workspace/g.py]\n",
                stderr="",
                exit_code=0,
            ),
            DockerProcessResult(
                stdout="name: bad\ncommand: [python\n",  # unclosed bracket
                stderr="",
                exit_code=0,
            ),
            DockerProcessResult(
                stdout="name: incomplete\n",  # missing required `command`
                stderr="",
                exit_code=0,
            ),
        ]
    )
    runtime = DockerUserRuntime(runner=runner, spawned_process_store=InMemorySpawnedProcessStore())  # type: ignore[arg-type]

    servers = await runtime.list_mcp_servers("u:1")

    assert [s.name for s in servers] == ["good"]


@pytest.mark.asyncio
async def test_list_mcp_servers_raises_when_find_fails() -> None:
    # Catastrophic listing failure (e.g. container down) must still surface --
    # only individual malformed files are silently skipped.
    runner = _ScriptedDockerRunner(
        [DockerProcessResult(stdout="", stderr="container is gone", exit_code=1)]
    )
    runtime = DockerUserRuntime(runner=runner, spawned_process_store=InMemorySpawnedProcessStore())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="container is gone"):
        await runtime.list_mcp_servers("u:1")
