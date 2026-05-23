"""Per-user MCP server discovery from `/workspace/mcp/*.yaml`.

A single malformed YAML file is logged and skipped so it cannot suppress
the rest; a failure to even list the directory still raises so the
caller learns the container is unreachable."""

from collections.abc import Awaitable, Callable

from loguru import logger

from harness_agent.mcp_models import McpServerConfig, parse_mcp_server_yaml
from harness_agent.runtime.models import ExecInContainer


ReadText = Callable[[str, str], Awaitable[str]]


class DockerMcpDiscovery:
    """List user-scope MCP server configs from the workspace."""

    def __init__(
        self,
        *,
        exec_in_container: ExecInContainer,
        read_text: ReadText,
    ) -> None:
        self._exec = exec_in_container
        self._read_text = read_text

    async def list_servers(self, user_id: str) -> list[McpServerConfig]:
        result = await self._exec(
            user_id,
            [
                "sh",
                "-lc",
                "find /workspace/mcp -name '*.yaml' -type f | sort",
            ],
        )
        if result.exit_code != 0:
            logger.warning(
                "Failed to list MCP servers for {user_id}: {stderr}",
                user_id=user_id,
                stderr=result.stderr,
            )
            raise RuntimeError(result.stderr)
        servers: list[McpServerConfig] = []
        for path in [line for line in result.stdout.splitlines() if line.strip()]:
            text = await self._read_text(user_id, path)
            server = parse_mcp_server_yaml(text, path=path)
            if server is not None:
                servers.append(server)
        return servers
