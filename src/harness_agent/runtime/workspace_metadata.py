"""User-supplied workspace metadata discovery: agent prompt files,
markdown skill definitions, and per-user MCP server YAML."""

from pathlib import PurePosixPath

from loguru import logger

from harness_agent.context import AgentFileSet, Skill
from harness_agent.mcp_models import McpServerConfig, parse_mcp_server_yaml
from harness_agent.runtime.container import DockerContainerExecutor
from harness_agent.runtime.skills import parse_skill_markdown
from harness_agent.runtime.workspace_files import DockerWorkspaceFiles


class DockerWorkspaceMetadata:
    def __init__(
        self,
        *,
        executor: DockerContainerExecutor,
        files: DockerWorkspaceFiles,
    ) -> None:
        self._executor = executor
        self._files = files

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        return AgentFileSet(
            soul=await self._files.read_text(user_id, "/workspace/agent/SOUL.md"),
            agents=await self._files.read_text(user_id, "/workspace/agent/AGENTS.md"),
            user=await self._files.read_text(user_id, "/workspace/agent/USER.md"),
            tools=await self._files.read_text(user_id, "/workspace/agent/TOOLS.md"),
        )

    async def list_skills(self, user_id: str) -> list[Skill]:
        result = await self._executor.exec(
            user_id,
            [
                "sh",
                "-lc",
                "find /workspace/skills -name SKILL.md -type f | sort",
            ],
        )
        if result.exit_code != 0:
            logger.warning(
                "Failed to list skills for {user_id}: {stderr}",
                user_id=user_id,
                stderr=result.stderr,
            )
            raise RuntimeError(result.stderr)
        skills: list[Skill] = []
        for path in [line for line in result.stdout.splitlines() if line.strip()]:
            text = await self._files.read_text(user_id, path)
            skills.append(
                parse_skill_markdown(text, file_name=PurePosixPath(path).parent.name)
            )
        return skills

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        result = await self._executor.exec(
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
            text = await self._files.read_text(user_id, path)
            server = parse_mcp_server_yaml(text, path=path)
            if server is not None:
                servers.append(server)
        return servers
