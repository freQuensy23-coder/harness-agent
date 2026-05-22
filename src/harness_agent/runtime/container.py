"""Per-user Docker container lifecycle + raw `docker exec` plumbing.

Owns: container ensure/start/bootstrap, container name derivation, raw
`docker exec` (foreground / detached / interactive stdio), and the
runner timeout/argv assembly. Does NOT know about workspace files,
shell-spawned processes, or MCP/skill discovery.
"""

import asyncio
import re

from loguru import logger

from harness_agent.runtime.models import DockerProcessResult
from harness_agent.runtime.paths import workspace_path
from harness_agent.runtime.protocols import DockerRunner


class DockerContainerExecutor:
    def __init__(
        self,
        *,
        runner: DockerRunner,
        image: str = "python:3.14-slim",
        container_prefix: str = "harness",
        ensure_container: bool = False,
        network: str | None = None,
        memory: str | None = None,
        cpus: str | None = None,
    ) -> None:
        self._runner = runner
        self._image = image
        self._container_prefix = container_prefix
        self._ensure_container = ensure_container
        self._network = network
        self._memory = memory
        self._cpus = cpus

    def container_name(self, user_id: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", user_id).strip("-")
        return f"{self._container_prefix}-{safe_user_id}"

    async def ensure(self, user_id: str) -> None:
        name = self.container_name(user_id)
        inspect = await self._runner.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name]
        )
        if inspect.exit_code == 0:
            if inspect.stdout.strip() != "true":
                await self._must_run(["docker", "start", name])
            await self._bootstrap_workspace(user_id)
            return

        argv = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-v",
            f"{name}-workspace:/workspace",
            "-w",
            "/workspace",
        ]
        if self._network is not None:
            argv.extend(["--network", self._network])
        if self._memory is not None:
            argv.extend(["--memory", self._memory])
        if self._cpus is not None:
            argv.extend(["--cpus", self._cpus])
        argv.extend([self._image, "sleep", "infinity"])
        await self._must_run(argv)
        await self._bootstrap_workspace(user_id)

    async def exec(
        self,
        user_id: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        if self._ensure_container:
            await self.ensure(user_id)
        return await self.exec_in_container(
            self.container_name(user_id),
            argv,
            workdir=workdir,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )

    async def exec_in_container(
        self,
        container_name: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        docker_argv = ["docker", "exec"]
        if stdin is not None:
            docker_argv.append("-i")
        if workdir is not None:
            docker_argv.extend(["-w", workdir])
        docker_argv.append(container_name)
        docker_argv.extend(argv)
        return await self._runner.run(
            docker_argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )

    async def exec_detached(
        self,
        container_name: str,
        argv: list[str],
        *,
        workdir: str | None = None,
    ) -> DockerProcessResult:
        docker_argv = ["docker", "exec", "-d"]
        if workdir is not None:
            docker_argv.extend(["-w", workdir])
        docker_argv.append(container_name)
        docker_argv.extend(argv)
        return await self._runner.run(docker_argv)

    async def open_stdio(
        self,
        *,
        user_id: str,
        argv: list[str],
        cwd: str = "/workspace",
    ) -> asyncio.subprocess.Process:
        if self._ensure_container:
            await self.ensure(user_id)
        return await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "-w",
            workspace_path(cwd),
            self.container_name(user_id),
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _bootstrap_workspace(self, user_id: str) -> None:
        script = """
mkdir -p /workspace/agent /workspace/skills /workspace/files /workspace/content
mkdir -p /workspace/mcp
touch /workspace/agent/SOUL.md /workspace/agent/AGENTS.md /workspace/agent/USER.md /workspace/agent/TOOLS.md
"""
        await self._must_run(
            [
                "docker",
                "exec",
                self.container_name(user_id),
                "sh",
                "-lc",
                script,
            ]
        )

    async def _must_run(self, argv: list[str]) -> DockerProcessResult:
        result = await self._runner.run(argv)
        if result.exit_code != 0:
            logger.error(
                "Docker command failed: {argv} stderr={stderr}",
                argv=argv,
                stderr=result.stderr,
            )
            raise RuntimeError(result.stderr)
        return result
