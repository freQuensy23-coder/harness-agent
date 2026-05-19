import asyncio
import base64
import posixpath
import re
import shlex
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Protocol
from uuid import uuid4

import yaml
from loguru import logger
from pydantic import BaseModel

from harness_agent.context import AgentFileSet, Skill, UserContextRuntime
from harness_agent.content import WorkspaceFile
from harness_agent.mcp_models import McpServerConfig
from harness_agent.tools import (
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
    ShellKillInput,
    ShellReadInput,
    ShellSpawnInput,
)


class RuntimeToolResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    def render_for_llm(self, tool_name: str) -> str:
        return (
            f"{tool_name} stdout:\n{self.stdout}"
            f"stderr:\n{self.stderr}\n"
            f"exit_code: {self.exit_code}"
        )


class DockerProcessResult(RuntimeToolResult):
    pass


class SkillFrontmatter(BaseModel):
    name: str
    description: str


class DockerRunner(Protocol):
    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        pass


class AsyncioDockerRunner:
    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if timeout_seconds is None:
            stdout, stderr = await process.communicate(stdin)
            return DockerProcessResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode,
            )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin),
            timeout=timeout_seconds,
        )
        return DockerProcessResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=process.returncode,
        )


class UserRuntime(UserContextRuntime):
    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        raise NotImplementedError

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int,
    ) -> WorkspaceFile:
        raise NotImplementedError

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        raise NotImplementedError


class DockerUserRuntime(UserRuntime):
    def __init__(
        self,
        *,
        runner: DockerRunner | None = None,
        image: str = "python:3.14-slim",
        container_prefix: str = "harness",
        ensure_container: bool = False,
        network: str | None = None,
        memory: str | None = None,
        cpus: str | None = None,
    ) -> None:
        if runner is None:
            runner = AsyncioDockerRunner()
        self._runner = runner
        self._image = image
        self._container_prefix = container_prefix
        self._ensure_container = ensure_container
        self._network = network
        self._memory = memory
        self._cpus = cpus
        self._spawned: dict[str, asyncio.subprocess.Process] = {}

    async def ensure_user_container(self, user_id: str) -> None:
        name = self._container_name(user_id)
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

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        return AgentFileSet(
            soul=await self._read_text(user_id, "/workspace/agent/SOUL.md"),
            agents=await self._read_text(user_id, "/workspace/agent/AGENTS.md"),
            user=await self._read_text(user_id, "/workspace/agent/USER.md"),
            tools=await self._read_text(user_id, "/workspace/agent/TOOLS.md"),
        )

    async def list_skills(self, user_id: str) -> list[Skill]:
        result = await self._exec(
            user_id,
            [
                "sh",
                "-lc",
                "find /workspace/skills -name SKILL.md -type f | sort",
            ],
        )
        if result.exit_code != 0:
            logger.warning("Failed to list skills for {user_id}: {stderr}", user_id=user_id, stderr=result.stderr)
            raise RuntimeError(result.stderr)
        skills: list[Skill] = []
        for path in [line for line in result.stdout.splitlines() if line.strip()]:
            text = await self._read_text(user_id, path)
            skills.append(
                parse_skill_markdown(text, file_name=PurePosixPath(path).parent.name)
            )
        return skills

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        result = await self._exec(
            user_id,
            [
                "sh",
                "-lc",
                "find /workspace/mcp -name '*.yaml' -type f | sort",
            ],
        )
        if result.exit_code != 0:
            logger.warning("Failed to list MCP servers for {user_id}: {stderr}", user_id=user_id, stderr=result.stderr)
            raise RuntimeError(result.stderr)
        servers: list[McpServerConfig] = []
        for path in [line for line in result.stdout.splitlines() if line.strip()]:
            text = await self._read_text(user_id, path)
            servers.append(McpServerConfig.model_validate(yaml.safe_load(text)))
        return servers

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        cwd = _workspace_path(input.cwd)
        return await self._exec(
            user_id,
            ["sh", "-lc", input.command],
            workdir=cwd,
            timeout_seconds=input.timeout_seconds,
        )

    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        path = _content_path(path)
        parent = posixpath.dirname(path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(path)}"],
            stdin=content,
        )

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        await self._maybe_ensure(user_id)
        cwd = _workspace_path(input.cwd)
        process_id = uuid4().hex
        process = await self.open_stdio(
            user_id=user_id,
            argv=["sh", "-lc", input.command],
            cwd=cwd,
        )
        self._spawned[process_id] = process
        return RuntimeToolResult(stdout=process_id)

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        process = self._spawned.get(input.process_id)
        if process is None:
            raise KeyError(f"Unknown process: {input.process_id}")
        stdout = await _read_available_now(process.stdout, input.max_bytes)
        stderr = await _read_available_now(process.stderr, input.max_bytes)
        return RuntimeToolResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode if process.returncode is not None else 0,
        )

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        process = self._spawned.pop(input.process_id, None)
        if process is None:
            raise KeyError(f"Unknown process: {input.process_id}")
        if process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=2)
        return RuntimeToolResult(stdout=f"killed {input.process_id}\n")

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        path = _workspace_path(input.path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"head -c {int(input.max_bytes)} -- {shlex.quote(path)}"],
        )

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int,
    ) -> WorkspaceFile:
        path = _workspace_path(path)
        code = (
            "import base64, pathlib, sys;"
            "data=pathlib.Path(sys.argv[1]).read_bytes()[:int(sys.argv[2])];"
            "sys.stdout.write(base64.b64encode(data).decode('ascii'))"
        )
        result = await self._exec(
            user_id,
            ["python", "-c", code, path, str(max_bytes)],
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
        return WorkspaceFile(
            path=path,
            content=base64.b64decode(result.stdout),
        )

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        path = _workspace_path(input.path)
        parent = posixpath.dirname(path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(path)}"],
            stdin=input.content.encode("utf-8"),
        )

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        current = await self.file_read(
            user_id,
            FileReadInput(path=input.path, max_bytes=5_000_000),
        )
        if current.exit_code != 0:
            return current
        if input.old not in current.stdout:
            raise ValueError("old text not found")
        count = -1 if input.replace_all else 1
        updated = current.stdout.replace(input.old, input.new, count)
        return await self.file_write(user_id, FileWriteInput(path=input.path, content=updated))

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        current = await self.file_read(
            user_id,
            FileReadInput(path=input.path, max_bytes=5_000_000),
        )
        if current.exit_code != 0:
            return current
        updated = current.stdout
        for edit in input.edits:
            if edit.old not in updated:
                raise ValueError(f"old text not found: {edit.old}")
            count = -1 if edit.replace_all else 1
            updated = updated.replace(edit.old, edit.new, count)
        return await self.file_write(user_id, FileWriteInput(path=input.path, content=updated))

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        cwd = _workspace_path(input.cwd)
        code = (
            "import glob, sys;"
            "pattern=sys.argv[1]; limit=int(sys.argv[2]);"
            "print('\\n'.join(sorted(glob.glob(pattern, recursive=True))[:limit]))"
        )
        return await self._exec(
            user_id,
            ["python", "-c", code, input.pattern, str(input.max_results)],
            workdir=cwd,
        )

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        path = _workspace_path(input.path)
        command = (
            f"grep -RIn -- {shlex.quote(input.pattern)} {shlex.quote(path)} "
            f"| head -n {int(input.max_results)}"
        )
        return await self._exec(user_id, ["sh", "-lc", command])

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        path = _workspace_path(input.path)
        command = (
            f"find {shlex.quote(path)} -maxdepth 1 -mindepth 1 "
            f"| sort | head -n {int(input.max_results)}"
        )
        return await self._exec(user_id, ["sh", "-lc", command])

    async def _read_text(self, user_id: str, path: str) -> str:
        result = await self._exec(
            user_id,
            ["sh", "-lc", f"cat -- {shlex.quote(_workspace_path(path))}"],
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
        return result.stdout

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
                self._container_name(user_id),
                "sh",
                "-lc",
                script,
            ]
        )

    async def _exec(
        self,
        user_id: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        await self._maybe_ensure(user_id)
        docker_argv = ["docker", "exec"]
        if stdin is not None:
            docker_argv.append("-i")
        if workdir is not None:
            docker_argv.extend(["-w", workdir])
        docker_argv.append(self._container_name(user_id))
        docker_argv.extend(argv)
        return await self._runner.run(
            docker_argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )

    async def open_stdio(
        self,
        *,
        user_id: str,
        argv: list[str],
        cwd: str = "/workspace",
    ) -> asyncio.subprocess.Process:
        await self._maybe_ensure(user_id)
        return await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "-w",
            _workspace_path(cwd),
            self._container_name(user_id),
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _must_run(self, argv: list[str]) -> DockerProcessResult:
        result = await self._runner.run(argv)
        if result.exit_code != 0:
            logger.error("Docker command failed: {argv} stderr={stderr}", argv=argv, stderr=result.stderr)
            raise RuntimeError(result.stderr)
        return result

    async def _maybe_ensure(self, user_id: str) -> None:
        if self._ensure_container:
            await self.ensure_user_container(user_id)

    def _container_name(self, user_id: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", user_id).strip("-")
        return f"{self._container_prefix}-{safe_user_id}"


class FakeUserRuntime(UserRuntime):
    def __init__(
        self,
        *,
        files: dict[str, str | bytes] | None = None,
        agent_files: AgentFileSet | None = None,
        skills: Sequence[Skill] = (),
        shell_results: Sequence[RuntimeToolResult] = (),
    ) -> None:
        self._files = {} if files is None else files
        self._agent_files = agent_files
        self._skills = list(skills)
        self._shell_results = list(shell_results)
        self.read_agent_files_calls: list[str] = []
        self.list_skills_calls: list[str] = []
        self.shell_exec_calls: list[ShellExecInput] = []
        self.file_write_calls: list[FileWriteInput] = []
        self.content_write_calls: list[tuple[str, bytes]] = []

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        self.read_agent_files_calls.append(user_id)
        if self._agent_files is not None:
            return self._agent_files
        required = [
            "/workspace/agent/SOUL.md",
            "/workspace/agent/AGENTS.md",
            "/workspace/agent/USER.md",
            "/workspace/agent/TOOLS.md",
        ]
        missing = [path for path in required if path not in self._files]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        return AgentFileSet(
            soul=self._files["/workspace/agent/SOUL.md"],
            agents=self._files["/workspace/agent/AGENTS.md"],
            user=self._files["/workspace/agent/USER.md"],
            tools=self._files["/workspace/agent/TOOLS.md"],
        )

    async def list_skills(self, user_id: str) -> list[Skill]:
        self.list_skills_calls.append(user_id)
        return self._skills

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        return []

    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        self.content_write_calls.append((path, content))
        self._files[path] = content
        return RuntimeToolResult()

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int,
    ) -> WorkspaceFile:
        content = self._files[path]
        try:
            data = content.encode("latin1")
        except AttributeError:
            data = content
        return WorkspaceFile(path=path, content=data[:max_bytes])

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        self.shell_exec_calls.append(input)
        if self._shell_results:
            return self._shell_results.pop(0)
        return RuntimeToolResult()

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        return RuntimeToolResult(stdout="fake-process")

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        return RuntimeToolResult()

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        return RuntimeToolResult()

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        content = self._files[input.path]
        try:
            return RuntimeToolResult(stdout=content.decode("utf-8", errors="replace"))
        except AttributeError:
            pass
        return RuntimeToolResult(stdout=content)

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        self.file_write_calls.append(input)
        self._files[input.path] = input.content
        return RuntimeToolResult()

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        content = self._files[input.path]
        count = -1 if input.replace_all else 1
        self._files[input.path] = content.replace(input.old, input.new, count)
        return RuntimeToolResult()

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        content = self._files[input.path]
        for edit in input.edits:
            count = -1 if edit.replace_all else 1
            content = content.replace(edit.old, edit.new, count)
        self._files[input.path] = content
        return RuntimeToolResult()

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        return RuntimeToolResult(
            stdout="\n".join(path for path in sorted(self._files) if path.startswith(input.cwd))
        )

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        lines: list[str] = []
        for path, content in sorted(self._files.items()):
            if not path.startswith(input.path):
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                if input.pattern in line:
                    lines.append(f"{path}:{line_no}:{line}")
        return RuntimeToolResult(stdout="\n".join(lines[: input.max_results]))

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        return RuntimeToolResult(
            stdout="\n".join(path for path in sorted(self._files) if path.startswith(input.path))
        )


def parse_skill_markdown(text: str, *, file_name: str) -> Skill:
    if text.startswith("---\n"):
        _, frontmatter, body = text.split("---", 2)
        metadata = SkillFrontmatter.model_validate(yaml.safe_load(frontmatter))
        return Skill(
            name=metadata.name,
            description=metadata.description,
            body=body.lstrip("\n"),
        )
    raise ValueError(f"Missing skill frontmatter in {file_name}")


def _workspace_path(path: str) -> str:
    normalized = posixpath.normpath(path)
    if not normalized.startswith("/"):
        normalized = posixpath.normpath(f"/workspace/{normalized}")
    if normalized != "/workspace" and not normalized.startswith("/workspace/"):
        raise ValueError(f"path must stay inside /workspace: {path}")
    return normalized


def _content_path(path: str) -> str:
    normalized = _workspace_path(path)
    if normalized != "/workspace/content" and not normalized.startswith("/workspace/content/"):
        raise ValueError(f"content path must stay inside /workspace/content: {path}")
    return normalized


async def _read_available_now(
    reader: asyncio.StreamReader | None,
    max_bytes: int,
) -> str:
    if reader is None:
        raise RuntimeError("process stream is not available")
    if not reader._buffer:
        return ""
    data = await reader.read(max_bytes)
    return data.decode("utf-8", errors="replace")
