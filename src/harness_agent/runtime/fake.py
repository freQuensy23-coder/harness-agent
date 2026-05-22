from collections.abc import Sequence

from harness_agent.content import WorkspaceFile
from harness_agent.context import AgentFileSet, Skill
from harness_agent.mcp_models import McpServerConfig
from harness_agent.memory import MemoryTarget
from harness_agent.runtime.models import RuntimeFileRead, RuntimeToolResult
from harness_agent.runtime.paths import safe_conversation_id_part
from harness_agent.runtime.protocols import UserRuntime
from harness_agent.text import as_str
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


class FakeUserRuntime(UserRuntime):
    def __init__(
        self,
        *,
        files: dict[str, str | bytes] | None = None,
        agent_files: AgentFileSet | None = None,
        skills: Sequence[Skill] = (),
        shell_results: Sequence[RuntimeToolResult] = (),
        file_write_results: Sequence[RuntimeToolResult | Exception] = (),
    ) -> None:
        self._files = {} if files is None else files
        self._agent_files = agent_files
        self._skills = list(skills)
        self._shell_results = list(shell_results)
        self._file_write_results: list[RuntimeToolResult | Exception] = list(
            file_write_results
        )
        self.read_agent_files_calls: list[str] = []
        self.list_skills_calls: list[str] = []
        self.shell_exec_calls: list[ShellExecInput] = []
        self.file_write_calls: list[FileWriteInput] = []
        self.content_write_calls: list[tuple[str, bytes]] = []
        self._memory_files: dict[tuple[str, MemoryTarget], str] = {}
        self._session_logs: dict[tuple[str, str], list[str]] = {}

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        self.read_agent_files_calls.append(user_id)
        if self._agent_files is not None:
            base = self._agent_files
            return base.model_copy(
                update={
                    "user": self._memory_files.get((user_id, "user"), base.user),
                    "memory": self._memory_files.get((user_id, "memory"), base.memory),
                }
            )
        required = [
            "/workspace/agent/SOUL.md",
            "/workspace/agent/AGENTS.md",
            "/workspace/agent/USER.md",
            "/workspace/agent/TOOLS.md",
        ]
        missing = [path for path in required if path not in self._files]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        memory_raw = self._files.get("/workspace/agent/MEMORY.md", "")
        return AgentFileSet(
            soul=as_str(self._files["/workspace/agent/SOUL.md"]),
            agents=as_str(self._files["/workspace/agent/AGENTS.md"]),
            user=as_str(self._files["/workspace/agent/USER.md"]),
            tools=as_str(self._files["/workspace/agent/TOOLS.md"]),
            memory=as_str(memory_raw),
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
        max_bytes: int | None,
    ) -> RuntimeFileRead:
        if path not in self._files:
            return RuntimeFileRead(
                result=RuntimeToolResult(
                    stderr=f"No such file: {path}\n",
                    exit_code=1,
                )
            )
        content = self._files[path]
        data = content.encode("latin1") if isinstance(content, str) else content
        if max_bytes is None:
            return RuntimeFileRead(file=WorkspaceFile(path=path, content=data))
        return RuntimeFileRead(file=WorkspaceFile(path=path, content=data[:max_bytes]))

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
        return RuntimeToolResult(stdout=as_str(self._files[input.path]))

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        self.file_write_calls.append(input)
        if self._file_write_results:
            outcome = self._file_write_results.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        self._files[input.path] = input.content
        return RuntimeToolResult()

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        content = as_str(self._files[input.path])
        count = -1 if input.replace_all else 1
        self._files[input.path] = content.replace(input.old, input.new, count)
        return RuntimeToolResult()

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        content = as_str(self._files[input.path])
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
            for line_no, line in enumerate(as_str(content).splitlines(), start=1):
                if input.pattern in line:
                    lines.append(f"{path}:{line_no}:{line}")
        return RuntimeToolResult(stdout="\n".join(lines))

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        return RuntimeToolResult(
            stdout="\n".join(path for path in sorted(self._files) if path.startswith(input.path))
        )

    async def read_memory_file(self, user_id: str, target: MemoryTarget) -> str:
        return self._memory_files.get((user_id, target), "")

    async def write_memory_file(
        self,
        user_id: str,
        target: MemoryTarget,
        content: str,
    ) -> None:
        self._memory_files[(user_id, target)] = content

    async def append_session_log(
        self,
        user_id: str,
        conversation_id: str,
        line: str,
    ) -> None:
        safe = safe_conversation_id_part(conversation_id)
        if not safe:
            return
        self._session_logs.setdefault((user_id, safe), []).append(line)

    async def list_session_logs(self, user_id: str) -> list[str]:
        import urllib.parse

        return sorted(
            urllib.parse.unquote(safe_id)
            for stored_user, safe_id in self._session_logs
            if stored_user == user_id
        )

    async def read_session_log(self, user_id: str, conversation_id: str) -> str:
        safe = safe_conversation_id_part(conversation_id)
        lines = self._session_logs.get((user_id, safe), [])
        return "\n".join(lines)
