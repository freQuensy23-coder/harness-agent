from pydantic import BaseModel, Field

from harness_agent.content import WorkspaceFile


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


class RuntimeFileRead(BaseModel):
    file: WorkspaceFile | None = None
    result: RuntimeToolResult = Field(default_factory=RuntimeToolResult)


class SpawnedProcessRecord(BaseModel):
    """Persisted `shell.spawn` metadata.

    Stdout/stderr offsets are byte positions in container output files. `shell.read`
    starts at those offsets, returns new bytes, then stores the next offsets so a
    core restart does not replay already-delivered output.
    """

    process_id: str
    user_id: str
    container_name: str
    command: str
    cwd: str
    base_path: str
    stdout_path: str
    stderr_path: str
    pid_path: str
    exit_code_path: str
    stdout_offset: int = 0
    stderr_offset: int = 0
