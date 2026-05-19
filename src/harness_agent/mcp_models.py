from pydantic import BaseModel, ConfigDict


class McpServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    command: list[str]
    cwd: str = "/workspace"
