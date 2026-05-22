from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from harness_agent.mcp_models import McpServerConfig


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8080


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path = Path("./data/harness.sqlite3")


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = "replace-me"
    model: str = "z-ai/glm-5v-turbo"


class TelegramConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    bot_token: str | None = None


class DockerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = "python:3.14-slim"
    container_prefix: str = "harness"
    network: str | None = None
    memory: str | None = None
    cpus: str | None = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docker: DockerConfig = Field(default_factory=DockerConfig)


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poll_seconds: float = 5


class McpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: list[McpServerConfig] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    nudge_interval: int = Field(default=10, ge=1)
    review_max_iterations: int = Field(default=5, ge=1)


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


def load_config(path: Path) -> HarnessConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return HarnessConfig.model_validate(data)
