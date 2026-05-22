import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError


class McpServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    command: list[str]
    cwd: str = "/workspace"


def parse_mcp_server_yaml(text: str, *, path: str) -> McpServerConfig | None:
    """Parse one user MCP YAML. Returns None and logs a warning on failure so
    a single malformed file in /workspace/mcp/ cannot suppress the rest.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("Skipping MCP server {path}: invalid YAML: {exc}", path=path, exc=exc)
        return None
    try:
        return McpServerConfig.model_validate(data)
    except ValidationError as exc:
        logger.warning("Skipping MCP server {path}: invalid schema: {exc}", path=path, exc=exc)
        return None
