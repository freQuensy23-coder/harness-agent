"""Skill discovery from `/workspace/skills/<name>/SKILL.md` files.

Each SKILL.md has YAML frontmatter (name, description) followed by a
markdown body shown to the model. A find shells out once, then each
file is read individually through the shared text reader."""

from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath

from loguru import logger

from harness_agent.context import Skill
from harness_agent.runtime.models import DockerProcessResult
from harness_agent.runtime.skills import parse_skill_markdown


ExecInContainer = Callable[..., Awaitable[DockerProcessResult]]
ReadText = Callable[[str, str], Awaitable[str]]


class DockerSkills:
    """List skills installed in the user's workspace."""

    def __init__(
        self,
        *,
        exec_in_container: ExecInContainer,
        read_text: ReadText,
    ) -> None:
        self._exec = exec_in_container
        self._read_text = read_text

    async def list(self, user_id: str) -> list[Skill]:
        result = await self._exec(
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
            text = await self._read_text(user_id, path)
            skills.append(
                parse_skill_markdown(text, file_name=PurePosixPath(path).parent.name)
            )
        return skills
