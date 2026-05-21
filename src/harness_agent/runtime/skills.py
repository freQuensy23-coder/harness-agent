import yaml
from pydantic import BaseModel

from harness_agent.context import Skill


class SkillFrontmatter(BaseModel):
    name: str
    description: str


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
