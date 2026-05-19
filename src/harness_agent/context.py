from pydantic import BaseModel


class AgentFileSet(BaseModel):
    soul: str = ""
    agents: str = ""
    user: str = ""
    tools: str = ""


class Skill(BaseModel):
    name: str
    description: str
    body: str

    def render_for_prompt(self) -> str:
        return f"Skill: {self.name}\n{self.description}\n{self.body}"


class AgentContext(BaseModel):
    system: str
    skills: list[Skill]


class ContextBuilder:
    def __init__(self, runtime: UserContextRuntime) -> None:
        self._runtime = runtime

    async def build(self, user_id: str) -> AgentContext:
        files = await self._runtime.read_agent_files(user_id)
        skills = await self._runtime.list_skills(user_id)
        tool_contract = "\n".join(
            [
                "Tools:",
                "- shell.exec runs commands in /workspace.",
                "- shell.spawn starts long-running commands in /workspace.",
                "- shell.read reads spawned command output.",
                "- shell.kill stops spawned commands.",
                "- file.read reads files under /workspace.",
                "- file.write writes files under /workspace.",
                "- file.edit replaces exact text in one file.",
                "- file.multi_edit applies exact replacements to one file.",
                "- file.glob finds files under /workspace.",
                "- file.grep searches files under /workspace.",
                "- file.list lists paths under /workspace.",
                "- web.fetch fetches an HTTP/HTTPS URL and answers a specific extraction prompt.",
                "- task.* manages the conversation checklist.",
                "- schedule.once schedules one future synthetic user message.",
                "- schedule.cron schedules recurring synthetic user messages.",
                "- schedule.list and schedule.cancel manage scheduled messages.",
                "- skill.* reads enabled markdown skills.",
                "- agent.* runs sub-agents that can use workspace file and shell tools.",
            ]
        )
        blocks = [
            files.soul,
            files.agents,
            files.user,
            files.tools,
            *[skill.render_for_prompt() for skill in skills],
            tool_contract,
            "Runtime: tools run in the user's workspace. Container details are not part of the model context.",
            "Incoming Telegram files are saved under /workspace/content. Use file.read for saved text files. Images are also attached to multimodal user messages when available.",
            "Do not use sleep, wait, or long-running bash commands to schedule future work. Use schedule.once or schedule.cron.",
        ]
        return AgentContext(
            system="\n\n".join(block for block in blocks if block),
            skills=skills,
        )


class UserContextRuntime:
    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        raise NotImplementedError

    async def list_skills(self, user_id: str) -> list[Skill]:
        raise NotImplementedError
