from pydantic import BaseModel

from harness_agent.tools import ToolSpec


class AgentFileSet(BaseModel):
    soul: str = ""
    agents: str = ""
    user: str = ""
    tools: str = ""
    memory: str = ""


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
        blocks = [
            files.soul,
            files.agents,
            files.user,
            files.memory,
            files.tools,
            *[skill.render_for_prompt() for skill in skills],
            "Runtime: tools run in the user's workspace. Container details are not part of the model context.",
            "Incoming Telegram files are saved under /workspace/content. Use file.read for saved text files. Images are also attached to multimodal user messages when available.",
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


def system_prompt_with_tools(base_system: str, tools: list[ToolSpec]) -> str:
    blocks = [base_system, render_tool_contract(tools), render_tool_guidance(tools)]
    return "\n\n".join(block for block in blocks if block)


def render_tool_contract(tools: list[ToolSpec]) -> str:
    names = {tool.name for tool in tools}
    lines = ["Tools:"]
    if "shell.exec" in names:
        lines.append("- shell.exec runs commands in /workspace.")
    if "shell.spawn" in names:
        lines.append("- shell.spawn starts long-running commands in /workspace.")
    if "shell.read" in names:
        lines.append("- shell.read reads spawned command output.")
    if "shell.kill" in names:
        lines.append("- shell.kill stops spawned commands.")
    if "file.read" in names:
        lines.append("- file.read reads files under /workspace.")
    if "file.write" in names:
        lines.append("- file.write writes files under /workspace.")
    if "file.edit" in names:
        lines.append("- file.edit replaces exact text in one file.")
    if "file.multi_edit" in names:
        lines.append("- file.multi_edit applies exact replacements to one file.")
    if "file.glob" in names:
        lines.append("- file.glob finds files under /workspace.")
    if "file.grep" in names:
        lines.append("- file.grep searches files under /workspace.")
    if "file.list" in names:
        lines.append("- file.list lists paths under /workspace.")
    if "web.fetch" in names:
        lines.append("- web.fetch fetches HTTP/HTTPS text.")
    if "image.generate" in names:
        lines.append(
            "- image.generate starts an async Gemini (Nano Banana, flex tier) image render and returns image_id."
        )
    if "image.status" in names:
        lines.append(
            "- image.status returns the job status; on completion the rendered image is attached to the next turn."
        )
    if any(name.startswith("task.") for name in names):
        lines.append("- task.* manages the conversation checklist.")
    if "schedule.once" in names:
        lines.append("- schedule.once schedules one future synthetic user message.")
    if "schedule.cron" in names:
        lines.append("- schedule.cron schedules recurring synthetic user messages.")
    if "schedule.list" in names or "schedule.cancel" in names:
        lines.append("- schedule.list and schedule.cancel manage scheduled messages.")
    if any(name.startswith("skill.") for name in names):
        lines.append("- skill.* reads enabled markdown skills.")
    if "memory" in names:
        lines.append(
            "- memory writes durable notes the agent should remember across sessions (action: add/replace/remove, target: memory/user)."
        )
    if "session.search" in names:
        lines.append("- session.search recalls focused summaries of past conversations.")
    if any(name.startswith("agent.") for name in names):
        lines.append(
            "- agent.* runs sub-agents that can use workspace, web, task, schedule, skill, and MCP tools but cannot spawn further sub-agents."
        )
    if any(name.startswith("mcp.") for name in names):
        lines.append("- mcp.* calls configured MCP tools.")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def render_tool_guidance(tools: list[ToolSpec]) -> str:
    names = {tool.name for tool in tools}
    guidance: list[str] = []
    if "memory" in names:
        memory_guidance = (
            "Persistent memory: USER.md holds what you know about the user "
            "(role, preferences, communication style, recurring corrections). "
            "MEMORY.md holds your own durable notes about the environment, "
            "conventions, tool quirks. Write through the `memory` tool. "
            "Save declarative facts, not imperatives: "
            "'User prefers concise responses' yes; 'Always be concise' no - "
            "imperatives re-injected into future sessions act as directives "
            "and can override the user's actual current request. "
            "Do not save task progress, outcomes, or completed-work logs to memory."
        )
        if "session.search" in names:
            memory_guidance += " Use session.search to recall those from past transcripts."
        guidance.append(memory_guidance)
    if "schedule.once" in names or "schedule.cron" in names:
        guidance.append(
            "Do not use sleep, wait, or long-running bash commands to schedule future work. "
            "Use schedule.once or schedule.cron."
        )
    return "\n\n".join(guidance)
