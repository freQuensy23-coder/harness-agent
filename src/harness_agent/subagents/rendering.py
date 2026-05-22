"""JSON renderers for agent.* tool results consumed by the LLM."""

from harness_agent.subagents.models import SubAgentRecord


def render_sub_agent_record(record: SubAgentRecord) -> str:
    return record.model_dump_json(indent=2)


def render_sub_agent_records(records: list[SubAgentRecord]) -> str:
    return "[\n" + ",\n".join(record.model_dump_json(indent=2) for record in records) + "\n]"
