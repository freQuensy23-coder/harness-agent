from uuid import uuid4

from harness_agent.events import CliTextReceived


def event_from_cli_send(
    *,
    text: str,
    user_id: str,
    conversation_id: str | None,
) -> CliTextReceived:
    resolved_conversation_id = f"cli:{user_id}" if conversation_id is None else conversation_id
    return CliTextReceived(
        cli_user_id=user_id,
        conversation_id=resolved_conversation_id,
        request_id=uuid4().hex,
        text=text,
    )
