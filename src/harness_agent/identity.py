from pydantic import BaseModel


class ResolvedIdentity(BaseModel):
    user_id: str
    conversation_id: str


class StaticIdentityResolver:
    async def resolve_telegram(
        self,
        *,
        telegram_user_id: int,
        telegram_chat_id: int,
    ) -> ResolvedIdentity:
        return ResolvedIdentity(
            user_id=f"u:{telegram_user_id}",
            conversation_id=f"tg:{telegram_chat_id}",
        )

    async def resolve_cli(
        self,
        *,
        cli_user_id: str,
        conversation_id: str,
    ) -> ResolvedIdentity:
        return ResolvedIdentity(
            user_id=f"u:{cli_user_id}",
            conversation_id=conversation_id,
        )
