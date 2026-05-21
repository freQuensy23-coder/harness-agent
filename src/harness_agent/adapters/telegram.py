import base64
import re
from typing import Literal

from aiogram import Bot, Dispatcher, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import Message

from harness_agent.bus import EventBus
from harness_agent.events import (
    AgentGenerationStarted,
    AssistantTextProduced,
    InboundAttachment,
    ReplyTarget,
    TelegramTextReceived,
)
from harness_agent.handlers import EventBatch


def event_from_aiogram_message(
    message: Message,
    *,
    attachments: list[InboundAttachment] | None = None,
) -> TelegramTextReceived:
    if message.from_user is None:
        raise ValueError("Telegram message has no sender")
    text = message.text or message.caption or ""
    return TelegramTextReceived(
        telegram_user_id=message.from_user.id,
        telegram_chat_id=message.chat.id,
        telegram_message_id=message.message_id,
        text=text,
        attachments=[] if attachments is None else attachments,
    )


async def event_from_aiogram_message_with_files(
    message: Message,
    bot: Bot,
) -> TelegramTextReceived:
    attachments: list[InboundAttachment] = []
    if message.photo:
        photo = message.photo[-1]
        file_name = f"photo_{photo.file_unique_id}.jpg"
        attachments.append(
            await download_attachment(
                bot=bot,
                file_id=photo.file_id,
                file_unique_id=photo.file_unique_id,
                kind="image",
                file_name=file_name,
                mime_type="image/jpeg",
                size_bytes=photo.file_size or 0,
                chat_id=message.chat.id,
                message_id=message.message_id,
            )
        )
    if message.document:
        document = message.document
        mime_type = document.mime_type
        kind: Literal["image", "file"] = (
            "image" if mime_type is not None and mime_type.startswith("image/") else "file"
        )
        attachments.append(
            await download_attachment(
                bot=bot,
                file_id=document.file_id,
                file_unique_id=document.file_unique_id,
                kind=kind,
                file_name=document.file_name or f"document_{document.file_unique_id}",
                mime_type=mime_type,
                size_bytes=document.file_size or 0,
                chat_id=message.chat.id,
                message_id=message.message_id,
            )
        )
    return event_from_aiogram_message(message, attachments=attachments)


async def download_attachment(
    *,
    bot: Bot,
    file_id: str,
    file_unique_id: str,
    kind: Literal["image", "file"],
    file_name: str,
    mime_type: str | None,
    size_bytes: int,
    chat_id: int,
    message_id: int,
) -> InboundAttachment:
    telegram_file = await bot.get_file(file_id)
    if telegram_file.file_path is None:
        raise RuntimeError(f"Telegram file has no file_path: {file_id}")
    content = await bot.download_file(telegram_file.file_path)
    if content is None:
        raise RuntimeError(f"Telegram download_file returned no content: {file_id}")
    data = content.read()
    safe_name = safe_file_name(file_name)
    return InboundAttachment(
        kind=kind,
        file_name=safe_name,
        mime_type=mime_type,
        size_bytes=len(data) if size_bytes == 0 else size_bytes,
        workspace_path=f"/workspace/content/telegram/{chat_id}/{message_id}/{safe_name}",
        content_base64=base64.b64encode(data).decode("ascii"),
        source_id=file_unique_id,
    )


def safe_file_name(file_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", file_name).strip("._")
    if sanitized:
        return sanitized
    return "attachment"


class AiogramTelegramAdapter:
    def __init__(self, *, token: str, bus: EventBus) -> None:
        self._bot = Bot(token=token)
        self._dispatcher = Dispatcher()
        self._router = Router()
        self._bus = bus
        self._router.message(CommandStart())(self._on_start)
        self._router.message()(self._on_message)
        self._dispatcher.include_router(self._router)

    async def start_polling(self) -> None:
        await self._dispatcher.start_polling(self._bot)  # pyright: ignore[reportUnknownMemberType]

    def register_outbound_handlers(self) -> None:
        self._bus.subscribe(AgentGenerationStarted, self.handle_generation_started)
        self._bus.subscribe(AssistantTextProduced, self.handle_assistant_text)

    async def handle_generation_started(self, event: AgentGenerationStarted) -> EventBatch:
        chat_id = telegram_chat_id(event.reply_target)
        if chat_id is None:
            return ()
        await self._bot.send_chat_action(chat_id=chat_id, action="typing")
        return ()

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        chat_id = telegram_chat_id(event.reply_target)
        if chat_id is None:
            return ()
        await self._send_markdown_or_plain(chat_id=chat_id, text=event.text)
        return ()

    async def send_assistant_text(self, event: AssistantTextProduced) -> None:
        chat_id = telegram_chat_id(event.reply_target)
        if chat_id is None:
            raise ValueError("Assistant text has no Telegram reply target")
        await self._send_markdown_or_plain(chat_id=chat_id, text=event.text)

    async def _send_markdown_or_plain(self, *, chat_id: int, text: str) -> None:
        try:
            await self._bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except TelegramBadRequest:
            await self._bot.send_message(chat_id=chat_id, text=text)

    async def _on_start(self, message: Message) -> None:
        await self._on_message(message)

    async def _on_message(self, message: Message) -> None:
        await self._bus.publish(await event_from_aiogram_message_with_files(message, self._bot))


def telegram_chat_id(reply_target: ReplyTarget | None) -> int | None:
    if reply_target is None:
        return None
    if reply_target.kind != "telegram":
        return None
    return reply_target.chat_id
