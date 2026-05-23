import base64
import io
from types import SimpleNamespace
from typing import cast

import pytest

from harness_agent.adapters.telegram import (
    download_attachment,
    event_from_aiogram_message_with_files,
    safe_file_name,
)


class _FakeBot:
    """Inline fake aiogram Bot for `event_from_aiogram_message_with_files`.

    Maps file_id → (telegram-side file_path or None, downloaded bytes or None).
    Mirrors the two-call contract: `get_file(file_id)` then
    `download_file(file_path)`.
    """

    def __init__(self, files: dict[str, tuple[str | None, bytes | None]]) -> None:
        self._files = files
        self.get_file_calls: list[str] = []
        self.download_calls: list[str] = []

    async def get_file(self, file_id: str) -> SimpleNamespace:
        self.get_file_calls.append(file_id)
        file_path, _ = self._files[file_id]
        return SimpleNamespace(file_path=file_path)

    async def download_file(self, file_path: str) -> io.BytesIO | None:
        self.download_calls.append(file_path)
        for stored_path, payload in self._files.values():
            if stored_path == file_path:
                return None if payload is None else io.BytesIO(payload)
        raise AssertionError(f"unexpected download path: {file_path}")


def _message(
    *,
    photo: list[object] | None = None,
    document: object | None = None,
    text: str = "",
    caption: str | None = None,
    chat_id: int = 456,
    message_id: int = 999,
    user_id: int = 123,
) -> SimpleNamespace:
    return SimpleNamespace(
        photo=photo,
        document=document,
        text=text,
        caption=caption,
        message_id=message_id,
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=chat_id),
    )


@pytest.mark.asyncio
async def test_single_photosize_becomes_image_attachment_with_decoded_bytes() -> None:
    photo = SimpleNamespace(file_id="A", file_unique_id="UA", file_size=1024)
    payload = b"\xff\xd8\xff\xe0jpeg-bytes"
    bot = _FakeBot({"A": ("photos/a.jpg", payload)})

    event = await event_from_aiogram_message_with_files(
        cast(object, _message(photo=[photo])),  # type: ignore[arg-type]
        cast(object, bot),  # type: ignore[arg-type]
    )

    assert len(event.attachments) == 1
    attachment = event.attachments[0]
    assert attachment.kind == "image"
    assert attachment.file_name == "photo_UA.jpg"
    assert attachment.mime_type == "image/jpeg"
    assert attachment.size_bytes == 1024
    assert attachment.workspace_path == "/workspace/content/telegram/456/999/photo_UA.jpg"
    assert attachment.source_id == "UA"
    assert base64.b64decode(attachment.content_base64) == payload


@pytest.mark.asyncio
async def test_multiple_photosizes_select_the_last_one() -> None:
    small = SimpleNamespace(file_id="SMALL", file_unique_id="US", file_size=100)
    medium = SimpleNamespace(file_id="MED", file_unique_id="UM", file_size=400)
    large = SimpleNamespace(file_id="LARGE", file_unique_id="UL", file_size=1600)
    bot = _FakeBot(
        {
            "SMALL": ("photos/small.jpg", b"s"),
            "MED": ("photos/med.jpg", b"m"),
            "LARGE": ("photos/large.jpg", b"L" * 1600),
        }
    )

    event = await event_from_aiogram_message_with_files(
        cast(object, _message(photo=[small, medium, large])),  # type: ignore[arg-type]
        cast(object, bot),  # type: ignore[arg-type]
    )

    # aiogram orders PhotoSize ascending by resolution; the adapter must take
    # the last (highest-res) entry.
    assert bot.get_file_calls == ["LARGE"]
    assert event.attachments[0].file_name == "photo_UL.jpg"
    assert event.attachments[0].size_bytes == 1600


@pytest.mark.asyncio
async def test_document_with_image_mime_is_saved_as_image_kind() -> None:
    document = SimpleNamespace(
        file_id="D-IMG",
        file_unique_id="UDI",
        file_size=2048,
        mime_type="image/png",
        file_name="picture.png",
    )
    bot = _FakeBot({"D-IMG": ("docs/picture.png", b"\x89PNG\r\n\x1a\n")})

    event = await event_from_aiogram_message_with_files(
        cast(object, _message(document=document)),  # type: ignore[arg-type]
        cast(object, bot),  # type: ignore[arg-type]
    )

    attachment = event.attachments[0]
    assert attachment.kind == "image"
    assert attachment.mime_type == "image/png"
    assert attachment.file_name == "picture.png"


@pytest.mark.asyncio
async def test_document_with_non_image_mime_is_saved_as_file_kind() -> None:
    document = SimpleNamespace(
        file_id="D-PDF",
        file_unique_id="UDP",
        file_size=4096,
        mime_type="application/pdf",
        file_name="report.pdf",
    )
    bot = _FakeBot({"D-PDF": ("docs/report.pdf", b"%PDF-1.4")})

    event = await event_from_aiogram_message_with_files(
        cast(object, _message(document=document)),  # type: ignore[arg-type]
        cast(object, bot),  # type: ignore[arg-type]
    )

    attachment = event.attachments[0]
    assert attachment.kind == "file"
    assert attachment.mime_type == "application/pdf"
    assert attachment.file_name == "report.pdf"


@pytest.mark.asyncio
async def test_document_without_mime_type_defaults_to_file_kind() -> None:
    document = SimpleNamespace(
        file_id="D-RAW",
        file_unique_id="UDR",
        file_size=10,
        mime_type=None,
        file_name="raw.bin",
    )
    bot = _FakeBot({"D-RAW": ("docs/raw.bin", b"raw-bytes!")})

    event = await event_from_aiogram_message_with_files(
        cast(object, _message(document=document)),  # type: ignore[arg-type]
        cast(object, bot),  # type: ignore[arg-type]
    )

    assert event.attachments[0].kind == "file"
    assert event.attachments[0].mime_type is None


@pytest.mark.asyncio
async def test_download_raises_when_telegram_returns_no_file_path() -> None:
    bot = _FakeBot({"NOFP": (None, b"never-read")})

    with pytest.raises(RuntimeError, match=r"file_path"):
        await download_attachment(
            bot=cast(object, bot),  # type: ignore[arg-type]
            file_id="NOFP",
            file_unique_id="U",
            kind="image",
            file_name="x.jpg",
            mime_type="image/jpeg",
            size_bytes=10,
            chat_id=1,
            message_id=2,
        )

    # File path is None, so we never reach `download_file`.
    assert bot.download_calls == []


@pytest.mark.asyncio
async def test_download_raises_when_telegram_download_returns_none() -> None:
    bot = _FakeBot({"NOBYTES": ("path/x.jpg", None)})

    with pytest.raises(RuntimeError, match=r"no content"):
        await download_attachment(
            bot=cast(object, bot),  # type: ignore[arg-type]
            file_id="NOBYTES",
            file_unique_id="U",
            kind="image",
            file_name="x.jpg",
            mime_type="image/jpeg",
            size_bytes=10,
            chat_id=1,
            message_id=2,
        )


@pytest.mark.asyncio
async def test_size_bytes_falls_back_to_actual_length_when_telegram_reports_zero() -> None:
    photo = SimpleNamespace(file_id="P0", file_unique_id="UP0", file_size=0)
    payload = b"abcdef"
    bot = _FakeBot({"P0": ("photos/p0.jpg", payload)})

    event = await event_from_aiogram_message_with_files(
        cast(object, _message(photo=[photo])),  # type: ignore[arg-type]
        cast(object, bot),  # type: ignore[arg-type]
    )

    assert event.attachments[0].size_bytes == len(payload)


def test_safe_file_name_strips_path_separators_and_traversal() -> None:
    # Slashes, dots, and backslashes all get squashed to '_'.
    assert safe_file_name("../../../etc/passwd.png") == "etc_passwd.png"
    assert safe_file_name("a/b/c.png") == "a_b_c.png"
    assert safe_file_name(r"foo\\bar.txt") == "foo_bar.txt"


def test_safe_file_name_collapses_runs_of_unsafe_chars_to_single_underscore() -> None:
    assert safe_file_name("a$$$b") == "a_b"
    assert safe_file_name("hi there.png") == "hi_there.png"


def test_safe_file_name_falls_back_to_attachment_for_empty_or_junk() -> None:
    assert safe_file_name("") == "attachment"
    assert safe_file_name(".") == "attachment"
    assert safe_file_name("...") == "attachment"
    assert safe_file_name("////") == "attachment"
    assert safe_file_name("___") == "attachment"


def test_safe_file_name_preserves_already_safe_names() -> None:
    assert safe_file_name("report-2026.pdf") == "report-2026.pdf"
    assert safe_file_name("photo_UA.jpg") == "photo_UA.jpg"
