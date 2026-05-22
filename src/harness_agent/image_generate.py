import base64
from typing import Any

import httpx
from pydantic import BaseModel

from harness_agent.tools import ImageGenerateInput


class GeneratedImage(BaseModel):
    mime_type: str
    data: bytes
    text: str | None = None


class ImageGenerator:
    async def generate(self, input: ImageGenerateInput) -> GeneratedImage:
        raise NotImplementedError


class ImageGenerationError(RuntimeError):
    pass


class GeminiImageGenerator(ImageGenerator):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        model: str = "gemini-2.5-flash-image",
        service_tier: str = "flex",
        timeout_seconds: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._service_tier = service_tier
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def generate(self, input: ImageGenerateInput) -> GeneratedImage:
        if not self._api_key or self._api_key == "replace-me":
            raise ImageGenerationError(
                "image.generate is not configured: set image.api_key in harness.yaml"
            )
        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": input.prompt}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": input.aspect_ratio},
            },
            "serviceTier": self._service_tier,
        }
        url = f"{self._base_url}/models/{self._model}:generateContent"
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            headers={
                "content-type": "application/json",
                "x-goog-api-key": self._api_key,
            },
        ) as client:
            response = await client.post(url, json=body)
        if not response.is_success:
            raise ImageGenerationError(
                f"Gemini API returned HTTP {response.status_code}: {response.text}"
            )
        return _parse_gemini_response(response.json())


def _parse_gemini_response(payload: dict[str, Any]) -> GeneratedImage:
    candidates: list[dict[str, Any]] = list(payload.get("candidates") or [])
    if not candidates:
        prompt_feedback = payload.get("promptFeedback")
        raise ImageGenerationError(
            f"Gemini returned no candidates: promptFeedback={prompt_feedback}"
        )
    content: dict[str, Any] = candidates[0].get("content") or {}
    parts: list[dict[str, Any]] = list(content.get("parts") or [])
    image_part: dict[str, Any] | None = None
    text_chunks: list[str] = []
    for part in parts:
        inline_raw: object = part.get("inlineData") or part.get("inline_data")
        inline = _coerce_dict(inline_raw)
        if inline:
            mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
            if mime.startswith("image/"):
                image_part = inline
                continue
        text_value = part.get("text")
        if isinstance(text_value, str) and text_value:
            text_chunks.append(text_value)
    if image_part is None:
        joined = "\n".join(text_chunks) if text_chunks else ""
        raise ImageGenerationError(
            f"Gemini returned no image part. Text response: {joined!r}"
        )
    mime_type = str(image_part.get("mimeType") or image_part.get("mime_type") or "")
    data_b64 = str(image_part.get("data") or "")
    return GeneratedImage(
        mime_type=mime_type,
        data=base64.b64decode(data_b64),
        text="\n".join(text_chunks) if text_chunks else None,
    )


def _coerce_dict(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}  # type: ignore[reportUnknownVariableType]
