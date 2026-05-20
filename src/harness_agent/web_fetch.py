import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel

from harness_agent.runtime import RuntimeToolResult
from harness_agent.tools import WebFetchInput


class WebFetchedPage(BaseModel):
    url: str
    markdown: str


class WebFetcher:
    async def fetch_markdown(self, input: WebFetchInput) -> WebFetchedPage:
        raise NotImplementedError

    async def fetch(self, input: WebFetchInput) -> RuntimeToolResult:
        page = await self.fetch_markdown(input)
        return RuntimeToolResult(stdout=page.markdown)


class HttpxWebFetcher(WebFetcher):
    async def fetch_markdown(self, input: WebFetchInput) -> WebFetchedPage:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=20,
            headers={
                "User-Agent": "Claude-User/1.0 Harness-Agent-WebFetch/0.1",
                "Accept": "text/markdown, text/plain;q=0.9, text/html;q=0.8, */*;q=0.5",
            },
        ) as client:
            response = await client.get(input.url)
        if not response.is_success:
            raise RuntimeError(f"HTTP {response.status_code}")
        text = response.text
        content_type = response.headers.get("content-type", "")
        markdown = (
            html_to_markdown(text, base_url=str(response.url))
            if _is_html(content_type, text)
            else text
        )
        return WebFetchedPage(url=str(response.url), markdown=markdown)


def html_to_markdown(html: str, *, base_url: str) -> str:
    parser = _MarkdownExtractor(base_url=base_url)
    parser.feed(html)
    parser.close()
    return parser.markdown()


def _is_html(content_type: str, text: str) -> bool:
    return "html" in content_type.lower() or bool(
        re.search(r"<\s*(html|body|article|main|p|div|h[1-6])\b", text, re.I)
    )


class _MarkdownExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside"}
    _BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "body",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "form",
        "hr",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }

    def __init__(self, *, base_url: str) -> None:
        super().__init__(convert_charrefs=False)
        self._base_url = base_url
        self._parts: list[str] = []
        self._skip_depth = 0
        self._links: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        attrs_by_name = {name.lower(): value for name, value in attrs if value is not None}
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._break()
            self._parts.append(f"{'#' * int(tag[1])} ")
            return
        if tag == "br":
            self._parts.append("\n")
            return
        if tag == "li":
            self._line_break()
            self._parts.append("- ")
            return
        if tag == "a":
            self._links.append((attrs_by_name.get("href", ""), []))
            return
        if tag in self._BLOCK_TAGS:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._links:
            href, text_parts = self._links.pop()
            text = " ".join(part for part in text_parts if part).strip()
            if text:
                if href:
                    self._append_text(f"[{text}]({urljoin(self._base_url, href)})")
                else:
                    self._append_text(text)
            return
        if tag == "li":
            self._line_break()
            return
        if tag in self._BLOCK_TAGS or tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(unescape(data).split())
        if not text:
            return
        if self._links:
            self._links[-1][1].append(text)
            return
        self._append_text(text)

    def markdown(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _append_text(self, text: str) -> None:
        if (
            self._parts
            and not self._parts[-1].endswith((" ", "\n"))
            and not text.startswith((".", ",", ":", ";", "!", "?", ")", "]"))
        ):
            self._parts.append(" ")
        self._parts.append(text)

    def _line_break(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    def _break(self) -> None:
        if not self._parts:
            return
        current = "".join(self._parts)
        if current.endswith("\n\n"):
            return
        if current.endswith("\n"):
            self._parts.append("\n")
            return
        self._parts.append("\n\n")
