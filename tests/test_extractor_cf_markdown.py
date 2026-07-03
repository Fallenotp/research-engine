from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from research_engine import extractor


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def _payload(char_count: int, *, title: str = "Example Domain") -> dict[str, str | None]:
    return extractor._payload(title, "A" * char_count, "Tester", "2026-05-12")


def test_cloudflare_markdown_preflight_remains_importable() -> None:
    markdown_body = "\n".join(
        [
            "---",
            "title: Example Domain",
            "---",
            "# Navigation",
            "Home | Pricing | Login",
            "",
            "# Example Domain",
            "This is the actual article body. " * 20,
        ]
    )
    cleaned_markdown = "Example Domain\n\n" + ("This is the actual article body. " * 20)
    response = _Response(
        text=markdown_body,
        headers={
            "content-type": "text/markdown; charset=utf-8",
            "x-markdown-tokens": "321",
        },
    )

    with patch.object(extractor.requests, "get", return_value=response) as get_mock, patch.object(
        extractor.trafilatura, "extract", return_value=cleaned_markdown
    ) as extract_mock:
        result = extractor._cloudflare_markdown_preflight("https://example.com/article")

    assert result is not None
    payload, extra = result
    assert extra["cf_markdown_tokens"] == "321"
    assert "Home | Pricing | Login" not in (payload["text"] or "")
    assert cleaned_markdown.strip() in (payload["text"] or "")
    extract_mock.assert_called_once_with(markdown_body)
    assert get_mock.call_count == 1
    assert get_mock.call_args.kwargs["headers"]["Accept"] == "text/markdown"


def test_extract_clean_text_falls_through_when_preflight_is_not_markdown() -> None:
    response = _Response(
        text="<html><body>Not markdown</body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
    )

    with patch.object(extractor.requests, "get", return_value=response) as get_mock, patch.object(
        extractor.trafilatura,
        "extract",
        side_effect=AssertionError("markdown cleanup should not run"),
    ):
        result = extractor._cloudflare_markdown_preflight("https://example.com/article")

    assert result is None
    assert get_mock.call_count == 1


def test_extract_clean_text_skips_cloudflare_markdown_auto_chain(tmp_path) -> None:
    with patch.object(extractor, "_is_pdf", return_value=False), patch.object(
        extractor,
        "_cloudflare_markdown_preflight",
        side_effect=AssertionError("cloudflare-md is not in the web ladder"),
    ), patch.object(
        extractor,
        "_crawl4ai",
        return_value=_payload(2400),
    ):
        result = extractor.extract_clean_text(
            "https://example.com/article",
            seen_urls_path=tmp_path / "seen.txt",
        )

    assert result is not None
    assert result["extraction_method"] == "crawl4ai"
    assert Path(result["raw_text_path"]).exists()
