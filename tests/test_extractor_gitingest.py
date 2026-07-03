from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from research_engine import extractor


def _payload(char_count: int, *, title: str = "Example Domain") -> dict[str, str | None]:
    return extractor._payload(title, "A" * char_count, "Tester", "2026-05-12")


def test_extract_clean_text_uses_gitingest_for_github_repo() -> None:
    fake_content = "CONTENT-XYZ " * 30

    with patch.object(extractor, "_is_pdf", return_value=False), patch(
        "gitingest.ingest",
        return_value=("SUMMARY", "TREE", fake_content),
    ) as ingest_mock, patch.object(
        extractor, "_cloudflare_markdown_preflight",
        side_effect=AssertionError("HTML cascade should not run"),
    ):
        result = extractor.extract_clean_text("https://github.com/octo/repo")

    assert result is not None
    assert result["extraction_method"] == "gitingest"
    cached_text = Path(result["raw_text_path"]).read_text(encoding="utf-8")
    assert "CONTENT-XYZ" in cached_text
    ingest_mock.assert_called_once()
    call_args, call_kwargs = ingest_mock.call_args
    assert call_args[0] == "https://github.com/octo/repo"
    assert call_kwargs.get("branch") is None
    assert call_kwargs.get("exclude_patterns") == extractor._GITINGEST_EXCLUDE_PATTERNS


def test_is_github_repo_url_rejects_subresources_and_non_repos() -> None:
    assert extractor._is_github_repo_url("https://github.com/octo/repo/issues/3") is False
    assert extractor._is_github_repo_url("https://github.com/octo/repo/blob/main/x.py") is False
    assert extractor._is_github_repo_url("https://gist.github.com/octo/abc123") is False
    assert extractor._is_github_repo_url("https://github.com/octo") is False


def test_is_github_repo_url_accepts_repo_root_and_tree_branch() -> None:
    assert extractor._is_github_repo_url("https://github.com/octo/repo") is True
    assert extractor._is_github_repo_url("https://github.com/octo/repo/") is True
    assert extractor._is_github_repo_url("https://github.com/octo/repo.git") is True
    assert extractor._is_github_repo_url("https://github.com/octo/repo/tree/main") is True


def test_extract_clean_text_falls_back_when_gitingest_raises() -> None:
    with patch.object(extractor, "_is_pdf", return_value=False), patch(
        "gitingest.ingest",
        side_effect=RuntimeError("network down"),
    ), patch.object(
        extractor, "_crawl4ai", return_value=_payload(2400)
    ) as crawl4ai_mock:
        result = extractor.extract_clean_text("https://github.com/octo/repo")

    assert result is not None
    assert result["extraction_method"] != "gitingest"
    assert result["extraction_method"] == "crawl4ai"
    crawl4ai_mock.assert_called_once_with("https://github.com/octo/repo")


def test_extract_clean_text_falls_back_when_gitingest_import_unavailable() -> None:
    with patch.object(extractor, "_is_pdf", return_value=False), patch(
        "gitingest.ingest",
        side_effect=ImportError("gitingest not installed"),
    ), patch.object(
        extractor, "_crawl4ai", return_value=_payload(2400)
    ) as crawl4ai_mock:
        result = extractor.extract_clean_text("https://github.com/octo/repo")

    assert result is not None
    assert result["extraction_method"] != "gitingest"
    assert result["extraction_method"] == "crawl4ai"
    crawl4ai_mock.assert_called_once_with("https://github.com/octo/repo")


def test_gitingest_char_cap_truncates_cached_text() -> None:
    huge_content = "X" * (extractor._GITINGEST_CHAR_CAP + 50_000)

    with patch.object(extractor, "_is_pdf", return_value=False), patch(
        "gitingest.ingest",
        return_value=("SUMMARY", "TREE", huge_content),
    ), patch.object(
        extractor, "_cloudflare_markdown_preflight",
        side_effect=AssertionError("HTML cascade should not run"),
    ):
        result = extractor.extract_clean_text("https://github.com/octo/repo")

    assert result is not None
    assert result["extraction_method"] == "gitingest"
    cached_text = Path(result["raw_text_path"]).read_text(encoding="utf-8")
    marker = f"[truncated at {extractor._GITINGEST_CHAR_CAP} chars]"
    assert marker in cached_text
    title_overhead = len("octo/repo — GitHub repo (gitingest)") + 10
    assert len(cached_text) <= extractor._GITINGEST_CHAR_CAP + len(marker) + title_overhead


def test_extract_clean_text_passes_tree_branch_to_gitingest() -> None:
    with patch.object(extractor, "_is_pdf", return_value=False), patch(
        "gitingest.ingest",
        return_value=("SUMMARY", "TREE", "CONTENT " * 30),
    ) as ingest_mock, patch.object(
        extractor, "_cloudflare_markdown_preflight",
        side_effect=AssertionError("HTML cascade should not run"),
    ):
        result = extractor.extract_clean_text("https://github.com/octo/repo/tree/feat/x")

    assert result is not None
    assert result["extraction_method"] == "gitingest"
    _, call_kwargs = ingest_mock.call_args
    assert call_kwargs.get("branch") == "feat/x"


def test_gitingest_timeout_does_not_hang(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_ingest(*_args, **_kwargs):
        time.sleep(3)
        return ("SUMMARY", "TREE", "CONTENT " * 30)

    monkeypatch.setattr(extractor, "_GITINGEST_TIMEOUT_S", 1)

    with patch.object(extractor, "_is_pdf", return_value=False), patch(
        "gitingest.ingest",
        side_effect=slow_ingest,
    ), patch.object(
        extractor, "_crawl4ai", return_value=_payload(2400)
    ):
        started = time.monotonic()
        result = extractor.extract_clean_text("https://github.com/octo/repo")
        elapsed = time.monotonic() - started

    assert elapsed < 2.5
    assert result is not None
    assert result["extraction_method"] != "gitingest"
    assert result["extraction_method"] == "crawl4ai"
