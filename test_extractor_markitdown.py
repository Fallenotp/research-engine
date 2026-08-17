import builtins
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from research_engine import extractor
from research_engine.schema import ExtractionMethod


def _write_sample_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_markitdown_convert_pdf_returns_markdown(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    expected_text = "MarkItDown sample PDF text"
    _write_sample_pdf(pdf_path, expected_text)

    markdown = extractor._markitdown_convert(str(pdf_path))

    assert markdown is not None
    assert expected_text in markdown


def test_chain_uses_markitdown_when_pdf_rungs_fail(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    expected_text = "MarkItDown fallback PDF text " * 12
    _write_sample_pdf(pdf_path, expected_text)

    with patch.object(extractor, "_pdf_docling", return_value={}) as docling_mock, patch.object(
        extractor,
        "_pdf_pymupdf",
        return_value={},
    ) as pymupdf_mock:
        out = extractor.extract_clean_text(
            str(pdf_path),
            seen_urls_path=tmp_path / "seen.txt",
        )

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.MARKITDOWN.value
    assert "MarkItDown fallback PDF text" in out["char_text_preview"]
    assert expected_text.strip() in Path(out["raw_text_path"]).read_text(encoding="utf-8")
    docling_mock.assert_called_once()
    pymupdf_mock.assert_called_once()


def test_pdf_default_uses_pymupdf_before_docling(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_sample_pdf(pdf_path, "PyMuPDF first PDF text")
    calls: list[str] = []

    def fake_pymupdf(*args, **kwargs):
        calls.append("pymupdf")
        return {}

    def fake_docling(*args, **kwargs):
        calls.append("docling")
        return {"title": "Docling", "text": "docling fallback text " * 20}

    with patch.object(extractor, "_pdf_pymupdf", side_effect=fake_pymupdf), patch.object(
        extractor,
        "_pdf_docling",
        side_effect=fake_docling,
    ):
        out = extractor.extract_clean_text(
            str(pdf_path),
            seen_urls_path=tmp_path / "seen.txt",
        )

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.DOCLING.value
    assert calls == ["pymupdf", "docling"]


def test_pdf_without_pymupdf_extra_raises_actionable_error(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_sample_pdf(pdf_path, "PyMuPDF missing PDF text")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fitz":
            raise ModuleNotFoundError("No module named 'fitz'")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match=r"pip install .*pdf"):
            extractor.extract_clean_text(
                str(pdf_path),
                seen_urls_path=tmp_path / "seen.txt",
            )


def test_pdf_helper_none_does_not_fall_through_to_web_ladder(monkeypatch) -> None:
    web_ladder_called = False

    def fake_web_ladder(*args, **kwargs):
        nonlocal web_ladder_called
        web_ladder_called = True
        return {"title": "wrong", "text": "wrong " * 100}

    monkeypatch.setattr(extractor, "_extract_pdf_or_document", lambda *args, **kwargs: None)
    monkeypatch.setattr(extractor, "_extract_github_repo", lambda *args, **kwargs: None)
    monkeypatch.setattr(extractor, "_extract_web_ladder", fake_web_ladder)

    result = extractor.extract_clean_text("https://example.com/paper.pdf")

    assert result is None
    assert web_ladder_called is False


def test_missing_pymupdf_falls_through_to_docling(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_sample_pdf(pdf_path, "PyMuPDF missing PDF text")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fitz":
            raise ModuleNotFoundError("No module named 'fitz'")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import), patch.object(
        extractor,
        "_pdf_docling",
        return_value={"title": "Docling", "text": "docling fallback text " * 20},
    ):
        out = extractor.extract_clean_text(
            str(pdf_path),
            seen_urls_path=tmp_path / "seen.txt",
        )

    assert out is not None
    assert out["extraction_method"] == ExtractionMethod.DOCLING.value
