from __future__ import annotations

from pathlib import Path

from research_engine import extractor


def _record_from_rung(
    monkeypatch, tmp_path: Path, text: str, tier: extractor.SourceTier
) -> dict:
    monkeypatch.setattr(extractor, "CACHE_DIR", tmp_path / "cache", raising=False)
    monkeypatch.setattr(extractor, "_is_document_source", lambda *_args: False)
    monkeypatch.setattr(extractor, "_extract_github_repo", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(extractor, "_extract_apify_route", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        extractor,
        "_extract_web_ladder",
        lambda url, **kwargs: extractor._finalize_record(
            url,
            "jina",
            {
                "title": "Example",
                "text": text,
                "author": None,
                "published_date": None,
            },
            tier=kwargs["tier"],
        ),
    )
    result = extractor.extract_clean_text("https://example.com/article", tier=tier)
    assert result is not None
    return result


def test_t3_record_keeps_full_text_on_disk(monkeypatch, tmp_path: Path) -> None:
    text = "T" * 20_000

    record = _record_from_rung(monkeypatch, tmp_path, text, extractor.SourceTier.T3)

    assert Path(record["raw_text_path"]).read_text(encoding="utf-8") == text
    assert record["char_count"] == 20_000
    assert record["text"] == text
    assert len(record["excerpt"]) == 1_500


def test_t2_record_is_not_stitched(monkeypatch, tmp_path: Path) -> None:
    text = "A" * 15_000 + "B" * 15_000

    record = _record_from_rung(monkeypatch, tmp_path, text, extractor.SourceTier.T2)

    cached_text = Path(record["raw_text_path"]).read_text(encoding="utf-8")
    assert len(cached_text) == 30_000
    assert "[... trimmed" not in cached_text
    assert "[... trimmed" not in record["text"]
    assert len(record["excerpt"]) == 1_500


def test_t1_record_unchanged(monkeypatch, tmp_path: Path) -> None:
    text = "T1" * 1_000

    record = _record_from_rung(monkeypatch, tmp_path, text, extractor.SourceTier.T1)

    assert record["text"] == text
    assert record["excerpt"] == text[:1_500]


def test_excerpt_length_is_constant(monkeypatch, tmp_path: Path) -> None:
    assert extractor.EXCERPT_CHARS == 1_500

    for tier in extractor.SourceTier:
        record = _record_from_rung(monkeypatch, tmp_path, "X" * 2_000, tier)
        assert len(record["excerpt"]) <= extractor.EXCERPT_CHARS
