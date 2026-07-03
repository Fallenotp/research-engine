from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path

import pytest
import requests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


STEEL_SCRAPE_URL = "http://127.0.0.1:3000/v1/scrape"


def _require_steel() -> None:
    try:
        with socket.create_connection(("127.0.0.1", 3000), timeout=1):
            return
    except OSError as exc:
        pytest.skip(f"Steel not reachable on 127.0.0.1:3000: {exc}")


def _load_live_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("WEBREAD_ALLOW_L3", "1")
    monkeypatch.setenv("WEBREAD_L3_MIN_FREE_MB", "1")
    monkeypatch.setenv("WEBREAD_L3_MAX_SWAP_MB", "99999999")
    monkeypatch.setenv("WEBREAD_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("STEEL_SCRAPE_URL", STEEL_SCRAPE_URL)

    l3_guard = importlib.import_module("research_engine.l3_guard")
    extractor = importlib.import_module("research_engine.extractor")
    l3_guard = importlib.reload(l3_guard)
    extractor = importlib.reload(extractor)
    return l3_guard, extractor


def test_live_steel_scrape_and_extractor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _require_steel()

    response = requests.post(
        STEEL_SCRAPE_URL,
        headers={"Content-Type": "application/json"},
        json={"url": "https://example.com", "format": ["markdown", "html"]},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    content = data.get("content")
    assert isinstance(content, dict)
    markdown = str(content.get("markdown") or "").strip()
    html = str(content.get("html") or "").strip()
    assert "documentation examples" in markdown
    assert "Example Domain" in html

    _, extractor = _load_live_modules(monkeypatch, tmp_path)
    payload = extractor._steel_scrape("https://example.com")

    assert payload is not None
    assert payload["title"] == "Example Domain"
    assert "documentation examples" in (payload.get("text") or "")

