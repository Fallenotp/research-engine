import os

import pytest


@pytest.fixture(autouse=True)
def stub_wayback(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("RUN_LIVE_WAYBACK") == "1":
        return

    monkeypatch.setattr(
        "research_engine.wayback_fallback.try_wayback",
        lambda _url: None,
    )
