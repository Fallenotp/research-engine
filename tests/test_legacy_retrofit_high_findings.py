from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from research_engine import extractor, paths, research_cli, router, telemetry_observer
from research_engine.fetch_proxy import NoProxyBackend, load_proxy_backend
from research_engine.schema import Protocol


def test_wayback_import_does_not_mutate_environment(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / "engine.env"
    env_file.write_text(
        "WAYBACK_ACCESS_KEY=file-access\nWAYBACK_SECRET_KEY=file-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "env_file", lambda: env_file)
    monkeypatch.delenv("WAYBACK_ACCESS_KEY", raising=False)
    monkeypatch.delenv("WAYBACK_SECRET_KEY", raising=False)
    baseline = dict(os.environ)
    sys.modules.pop("research_engine.wayback_fallback", None)

    try:
        importlib.import_module("research_engine.wayback_fallback")
        assert dict(os.environ) == baseline
        assert os.environ.get("WAYBACK_ACCESS_KEY") is None
        assert os.environ.get("WAYBACK_SECRET_KEY") is None
    finally:
        sys.modules.pop("research_engine.wayback_fallback", None)


def test_telemetry_observer_append_row_logs_failures(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        telemetry_observer,
        "MASTER_LOG",
        Path("/private/tmp/research-telemetry-test.jsonl"),
    )
    monkeypatch.setattr(
        telemetry_observer.json,
        "dumps",
        lambda _row: (_ for _ in ()).throw(OSError("disk full")),
    )

    with caplog.at_level(logging.WARNING, logger="research_engine.telemetry_observer"):
        telemetry_observer._append_row({"session_id": "s1"})

    assert "telemetry append failed: disk full" in caplog.text


def test_telemetry_observer_log_buzz_logs_failures(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        telemetry_observer,
        "MASTER_LOG",
        Path("/private/tmp/research-telemetry-test.jsonl"),
    )
    monkeypatch.setattr(
        telemetry_observer.json,
        "dumps",
        lambda _row: (_ for _ in ()).throw(OSError("buzz write blocked")),
    )

    with caplog.at_level(logging.WARNING, logger="research_engine.telemetry_observer"):
        telemetry_observer.log_buzz("topic", n_signals=1, platforms_with_data=["reddit"])

    assert "buzz telemetry append failed: buzz write blocked" in caplog.text


def test_extractor_append_telemetry_row_logs_failures(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        extractor,
        "READER_TELEMETRY_LOG",
        Path("/private/tmp/research-reader-telemetry-test.jsonl"),
    )
    monkeypatch.setattr(
        extractor.json,
        "dumps",
        lambda _row: (_ for _ in ()).throw(OSError("reader log blocked")),
    )

    with caplog.at_level(logging.WARNING, logger="extractor"):
        extractor._append_telemetry_row({"method": "curl"})

    assert "reader telemetry append failed: reader log blocked" in caplog.text


def test_load_router_or_none_logs_when_router_load_fails(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        research_cli,
        "load_router",
        lambda: (_ for _ in ()).throw(RuntimeError("router config missing")),
    )

    with caplog.at_level(logging.WARNING, logger="research_engine.research_cli"):
        loaded = research_cli.load_router_or_none()

    assert loaded is None
    assert "router load failed; continuing without router: router config missing" in caplog.text


def test_telemetry_safely_logs_when_observer_fails(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        research_cli.telemetry_observer,
        "run",
        lambda: (_ for _ in ()).throw(RuntimeError("telemetry offline")),
    )

    with caplog.at_level(logging.WARNING, logger="research_engine.research_cli"):
        research_cli.telemetry_safely()

    assert "telemetry observer failed: telemetry offline" in caplog.text


def test_run_multi_territory_research_logs_when_router_route_fails(caplog, monkeypatch) -> None:
    class BrokenRouter:
        def route(self, _question: str):
            raise RuntimeError("route explosion")

    fake_session = SimpleNamespace(sources=[], queries_run=[])
    monkeypatch.setattr(research_cli, "load_router_or_none", lambda: BrokenRouter())
    monkeypatch.setattr(
        research_cli,
        "run_gemini_scout",
        lambda *args, **kwargs: SimpleNamespace(success=False, output_text=None),
    )
    monkeypatch.setattr(research_cli, "provider_for_question", lambda _question: "tavily")
    monkeypatch.setattr(research_cli, "decompose_question", lambda *args, **kwargs: [])
    monkeypatch.setattr(research_cli, "build_territories", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        research_cli,
        "build_and_save_research_session",
        lambda *args, **kwargs: (fake_session, Path("/tmp/fake-session.json"), "codex", False),
    )
    monkeypatch.setattr(research_cli, "telemetry_safely", lambda: None)

    with caplog.at_level(logging.WARNING, logger="research_engine.research_cli"):
        result = research_cli.run_multi_territory_research(
            "question",
            protocol=Protocol.RESEARCH,
            territory_specs=[],
        )

    assert result.backend == "codex"
    assert (
        "router route failed; disabling authority topic and Tier-1 enforcement: route explosion"
        in caplog.text
    )


def test_provider_for_question_logs_when_router_route_fails(caplog, monkeypatch) -> None:
    class BrokenRouter:
        def route(self, _question: str):
            raise RuntimeError("route unavailable")

    monkeypatch.setattr(research_cli, "load_router", lambda: BrokenRouter())

    with caplog.at_level(logging.WARNING, logger="research_engine.research_cli"):
        provider = research_cli.provider_for_question("question")

    assert provider == "tavily"
    assert "router provider selection failed; falling back to tavily: route unavailable" in caplog.text


def test_source_authority_score_logs_default_fallback_when_router_load_fails(
    caplog, monkeypatch
) -> None:
    monkeypatch.setattr(router, "_DEFAULT_ROUTER_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(router, "_DEFAULT_ROUTER", None)
    monkeypatch.setattr(
        router,
        "load_router",
        lambda: (_ for _ in ()).throw(RuntimeError("bad router config")),
    )

    with caplog.at_level(logging.WARNING, logger="research_engine.router"):
        score = router.source_authority_score("example.com", "legal")

    assert score == router.DEFAULT_AUTHORITY_SCORE
    assert "router load failed; using default authority score for topic 'legal': bad router config" in caplog.text


def test_source_authority_score_without_topic_uses_default_score() -> None:
    assert router.source_authority_score("example.com", None) == router.DEFAULT_AUTHORITY_SCORE


def test_source_authority_score_logs_default_fallback_when_authority_call_fails(
    caplog, monkeypatch
) -> None:
    monkeypatch.setattr(router, "_DEFAULT_ROUTER_LOAD_ATTEMPTED", True)
    monkeypatch.setattr(
        router,
        "_DEFAULT_ROUTER",
        SimpleNamespace(
            authority_score=lambda _domain, _topic: (_ for _ in ()).throw(
                RuntimeError("authority lookup failed")
            )
        ),
    )

    with caplog.at_level(logging.WARNING, logger="research_engine.router"):
        score = router.source_authority_score("example.com", "legal")

    assert score == router.DEFAULT_AUTHORITY_SCORE
    assert (
        "authority scoring failed for domain 'example.com' topic 'legal'; using default score: authority lookup failed"
        in caplog.text
    )


def test_nordvpn_fallback_returns_honest_no_proxy_label_and_warning(
    caplog, monkeypatch
) -> None:
    monkeypatch.setattr(
        "research_engine.fetch_proxy.NordVPNSocksBackend",
        lambda: (_ for _ in ()).throw(RuntimeError("NordVPN credentials missing")),
    )
    monkeypatch.setattr(
        "research_engine.fetch_proxy.ScraperAPIBackend",
        lambda: (_ for _ in ()).throw(RuntimeError("scraperapi unavailable")),
    )

    with caplog.at_level(logging.WARNING, logger="fetch_proxy"):
        backend = load_proxy_backend({"proxy": {"backend": "nordvpn"}})
        session = backend.acquire(domain="example.com", sticky=False)

    assert isinstance(backend, NoProxyBackend)
    assert session.proxy_url is None
    assert session.label == "no_proxy:nordvpn_and_scraperapi_unavailable"
    assert (
        "nordvpn backend unavailable, no proxy will be used after scraperapi fallback failed: scraperapi unavailable"
        in caplog.text
    )


def test_deleted_dead_helpers_stay_deleted() -> None:
    assert not hasattr(research_cli, "gemini_home_for_spec")
    assert not hasattr(research_cli, "gemini_pro_synthesis_brief")
    assert not hasattr(sys.modules["research_engine.dispatcher"], "_match_lane_rule")
    assert not hasattr(sys.modules["research_engine.dispatcher"], "_load_env_file")
    assert not hasattr(sys.modules["research_engine.sufficiency"], "should_downgrade")
    assert not hasattr(sys.modules["research_engine.sufficiency"], "collect_items_by_source")
