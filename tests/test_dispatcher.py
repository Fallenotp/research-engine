from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time
from unittest.mock import patch
from urllib.parse import quote

import pytest

import research_engine.dispatcher as dispatcher
import research_engine.logged_search as logged_search
from research_engine import paths
from research_engine.dispatcher import (
    build_api_lane_request,
    discover_gemini_pro_model,
    dispatch,
    dispatch_pro_synthesis_fallback,
    dispatch_scout,
    routing_table,
)
from research_engine.schema import AgentRole, Protocol, Territory, WorkerModel


@pytest.fixture(autouse=True)
def _isolate_gemini_budget(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        dispatcher,
        "GEMINI_DAILY_COUNTER_FILE",
        tmp_path / "gemini-counter.json",
    )
    monkeypatch.setenv("RESEARCH_ENGINE_GEMINI_DAILY_BUDGET", "300")
    monkeypatch.delenv("RESEARCH_ENGINE_UNATTENDED", raising=False)
    monkeypatch.delenv("MENTOR_NIGHTLY_RUN", raising=False)


def _completed(
    returncode: int,
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[dispatcher.AGY_CLI],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class _RouterStub:
    def __init__(self, cli_home: str) -> None:
        self._config = {
            "cli_home": cli_home,
            "health_check_timeout_seconds": 30,
            "model_candidates": list(dispatcher.GEMINI_PRO_MODEL_CANDIDATES),
        }

    def scout_config(self) -> dict:
        return dict(self._config)


def test_discover_gemini_pro_model_uses_canonical_flash_candidate(tmp_path) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None

    call_count = {"n": 0}

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        cmd = args[0]
        assert cmd == [
            dispatcher.AGY_CLI,
            "--dangerously-skip-permissions",
            "-p",
            "Reply with exactly OK.",
            "--model",
            "Gemini 3.7 Flash (Medium)",
        ]
        assert "env" not in kwargs
        return _completed(0, stdout="OK\n")

    result = discover_gemini_pro_model(
        cli_home=str(tmp_path),
        runner=runner,
        use_cache=False,
    )

    assert result.ok is True
    assert result.model_id == dispatcher.GEMINI_PRO_MODEL_CANDIDATES[0]
    assert call_count["n"] == 1
    counter = json.loads(dispatcher.GEMINI_DAILY_COUNTER_FILE.read_text(encoding="utf-8"))
    assert counter == {
        "date": datetime.now().date().isoformat(),
        "used": 1,
        "reserved": 0,
        "reservation_leases": [],
    }


def test_discover_gemini_pro_model_retries_transient_failures(tmp_path) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None

    calls = {"count": 0}
    sleeps: list[float] = []

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        if calls["count"] < 3:
            return _completed(
                1,
                stderr="429 MODEL_CAPACITY_EXHAUSTED: try again later",
            )
        return _completed(0, stdout="OK\n")

    result = discover_gemini_pro_model(
        candidates=("Gemini 3.7 Flash (Medium)",),
        cli_home=str(tmp_path),
        runner=runner,
        sleeper=sleeps.append,
        use_cache=False,
    )

    assert result.model_id == "Gemini 3.7 Flash (Medium)"
    assert calls["count"] == 3
    assert sleeps == [2, 6]


def test_discover_gemini_pro_model_does_not_retry_auth_failure(tmp_path) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None

    calls = {"count": 0}

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return _completed(1, stderr="Opening authentication page in your browser")

    result = discover_gemini_pro_model(
        candidates=("Gemini 3.7 Flash (Medium)",),
        cli_home=str(tmp_path),
        runner=runner,
        use_cache=False,
    )

    assert calls["count"] == 1
    assert result.ok is False
    assert result.model_id is None
    assert "browser authentication" in result.reason.lower()


def test_dispatch_scout_degrades_to_skip_when_gemini_is_down(tmp_path) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None
    router = _RouterStub(str(tmp_path))

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _completed(1, stderr="Opening authentication page in your browser")

    spec = dispatch_scout(
        "test question",
        router,
        protocol=Protocol.RESEARCH,
        topic_slug="gemini-down",
        runner=runner,
    )

    assert spec is None


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("RESEARCH_ENGINE_UNATTENDED", "nightly"),
        ("MENTOR_NIGHTLY_RUN", "1"),
    ],
)
def test_dispatch_scout_never_probes_gemini_when_unattended(
    monkeypatch,
    tmp_path,
    env_name,
    env_value,
) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None
    monkeypatch.setenv(env_name, env_value)

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Gemini probe must not run unattended")

    spec = dispatch_scout(
        "nightly question",
        _RouterStub(str(tmp_path)),
        protocol=Protocol.RESEARCH,
        runner=runner,
    )

    assert spec is None


def test_dispatch_scout_ignores_warm_cache_when_unattended(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        dispatcher,
        "_GEMINI_PRO_MODEL_ID_CACHE",
        dispatcher.GEMINI_PRO_MODEL_CANDIDATES[0],
    )
    monkeypatch.setenv("MENTOR_NIGHTLY_RUN", "1")

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Gemini probe must not run unattended")

    spec = dispatch_scout(
        "nightly question with a warm cache",
        _RouterStub(str(tmp_path)),
        protocol=Protocol.RESEARCH,
        runner=runner,
    )

    assert spec is None


def test_dispatch_scout_never_probes_gemini_when_budget_exhausted(
    monkeypatch,
    tmp_path,
) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None
    monkeypatch.setenv("RESEARCH_ENGINE_GEMINI_DAILY_BUDGET", "1")
    dispatcher.GEMINI_DAILY_COUNTER_FILE.write_text(
        json.dumps(
            {
                "date": datetime.now().date().isoformat(),
                "used": 1,
                "reserved": 0,
            }
        ),
        encoding="utf-8",
    )

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Gemini probe must not run after the cap")

    spec = dispatch_scout(
        "capped question",
        _RouterStub(str(tmp_path)),
        protocol=Protocol.RESEARCH,
        runner=runner,
    )

    assert spec is None


def test_dispatch_pro_synthesis_fallback_soft_fails_without_assert(tmp_path) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _completed(1, stderr="Opening authentication page in your browser")

    spec = dispatch_pro_synthesis_fallback(
        "test question",
        _RouterStub(str(tmp_path)),
        protocol=Protocol.RESEARCH,
        runner=runner,
    )

    assert spec is None


def test_corrupt_counter_fails_closed(tmp_path) -> None:
    counter_path = tmp_path / "corrupt-counter.json"
    counter_path.write_text("{not-json", encoding="utf-8")

    assert not dispatcher.gemini_daily_budget_available(path=counter_path, limit=3)
    assert not dispatcher.reserve_gemini_daily_budget(path=counter_path, limit=3)


def test_locked_budget_reservations_do_not_exceed_cap(tmp_path) -> None:
    counter_path = tmp_path / "concurrent-counter.json"

    with ThreadPoolExecutor(max_workers=12) as pool:
        reservations = list(
            pool.map(
                lambda _index: dispatcher.reserve_gemini_daily_budget(
                    path=counter_path,
                    limit=5,
                ),
                range(30),
            )
        )

    assert reservations.count(True) == 5
    payload = json.loads(counter_path.read_text(encoding="utf-8"))
    assert payload["used"] == 0
    assert payload["reserved"] == 5


def test_stale_budget_reservation_expires_on_load(tmp_path) -> None:
    counter_path = tmp_path / "stale-counter.json"
    stale_lease = time.time() - dispatcher.GEMINI_RESERVATION_TTL_SECONDS - 1
    counter_path.write_text(
        json.dumps(
            {
                "date": datetime.now().date().isoformat(),
                "used": 0,
                "reserved": 1,
                "reservation_leases": [stale_lease],
            }
        ),
        encoding="utf-8",
    )

    assert dispatcher.gemini_daily_budget_available(path=counter_path, limit=1)
    assert dispatcher.reserve_gemini_daily_budget(path=counter_path, limit=1)

    payload = json.loads(counter_path.read_text(encoding="utf-8"))
    assert payload["reserved"] == 1
    assert len(payload["reservation_leases"]) == 1
    assert payload["reservation_leases"][0] > stale_lease


def test_invalid_budget_env_is_parsed_lazily(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv("RESEARCH_ENGINE_GEMINI_DAILY_BUDGET", "not-a-number")

    assert dispatcher.gemini_daily_budget_available(path=tmp_path / "counter.json")
    assert "using 300" in caplog.text


def test_dispatch_scout_emits_agy_command(tmp_path) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None
    router = _RouterStub(str(tmp_path))

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _completed(0, stdout="OK\n")

    spec = dispatch_scout(
        "test question",
        router,
        protocol=Protocol.DEEP_RESEARCH,
        topic_slug="gemini-smoke",
        runner=runner,
    )

    assert spec is not None
    assert spec.provider == "agy_cli"
    assert spec.worker_model == WorkerModel.GEMINI_FLASH.value
    assert spec.model_id == "Gemini 3.7 Flash (Medium)"
    assert spec.invocation_hint.startswith(
        f"{dispatcher.AGY_CLI} --dangerously-skip-permissions -p "
    )
    assert spec.brief_path in spec.invocation_hint
    assert spec.output_path in spec.invocation_hint
    assert "HOME=" not in spec.invocation_hint
    assert "--model 'Gemini 3.7 Flash (Medium)'" in spec.invocation_hint
    assert '-p "$(cat ' in spec.invocation_hint
    assert "--yolo" not in spec.invocation_hint
    assert "--skip-trust" not in spec.invocation_hint
    assert "--dangerously-skip-permissions" in spec.invocation_hint


def test_social_lane_emits_gemini_invocation_hint() -> None:
    territory = Territory(
        territory_id="social",
        description="Social and zeitgeist territory",
        queries=["community reaction"],
        assigned_agent_role=AgentRole.DOMAIN_SPECIALIST,
        assigned_lanes=["reddit_rss", "x_pulse"],
        assigned_worker_model=WorkerModel.GEMINI_FLASH,
    )

    spec = dispatch(territory, router=None, topic_slug="gemini-social")

    assert spec.worker_model == WorkerModel.GEMINI_FLASH.value
    assert spec.provider == "agy_cli"
    assert spec.model_id == "Gemini 3.7 Flash (Medium)"
    assert spec.invocation_hint.startswith(
        f"{dispatcher.AGY_CLI} --dangerously-skip-permissions -p "
    )
    assert spec.brief_path in spec.invocation_hint
    assert spec.output_path in spec.invocation_hint
    assert "HOME=" not in spec.invocation_hint
    assert "--model 'Gemini 3.7 Flash (Medium)'" in spec.invocation_hint
    assert '-p "$(cat ' in spec.invocation_hint
    assert "--yolo" not in spec.invocation_hint
    assert "--skip-trust" not in spec.invocation_hint
    assert "--dangerously-skip-permissions" in spec.invocation_hint


def test_unattended_gemini_worker_routes_to_full_quota_model(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_ENGINE_UNATTENDED", "launchd")
    territory = Territory(
        territory_id="nightly",
        description="Scheduled territory",
        queries=["nightly research"],
        assigned_agent_role=AgentRole.DOMAIN_SPECIALIST,
        assigned_lanes=["reddit_rss"],
        assigned_worker_model=WorkerModel.GEMINI_FLASH,
    )

    spec = dispatch(territory, router=None, topic_slug="nightly")

    assert spec.provider == "agy_cli"
    assert spec.model_id == "GPT-OSS 120B (Medium)"
    assert "--model 'GPT-OSS 120B (Medium)'" in spec.invocation_hint


def test_mistral_worker_emits_free_api_invocation_hint() -> None:
    territory = Territory(
        territory_id="mistral",
        description="Mistral tool-calling territory",
        queries=["summarize grounded sources"],
        assigned_agent_role=AgentRole.DOMAIN_SPECIALIST,
        assigned_lanes=["reddit_rss", "searxng_general"],
        assigned_worker_model=WorkerModel.MISTRAL,
    )

    spec = dispatch(territory, router=None, topic_slug="mistral-worker")

    assert spec.worker_model == WorkerModel.MISTRAL.value
    assert spec.provider == "mistral_free_api"
    assert paths.MISTRAL_KEYS_FILE_ENV in spec.invocation_hint


def test_dispatch_honors_assigned_worker_even_with_exa_lane() -> None:
    territory = Territory(
        territory_id="exa-codex",
        description="Assigned Codex territory with Exa search lane",
        queries=["Find the borrowed product model evidence"],
        assigned_agent_role=AgentRole.KEYWORD,
        assigned_lanes=["exa", "searxng_general"],
        assigned_worker_model=WorkerModel.CODEX_5_4,
    )

    spec = dispatch(territory, router=None, topic_slug="assigned-worker")

    assert spec.worker_model == WorkerModel.CODEX_5_4.value
    assert spec.provider == "codex_cli"


def test_dispatch_honors_explicit_grok_worker() -> None:
    territory = Territory(
        territory_id="grok-extra",
        description="Extra independent reasoning hand for broad web coverage",
        queries=["What did the other workers miss about model evaluation drift?"],
        assigned_agent_role=AgentRole.SEMANTIC,
        assigned_lanes=["arxiv", "semantic_scholar", "linkup_direct"],
        assigned_worker_model=WorkerModel.GROK,
    )

    spec = dispatch(territory, router=None, topic_slug="grok-smoke")

    assert spec.worker_model == WorkerModel.GROK.value
    assert spec.provider == "grok_cli"
    assert spec.invocation_hint.startswith(
        (paths.executable(paths.GROK_BIN_ENV, "grok") or "grok") + " --single "
    )
    assert spec.brief_path in spec.invocation_hint
    assert spec.output_path in spec.invocation_hint


def test_counter_evidence_role_is_standing_grok_lane() -> None:
    territory = Territory(
        territory_id="counter",
        description="Counter-evidence and real-time objections",
        queries=["What evidence argues against the main claim?"],
        assigned_agent_role=AgentRole.COUNTER_EVIDENCE,
        assigned_lanes=["counter_evidence", "reddit_failures", "x_pulse"],
        assigned_worker_model=WorkerModel.HAIKU,
    )

    spec = dispatch(territory, router=None, topic_slug="grok-counter")

    assert spec.worker_model == WorkerModel.GROK.value
    assert spec.provider == "grok_cli"
    assert spec.invocation_hint.startswith(
        (paths.executable(paths.GROK_BIN_ENV, "grok") or "grok") + " --single "
    )


@pytest.mark.parametrize(
    ("protocol", "role", "worker_model", "lanes", "expected_model"),
    [
        (
            Protocol.RESEARCH,
            AgentRole.COUNTER_EVIDENCE,
            WorkerModel.HAIKU,
            ["counter_evidence", "searxng_general"],
            dispatcher.GROK_RESEARCH_MODEL,
        ),
        (
            Protocol.RESEARCH,
            AgentRole.SEMANTIC,
            WorkerModel.GROK,
            ["semantic", "searxng_general"],
            dispatcher.GROK_REASONING_MODEL,
        ),
        (
            Protocol.RESEARCH,
            AgentRole.DOMAIN_SPECIALIST,
            WorkerModel.GROK,
            ["grok_x_search", "x_pulse"],
            dispatcher.GROK_RESEARCH_MODEL,
        ),
        (
            Protocol.DEEP_RESEARCH,
            AgentRole.COUNTER_EVIDENCE,
            WorkerModel.HAIKU,
            ["counter_evidence", "searxng_general"],
            dispatcher.GROK_RESEARCH_MODEL,
        ),
        (
            Protocol.DEEP_RESEARCH,
            AgentRole.SEMANTIC,
            WorkerModel.GROK,
            ["semantic", "searxng_general"],
            dispatcher.GROK_DEEP_REASONING_MODEL,
        ),
        (
            Protocol.DEEP_RESEARCH,
            AgentRole.DOMAIN_SPECIALIST,
            WorkerModel.GROK,
            ["grok_x_search", "x_pulse"],
            dispatcher.GROK_RESEARCH_MODEL,
        ),
    ],
)
def test_grok_model_matrix_respects_protocol_and_role(
    protocol: Protocol,
    role: AgentRole,
    worker_model: WorkerModel,
    lanes: list[str],
    expected_model: str,
) -> None:
    territory = Territory(
        territory_id=f"{protocol.value.strip('/')}-{role.value}",
        description="Grok model matrix territory",
        queries=["Which Grok model should this use?"],
        assigned_agent_role=role,
        assigned_lanes=lanes,
        assigned_worker_model=worker_model,
    )

    spec = dispatch(
        territory,
        router=None,
        protocol=protocol,
        topic_slug="grok-model-matrix",
    )

    assert spec.worker_model == WorkerModel.GROK.value
    assert spec.provider == "grok_cli"
    assert spec.model_id == expected_model


def test_routing_table_marks_counter_evidence_as_grok_cli() -> None:
    table = routing_table()

    assert table["counter_evidence"]["worker_model"] == WorkerModel.GROK.value
    assert table["counter_evidence"]["provider"] == "grok_cli"
    assert table["counter_evidence"]["resolves_to"] == "grok_cli"
    assert table["exa"]["provider"] == "exa_direct"
    assert table["exa"]["resolves_to"] == "exa_direct"
    assert table["grok_x_search"]["worker_model"] == WorkerModel.GROK.value
    assert table["grok_x_search"]["provider"] == "grok_cli"
    assert table["mistral_tool_worker"]["worker_model"] == WorkerModel.MISTRAL.value
    assert table["mistral_tool_worker"]["provider"] == "mistral_free_api"


@pytest.mark.parametrize(
    ("query", "expected_stream"),
    [
        ("NASA patent license opportunities", "patent"),
        ("NASA software catalog for robotics", "software"),
        ("NASA spinoff medical technology", "spinoff"),
    ],
)
def test_build_api_lane_request_uses_correct_nasa_stream(
    query: str,
    expected_stream: str,
) -> None:
    lane_config = {
        "type": "api",
        "endpoint": "https://technology.nasa.gov/api/query/{stream}/{query}",
        "auth": "none",
    }

    request = build_api_lane_request("nasa_techtransfer", lane_config, query)

    assert request.method == "GET"
    assert request.body is None
    assert request.url == (
        f"https://technology.nasa.gov/api/query/{expected_stream}/{quote(query, safe='')}"
    )


@pytest.mark.parametrize(
    ("query", "expected_resource", "expected_q_clause"),
    [
        (
            "DoD open source repositories for logistics",
            "repositories",
            f"org:deptofdefense+{quote('DoD open source repositories for logistics', safe='')}",
        ),
        (
            "deptofdefense code implementation for kubernetes",
            "code",
            f"{quote('deptofdefense code implementation for kubernetes', safe='')}+org:deptofdefense",
        ),
    ],
)
def test_build_api_lane_request_uses_correct_dod_oss_resource(
    query: str,
    expected_resource: str,
    expected_q_clause: str,
) -> None:
    lane_config = {
        "type": "api",
        "endpoint": "https://api.github.com/search/{resource}?q={q_clause}",
        "auth": "env:GITHUB_TOKEN",
        "headers": {
            "Authorization": "Bearer env:GITHUB_TOKEN",
        },
    }

    request = build_api_lane_request("dod_oss", lane_config, query)

    assert request.method == "GET"
    assert request.body is None
    assert request.url == (
        f"https://api.github.com/search/{expected_resource}?q={expected_q_clause}"
    )


def test_build_api_lane_request_loads_data_gov_key_from_env_file(tmp_path) -> None:
    env_file = tmp_path / "endpoints.env"
    env_file.write_text('DATA_GOV_API_KEY="demo-key"\n', encoding="utf-8")
    lane_config = {
        "type": "api",
        "endpoint": "https://api.congress.gov/v3/?api_key={DATA_GOV_API_KEY}&query={query}",
        "auth": "env:DATA_GOV_API_KEY",
    }

    with patch.dict(dispatcher.os.environ, {}, clear=True):
        with patch.object(dispatcher, "LANE_ENV_PATH", Path(env_file)):
            request = build_api_lane_request("congress_gov", lane_config, "mars")

    assert request.url == "https://api.congress.gov/v3/?api_key=demo-key&query=mars"


def test_build_api_lane_request_uses_reddit_rss_url_ua_and_format() -> None:
    lane_config = {
        "type": "api",
        "endpoint": "https://www.reddit.com/search.rss?q={query}&sort=relevance&limit=25",
        "headers": {
            "User-Agent": "{RESEARCH_ENGINE_USER_AGENT}",
        },
        "response_format": "atom",
    }

    request = build_api_lane_request("reddit_rss", lane_config, "local ai news")

    assert request.method == "GET"
    assert request.body is None
    assert request.url == (
        "https://www.reddit.com/search.rss?q=local%20ai%20news&sort=relevance&limit=25"
    )
    assert request.headers == {"User-Agent": paths.user_agent()}
    assert request.response_format == "atom"


def test_reddit_rss_atom_parser_emits_existing_result_shape() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>First Reddit result</title>
    <link href="https://www.reddit.com/r/example/comments/abc/first/"/>
    <author><name>/u/example_user</name></author>
  </entry>
  <entry>
    <title>Second Reddit result</title>
    <link href="https://www.reddit.com/r/example/comments/def/second/"/>
    <author><name>plain_author</name></author>
  </entry>
</feed>
"""

    assert logged_search._parse_atom_results(feed) == [
        {
            "title": "First Reddit result",
            "url": "https://www.reddit.com/r/example/comments/abc/first/",
            "author": "example_user",
        },
        {
            "title": "Second Reddit result",
            "url": "https://www.reddit.com/r/example/comments/def/second/",
            "author": "plain_author",
        },
    ]
