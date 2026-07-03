from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest

import research_engine.dispatcher as dispatcher
import research_engine.logged_search as logged_search
from research_engine.dispatcher import (
    build_api_lane_request,
    discover_gemini_pro_model,
    dispatch,
    dispatch_scout,
    routing_table,
)
from research_engine.schema import AgentRole, Protocol, Territory, WorkerModel


def _completed(
    returncode: int,
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["agy-cli-1"],
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
    # The command never contains a model_id token. It relies on agy-cli-1's
    # configured default and returns GEMINI_PRO_MODEL_CANDIDATES[0] as the
    # logical model_id.
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None

    call_count = {"n": 0}

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        cmd = args[0]
        assert cmd == [
            "agy-cli-1",
            "-p",
            "Reply with exactly OK.",
            "--dangerously-skip-permissions",
        ]
        assert "--model" not in cmd, f"--model must not appear in agy command: {cmd}"
        assert "-m" not in cmd, f"-m must not appear in agy command: {cmd}"
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
        candidates=("gemini-3-flash",),
        cli_home=str(tmp_path),
        runner=runner,
        sleeper=sleeps.append,
        use_cache=False,
    )

    assert result.model_id == "gemini-3-flash"
    assert calls["count"] == 3
    assert sleeps == [2, 6]


def test_discover_gemini_pro_model_does_not_retry_auth_failure(tmp_path) -> None:
    dispatcher._GEMINI_PRO_MODEL_ID_CACHE = None

    calls = {"count": 0}

    def runner(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return _completed(1, stderr="Opening authentication page in your browser")

    result = discover_gemini_pro_model(
        candidates=("gemini-3-flash",),
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
    assert spec.invocation_hint.startswith("agy-cli-1 --dangerously-skip-permissions --print ")
    assert spec.brief_path in spec.invocation_hint
    assert spec.output_path in spec.invocation_hint
    assert "HOME=" not in spec.invocation_hint
    assert "--model" not in spec.invocation_hint
    assert '--print "$(cat ' in spec.invocation_hint
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
    assert spec.invocation_hint.startswith("agy-cli-1 --dangerously-skip-permissions --print ")
    assert spec.brief_path in spec.invocation_hint
    assert spec.output_path in spec.invocation_hint
    assert "HOME=" not in spec.invocation_hint
    assert "--model" not in spec.invocation_hint
    assert '--print "$(cat ' in spec.invocation_hint
    assert "--yolo" not in spec.invocation_hint
    assert "--skip-trust" not in spec.invocation_hint
    assert "--dangerously-skip-permissions" in spec.invocation_hint


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
    assert "/Users/cleo/.secrets/mistral-free-keys.env" in spec.invocation_hint


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
    assert spec.invocation_hint.startswith("hermes -m ")
    assert f"-m {dispatcher.GROK_REASONING_MODEL}" in spec.invocation_hint
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
    assert spec.invocation_hint.startswith("hermes -m ")
    assert f"-m {dispatcher.GROK_RESEARCH_MODEL}" in spec.invocation_hint


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
    assert f"-m {expected_model}" in spec.invocation_hint


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
            "User-Agent": "IanResearch/1.0 (123icpe@gmail.com)",
        },
        "response_format": "atom",
    }

    request = build_api_lane_request("reddit_rss", lane_config, "local ai news")

    assert request.method == "GET"
    assert request.body is None
    assert request.url == (
        "https://www.reddit.com/search.rss?q=local%20ai%20news&sort=relevance&limit=25"
    )
    assert request.headers == {"User-Agent": "IanResearch/1.0 (123icpe@gmail.com)"}
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
