from __future__ import annotations

from research_engine import research_cli
import research_engine.dispatcher as dispatcher
from research_engine.schema import AgentRole, Protocol, Territory, WorkerModel


FORGE_PROMPT_LINE = (
    "When the question touches code, libraries, or engineering, search MULTIPLE "
    "forges — GitHub, GitLab, Codeberg, and SourceHut — not GitHub alone."
)


def test_api_lane_auth_field_does_not_inject_headers(monkeypatch) -> None:
    lane_config = {
        "type": "api",
        "endpoint": "https://api.github.com/search/code?q={query}",
        "auth": "env:GITHUB_TOKEN",
    }

    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    request = dispatcher.build_api_lane_request("github_code", lane_config, "asyncio")

    assert request.headers == {}


def test_api_lane_headers_resolve_env_placeholders(monkeypatch) -> None:
    lane_config = {
        "type": "api",
        "endpoint": "https://api.github.com/search/code?q={query}",
        "headers": {
            "Authorization": "Bearer env:GITHUB_TOKEN",
            "X-Static": "keep-me",
        },
    }

    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    request = dispatcher.build_api_lane_request("github_code", lane_config, "asyncio")

    assert request.headers == {
        "Authorization": "Bearer secret-token",
        "X-Static": "keep-me",
    }


def test_api_lane_headers_drop_empty_env_placeholders(monkeypatch) -> None:
    lane_config = {
        "type": "api",
        "endpoint": "https://gitlab.com/api/v4/search?scope=blobs&search={query}",
        "headers": {
            "PRIVATE-TOKEN": "env:GITLAB_TOKEN",
            "X-Static": "keep-me",
        },
    }

    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    request = dispatcher.build_api_lane_request("gitlab_code", lane_config, "asyncio")

    assert request.headers == {"X-Static": "keep-me"}


def test_scout_brief_mentions_multiple_forges() -> None:
    brief = dispatcher._scout_brief_text("how do I use this SDK", protocol=Protocol.RESEARCH)

    assert FORGE_PROMPT_LINE in brief


def test_code_worker_brief_mentions_multiple_forges() -> None:
    territory = Territory(
        territory_id="code",
        description="Investigate SDK usage patterns",
        queries=["sdk usage"],
        assigned_agent_role=AgentRole.KEYWORD,
        assigned_lanes=["github_code", "gitlab_code", "codeberg_code", "sourcehut_code"],
        assigned_worker_model=WorkerModel.CODEX_5_4,
    )

    brief = research_cli.worker_territory_brief(
        "how do I use this SDK",
        territory,
        protocol=Protocol.RESEARCH,
        source_pairs=[],
        worker_model=WorkerModel.CODEX_5_4,
    )

    assert FORGE_PROMPT_LINE in brief
