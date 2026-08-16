from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from threading import Lock
import time
from typing import Any, Callable, Iterator, Literal, Mapping, Optional, Sequence
from urllib.parse import quote

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import AgentRole, Protocol, Territory, WorkerModel
else:
    from . import paths
    from .schema import AgentRole, Protocol, Territory, WorkerModel

if __package__ in (None, ""):
    import paths

__all__ = [
    "ApiLaneRequest",
    "GeminiProHealthResult",
    "GeminiProScoutError",
    "WorkerSpec",
    "build_api_lane_request",
    "build_blocking_scout_alert",
    "discover_gemini_pro_model",
    "dispatch",
    "dispatch_pro_synthesis_fallback",
    "dispatch_scout",
    "finalize_gemini_daily_budget",
    "gemini_daily_budget_available",
    "is_unattended_research_run",
    "reserve_gemini_daily_budget",
    "resolve_lane_auth",
    "resolve_agy_model",
    "routing_table",
]


Provider = Literal[
    "anthropic_subagent",
    "agy_cli",
    "codex_cli",
    "grok_cli",
    "mistral_free_api",
    "sonnet_inline",
    "opus_subagent",
]

_BRIEF_DIR_TEMPLATE = "/tmp/deep-research-briefs-{topic_slug}"
_BRIEF_PATH_TEMPLATE = _BRIEF_DIR_TEMPLATE + "/brief-{role}-{territory_id}.md"
_OUTPUT_PATH_TEMPLATE = "/tmp/deep-research-{topic_slug}-agent-{role}-{territory_id}.md"
# Gemini is reachable fleet-wide only through agy. Scheduled workers pin a
# full-quota agy model so the configured Gemini lane remains for daytime use.
AGY_PROVIDER = "agy_cli"
AGY_CLI = paths.executable(paths.AGY_BIN_ENV, "agy-cli-1", "agy-cli-2", "agy") or "agy"
# Use one canonical agy binary. Account/model fallback is handled by agy itself.
AGY_SKIP_PERMISSIONS_FLAG = "--dangerously-skip-permissions"
AGY_INTERACTIVE_GEMINI_MODEL = os.environ.get(
    "RESEARCH_ENGINE_GEMINI_MODEL",
    "Gemini 3.7 Flash (Medium)",
)
AGY_SCHEDULED_WORKER_MODEL = os.environ.get(
    "RESEARCH_ENGINE_SCHEDULED_WORKER_MODEL",
    "GPT-OSS 120B (Medium)",
)
GEMINI_CLI_HOME = AGY_CLI
GEMINI_SCOUT_CLI_HOME = AGY_CLI
RESERVED_GEMINI_CLI_HOME = ""
# Per Ian's locked decision: use Gemini 3.7 Flash for the scout too. Pro is not
# allowed for /research or /deep-research scouting.
# (Name kept as *_PRO_* to avoid breaking references; the value is the Flash model.)
GEMINI_PRO_MODEL_CANDIDATES = (
    AGY_INTERACTIVE_GEMINI_MODEL,
)
# AGY INVOCATION RULE:
# Use the canonical agy binary with a friendly --model name. Do NOT use the
# retired gemini CLI, HOME switching, --yolo, or --skip-trust.
GEMINI_TRANSIENT_BACKOFF_SECONDS = (2, 6, 15)
GEMINI_TIMEOUT_SECONDS = 180
GROK_TIMEOUT_SECONDS = 120
# Grok runs through the Grok CLI (grok -p), NOT Hermes.
# Hermes' xAI auth is dead (invalid_grant, no working refresh). Fallback seat is
# cursor-agent -p --model cursor-grok-4.5-high.
GROK_CLI_MODEL = "grok-4.5"
GROK_REASONING_MODEL = GROK_CLI_MODEL
GROK_RESEARCH_MODEL = GROK_CLI_MODEL
GROK_DEEP_REASONING_MODEL = GROK_CLI_MODEL
_GEMINI_PRO_MODEL_ID_CACHE: str | None = None
_DEFAULT_GEMINI_DAILY_BUDGET = 300
GEMINI_RESERVATION_TTL_SECONDS = 30 * 60
GEMINI_DAILY_COUNTER_FILE = Path(
    os.environ.get(
        "RESEARCH_ENGINE_GEMINI_COUNTER_FILE",
        str(Path.home() / ".cache" / "research_engine" / "gemini_daily_budget.json"),
    )
)
_GEMINI_AUTH_MARKERS = ("opening authentication page", "authenticate")
_GEMINI_TRANSIENT_ERROR_MARKERS = (
    "429",
    "bad record mac",
    "model_capacity_exhausted",
    "rate limit",
    "resource exhausted",
    "resource_exhausted",
    "temporarily unavailable",
    "try again later",
)
LANE_ENV_PATH = paths.optional_path(paths.LANE_ENV_FILE_ENV) or paths.env_file() or paths.data_path(
    "missing-lane-endpoints.env"
)
_LANE_TEMPLATE_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LANE_HEADER_ENV_TOKEN_RE = re.compile(r"env:([A-Z0-9_]+)")
_DEFAULT_LANE_RATE_LIMIT_QPS: float | None = None
_DEFAULT_DISCORD_ALERT_CHANNEL_ID = "1495166380163596298"
_IGNORED_API_LANE_FIELDS = ("command", "args", "profile", "db_path", "glob", "mcp_server")
_LANE_STATE_LOCK = Lock()
_LANE_LAST_CALL_AT: dict[str, float] = {}
_WARNED_IGNORED_API_FIELDS: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class WorkerSpec:
    """The orchestrator uses this to spawn the right worker the right way."""

    worker_model: str
    provider: Provider
    invocation_hint: str
    brief_path: str
    output_path: str
    lanes: list[str]
    rationale: str
    model_id: str | None = None


@dataclass(frozen=True)
class ApiLaneRequest:
    """A formatted HTTP request derived from one api lane config."""

    method: str
    url: str
    headers: dict[str, str]
    body: str | None = None
    response_format: str = "json"
    credential: LaneCredential | None = None


@dataclass(frozen=True)
class LaneCredential:
    """Resolved auth metadata for one lane without mutating os.environ."""

    auth_spec: str
    kind: str
    material: str | None = None
    source: str | None = None
    present: bool = False


@dataclass(frozen=True)
class GeminiProHealthResult:
    ok: bool
    model_id: str | None
    checked_models: tuple[str, ...]
    reason: str = ""


class GeminiProScoutError(RuntimeError):
    """Raised by legacy Gemini interlock paths."""


@dataclass(frozen=True)
class _LaneRule:
    lanes: tuple[str, ...]
    worker_model: WorkerModel
    provider: Provider
    rationale: str


_LANE_RULES = (
    _LaneRule(
        lanes=("github_code", "sourcegraph", "stack_exchange"),
        worker_model=WorkerModel.HAIKU,
        provider="anthropic_subagent",
        rationale="Code and repo lanes route to Haiku; Codex dropped from /research workers.",
    ),
    _LaneRule(
        lanes=(
            "reddit_rss",
            "reddit_failures",
            "x_pulse",
            "bluesky_jetstream",
            "hn_algolia",
            "searxng_forums",
        ),
        worker_model=WorkerModel.GROK,
        provider="grok_cli",
        rationale="Social/zeitgeist lanes route to Grok as the social-platform agent; Gemini is scout-only.",
    ),
    _LaneRule(
        lanes=("arxiv", "semantic_scholar", "core", "papers_with_code", "pubmed"),
        worker_model=WorkerModel.HAIKU,
        provider="anthropic_subagent",
        rationale="Academic lanes route to Haiku for broad, fast literature coverage.",
    ),
    _LaneRule(
        lanes=(
            "courtlistener",
            "nasa_techtransfer",
            "dod_oss",
            "sec_edgar",
            "sec_gov",
            "congress_gov",
            "fec_gov",
            "openstates",
            "govinfo_gov",
            "fda_gov",
            "nih_gov",
        ),
        worker_model=WorkerModel.HAIKU,
        provider="anthropic_subagent",
        rationale="Government and regulatory lanes route to Haiku for primary-source search.",
    ),
)

_DEFAULT_RULE = _LaneRule(
    lanes=(),
    worker_model=WorkerModel.HAIKU,
    provider="anthropic_subagent",
    rationale="Unclassified lanes fall back to Haiku as the default research worker.",
)

_MODEL_PROVIDERS: dict[WorkerModel, Provider] = {
    WorkerModel.HAIKU: "anthropic_subagent",
    WorkerModel.CODEX_MINI: "codex_cli",
    WorkerModel.CODEX_5_3: "codex_cli",
    WorkerModel.CODEX_5_4: "codex_cli",
    WorkerModel.CODEX_5_5: "codex_cli",
    WorkerModel.GEMINI_PRO: AGY_PROVIDER,
    WorkerModel.GEMINI_FLASH: AGY_PROVIDER,
    WorkerModel.GROK: "grok_cli",
    WorkerModel.MISTRAL: "mistral_free_api",
    WorkerModel.SONNET: "sonnet_inline",
    WorkerModel.OPUS: "opus_subagent",
}

logger = logging.getLogger("research_engine.dispatcher")

def routing_table() -> dict:
    """Return the lane→LLM mapping table. Pure function; for inspection."""

    table: dict[str, dict[str, str]] = {
        "__cross_model_verifier__": {
            "worker_model": WorkerModel.SONNET.value,
            "provider": "sonnet_inline",
            "opus_worker_model": WorkerModel.OPUS.value,
            "opus_provider": "opus_subagent",
        },
        "__default__": {
            "worker_model": _DEFAULT_RULE.worker_model.value,
            "provider": _DEFAULT_RULE.provider,
        },
        "mistral_tool_worker": {
            "worker_model": WorkerModel.MISTRAL.value,
            "provider": "mistral_free_api",
            "key_file": os.environ.get(
                paths.MISTRAL_KEYS_FILE_ENV,
                str(paths.data_path("missing-mistral-free-keys.env")),
            ),
        },
        "counter_evidence": {
            "worker_model": WorkerModel.GROK.value,
            "provider": "grok_cli",
            "resolves_to": "grok_cli",
        },
        "exa": {
            "worker_model": WorkerModel.HAIKU.value,
            "provider": "exa_direct",
            "resolves_to": "exa_direct",
        },
        "grok_x_search": {
            "worker_model": WorkerModel.GROK.value,
            "provider": "grok_cli",
        },
        "gemini_pro_scout": {
            "worker_model": WorkerModel.GEMINI_FLASH.value,
            "provider": AGY_PROVIDER,
            "home": GEMINI_SCOUT_CLI_HOME,
            "model_id": _agy_model_for_run(WorkerModel.GEMINI_FLASH),
            "profile": "canonical-agy",
        },
    }
    for rule in _LANE_RULES:
        for lane in rule.lanes:
            table[lane] = {
                "worker_model": rule.worker_model.value,
                "provider": rule.provider,
            }
    return table


def build_api_lane_request(
    lane_name: str,
    lane_config: dict[str, Any],
    query: str,
) -> ApiLaneRequest:
    """Render one api lane config into a concrete HTTP request."""

    if str(lane_config.get("type")) != "api":
        raise ValueError(f"Lane '{lane_name}' is not an api lane")

    endpoint = str(lane_config.get("endpoint") or "").strip()
    if not endpoint:
        raise ValueError(f"Lane '{lane_name}' is missing endpoint")

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError(f"Lane '{lane_name}' requires a non-empty query")

    _warn_ignored_api_lane_fields(lane_name, lane_config)
    _apply_lane_rate_limit(lane_name, lane_config)
    env_values = _lane_env_values(LANE_ENV_PATH)
    credential = resolve_lane_auth(lane_config.get("auth"), env_values=env_values)
    template_values = {"query": quote(normalized_query, safe="")}
    if lane_name == "nasa_techtransfer":
        template_values["stream"] = _nasa_techtransfer_stream(normalized_query)
    if lane_name == "dod_oss":
        resource = _dod_oss_resource(normalized_query)
        template_values["resource"] = resource
        template_values["q_clause"] = _dod_oss_q_clause(normalized_query, resource)

    method = str(lane_config.get("method", "GET")).upper()
    headers = _lane_headers(lane_config.get("headers"), env_values=env_values)
    body_template = lane_config.get("body_template")
    body = None
    if body_template is not None:
        body = _render_lane_template(
            str(body_template),
            lane_name,
            template_values,
            env_values=env_values,
        )

    return ApiLaneRequest(
        method=method,
        url=_render_lane_template(
            endpoint,
            lane_name,
            template_values,
            env_values=env_values,
        ),
        headers=headers,
        body=body,
        response_format=str(lane_config.get("response_format") or "json").strip().lower(),
        credential=credential,
    )


def discover_gemini_pro_model(
    *,
    candidates: Sequence[str] | None = None,
    cli_home: str = GEMINI_SCOUT_CLI_HOME,
    timeout_seconds: int = 60,
    alert_channel_id: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    backoff_seconds: Sequence[int] = GEMINI_TRANSIENT_BACKOFF_SECONDS,
    use_cache: bool = True,
) -> GeminiProHealthResult:
    """Health-check agy and cache the logical Gemini 3.7 Flash model id.

    The scout uses agy with an explicit friendly --model and no HOME switching.
    Gemini outage now degrades instead of raising:
    callers receive ok=False and can continue without the scout.
    """

    global _GEMINI_PRO_MODEL_ID_CACHE

    candidate_tuple = tuple(candidates or GEMINI_PRO_MODEL_CANDIDATES)
    if is_unattended_research_run():
        reason = "Gemini health probe skipped for unattended research"
        logger.info(reason)
        return GeminiProHealthResult(False, None, candidate_tuple, reason)
    if not gemini_daily_budget_available():
        reason = "Gemini health probe skipped because the daily budget is exhausted"
        logger.info(reason)
        return GeminiProHealthResult(False, None, candidate_tuple, reason)
    if use_cache and _GEMINI_PRO_MODEL_ID_CACHE in candidate_tuple:
        return GeminiProHealthResult(True, _GEMINI_PRO_MODEL_ID_CACHE, candidate_tuple)

    _ = cli_home
    failures: list[str] = []

    for requested_model_id in candidate_tuple:
        model_id = resolve_agy_model(requested_model_id)
        completed, combined_output, exc, attempts_used = _run_agy_gemini_with_backoff(
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            runner=runner,
            sleeper=sleeper,
            backoff_seconds=backoff_seconds,
        )
        if exc is not None:
            failures.append(
                f"{model_id}: {type(exc).__name__}: {_trim_output(str(exc))} "
                f"after {attempts_used} attempt(s)"
            )
            continue

        assert completed is not None
        lowered = combined_output.lower()
        if completed.returncode == 0 and "ok" in lowered:
            _GEMINI_PRO_MODEL_ID_CACHE = model_id
            return GeminiProHealthResult(True, model_id, candidate_tuple)
        if _is_gemini_auth_failure(lowered):
            failures.append(f"{model_id}: agy requested browser authentication")
            break
        failures.append(
            f"{model_id}: exit={completed.returncode} after {attempts_used} attempt(s); "
            f"output={_trim_output(combined_output)}"
        )

    reason = "; ".join(failures) or "no Gemini 3.7 Flash model candidates were checked"
    _maybe_alert_scout_failure(reason, channel_id=alert_channel_id)
    logger.warning("Gemini 3.7 Flash scout unavailable; continuing without scout: %s", reason)
    return GeminiProHealthResult(False, None, candidate_tuple, reason)


def build_blocking_scout_alert(reason: str) -> str:
    return (
        f"Gemini 3.7 Flash scout via {AGY_CLI} unavailable: {reason}. "
        "Continuing with the configured fallback path; fix agy to restore scout coverage."
    )


def dispatch_scout(
    question: str,
    router,
    *,
    protocol: Protocol,
    topic_slug: str = "topic",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkerSpec | None:
    if protocol not in (Protocol.RESEARCH, Protocol.DEEP_RESEARCH):
        raise ValueError("Gemini 3.7 Flash scout is only for /research and /deep-research")

    scout_config = _router_config(router, "scout_config")
    candidates = scout_config.get("model_candidates") or GEMINI_PRO_MODEL_CANDIDATES
    cli_home = str(scout_config.get("cli_home") or GEMINI_SCOUT_CLI_HOME)
    timeout_seconds = _gemini_config_timeout(scout_config)
    alert_channel_id = str(scout_config.get("discord_alert_channel_id") or "").strip() or None
    health = discover_gemini_pro_model(
        candidates=candidates,
        cli_home=cli_home,
        timeout_seconds=timeout_seconds,
        alert_channel_id=alert_channel_id,
        runner=runner,
    )
    if not health.ok or health.model_id is None:
        logger.warning(
            "Skipping Gemini 3.7 Flash scout for %s after health check failure: %s",
            protocol.value,
            health.reason,
        )
        if _config_flag(scout_config.get("fail_loud")):
            raise GeminiProScoutError(health.reason or "Gemini scout health check failed")
        return None

    safe_topic_slug = _slugify(topic_slug)
    brief_path = (
        f"/tmp/deep-research-briefs-{safe_topic_slug}/brief-scout-gemini-flash.md"
    )
    output_path = f"/tmp/deep-research-{safe_topic_slug}-agent-scout-gemini-flash.md"
    _ensure_parent_dir(brief_path)
    _ensure_parent_dir(output_path)
    Path(brief_path).write_text(
        _scout_brief_text(question, protocol=protocol),
        encoding="utf-8",
    )

    return WorkerSpec(
        worker_model=WorkerModel.GEMINI_FLASH.value,
        provider=AGY_PROVIDER,
        invocation_hint=_gemini_invocation_hint(
            cli_home=cli_home,
            model_id=health.model_id,
            brief_path=brief_path,
            output_path=output_path,
        ),
        brief_path=brief_path,
        output_path=output_path,
        lanes=["gemini_pro_scout", "searxng_general", "linkup_direct", "firecrawl_direct"],
        rationale="Gemini 3.7 Flash scout runs first through agy and returns a first-picture summary plus depth targets.",
        model_id=health.model_id,
    )


def dispatch_pro_synthesis_fallback(
    question: str,
    router,
    *,
    protocol: Protocol,
    topic_slug: str = "topic",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkerSpec | None:
    if protocol not in (Protocol.RESEARCH, Protocol.DEEP_RESEARCH):
        raise ValueError("Gemini synthesis fallback is only for /research and /deep-research")

    scout_config = _router_config(router, "scout_config")
    health = discover_gemini_pro_model(
        candidates=scout_config.get("model_candidates") or GEMINI_PRO_MODEL_CANDIDATES,
        cli_home=str(scout_config.get("cli_home") or GEMINI_SCOUT_CLI_HOME),
        timeout_seconds=_gemini_config_timeout(scout_config),
        alert_channel_id=str(scout_config.get("discord_alert_channel_id") or "").strip() or None,
        runner=runner,
    )
    if not health.ok or health.model_id is None:
        logger.warning("Skipping Gemini synthesis fallback: %s", health.reason)
        return None

    safe_topic_slug = _slugify(topic_slug)
    brief_path = f"/tmp/deep-research-briefs-{safe_topic_slug}/brief-final-synthesis-pro.md"
    output_path = f"/tmp/deep-research-{safe_topic_slug}-final-synthesis-pro.md"
    _ensure_parent_dir(brief_path)
    _ensure_parent_dir(output_path)
    Path(brief_path).write_text(
        _pro_synthesis_fallback_brief_text(question, protocol=protocol),
        encoding="utf-8",
    )

    return WorkerSpec(
        worker_model=WorkerModel.GEMINI_FLASH.value,
        provider=AGY_PROVIDER,
        invocation_hint=_gemini_invocation_hint(
            cli_home=str(scout_config.get("cli_home") or GEMINI_SCOUT_CLI_HOME),
            model_id=health.model_id,
            brief_path=brief_path,
            output_path=output_path,
        ),
        brief_path=brief_path,
        output_path=output_path,
        lanes=["gemini_pro_scout"],
        rationale="--pro-synthesis-fallback skips the scout and uses Gemini 3.7 Flash through agy for final synthesis.",
        model_id=health.model_id,
    )


def dispatch(
    territory: Territory,
    router,
    *,
    topic_slug: str = "topic",
    protocol: Protocol = Protocol.RESEARCH,
) -> WorkerSpec:
    """Decide which LLM handles this worker, return a complete spec."""

    assigned_worker_model = _coerce_worker_model(territory.assigned_worker_model)
    assigned_lanes = _normalize_lanes(getattr(territory, "assigned_lanes", []))
    role = _coerce_role(getattr(territory, "assigned_agent_role", None))

    if role == AgentRole.CROSS_MODEL_VERIFIER:
        if assigned_worker_model == WorkerModel.OPUS:
            worker_model = WorkerModel.OPUS
            provider: Provider = "opus_subagent"
            rationale = "Cross-model verification uses Opus when the territory explicitly requests it."
        else:
            worker_model = WorkerModel.SONNET
            provider = "sonnet_inline"
            rationale = "Cross-model verification overrides lane routing to use judgment, not search."
    elif role == AgentRole.COUNTER_EVIDENCE:
        worker_model = WorkerModel.GROK
        provider = "grok_cli"
        rationale = (
            "Counter-evidence uses the strong Grok API wrapper as the standing live web/X "
            "plus reasoning worker."
        )
    elif assigned_worker_model == WorkerModel.GROK:
        worker_model = WorkerModel.GROK
        provider = "grok_cli"
        rationale = (
            "Territory explicitly requests the strong Grok API wrapper for live web/X "
            "plus reasoning coverage."
        )
    else:
        worker_model = assigned_worker_model
        provider = _MODEL_PROVIDERS.get(worker_model, _DEFAULT_RULE.provider)
        rationale = (
            f"Territory explicitly assigned {worker_model.value}; lane choice only controls "
            "where evidence is searched."
        )

    role_name = role.value if role is not None else str(territory.assigned_agent_role)
    safe_topic_slug = _slugify(topic_slug)
    territory_id = _slugify(territory.territory_id)
    brief_path = _BRIEF_PATH_TEMPLATE.format(
        topic_slug=safe_topic_slug,
        role=_slugify(role_name),
        territory_id=territory_id,
    )
    output_path = _OUTPUT_PATH_TEMPLATE.format(
        topic_slug=safe_topic_slug,
        role=_slugify(role_name),
        territory_id=territory_id,
    )
    _ensure_parent_dir(brief_path)
    _ensure_parent_dir(output_path)
    _warn_unknown_lanes(territory, router, assigned_lanes)

    grok_model = _grok_model_for_territory(role, protocol, assigned_lanes)
    return WorkerSpec(
        worker_model=worker_model.value,
        provider=provider,
        invocation_hint=_invocation_hint(
            provider,
            brief_path,
            output_path,
            grok_model=grok_model,
        ),
        brief_path=brief_path,
        output_path=output_path,
        lanes=assigned_lanes,
        rationale=rationale,
        model_id=(
            grok_model
            if provider == "grok_cli"
            else _agy_model_for_run(worker_model)
            if provider == AGY_PROVIDER
            else None
        ),
    )


def _coerce_worker_model(value: object) -> WorkerModel:
    if isinstance(value, WorkerModel):
        return value
    try:
        return WorkerModel(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid worker_model: {value!r}") from exc


def _coerce_role(value: object) -> Optional[AgentRole]:
    if isinstance(value, AgentRole):
        return value
    try:
        return AgentRole(str(value))
    except ValueError:
        return None


def _normalize_lanes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _warn_unknown_lanes(territory: Territory, router, assigned_lanes: list[str]) -> None:
    if router is None:
        return
    config = getattr(router, "config", None)
    if not isinstance(config, dict):
        return
    configured_lanes = config.get("lanes")
    if not isinstance(configured_lanes, dict):
        return
    territory_id = str(territory.territory_id)
    for lane in assigned_lanes:
        if lane not in configured_lanes:
            logger.warning(
                "Unknown lane '%s' in territory '%s'; keeping assigned worker model",
                lane,
                territory_id,
            )


def _invocation_hint(
    provider: Provider,
    brief_path: str,
    output_path: str,
    *,
    grok_model: str | None = None,
) -> str:
    quoted_brief_path = shlex.quote(brief_path)
    quoted_output_path = shlex.quote(output_path)
    if provider == "anthropic_subagent":
        return (
            "USE Task tool with subagent_type='general-purpose', "
            f"model='haiku', prompt that points at {brief_path}"
        )
    if provider == "codex_cli":
        return (
            "codex exec -m gpt-5.4-mini --skip-git-repo-check -s danger-full-access "
            f"< {quoted_brief_path} > {quoted_output_path} 2>&1"
        )
    if provider == AGY_PROVIDER:
        model = shlex.quote(_agy_model_for_run(WorkerModel.GEMINI_FLASH))
        return (
            f'{shlex.quote(AGY_CLI)} {AGY_SKIP_PERMISSIONS_FLAG} -p "$(cat {quoted_brief_path})" '
            f"--model {model} "
            f"> {quoted_output_path} 2>&1"
        )
    if provider == "grok_cli":
        return _grok_invocation_hint(
            brief_path=brief_path,
            output_path=output_path,
            model=grok_model or GROK_REASONING_MODEL,
        )
    if provider == "mistral_free_api":
        return (
            f"Call Mistral chat completions with tools using {paths.MISTRAL_KEYS_FILE_ENV}"
        )
    if provider == "sonnet_inline":
        return "Sonnet handles inline; no subprocess"
    return "USE Task tool with subagent_type='general-purpose', model='opus'"


def _gemini_invocation_hint(
    *,
    cli_home: str,
    model_id: str,
    brief_path: str,
    output_path: str,
) -> str:
    _ = cli_home
    quoted_brief_path = shlex.quote(brief_path)
    quoted_output_path = shlex.quote(output_path)
    model = shlex.quote(model_id)
    return (
        f'{shlex.quote(AGY_CLI)} {AGY_SKIP_PERMISSIONS_FLAG} -p "$(cat {quoted_brief_path})" '
        f"--model {model} "
        f"> {quoted_output_path} 2>&1"
    )


def _agy_model_for_run(worker_model: WorkerModel) -> str | None:
    if worker_model not in {WorkerModel.GEMINI_FLASH, WorkerModel.GEMINI_PRO}:
        return None
    return resolve_agy_model(AGY_INTERACTIVE_GEMINI_MODEL)


def _grok_model_for_territory(
    role: Optional[AgentRole],
    protocol: Protocol,
    lanes: list[str],
) -> str:
    if "grok_x_search" in set(lanes):
        return GROK_RESEARCH_MODEL
    if role == AgentRole.COUNTER_EVIDENCE:
        return GROK_RESEARCH_MODEL
    if protocol == Protocol.DEEP_RESEARCH:
        return GROK_DEEP_REASONING_MODEL
    return GROK_REASONING_MODEL


def _grok_invocation_hint(*, brief_path: str, output_path: str, model: str) -> str:
    return (
        f'{shlex.quote(paths.executable(paths.GROK_BIN_ENV, "grok") or "grok")} --single "$(cat {shlex.quote(brief_path)})" '
        f"> {shlex.quote(output_path)} 2>&1"
    )


def _router_config(router, method_name: str) -> dict:
    method = getattr(router, method_name, None)
    if callable(method):
        config = method()
        if isinstance(config, dict):
            return config
    config = getattr(router, "config", None)
    if isinstance(config, dict):
        key = method_name.removesuffix("_config")
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _gemini_config_timeout(config: dict[str, Any]) -> int:
    return max(
        int(config.get("health_check_timeout_seconds", GEMINI_TIMEOUT_SECONDS)),
        GEMINI_TIMEOUT_SECONDS,
    )


def _config_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values.setdefault(key, value)
    return values


def _lane_env_values(path: Path) -> dict[str, str]:
    values = _load_env_file_values(path)
    values.update(os.environ)
    values.setdefault(paths.USER_AGENT_ENV, paths.user_agent())
    email = paths.contact_email()
    if email:
        values.setdefault(paths.CONTACT_EMAIL_ENV, email)
    return values


def resolve_lane_auth(
    auth: object,
    *,
    env_values: Mapping[str, str] | None = None,
) -> LaneCredential:
    auth_spec = str(auth or "none").strip() or "none"
    env_map = env_values if env_values is not None else _lane_env_values(LANE_ENV_PATH)

    if auth_spec in {"none", "none_local"}:
        return LaneCredential(auth_spec=auth_spec, kind="none")
    if auth_spec in {"optional_key", "optional_token", "oauth_subscription"}:
        return LaneCredential(auth_spec=auth_spec, kind=auth_spec)
    if auth_spec.startswith("env:"):
        env_name = auth_spec[4:].strip()
        material = env_map.get(env_name, "").strip() if env_name else ""
        return LaneCredential(
            auth_spec=auth_spec,
            kind="env",
            material=material or None,
            source=env_name or None,
            present=bool(material),
        )
    if auth_spec.startswith("file:"):
        file_target = auth_spec[5:].strip()
        path = Path(file_target).expanduser() if file_target else None
        if path is None or not path.is_file():
            return LaneCredential(
                auth_spec=auth_spec,
                kind="file",
                source=str(path) if path is not None else None,
            )
        try:
            material = path.read_text(encoding="utf-8")
        except OSError:
            return LaneCredential(auth_spec=auth_spec, kind="file", source=str(path))
        return LaneCredential(
            auth_spec=auth_spec,
            kind="file",
            material=material,
            source=str(path),
            present=True,
        )
    return LaneCredential(auth_spec=auth_spec, kind="label")


def _lane_headers(
    value: object,
    *,
    env_values: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    headers: dict[str, str] = {}
    for key, item in value.items():
        resolved = _resolve_lane_header_value(str(item), env_values=env_values)
        if resolved is None:
            continue
        headers[str(key)] = resolved
    return headers


def _resolve_lane_header_value(
    value: str,
    *,
    env_values: Mapping[str, str],
) -> str | None:
    resolved = value
    for env_name in _LANE_HEADER_ENV_TOKEN_RE.findall(value):
        env_value = env_values.get(env_name, "").strip()
        if not env_value:
            return None
        resolved = resolved.replace(f"env:{env_name}", env_value)
    for token in _LANE_TEMPLATE_TOKEN_RE.findall(resolved):
        env_value = env_values.get(token)
        if env_value is None:
            continue
        resolved = resolved.replace(f"{{{token}}}", env_value)
    return resolved


def _render_lane_template(
    template: str,
    lane_name: str,
    template_values: dict[str, str],
    *,
    env_values: Mapping[str, str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in template_values:
            return template_values[token]
        env_value = env_values.get(token)
        if env_value is None:
            raise KeyError(f"Lane '{lane_name}' requires env var '{token}'")
        return env_value

    return _LANE_TEMPLATE_TOKEN_RE.sub(replace, template)


def _nasa_techtransfer_stream(query: str) -> str:
    lowered = query.lower()
    if "spinoff" in lowered:
        return "spinoff"
    if "software" in lowered or "open source" in lowered:
        return "software"
    return "patent"


def _dod_oss_resource(query: str) -> str:
    lowered = query.lower()
    if any(
        token in lowered
        for token in ("code", "implementation", "source file", "function", "class", "module")
    ):
        return "code"
    return "repositories"


def _dod_oss_q_clause(query: str, resource: str) -> str:
    encoded_query = quote(query.strip(), safe="")
    if resource == "repositories":
        return f"org:deptofdefense+{encoded_query}"
    return f"{encoded_query}+org:deptofdefense"


def _scout_brief_text(question: str, *, protocol: Protocol) -> str:
    return f"""You are the Gemini 3.7 Flash scout for {protocol.value}.

Run one broad, bounded first sweep before any parallel workers are assigned.
Use the normal research-engine search and fetch tools: free community/search
first, then direct-provider lanes only where needed, then clean extraction for
the best URLs. Stay within the run's total budget ceiling.
When the question touches code, libraries, or engineering, search MULTIPLE forges — GitHub, GitLab, Codeberg, and SourceHut — not GitHub alone.

Question:
{question}

Return exactly these sections:
1. FIRST-PICTURE SUMMARY — concise map of what appears true, contested, and current.
2. DEPTH TARGETS — structured bullet list of sub-areas needing deeper digging.
3. SUGGESTED WORKER TERRITORIES — non-overlapping sub-questions for cheap workers.
4. SOURCE STARTERS — URLs/search terms worth handing to workers.
5. KNOWN GAPS — what not to synthesize yet.
"""


def _pro_synthesis_fallback_brief_text(question: str, *, protocol: Protocol) -> str:
    return f"""You are Gemini 3.7 Flash through agy doing final synthesis for {protocol.value}.

The scout was intentionally skipped by --pro-synthesis-fallback. Read the
worker outputs already gathered by the normal lanes, synthesize only grounded
claims, and use graduated abstention: full answer, partial answer with explicit
confidence, or insufficient evidence with concrete next steps.

Question:
{question}
"""


def _maybe_alert_scout_failure(reason: str, *, channel_id: str | None = None) -> None:
    if not _is_unattended_run():
        return
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        return
    target_channel_id = (
        str(channel_id or "").strip()
        or os.environ.get("DISCORD_ALERT_CHANNEL_ID", _DEFAULT_DISCORD_ALERT_CHANNEL_ID)
    )
    try:
        subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                f"https://discord.com/api/v10/channels/{target_channel_id}/messages",
                "-H",
                f"Authorization: Bot {token}",
                "-F",
                f"content={build_blocking_scout_alert(reason)}",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.exception("Failed to post Gemini 3.7 Flash scout alert to Discord")


def _warn_ignored_api_lane_fields(lane_name: str, lane_config: dict[str, Any]) -> None:
    for field in _IGNORED_API_LANE_FIELDS:
        value = lane_config.get(field)
        if value in (None, "", [], {}, ()):
            continue
        lane_field = (lane_name, field)
        with _LANE_STATE_LOCK:
            if lane_field in _WARNED_IGNORED_API_FIELDS:
                continue
            _WARNED_IGNORED_API_FIELDS.add(lane_field)
        logger.warning(
            "Lane '%s' declares ignored field '%s' on the api path; dispatcher cannot use it",
            lane_name,
            field,
        )


def _lane_min_interval_seconds(lane_config: dict[str, Any]) -> float | None:
    raw_value = lane_config.get("rate_limit_qps", _DEFAULT_LANE_RATE_LIMIT_QPS)
    if raw_value in (None, ""):
        return None
    try:
        qps = float(raw_value)
    except (TypeError, ValueError):
        return None
    if qps <= 0:
        return None
    return 1.0 / qps


def _apply_lane_rate_limit(
    lane_name: str,
    lane_config: dict[str, Any],
    *,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> None:
    if clock is None:
        clock = time.monotonic
    if sleeper is None:
        sleeper = time.sleep

    min_interval = _lane_min_interval_seconds(lane_config)
    if min_interval is None:
        return

    while True:
        with _LANE_STATE_LOCK:
            now = clock()
            last_called = _LANE_LAST_CALL_AT.get(lane_name)
            remaining = 0.0 if last_called is None else min_interval - (now - last_called)
            if remaining <= 0:
                _LANE_LAST_CALL_AT[lane_name] = now
                return
        sleeper(remaining)


def is_unattended_research_run() -> bool:
    unattended_flag = os.environ.get("RESEARCH_ENGINE_UNATTENDED", "").strip().lower()
    mentor_flag = os.environ.get("MENTOR_NIGHTLY_RUN", "").strip().lower()
    enabled = {"1", "true", "yes", "cron", "launchd", "nightly"}
    return unattended_flag in enabled or mentor_flag in {"1", "true", "yes"}


def _gemini_daily_budget_limit() -> int:
    raw_value = os.environ.get(
        "RESEARCH_ENGINE_GEMINI_DAILY_BUDGET",
        str(_DEFAULT_GEMINI_DAILY_BUDGET),
    )
    try:
        return int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid RESEARCH_ENGINE_GEMINI_DAILY_BUDGET=%r; using %s",
            raw_value,
            _DEFAULT_GEMINI_DAILY_BUDGET,
        )
        return _DEFAULT_GEMINI_DAILY_BUDGET


def _fresh_gemini_counter() -> dict[str, Any]:
    return {
        "date": datetime.now().date().isoformat(),
        "used": 0,
        "reserved": 0,
        "reservation_leases": [],
    }


def _load_gemini_counter_locked(counter_path: Path, limit: int) -> dict[str, Any]:
    if not counter_path.exists():
        return _fresh_gemini_counter()
    try:
        payload = json.loads(counter_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("counter must be a JSON object")
        if payload.get("date") != datetime.now().date().isoformat():
            return _fresh_gemini_counter()
        used = max(0, int(payload.get("used", 0)))
        reserved = max(0, int(payload.get("reserved", 0)))
        raw_leases = payload.get("reservation_leases")
        if raw_leases is None:
            raw_leases = [counter_path.stat().st_mtime] * reserved
        if not isinstance(raw_leases, list):
            raise ValueError("reservation_leases must be a list")
        cutoff = time.time() - GEMINI_RESERVATION_TTL_SECONDS
        reservation_leases = [
            float(lease_timestamp)
            for lease_timestamp in raw_leases
            if float(lease_timestamp) > cutoff
        ]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        logger.error("Gemini daily counter is corrupt; treating budget as exhausted")
        return {
            "date": datetime.now().date().isoformat(),
            "used": max(0, limit),
            "reserved": 0,
            "reservation_leases": [],
        }
    return {
        "date": payload["date"],
        "used": used,
        "reserved": len(reservation_leases),
        "reservation_leases": reservation_leases,
    }


def _save_gemini_counter_locked(payload: dict[str, Any], counter_path: Path) -> None:
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{counter_path.name}.",
        suffix=".tmp",
        dir=counter_path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, counter_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


@contextmanager
def _locked_gemini_counter(
    path: Path | None = None,
    *,
    limit: int | None = None,
) -> Iterator[tuple[dict[str, Any], Path, int]]:
    counter_path = path or GEMINI_DAILY_COUNTER_FILE
    budget_limit = _gemini_daily_budget_limit() if limit is None else limit
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = counter_path.with_name(f"{counter_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        payload = _load_gemini_counter_locked(counter_path, budget_limit)
        yield payload, counter_path, budget_limit
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def gemini_daily_budget_available(
    *,
    path: Path | None = None,
    limit: int | None = None,
) -> bool:
    with _locked_gemini_counter(path, limit=limit) as (payload, _path, budget_limit):
        if budget_limit <= 0:
            return True
        return int(payload["used"]) + int(payload["reserved"]) < budget_limit


def reserve_gemini_daily_budget(
    *,
    path: Path | None = None,
    limit: int | None = None,
) -> bool:
    with _locked_gemini_counter(path, limit=limit) as (payload, counter_path, budget_limit):
        if budget_limit <= 0:
            return True
        if int(payload["used"]) + int(payload["reserved"]) >= budget_limit:
            return False
        payload["reservation_leases"].append(time.time())
        payload["reserved"] = len(payload["reservation_leases"])
        _save_gemini_counter_locked(payload, counter_path)
        return True


def finalize_gemini_daily_budget(
    *,
    success: bool,
    path: Path | None = None,
    limit: int | None = None,
) -> None:
    with _locked_gemini_counter(path, limit=limit) as (payload, counter_path, budget_limit):
        if budget_limit <= 0:
            return
        if payload["reservation_leases"]:
            payload["reservation_leases"].pop(0)
        payload["reserved"] = len(payload["reservation_leases"])
        if success:
            payload["used"] = int(payload["used"]) + 1
        _save_gemini_counter_locked(payload, counter_path)


def resolve_agy_model(
    requested_model: str | None,
    *,
    gemini_budget_available: bool = True,
) -> str:
    model_id = requested_model or AGY_INTERACTIVE_GEMINI_MODEL
    normalized = model_id.strip().lower()
    is_gemini = normalized.startswith("gemini ") or normalized.startswith("gemini-")
    if is_gemini and (is_unattended_research_run() or not gemini_budget_available):
        return AGY_SCHEDULED_WORKER_MODEL
    return model_id


# Backwards-compatible private alias for callers/tests that predate the shared helper.
_is_unattended_run = is_unattended_research_run


def _trim_output(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _combine_process_output(completed: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    )


def _is_gemini_auth_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _GEMINI_AUTH_MARKERS)


def _is_transient_gemini_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _GEMINI_TRANSIENT_ERROR_MARKERS)


def _run_agy_gemini_with_backoff(
    *,
    model_id: str,
    timeout_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    sleeper: Callable[[float], None],
    backoff_seconds: Sequence[int],
) -> tuple[Optional[subprocess.CompletedProcess[str]], str, Optional[BaseException], int]:
    total_attempts = len(backoff_seconds) + 1
    command = [
        AGY_CLI,
        AGY_SKIP_PERMISSIONS_FLAG,
        "-p",
        "Reply with exactly OK.",
        "--model",
        model_id,
    ]

    for attempt_number in range(1, total_attempts + 1):
        if not reserve_gemini_daily_budget():
            reason = "Gemini daily budget exhausted before health probe"
            return None, reason, RuntimeError(reason), attempt_number - 1
        try:
            completed = runner(
                command,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - release the reservation for any runner failure
            finalize_gemini_daily_budget(success=False)
            exc_text = str(exc)
            if attempt_number < total_attempts and _is_transient_gemini_failure(exc_text):
                logger.warning(
                    "Transient Gemini scout failure for %s on attempt %s/%s: %s",
                    model_id,
                    attempt_number,
                    total_attempts,
                    _trim_output(exc_text),
                )
                sleeper(backoff_seconds[attempt_number - 1])
                continue
            return None, exc_text, exc, attempt_number

        combined_output = _combine_process_output(completed)
        finalize_gemini_daily_budget(success=completed.returncode == 0)
        if (
            completed.returncode != 0
            and attempt_number < total_attempts
            and _is_transient_gemini_failure(combined_output)
            and not _is_gemini_auth_failure(combined_output)
        ):
            logger.warning(
                "Transient Gemini scout failure for %s on attempt %s/%s: %s",
                model_id,
                attempt_number,
                total_attempts,
                _trim_output(combined_output),
            )
            sleeper(backoff_seconds[attempt_number - 1])
            continue
        return completed, combined_output, None, attempt_number

    return None, "Gemini runner exhausted retries unexpectedly", RuntimeError(
        "Gemini runner exhausted retries unexpectedly"
    ), total_attempts


def _slugify(value: object) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip().lower()).strip("-")
    return slug or "topic"


def _ensure_parent_dir(path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    from unittest.mock import patch

    topic_slug = "dispatcher-self-test"

    keyword = Territory(
        territory_id="A",
        description="Code territory",
        queries=["python asyncio"],
        assigned_agent_role=AgentRole.KEYWORD,
        assigned_lanes=["github_code", "searxng_general"],
        assigned_worker_model=WorkerModel.CODEX_5_4,
    )
    spec = dispatch(keyword, None, topic_slug=topic_slug)
    assert spec.worker_model == WorkerModel.CODEX_5_4.value
    assert spec.provider == "codex_cli"
    assert "$(" not in spec.invocation_hint
    assert f"< {spec.brief_path}" in spec.invocation_hint
    keyword_b = Territory(
        territory_id="B",
        description="Code territory duplicate role",
        queries=["python asyncio"],
        assigned_agent_role=AgentRole.KEYWORD,
        assigned_lanes=["github_code"],
        assigned_worker_model=WorkerModel.CODEX_5_4,
    )
    spec_b = dispatch(keyword_b, None, topic_slug=topic_slug)
    assert spec.brief_path != spec_b.brief_path
    assert spec.output_path != spec_b.output_path

    domain = Territory(
        territory_id="C",
        description="Social territory",
        queries=["community reaction"],
        assigned_agent_role=AgentRole.DOMAIN_SPECIALIST,
        assigned_lanes=["reddit_rss", "x_pulse"],
        assigned_worker_model=WorkerModel.GEMINI_FLASH,
    )
    spec = dispatch(domain, None, topic_slug=topic_slug)
    assert spec.worker_model == WorkerModel.GEMINI_FLASH.value
    assert spec.provider == AGY_PROVIDER
    assert spec.invocation_hint.startswith(
        f"{AGY_CLI} --dangerously-skip-permissions -p "
    )
    assert '-p "$(cat ' in spec.invocation_hint
    assert "--model 'Gemini 3.7 Flash (Medium)'" in spec.invocation_hint
    assert "HOME=" not in spec.invocation_hint
    assert "--dangerously-skip-permissions" in spec.invocation_hint

    semantic = Territory(
        territory_id="D",
        description="Academic territory",
        queries=["new paper"],
        assigned_agent_role=AgentRole.SEMANTIC,
        assigned_lanes=["arxiv", "semantic_scholar"],
        assigned_worker_model=WorkerModel.HAIKU,
    )
    spec = dispatch(semantic, None, topic_slug=topic_slug)
    assert spec.worker_model == WorkerModel.HAIKU.value
    assert spec.provider == "anthropic_subagent"

    verifier = Territory(
        territory_id="E",
        description="Verification territory",
        queries=["verify claims"],
        assigned_agent_role=AgentRole.CROSS_MODEL_VERIFIER,
        assigned_lanes=["searxng_general"],
        assigned_worker_model=WorkerModel.SONNET,
    )
    spec = dispatch(verifier, None, topic_slug=topic_slug)
    assert spec.worker_model == WorkerModel.SONNET.value
    assert spec.provider == "sonnet_inline"

    fallback = Territory(
        territory_id="F",
        description="Default territory",
        queries=["general web"],
        assigned_agent_role=AgentRole.KEYWORD,
        assigned_lanes=["searxng_general"],
        assigned_worker_model=WorkerModel.HAIKU,
    )
    spec = dispatch(fallback, None, topic_slug=topic_slug)
    assert spec.worker_model == WorkerModel.HAIKU.value
    assert spec.provider == "anthropic_subagent"

    class _RouterStub:
        config = {"lanes": {"github_code": {}}}

    typo = Territory(
        territory_id="G",
        description="Typo lane territory",
        queries=["python asyncio"],
        assigned_agent_role=AgentRole.KEYWORD,
        assigned_lanes=["github_code", "github_codez_TYPO"],
        assigned_worker_model=WorkerModel.CODEX_5_4,
    )
    with patch(__name__ + ".logger.warning") as warning_mock:
        spec = dispatch(typo, _RouterStub(), topic_slug=topic_slug)
    assert spec.worker_model == WorkerModel.CODEX_5_4.value
    warning_mock.assert_called_once_with(
        "Unknown lane '%s' in territory '%s'; keeping assigned worker model",
        "github_codez_TYPO",
        "G",
    )

    print("PASS dispatcher self-test")
