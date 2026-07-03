from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Literal, Optional, Sequence
from urllib.parse import quote

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from schema import AgentRole, Protocol, Territory, WorkerModel
else:
    from .schema import AgentRole, Protocol, Territory, WorkerModel

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
# Gemini is reachable fleet-wide only through agy-cli-1. agy-cli-2 is the
# documented fallback account when it is unlocked, but this dispatcher emits
# agy-cli-1 directly so the orchestrator has one canonical command to run.
AGY_PROVIDER = "agy_cli"
AGY_CLI = "agy-cli-1"
AGY_FALLBACK_CLI = "agy-cli-2"
AGY_SKIP_PERMISSIONS_FLAG = "--dangerously-skip-permissions"
GEMINI_CLI_HOME = AGY_CLI
GEMINI_SCOUT_CLI_HOME = AGY_CLI
RESERVED_GEMINI_CLI_HOME = ""
# Per Ian's locked decision: use Gemini 3 Flash for the scout too. Pro is not
# allowed for /research or /deep-research scouting.
# (Name kept as *_PRO_* to avoid breaking references; the value is the Flash model.)
GEMINI_PRO_MODEL_CANDIDATES = (
    "gemini-3-flash",
)
# AGY INVOCATION RULE:
# Use agy-cli-1's configured default only (no -m/--model flag). Do NOT probe
# model ids with test calls. Do NOT substitute flash-lite. If the default ever
# fails, ask Ian for the exact agy account/model fix before retrying.
GEMINI_TRANSIENT_BACKOFF_SECONDS = (2, 6, 15)
GEMINI_TIMEOUT_SECONDS = 180
GROK_TIMEOUT_SECONDS = 120
GROK_4_2_HERMES_MODEL = "grok-4.20-0309-reasoning"
GROK_REASONING_MODEL = GROK_4_2_HERMES_MODEL
GROK_RESEARCH_MODEL = GROK_4_2_HERMES_MODEL
GROK_DEEP_REASONING_MODEL = GROK_4_2_HERMES_MODEL
_GEMINI_PRO_MODEL_ID_CACHE: str | None = None
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
LANE_ENV_PATH = Path("/Users/cleo/.openclaw/workspace/gov-tech-transfer/endpoints.env")
_LANE_TEMPLATE_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LANE_HEADER_ENV_TOKEN_RE = re.compile(r"env:([A-Z0-9_]+)")


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
            "key_file": "/Users/cleo/.secrets/mistral-free-keys.env",
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
            "profile": "CLI1",
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

    _load_env_file(LANE_ENV_PATH)
    template_values = {"query": quote(normalized_query, safe="")}
    if lane_name == "nasa_techtransfer":
        template_values["stream"] = _nasa_techtransfer_stream(normalized_query)
    if lane_name == "dod_oss":
        resource = _dod_oss_resource(normalized_query)
        template_values["resource"] = resource
        template_values["q_clause"] = _dod_oss_q_clause(normalized_query, resource)

    method = str(lane_config.get("method", "GET")).upper()
    headers = _lane_headers(lane_config.get("headers"))
    body_template = lane_config.get("body_template")
    body = None
    if body_template is not None:
        body = _render_lane_template(str(body_template), lane_name, template_values)

    return ApiLaneRequest(
        method=method,
        url=_render_lane_template(endpoint, lane_name, template_values),
        headers=headers,
        body=body,
        response_format=str(lane_config.get("response_format") or "json").strip().lower(),
    )


def discover_gemini_pro_model(
    *,
    candidates: Sequence[str] | None = None,
    cli_home: str = GEMINI_SCOUT_CLI_HOME,
    timeout_seconds: int = 60,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    backoff_seconds: Sequence[int] = GEMINI_TRANSIENT_BACKOFF_SECONDS,
    use_cache: bool = True,
) -> GeminiProHealthResult:
    """Health-check agy-cli-1 and cache the logical Gemini 3 Flash model id.

    The scout uses agy-cli-1's configured default model, with no -m/--model
    flag and no HOME switching. Gemini outage now degrades instead of raising:
    callers receive ok=False and can continue without the scout.
    """

    global _GEMINI_PRO_MODEL_ID_CACHE

    candidate_tuple = tuple(candidates or GEMINI_PRO_MODEL_CANDIDATES)
    if use_cache and _GEMINI_PRO_MODEL_ID_CACHE in candidate_tuple:
        return GeminiProHealthResult(True, _GEMINI_PRO_MODEL_ID_CACHE, candidate_tuple)

    _ = cli_home
    failures: list[str] = []

    for model_id in candidate_tuple:
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
            failures.append(f"{model_id}: agy-cli-1 requested browser authentication")
            break
        failures.append(
            f"{model_id}: exit={completed.returncode} after {attempts_used} attempt(s); "
            f"output={_trim_output(combined_output)}"
        )

    reason = "; ".join(failures) or "no Gemini 3 Flash model candidates were checked"
    _maybe_alert_scout_failure(reason)
    logger.warning("Gemini 3 Flash scout unavailable; continuing without scout: %s", reason)
    return GeminiProHealthResult(False, None, candidate_tuple, reason)


def build_blocking_scout_alert(reason: str) -> str:
    return (
        f"Gemini 3 Flash scout via {AGY_CLI} unavailable: {reason}. "
        "Continuing with the configured fallback path; fix agy-cli-1 to restore scout coverage."
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
        raise ValueError("Gemini 3 Flash scout is only for /research and /deep-research")

    scout_config = _router_config(router, "scout_config")
    candidates = scout_config.get("model_candidates") or GEMINI_PRO_MODEL_CANDIDATES
    cli_home = str(scout_config.get("cli_home") or GEMINI_SCOUT_CLI_HOME)
    timeout_seconds = _gemini_config_timeout(scout_config)
    health = discover_gemini_pro_model(
        candidates=candidates,
        cli_home=cli_home,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    if not health.ok or health.model_id is None:
        logger.warning(
            "Skipping Gemini 3 Flash scout for %s after health check failure: %s",
            protocol.value,
            health.reason,
        )
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
        rationale="Gemini 3 Flash scout runs first on CLI1 and returns first-picture summary plus depth targets.",
    )


def dispatch_pro_synthesis_fallback(
    question: str,
    router,
    *,
    protocol: Protocol,
    topic_slug: str = "topic",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkerSpec:
    if protocol not in (Protocol.RESEARCH, Protocol.DEEP_RESEARCH):
        raise ValueError("Gemini synthesis fallback is only for /research and /deep-research")

    scout_config = _router_config(router, "scout_config")
    health = discover_gemini_pro_model(
        candidates=scout_config.get("model_candidates") or GEMINI_PRO_MODEL_CANDIDATES,
        cli_home=str(scout_config.get("cli_home") or GEMINI_SCOUT_CLI_HOME),
        timeout_seconds=_gemini_config_timeout(scout_config),
        runner=runner,
    )
    assert health.model_id is not None

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
        rationale="--pro-synthesis-fallback skips the scout and uses CLI1 Gemini 3 Flash for final synthesis.",
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
        model_id=grok_model if provider == "grok_cli" else None,
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


def _match_lane_rule(lanes: list[str]) -> Optional[_LaneRule]:
    lane_set = set(lanes)
    for rule in _LANE_RULES:
        if lane_set.intersection(rule.lanes):
            return rule
    return None


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
        return (
            f'{AGY_CLI} {AGY_SKIP_PERMISSIONS_FLAG} --print "$(cat {quoted_brief_path})" '
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
            "Call Mistral chat completions with tools using only "
            "/Users/cleo/.secrets/mistral-free-keys.env"
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
    _ = (cli_home, model_id)
    quoted_brief_path = shlex.quote(brief_path)
    quoted_output_path = shlex.quote(output_path)
    return (
        f'{AGY_CLI} {AGY_SKIP_PERMISSIONS_FLAG} --print "$(cat {quoted_brief_path})" '
        f"> {quoted_output_path} 2>&1"
    )


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
        f"hermes -m {shlex.quote(model)} -z \"$(cat {shlex.quote(brief_path)})\" "
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


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _lane_headers(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    headers: dict[str, str] = {}
    for key, item in value.items():
        resolved = _resolve_lane_header_value(str(item))
        if resolved is None:
            continue
        headers[str(key)] = resolved
    return headers


def _resolve_lane_header_value(value: str) -> str | None:
    resolved = value
    for env_name in _LANE_HEADER_ENV_TOKEN_RE.findall(value):
        env_value = os.environ.get(env_name, "").strip()
        if not env_value:
            return None
        resolved = resolved.replace(f"env:{env_name}", env_value)
    return resolved


def _render_lane_template(
    template: str,
    lane_name: str,
    template_values: dict[str, str],
) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in template_values:
            return template_values[token]
        env_value = os.environ.get(token)
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
    return f"""You are the Gemini 3 Flash scout for {protocol.value}.

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
    return f"""You are Gemini 3 Flash on CLI1 doing final synthesis for {protocol.value}.

The scout was intentionally skipped by --pro-synthesis-fallback. Read the
worker outputs already gathered by the normal lanes, synthesize only grounded
claims, and use graduated abstention: full answer, partial answer with explicit
confidence, or insufficient evidence with concrete next steps.

Question:
{question}
"""


def _maybe_alert_scout_failure(reason: str) -> None:
    if not _is_unattended_run():
        return
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        return
    channel_id = os.environ.get("DISCORD_ALERT_CHANNEL_ID", "1495166380163596298")
    try:
        subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
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
        logger.exception("Failed to post Gemini 3 Flash scout alert to Discord")


def _is_unattended_run() -> bool:
    unattended_flag = os.environ.get("RESEARCH_ENGINE_UNATTENDED", "").lower()
    return (not sys.stdin.isatty()) and unattended_flag in {"1", "true", "yes", "cron", "launchd"}


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
    # model_id is retained as a parameter for logging/cache purposes but is NOT
    # passed to agy.
    command = [AGY_CLI, "-p", "Reply with exactly OK.", AGY_SKIP_PERMISSIONS_FLAG]

    for attempt_number in range(1, total_attempts + 1):
        try:
            completed = runner(
                command,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
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
    assert spec.invocation_hint.startswith("agy-cli-1 --dangerously-skip-permissions --print ")
    assert '--print "$(cat ' in spec.invocation_hint
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
