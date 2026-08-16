from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional

import yaml

from . import paths
from research_engine.schema import WorkerModel


DEFAULT_CONFIG_PATH = str(paths.package_path("router_config.yaml"))
EXPECTED_SCHEMA_VERSION = "1.0.0"
ALWAYS_MATCH_PATTERN = ".*"
DEFAULT_AUTHORITY_SCORE = 0.5
LEGACY_AUTHORITY_SCORE = 0.6
_REQUIRED_RULE_FIELDS = ("name", "patterns", "lanes", "worker_model", "topic")
_HOW_TO_PREFIXES = ("how do i ", "how can i ", "how should i ")
_DEFAULT_ROUTER: Router | None = None
_DEFAULT_ROUTER_LOAD_ATTEMPTED = False
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    rule_name: str
    lanes: list[str]
    worker_model: str
    topic: str
    require_tier_1: bool
    skip_web: bool
    bias: Optional[str]
    freshness_max_age_hours: Optional[int]
    matched_patterns: list[str]
    contributing_rules: list[str]


@dataclass(frozen=True)
class IterationPolicy:
    max_loops: int
    trigger_on_gap_severities: list[str]
    per_iteration_lane_cap: int
    cost_cap_usd_per_session: float


@dataclass(frozen=True)
class _CompiledRule:
    name: str
    patterns: tuple[str, ...]
    pattern_regexes: tuple[Optional[re.Pattern[str]], ...]
    lanes: tuple[str, ...]
    worker_model: str
    topic: str
    require_tier_1: bool = False
    skip_web: bool = False
    bias: Optional[str] = None
    freshness_max_age_hours: Optional[int] = None


def load_router(config_path: str = DEFAULT_CONFIG_PATH) -> "Router":
    path = Path(config_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"Router config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        config = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(_yaml_error_message(path, raw_text, exc)) from exc

    if not isinstance(config, dict):
        raise ValueError(f"Router config at {path} must load to a mapping")

    return Router(_resolve_config_paths(config))


def _resolve_config_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config_paths(item) for item in value]
    if not isinstance(value, str):
        return value
    replacements = {
        "{RESEARCH_ENGINE_AGY_BIN}": paths.executable(
            paths.AGY_BIN_ENV, "agy-cli-1", "agy-cli-2", "agy"
        )
        or "agy",
        "{RESEARCH_ENGINE_GROK_BIN}": paths.executable(paths.GROK_BIN_ENV, "grok") or "grok",
        "{RESEARCH_ENGINE_BUZZ_SCRIPT}": str(
            paths.optional_path(paths.BUZZ_SCRIPT_ENV) or paths.package_path("tools", "buzz-disabled")
        ),
        "{RESEARCH_ENGINE_MEMORY_DB}": str(paths.data_path("memory.db")),
        "{RESEARCH_ENGINE_CLAUDE_MEMORY_GLOB}": os.environ.get(
            paths.CLAUDE_MEMORY_GLOB_ENV,
            str(paths.data_path("claude-memory", "*.md")),
        ),
        "{RESEARCH_ENGINE_OBSIDIAN_GLOB}": os.environ.get(
            paths.OBSIDIAN_GLOB_ENV,
            str(paths.data_path("obsidian", "**", "*.md")),
        ),
        "{RESEARCH_ENGINE_MISTRAL_KEYS_FILE}": str(
            paths.optional_path(paths.MISTRAL_KEYS_FILE_ENV) or paths.data_path("missing-mistral-free-keys.env")
        ),
        "{RESEARCH_ENGINE_USER_AGENT}": paths.user_agent(),
    }
    for token, replacement in replacements.items():
        value = value.replace(token, replacement)
    return value


def source_authority_score(domain: str, topic: str | None) -> float:
    """Return configured authority, preserving the legacy score without a topic."""
    if not topic:
        return LEGACY_AUTHORITY_SCORE

    global _DEFAULT_ROUTER, _DEFAULT_ROUTER_LOAD_ATTEMPTED
    if not _DEFAULT_ROUTER_LOAD_ATTEMPTED:
        _DEFAULT_ROUTER_LOAD_ATTEMPTED = True
        try:
            _DEFAULT_ROUTER = load_router()
        except Exception as exc:
            logger.warning(
                "router load failed; using legacy authority score for topic %r: %s",
                topic,
                exc,
            )
            _DEFAULT_ROUTER = None

    if _DEFAULT_ROUTER is None:
        return LEGACY_AUTHORITY_SCORE
    try:
        return _DEFAULT_ROUTER.authority_score(domain, topic)
    except Exception as exc:
        logger.warning(
            "authority scoring failed for domain %r topic %r; using legacy score: %s",
            domain,
            topic,
            exc,
        )
        return LEGACY_AUTHORITY_SCORE


class Router:
    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError("Router config must be a mapping")

        self._config = deepcopy(config)
        _warn_if_unexpected_schema_version(self._config.get("schema_version"))
        self._lanes = self._require_mapping("lanes")
        self._authority_weights = self._mapping_or_empty("authority_weights")
        self._reranker = self._mapping_or_empty("reranker")
        self._failsafe = self._mapping_or_empty("failsafe")
        self._rules = self._compile_rules(self._config.get("rules"))

    @property
    def config(self) -> dict:
        return deepcopy(self._config)

    def route(self, question: str) -> RoutingDecision:
        normalized_question = _normalize_question(question)
        matches = []
        for rule_index, rule in enumerate(self._rules):
            matched_patterns = _matched_patterns(
                rule.patterns,
                rule.pattern_regexes,
                normalized_question,
            )
            if matched_patterns:
                match_count = (
                    0 if rule.patterns == (ALWAYS_MATCH_PATTERN,) else len(matched_patterns)
                )
                matches.append(
                    (
                        -match_count,
                        -sum(len(pattern) for pattern in matched_patterns),
                        rule_index,
                        rule,
                        matched_patterns,
                    )
                )

        if matches:
            matches.sort(key=lambda match: match[:3])
            primary_rule = matches[0][3]
            contributing_matches = matches
            if primary_rule.patterns != (ALWAYS_MATCH_PATTERN,):
                contributing_matches = [
                    match
                    for match in matches
                    if match[3].patterns != (ALWAYS_MATCH_PATTERN,)
                ]
            lanes = list(
                dict.fromkeys(
                    lane
                    for match in contributing_matches
                    for lane in match[3].lanes
                )
            )
            if not primary_rule.skip_web:
                for lane in ("searxng_general", "exa_direct"):
                    if lane not in lanes:
                        lanes.append(lane)

            return RoutingDecision(
                rule_name=primary_rule.name,
                lanes=lanes[:10],
                worker_model=primary_rule.worker_model,
                topic=primary_rule.topic,
                require_tier_1=primary_rule.require_tier_1,
                skip_web=primary_rule.skip_web,
                bias=primary_rule.bias,
                freshness_max_age_hours=primary_rule.freshness_max_age_hours,
                matched_patterns=matches[0][4],
                contributing_rules=[match[3].name for match in contributing_matches],
            )

        raise RuntimeError("No routing rule matched; define a catch-all rule with ['.*'].")

    def lane_endpoint(self, lane_name: str) -> dict:
        try:
            lane = self._lanes[lane_name]
        except KeyError as exc:
            raise KeyError(f"Unknown lane: {lane_name}") from exc
        return deepcopy(lane)

    def authority_score(self, domain: str, topic: str) -> float:
        topic_weights = self._authority_weights.get(topic)
        if not isinstance(topic_weights, dict):
            return DEFAULT_AUTHORITY_SCORE

        normalized_domain = _normalize_domain(domain)
        for configured_domain, score in topic_weights.items():
            if not isinstance(configured_domain, str):
                continue
            if _domain_matches(normalized_domain, configured_domain):
                return float(score)

        return DEFAULT_AUTHORITY_SCORE

    def reranker_config(self) -> dict:
        return deepcopy(self._reranker)

    def failsafe_config(self) -> dict:
        return deepcopy(self._failsafe)

    def scout_config(self) -> dict:
        return self._mapping_or_empty("scout")

    def recursion_config(self) -> dict:
        return self._mapping_or_empty("recursion")

    def judge_config(self) -> dict:
        return self._mapping_or_empty("judge")

    def graduated_answer_config(self) -> dict:
        return self._mapping_or_empty("graduated_answer")

    def fleets_config(self) -> dict:
        return self._mapping_or_empty("fleets")

    def fleet_worker_models(self, fleet_name: str) -> list[str]:
        """Flatten one `fleets:` entry into an ordered worker-model list."""
        fleet = self.fleets_config().get(fleet_name)
        if not isinstance(fleet, dict):
            raise ValueError(
                f"Router config 'fleets' has no mapping for fleet '{fleet_name}'"
            )

        workers = fleet.get("workers")
        if not isinstance(workers, list) or not workers:
            raise ValueError(f"Fleet '{fleet_name}' missing a non-empty 'workers' list")

        models: list[str] = []
        for position, worker in enumerate(workers, start=1):
            slot = f"fleets.{fleet_name}.workers[{position}]"
            if not isinstance(worker, dict):
                raise ValueError(f"Fleet slot '{slot}' must be a mapping")
            count = worker.get("count", 1)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(
                    f"Fleet slot '{slot}' has invalid 'count' {count!r}; "
                    "expected a positive integer"
                )
            models.extend([_resolve_worker_model(worker.get("model"), slot)] * count)

        declared_count = fleet.get("worker_count")
        if declared_count is not None:
            if not isinstance(declared_count, int) or isinstance(declared_count, bool):
                raise ValueError(
                    f"Fleet '{fleet_name}' field 'worker_count' must be an integer"
                )
            if declared_count != len(models):
                raise ValueError(
                    f"Fleet '{fleet_name}' declares worker_count {declared_count} "
                    f"but its workers sum to {len(models)}"
                )
        return models

    @property
    def iteration_policy(self) -> IterationPolicy:
        raw = self._mapping_or_empty("iteration")
        return IterationPolicy(
            max_loops=int(raw.get("max_loops", 0)),
            trigger_on_gap_severities=[
                str(item).lower() for item in raw.get("trigger_on_gap_severities", [])
            ],
            per_iteration_lane_cap=max(1, int(raw.get("per_iteration_lane_cap", 1))),
            cost_cap_usd_per_session=float(raw.get("cost_cap_usd_per_session", 0.0)),
        )

    def _compile_rules(self, rules: Any) -> tuple[_CompiledRule, ...]:
        if not isinstance(rules, list):
            raise ValueError("Router config missing required top-level field 'rules'")
        return tuple(self._compile_rule(rule) for rule in rules)

    def _compile_rule(self, rule: Any) -> _CompiledRule:
        if not isinstance(rule, dict):
            raise ValueError("Rule '<unnamed rule>' must be a mapping")

        rule_name = rule.get("name", "<unnamed rule>")
        for field_name in _REQUIRED_RULE_FIELDS:
            if field_name not in rule:
                raise ValueError(
                    f"Rule '{rule_name}' missing required field '{field_name}'"
                )

        patterns = _validated_string_list(
            rule["patterns"],
            field_name="patterns",
            rule_name=rule_name,
        )
        lanes = _validated_string_list(
            rule["lanes"],
            field_name="lanes",
            rule_name=rule_name,
        )
        for lane in lanes:
            if lane not in self._lanes:
                raise ValueError(f"Rule '{rule_name}' references unknown lane '{lane}'")

        worker_model = _resolve_worker_model(rule["worker_model"], rule_name)
        topic = _validated_string(rule["topic"], "topic", rule_name)
        require_tier_1 = _validated_bool(
            rule.get("require_tier_1", False),
            field_name="require_tier_1",
            rule_name=rule_name,
        )
        skip_web = _validated_bool(
            rule.get("skip_web", False),
            field_name="skip_web",
            rule_name=rule_name,
        )
        bias = _validated_optional_string(rule.get("bias"), "bias", rule_name)
        freshness = _validated_optional_int(
            rule.get("freshness_max_age_hours"),
            field_name="freshness_max_age_hours",
            rule_name=rule_name,
        )

        return _CompiledRule(
            name=_validated_string(rule["name"], "name", rule_name),
            patterns=tuple(patterns),
            pattern_regexes=tuple(_compile_pattern(pattern) for pattern in patterns),
            lanes=tuple(lanes),
            worker_model=worker_model,
            topic=topic,
            require_tier_1=require_tier_1,
            skip_web=skip_web,
            bias=bias,
            freshness_max_age_hours=freshness,
        )

    def _require_mapping(self, key: str) -> dict[str, Any]:
        value = self._config.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Router config missing required top-level field '{key}'")
        return value

    def _mapping_or_empty(self, key: str) -> dict[str, Any]:
        value = self._config.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"Router config field '{key}' must be a mapping")
        return value


def _validated_string_list(value: Any, field_name: str, rule_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Rule '{rule_name}' missing required field '{field_name}'")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Rule '{rule_name}' field '{field_name}' must contain non-empty strings"
            )
        normalized.append(item.strip())
    return normalized


def _validated_string(value: Any, field_name: str, rule_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Rule '{rule_name}' missing required field '{field_name}'")
    return value.strip()


def _validated_optional_string(
    value: Any, field_name: str, rule_name: str
) -> Optional[str]:
    if value is None:
        return None
    return _validated_string(value, field_name, rule_name)


def _validated_bool(value: Any, field_name: str, rule_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Rule '{rule_name}' field '{field_name}' must be a boolean")
    return value


def _validated_optional_int(value: Any, field_name: str, rule_name: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Rule '{rule_name}' field '{field_name}' must be an integer")
    return value


def _resolve_worker_model(value: Any, rule_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Rule '{rule_name}' missing required field 'worker_model'")

    candidate = value.strip()
    for model in WorkerModel:
        if candidate == model.value or candidate.lower() == model.name.lower():
            return model.value

    raise ValueError(f"Rule '{rule_name}' has unknown worker_model '{candidate}'")


def _warn_if_unexpected_schema_version(value: Any) -> None:
    if value == EXPECTED_SCHEMA_VERSION:
        return
    found = "<missing>" if value is None else str(value)
    logger.warning(
        "Router config schema_version mismatch: found %s, expected %s; continuing.",
        found,
        EXPECTED_SCHEMA_VERSION,
    )


def _normalize_question(question: str) -> str:
    normalized = question.strip().lower()
    for prefix in _HOW_TO_PREFIXES:
        if normalized.startswith(prefix):
            return f"how to {normalized[len(prefix):]}"
    return normalized


def _compile_pattern(pattern: str) -> Optional[re.Pattern[str]]:
    if pattern == ALWAYS_MATCH_PATTERN:
        return None

    expression = re.escape(pattern)
    if re.match(r"[0-9A-Za-z_]", pattern[0]):
        expression = rf"\b{expression}"
    if re.match(r"[0-9A-Za-z_]", pattern[-1]):
        expression = rf"{expression}(?:s|es)?\b"
    return re.compile(expression, re.IGNORECASE)


def _matched_patterns(
    patterns: tuple[str, ...],
    pattern_regexes: tuple[Optional[re.Pattern[str]], ...],
    question: str,
) -> list[str]:
    matched: list[str] = []
    for pattern, pattern_regex in zip(patterns, pattern_regexes):
        if pattern == ALWAYS_MATCH_PATTERN:
            matched.append(pattern)
            continue
        if pattern_regex is not None and pattern_regex.search(question):
            matched.append(pattern)
    return matched


def _normalize_domain(domain: str) -> str:
    normalized = domain.strip().lower()
    if "://" in normalized:
        normalized = normalized.split("://", 1)[1]
    normalized = normalized.split("/", 1)[0]
    normalized = normalized.split(":", 1)[0]
    return normalized.lstrip(".")


def _domain_matches(input_domain: str, configured_domain: str) -> bool:
    candidate = configured_domain.strip().lower().lstrip(".")
    return input_domain == candidate or input_domain.endswith(f".{candidate}")


def _yaml_error_message(path: Path, raw_text: str, exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if mark is None:
        return f"Failed to parse YAML in {path}: {exc}"

    line_number = mark.line + 1
    lines = raw_text.splitlines()
    context_line = lines[mark.line].rstrip() if mark.line < len(lines) else "<unknown>"
    return f"Failed to parse YAML in {path} at line {line_number}: {context_line}"


if __name__ == "__main__":
    router = load_router()

    decision = router.route("Supreme Court ruling on EPA case")
    assert decision.rule_name == "federal_court_ruling"
    assert "courtlistener" in decision.lanes

    decision = router.route("how do I use python asyncio")
    assert decision.rule_name == "code_pattern"
    assert "github_code" in decision.lanes

    decision = router.route("random question with no matching keywords")
    assert decision.rule_name == "general_web"

    assert router.authority_score("nejm.org", "medicine") == 1.0
    assert router.authority_score("unknown-site.xyz", "medicine") == 0.5
    assert router.lane_endpoint("courtlistener")["type"] == "api"

    print(
        f"router self-test passed: {len(router._rules)} rules, {len(router._lanes)} lanes"
    )
