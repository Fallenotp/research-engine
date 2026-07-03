from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from research_engine.schema import WorkerModel


DEFAULT_CONFIG_PATH = "/Users/cleo/lattice/research_engine/router_config.yaml"
ALWAYS_MATCH_PATTERN = ".*"
DEFAULT_AUTHORITY_SCORE = 0.5
_REQUIRED_RULE_FIELDS = ("name", "patterns", "lanes", "worker_model", "topic")
_HOW_TO_PREFIXES = ("how do i ", "how can i ", "how should i ")


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

    return Router(config)


class Router:
    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError("Router config must be a mapping")

        self._config = deepcopy(config)
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
        for rule in self._rules:
            matched_patterns = _matched_patterns(rule.patterns, normalized_question)
            if matched_patterns:
                return RoutingDecision(
                    rule_name=rule.name,
                    lanes=list(rule.lanes),
                    worker_model=rule.worker_model,
                    topic=rule.topic,
                    require_tier_1=rule.require_tier_1,
                    skip_web=rule.skip_web,
                    bias=rule.bias,
                    freshness_max_age_hours=rule.freshness_max_age_hours,
                    matched_patterns=matched_patterns,
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


def _normalize_question(question: str) -> str:
    normalized = question.strip().lower()
    for prefix in _HOW_TO_PREFIXES:
        if normalized.startswith(prefix):
            return f"how to {normalized[len(prefix):]}"
    return normalized


def _matched_patterns(patterns: tuple[str, ...], question: str) -> list[str]:
    matched: list[str] = []
    for pattern in patterns:
        if pattern == ALWAYS_MATCH_PATTERN:
            matched.append(pattern)
            continue
        if pattern.lower() in question:
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
