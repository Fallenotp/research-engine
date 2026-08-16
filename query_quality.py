from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable
from urllib.parse import urlsplit


SEARCH_OPERATOR_NAMES = ("site", "filetype", "intitle", "inurl")
LANE_SCOPED_OPERATOR_NAMES = ("site", "filetype")
BROADENING_OPERATOR_NAMES = frozenset({"site", "filetype", "inurl"})
TRIVIAL_QUERY_MIN_TOKENS = 2
BROADEN_LONG_QUERY_MIN_TOKENS = 5
# Lexical engines intersect keyword terms. Past roughly a dozen terms, these
# query strings usually become LLM spill rather than a retrievable keyword query.
MAX_MEANINGFUL_QUERY_TOKENS = 14
# Flag duplicates at two repeats so the known UK/UK spill is caught.
DUPLICATE_TERM_FLAG_THRESHOLD = 2
# Repair stays stricter. Collapsing a term that appears only twice can change
# meaning, while 3+ repeats are usually spill.
DUPLICATE_TERM_REPAIR_THRESHOLD = 3
# A retrieval set with fewer than 3 results is too brittle to treat as healthy.
GOOD_RESULT_MIN_COUNT = 3
# At least 2 domains avoids calling a single-site cluster "good" by default.
GOOD_RESULT_MIN_UNIQUE_DOMAINS = 2
# Once one domain owns ~80% of the set, retrieval is dominated by one source.
THIN_TOP_DOMAIN_SHARE = 0.8
_EXCEPTION_MARKER_KEYS = ("exception", "exception_type", "exception_message")
_HTTP_ERROR_STATUS_KEYS = ("status_code", "http_status", "status")

_WEB_OPERATOR_LANE_HINTS = (
    "searxng",
    "proxy",
    "exa",
    "linkup",
    "tavily",
    "youcom",
    "firecrawl",
    "serper",
    "web",
)
_QUERY_PARAM_NAMES = ("query", "q", "search_query", "term", "text", "keyword")
_OUTER_PUNCTUATION = ".,;:!?()[]{}<>"


class QueryIssueCode(str, Enum):
    TRUNCATED_OPERATOR = "TRUNCATED_OPERATOR"
    UNBALANCED_QUOTES = "UNBALANCED_QUOTES"
    EMPTY_OR_TRIVIAL = "EMPTY_OR_TRIVIAL"
    EXCESSIVE_LENGTH = "EXCESSIVE_LENGTH"
    DUPLICATE_TERMS = "DUPLICATE_TERMS"
    UNSUPPORTED_OPERATOR_FOR_LANE = "UNSUPPORTED_OPERATOR_FOR_LANE"


class ResultQualityVerdict(str, Enum):
    GOOD = "GOOD"
    THIN = "THIN"
    EMPTY = "EMPTY"
    ERROR = "ERROR"


@dataclass(frozen=True)
class QueryIssue:
    code: QueryIssueCode
    detail: str = ""


@dataclass(frozen=True)
class QueryIssues:
    issues: tuple[QueryIssue, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.issues)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code.value for issue in self.issues)

    def has(self, code: QueryIssueCode | str) -> bool:
        target = code.value if isinstance(code, QueryIssueCode) else str(code)
        return any(issue.code.value == target for issue in self.issues)

    def details_for(self, code: QueryIssueCode | str) -> tuple[str, ...]:
        target = code.value if isinstance(code, QueryIssueCode) else str(code)
        return tuple(issue.detail for issue in self.issues if issue.code.value == target)


@dataclass(frozen=True)
class ResultQuality:
    result_count: int
    empty: bool
    unique_domain_count: int
    top_domain_share: float
    has_error: bool
    error: str | None
    verdict: ResultQualityVerdict


def validate_query(query: str, lane: str | None = None) -> QueryIssues:
    text = str(query or "")
    tokens = _split_tokens(text)
    issues: list[QueryIssue] = []

    truncated = _truncated_operator_tokens(tokens)
    issues.extend(
        QueryIssue(QueryIssueCode.TRUNCATED_OPERATOR, token) for token in truncated
    )

    if _has_unbalanced_quotes(text):
        issues.append(QueryIssue(QueryIssueCode.UNBALANCED_QUOTES))

    meaningful_terms = _meaningful_terms(tokens)
    if len(meaningful_terms) < TRIVIAL_QUERY_MIN_TOKENS:
        issues.append(QueryIssue(QueryIssueCode.EMPTY_OR_TRIVIAL))
    if len(meaningful_terms) > MAX_MEANINGFUL_QUERY_TOKENS:
        issues.append(
            QueryIssue(
                QueryIssueCode.EXCESSIVE_LENGTH,
                str(len(meaningful_terms)),
            )
        )

    duplicate_terms = _duplicate_terms(meaningful_terms)
    issues.extend(
        QueryIssue(QueryIssueCode.DUPLICATE_TERMS, term) for term in duplicate_terms
    )

    if lane and not _lane_supports_web_operators(lane):
        unsupported = _unsupported_lane_operator_tokens(tokens)
        issues.extend(
            QueryIssue(QueryIssueCode.UNSUPPORTED_OPERATOR_FOR_LANE, token)
            for token in unsupported
        )

    return QueryIssues(tuple(issues))


def repair_query(query: str) -> tuple[str, list[str]]:
    """Drop obviously bad spill without changing meaningful word order.

    Repairs may remove malformed operators or repeated spill terms. The only
    non-removal edit allowed here is closing an unbalanced quote.
    """
    tokens = _split_tokens(str(query or ""))
    repairs: list[str] = []

    truncated_tokens = _truncated_operator_tokens(tokens)
    if truncated_tokens:
        truncated_set = set(truncated_tokens)
        tokens = [token for token in tokens if token not in truncated_set]
        repairs.extend(f"dropped truncated operator: {token}" for token in truncated_tokens)

    tokens, duplicate_repairs = _collapse_duplicate_terms(tokens)
    repairs.extend(duplicate_repairs)

    repaired = " ".join(tokens) if tokens else (query if query.isspace() else "")
    if _has_unbalanced_quotes(repaired) and repaired.strip():
        repaired = repaired.rstrip() + '"'
        repairs.append("closed unbalanced quote")

    return repaired, repairs


def score_result_quality(payload: dict, lane: str) -> ResultQuality:
    _ = lane
    error = _payload_error(payload)
    results = payload.get("results") if isinstance(payload, dict) else None
    result_items = results if isinstance(results, list) else []
    result_count = len(result_items)
    domain_counts = Counter(
        domain for domain in (_result_domain(item) for item in result_items) if domain
    )
    unique_domain_count = len(domain_counts)
    top_domain_share = 0.0
    if result_count:
        top_count = max(domain_counts.values(), default=result_count)
        top_domain_share = top_count / result_count

    if error:
        verdict = ResultQualityVerdict.ERROR
    elif result_count == 0:
        verdict = ResultQualityVerdict.EMPTY
    elif (
        result_count < GOOD_RESULT_MIN_COUNT
        or unique_domain_count < GOOD_RESULT_MIN_UNIQUE_DOMAINS
        or top_domain_share > THIN_TOP_DOMAIN_SHARE
    ):
        verdict = ResultQualityVerdict.THIN
    else:
        verdict = ResultQualityVerdict.GOOD

    return ResultQuality(
        result_count=result_count,
        empty=result_count == 0,
        unique_domain_count=unique_domain_count,
        top_domain_share=top_domain_share,
        has_error=error is not None,
        error=error,
        verdict=verdict,
    )


def broaden_query(
    query: str,
    verdict: ResultQualityVerdict,
) -> tuple[str, list[str]] | None:
    if verdict not in {ResultQualityVerdict.EMPTY, ResultQualityVerdict.THIN}:
        return None

    original = str(query or "")
    if not original.strip():
        return None

    attempts = (
        _drop_restrictive_operator,
        _unquote_quoted_phrase,
        _drop_last_modifier_token,
    )
    for attempt in attempts:
        broadened = attempt(original)
        if broadened is None:
            continue
        candidate, transforms = broadened
        if _normalize_query_spacing(candidate) == _normalize_query_spacing(original):
            continue
        return candidate, transforms
    return None


def _split_tokens(query: str) -> list[str]:
    return query.split()


def _has_unbalanced_quotes(query: str) -> bool:
    return query.count('"') % 2 == 1


def _truncated_operator_tokens(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if _is_truncated_operator_token(token)]


def _unsupported_lane_operator_tokens(tokens: Iterable[str]) -> list[str]:
    return [
        token
        for token in tokens
        if _operator_name(token) in LANE_SCOPED_OPERATOR_NAMES
    ]


def _duplicate_terms(meaningful_terms: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    first_forms: dict[str, str] = {}
    ordered: list[str] = []
    for term in meaningful_terms:
        normalized = term.lower()
        counts[normalized] += 1
        first_forms.setdefault(normalized, term)
        if normalized not in ordered:
            ordered.append(normalized)
    return [
        first_forms[term]
        for term in ordered
        if counts[term] >= DUPLICATE_TERM_FLAG_THRESHOLD
    ]


def _collapse_duplicate_terms(tokens: list[str]) -> tuple[list[str], list[str]]:
    meaningful_terms = _meaningful_terms(tokens)
    counts = Counter(term.lower() for term in meaningful_terms)
    repairable_terms = {
        term for term, count in counts.items() if count >= DUPLICATE_TERM_REPAIR_THRESHOLD
    }
    if not repairable_terms:
        return tokens, []

    seen_terms: set[str] = set()
    repaired_tokens: list[str] = []
    repaired_names: list[str] = []
    for token in tokens:
        meaningful = _meaningful_token(token)
        if meaningful is None:
            repaired_tokens.append(token)
            continue
        normalized = meaningful.lower()
        if normalized not in repairable_terms:
            repaired_tokens.append(token)
            continue
        if normalized in seen_terms:
            if meaningful not in repaired_names:
                repaired_names.append(meaningful)
            continue
        seen_terms.add(normalized)
        repaired_tokens.append(token)

    repairs = [f"collapsed duplicated term: {term}" for term in repaired_names]
    return repaired_tokens, repairs


def _meaningful_terms(tokens: Iterable[str]) -> list[str]:
    return [meaningful for token in tokens if (meaningful := _meaningful_token(token)) is not None]


def _drop_restrictive_operator(query: str) -> tuple[str, list[str]] | None:
    tokens = _split_tokens(query)
    for index, token in enumerate(tokens):
        if not _is_broadening_operator_token(token):
            continue
        broadened_tokens = tokens[:index] + tokens[index + 1 :]
        if not broadened_tokens:
            return None
        return (
            " ".join(broadened_tokens),
            [f"dropped restrictive operator: {token}"],
        )
    return None


def _unquote_quoted_phrase(query: str) -> tuple[str, list[str]] | None:
    match = re.search(r'"([^"\n]+)"', query)
    if match is None:
        return None
    phrase = match.group(1)
    broadened = f"{query[: match.start()]}{phrase}{query[match.end() :]}"
    return broadened, [f'removed exact-phrase quotes: "{phrase}"']


def _drop_last_modifier_token(query: str) -> tuple[str, list[str]] | None:
    tokens = _split_tokens(query)
    if len(tokens) <= 1:
        return None
    if len(_meaningful_terms(tokens)) < BROADEN_LONG_QUERY_MIN_TOKENS:
        return None
    dropped = tokens[-1]
    broadened_tokens = tokens[:-1]
    if not broadened_tokens:
        return None
    return " ".join(broadened_tokens), [f"dropped trailing modifier: {dropped}"]


def _meaningful_token(token: str) -> str | None:
    raw = token.strip()
    if not raw:
        return None
    if raw == "-":
        return None
    negative = raw.startswith("-")
    core = raw[1:] if negative else raw
    core = core.strip('"')
    if _operator_name(raw):
        return None
    core = core.strip(_OUTER_PUNCTUATION)
    if not core or not re.search(r"[A-Za-z0-9]", core):
        return None
    return core


def _is_broadening_operator_token(token: str) -> bool:
    raw = token.strip()
    if not raw:
        return False
    if raw.startswith("-") and raw != "-":
        return True
    return _operator_name(raw) in BROADENING_OPERATOR_NAMES


def _operator_name(token: str) -> str | None:
    raw = token.strip()
    if not raw:
        return None
    if raw == "-":
        return "-"
    core = raw[1:] if raw.startswith("-") else raw
    core = core.strip('"')
    lowered = core.lower()
    for operator in SEARCH_OPERATOR_NAMES:
        if lowered.startswith(f"{operator}:"):
            return operator
    return None


def _is_truncated_operator_token(token: str) -> bool:
    raw = token.strip()
    if not raw:
        return False
    if raw.strip('"') == "-":
        return True
    operator = _operator_name(raw)
    if operator is None or operator == "-":
        return False

    core = raw[1:] if raw.startswith("-") else raw
    core = core.strip('"')
    value = core[len(operator) + 1 :].strip().strip('"')
    if not value:
        return True

    if operator == "site":
        return "." not in value
    if operator == "filetype":
        return len(value.strip(".")) < 2
    if operator in {"intitle", "inurl"}:
        return len(value) < 2
    return False


def _lane_supports_web_operators(lane: str) -> bool:
    lowered = str(lane or "").strip().lower()
    return any(hint in lowered for hint in _WEB_OPERATOR_LANE_HINTS)


def _normalize_query_spacing(query: str) -> str:
    return " ".join(str(query or "").split())


def _payload_error(payload: dict | object) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if "error" in payload:
        return _stringify_error_marker(error, default="error")
    for key in _EXCEPTION_MARKER_KEYS:
        marker = payload.get(key)
        if key in payload:
            return _stringify_error_marker(marker, default=key)
    for key in _HTTP_ERROR_STATUS_KEYS:
        if (status := _http_error_status(payload.get(key))) is not None:
            return f"HTTP {status}"
    return None


def _stringify_error_marker(marker: object, *, default: str) -> str:
    if isinstance(marker, str):
        text = marker.strip()
        return text or default
    text = str(marker).strip()
    return text or default


def _http_error_status(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 400 else None
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    match = re.search(r"\b([1-5][0-9]{2})\b", text)
    if match is None:
        return None
    status = int(match.group(1))
    return status if status >= 400 else None


def _result_domain(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    raw_domain = item.get("domain")
    if isinstance(raw_domain, str) and raw_domain.strip():
        return _normalize_domain(raw_domain)
    for field in ("url", "link", "absolute_url", "html_url", "paper_url", "story_url"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return _domain_from_url(value)
    return None


def _domain_from_url(url: str) -> str | None:
    parsed = urlsplit(url)
    host = parsed.netloc or parsed.path
    if not host:
        return None
    return _normalize_domain(host)


def _normalize_domain(value: str) -> str | None:
    lowered = value.strip().lower()
    if not lowered:
        return None
    if "@" in lowered and "/" not in lowered:
        return None
    host = lowered.split("/", 1)[0]
    host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None
