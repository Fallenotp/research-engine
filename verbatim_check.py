"""Cheap verbatim triage for sourced synthesis.

This module catches fabricated or altered numbers, dates, and quoted phrases by
checking whether each token appears word-for-word in captured source text. It is
the cheap first net, not proof of truth: it does not catch correctly quoted but
wrongly reasoned conclusions.

Heuristics are intentionally conservative:
- tokens in sections headed uncertain, unknown, caveats, limitations, open
  questions, still unknown, or what we do not know are skipped;
- trivial single-digit numbering such as "1." bullets is ignored;
- whitespace and smart quotes are normalized before comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any


HONEST_SCOPE_NOTE = (
    "This catches fabricated or altered numbers, dates, and quotes. It does not "
    "catch correctly-quoted-but-wrongly-reasoned conclusions. It is the cheap "
    "first net, not proof of truth."
)


@dataclass(frozen=True)
class UnsupportedToken:
    token_type: str
    token: str
    context: str


@dataclass(frozen=True)
class VerbatimResult:
    checked_count: int
    supported_count: int
    unsupported_count: int
    pass_rate: float
    unsupported: list[UnsupportedToken] = field(default_factory=list)
    scope_note: str = HONEST_SCOPE_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "unsupported": [asdict(item) for item in self.unsupported],
        }


_SMART_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
)

_UNCERTAIN_HEADINGS = (
    "uncertain",
    "unknown",
    "open question",
    "open questions",
    "still unknown",
    "what we don't know",
    "what we do not know",
    "limitation",
    "limitations",
    "caveat",
    "caveats",
)

_QUOTE_RE = re.compile(r'"([^"\n]{3,240})"')
_DATE_RES = (
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(
        r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\.?(?:\s+\d{4})?\b",
        re.IGNORECASE,
    ),
)
_NUMBER_RE = re.compile(
    r"(?<![\w/])(?:[$€£]\s*)?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"(?:\s?(?:%|percent|million|billion|trillion|thousand|k|m|bn))?"
    r"|(?<![\w/])(?:[$€£]\s*)?\d+(?:\.\d+)?"
    r"(?:\s?(?:%|percent|million|billion|trillion|thousand|k|m|bn))?"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def check_verbatim(synthesis_text: str, source_texts: list[str]) -> VerbatimResult:
    """Check extracted numbers, dates, and quoted spans against source text."""

    comparable_sources = [_normalize_text(text) for text in source_texts if text]
    tokens = _extract_tokens(synthesis_text)
    unsupported: list[UnsupportedToken] = []
    supported_count = 0

    for token_type, token, context in tokens:
        comparable = _normalize_text(token)
        if comparable and any(comparable in source for source in comparable_sources):
            supported_count += 1
        else:
            unsupported.append(
                UnsupportedToken(
                    token_type=token_type,
                    token=token,
                    context=_compact(context, 260),
                )
            )

    checked_count = len(tokens)
    unsupported_count = len(unsupported)
    pass_rate = supported_count / checked_count if checked_count else 1.0
    return VerbatimResult(
        checked_count=checked_count,
        supported_count=supported_count,
        unsupported_count=unsupported_count,
        pass_rate=round(pass_rate, 4),
        unsupported=unsupported,
    )


def source_texts_from_paths(paths: list[str | Path]) -> tuple[list[str], list[str]]:
    """Read local source-text paths. Returns (texts, unavailable_labels)."""

    texts: list[str] = []
    unavailable: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            else:
                unavailable.append(str(raw_path))
        except OSError:
            unavailable.append(str(raw_path))
    return texts, unavailable


def result_to_markdown(result: VerbatimResult, unavailable_sources: list[str] | None = None) -> str:
    """Render a compact report section for user-facing answers."""

    lines = ["## Verbatim check", "", result.scope_note, ""]
    unavailable_sources = unavailable_sources or []
    if result.unsupported:
        lines.append("Not found word-for-word in captured source text:")
        for item in result.unsupported:
            lines.append(f"- {item.token_type}: `{item.token}` — {item.context}")
    else:
        lines.append("All checked tokens found in sources.")
    if unavailable_sources:
        lines.append("")
        lines.append("Unverifiable source text not retrieved:")
        for source in unavailable_sources:
            lines.append(f"- {source}")
    lines.append("")
    lines.append(
        f"Checked {result.checked_count} token(s); pass rate {result.pass_rate:.0%}."
    )
    return "\n".join(lines)


def _extract_tokens(text: str) -> list[tuple[str, str, str]]:
    filtered = _drop_uncertain_sections(text)
    tokens: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for match in _QUOTE_RE.finditer(_normalize_quotes(filtered)):
        token = match.group(1).strip()
        if token:
            _append_token(tokens, seen, "quote", token, _sentence_for(filtered, match.start()))

    for date_re in _DATE_RES:
        for match in date_re.finditer(filtered):
            token = match.group(0).strip()
            _append_token(tokens, seen, "date", token, _sentence_for(filtered, match.start()))

    for match in _NUMBER_RE.finditer(filtered):
        token = match.group(0).strip()
        if _is_trivial_number(token, filtered, match.start(), match.end()):
            continue
        _append_token(tokens, seen, "number", token, _sentence_for(filtered, match.start()))

    return tokens


def _append_token(
    tokens: list[tuple[str, str, str]],
    seen: set[tuple[str, str, str]],
    token_type: str,
    token: str,
    context: str,
) -> None:
    key = (token_type, _normalize_text(token), _normalize_text(context))
    if key in seen:
        return
    seen.add(key)
    tokens.append((token_type, token, context))


def _drop_uncertain_sections(text: str) -> str:
    lines: list[str] = []
    skipping = False
    for line in text.splitlines():
        stripped = line.strip()
        heading_text = stripped.lstrip("#").strip().lower()
        is_heading = stripped.startswith("#") or (
            stripped.endswith(":") and len(stripped) <= 80
        )
        if is_heading:
            skipping = any(marker in heading_text for marker in _UNCERTAIN_HEADINGS)
            if skipping:
                continue
        if not skipping:
            lines.append(line)
    return "\n".join(lines)


def _is_trivial_number(token: str, text: str, start: int, end: int) -> bool:
    bare = token.strip().strip("$€£").strip()
    if not bare.isdigit() or len(bare) != 1:
        return False
    before = text[max(0, start - 4) : start]
    after = text[end : min(len(text), end + 3)]
    if bool(re.search(r"(^|\n)\s*$", before)) and after.startswith("."):
        return True
    return before.endswith("(") and after.startswith(")")


def _sentence_for(text: str, index: int) -> str:
    start_candidates = [text.rfind(mark, 0, index) for mark in (".", "!", "?", "\n")]
    end_candidates = [
        pos for pos in (text.find(mark, index) for mark in (".", "!", "?", "\n")) if pos != -1
    ]
    start = max(start_candidates) + 1
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return _compact(text[start:end], 280)


def _normalize_quotes(text: str) -> str:
    return text.translate(_SMART_TRANSLATION)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", _normalize_quotes(text)).strip().lower()


def _compact(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."
