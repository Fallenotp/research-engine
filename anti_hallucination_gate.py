"""Small regex-based gate for Grok/Gemini research text."""

from __future__ import annotations

import re


URL_RE = re.compile(r"https?://[^\s<>()\]]+", re.IGNORECASE)
QUOTE_RE = re.compile(r'("[^"]{6,}"|“[^”]{6,}”|\'[^\']{6,}\'|‘[^’]{6,}’)')
HEDGE_RE = re.compile(
    r"\b(probably|i think|likely|seems|might)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?!https?://)[A-Z0-9\"'“‘])")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
VERBISH_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|will|would|can|could|did|does|"
    r"released|announced|launched|reported|confirmed|said|found|showed|"
    r"shipped|reached|reach|grew|fell|rose|declined|increased|decreased)\b",
    re.IGNORECASE,
)


def validate(
    text: str,
    source_required: bool = True,
) -> tuple[str, int, list[str]]:
    """Drop unsupported factual-looking claims and flag risky hedging.

    A source is intentionally simple here: either a URL or a verbatim quoted
    passage. This gate is a cheap guardrail, not a full fact checker.
    """

    if not text.strip():
        return text, 0, []

    kept_lines: list[str] = []
    flagged: list[str] = []
    dropped = 0

    for line in text.splitlines():
        if not line.strip() or _skip_line(line):
            kept_lines.append(line)
            continue

        kept_parts: list[str] = []
        for claim in _claim_units(line):
            stripped = claim.strip()
            if not stripped:
                continue
            reasons = _flag_reasons(stripped, source_required=source_required)
            if reasons:
                dropped += 1
                flagged.append(f"{'+'.join(reasons)}: {_preview(stripped)}")
                continue
            kept_parts.append(stripped)

        if kept_parts:
            kept_lines.append(" ".join(kept_parts))

    if not flagged:
        return text, 0, []

    return "\n".join(kept_lines).strip(), dropped, flagged


def _flag_reasons(claim: str, *, source_required: bool) -> list[str]:
    if not _looks_like_claim(claim):
        return []

    has_source = _has_source(claim)
    reasons: list[str] = []
    if HEDGE_RE.search(claim) and not has_source:
        reasons.append("hedging_without_source")
    if source_required and not has_source:
        reasons.append("missing_source")
    return reasons


def _claim_units(line: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(line) if part.strip()]


def _has_source(text: str) -> bool:
    return bool(URL_RE.search(text) or QUOTE_RE.search(text))


def _skip_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(("#", "```", "|")):
        return True
    return stripped.endswith(":") and len(WORD_RE.findall(stripped)) <= 8


def _looks_like_claim(text: str) -> bool:
    words = WORD_RE.findall(text)
    if len(words) < 4:
        return False
    return bool(VERBISH_RE.search(text) or re.search(r"\d", text))


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."
