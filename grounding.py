from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import research_engine.logged_search as logged_search
from research_engine.extractor import compact_search_results, extract_clean_text
from research_engine.schema import ExtractionMethod, SourceRecord, SourceTier


GroundStatus = Literal["grounded", "partial", "not_found"]

GROUNDING_PROTOCOL = "/grounding"
DEFAULT_GROK_BIN = Path("/Users/cleo/.local/bin/hermes")
DEFAULT_GROK_MODEL = "grok-4.20-0309-reasoning"
DEFAULT_AGY_BIN = Path("/Users/cleo/bin/agy-cli-1")
DEFAULT_AGY_FALLBACKS = (
    Path("/Users/cleo/bin/agy-cli-2"),
    Path("/Users/cleo/.local/bin/agy"),
)
AGY_SKIP_PERMISSIONS_FLAG = "--dangerously-skip-permissions"
TELEMETRY_PATH = Path(
    os.environ.get("NO_BLUFF_TELEMETRY_PATH", "/Users/cleo/.claude/no-bluff-telemetry.jsonl")
)
MAX_SEARCH_RESULTS = 5
MAX_SOURCE_URLS = 3
SOURCE_EXCERPT_CHARS = 2500
LOOKUP_TIMEOUT_SECONDS = 60
URL_RE = re.compile(r"https?://[^\s<>()\]]+")
JSON_RE = re.compile(r"\{[\s\S]*\}")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "latest",
    "of",
    "on",
    "or",
    "question",
    "the",
    "this",
    "to",
    "today",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
}

__all__ = [
    "BackendLookupResult",
    "GroundResult",
    "GroundSource",
    "ground",
    "main",
]


@dataclass(frozen=True, slots=True)
class GroundSource:
    url: str
    tier: SourceTier
    raw_text_path: str


@dataclass(frozen=True, slots=True)
class GroundResult:
    status: GroundStatus
    answer: str
    confidence: float
    sources: list[GroundSource]
    backends_used: list[str]


@dataclass(frozen=True, slots=True)
class BackendLookupResult:
    answer: str
    urls: list[str]


@dataclass(frozen=True, slots=True)
class _VerifiedSource:
    source: GroundSource
    record: SourceRecord
    full_text: str
    snippet_hint: str


def ground(query: str, *, topic_slug: str | None = None) -> GroundResult:
    cleaned_query = " ".join(query.split()).strip()
    if not cleaned_query:
        return GroundResult(
            status="not_found",
            answer="",
            confidence=0.0,
            sources=[],
            backends_used=["searxng"],
        )

    topic = topic_slug or _slugify_query(cleaned_query)
    backends_used = ["searxng"]
    search_payload = logged_search.searxng(
        _search_query(cleaned_query),
        protocol=GROUNDING_PROTOCOL,
        topic=topic,
    )
    search_results = compact_search_results(_payload_results(search_payload), max_results=MAX_SEARCH_RESULTS)
    search_urls = _candidate_search_urls(search_results, cleaned_query)
    search_snippets = {
        result["url"]: str(result.get("snippet") or "").strip()
        for result in search_results
        if result.get("url")
    }
    backend_answers: list[str] = []

    with tempfile.TemporaryDirectory(prefix=f"grounding-{topic}-") as tmpdir:
        seen_urls_path = Path(tmpdir) / "seen-urls.txt"
        sources = _extract_verified_sources(
            search_urls,
            seen_urls_path=seen_urls_path,
            snippet_by_url=search_snippets,
        )
        used_fallback = False
        if _should_escalate(search_results, cleaned_query) or not sources:
            sources = _merge_sources(
                sources,
                _lookup_fallback_sources(
                    cleaned_query,
                    seen_urls_path=seen_urls_path,
                    backends_used=backends_used,
                    backend_answers=backend_answers,
                ),
            )
            used_fallback = True

        synthesized = _synthesize_from_sources(
            cleaned_query,
            sources,
            search_results=search_results,
            backend_answers=backend_answers,
        )
        if synthesized[0] == "not_found" and not used_fallback:
            sources = _merge_sources(
                sources,
                _lookup_fallback_sources(
                    cleaned_query,
                    seen_urls_path=seen_urls_path,
                    backends_used=backends_used,
                    backend_answers=backend_answers,
                ),
            )
            synthesized = _synthesize_from_sources(
                cleaned_query,
                sources,
                search_results=search_results,
                backend_answers=backend_answers,
            )

    status, answer, confidence = synthesized
    if status == "not_found":
        return GroundResult(
            status="not_found",
            answer="",
            confidence=0.0,
            sources=[],
            backends_used=backends_used,
        )

    public_sources = [verified.source for verified in sources[:MAX_SOURCE_URLS]]
    return GroundResult(
        status=status,
        answer=answer,
        confidence=confidence,
        sources=public_sources,
        backends_used=backends_used,
    )


def _slugify_query(query: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug[:80].strip("-") or "grounding"


def _payload_results(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    return results if isinstance(results, list) else []


def _search_query(query: str) -> str:
    terms = _query_terms(query)
    if not terms:
        return query
    return " ".join(terms)


def _candidate_search_urls(results: list[dict], query: str) -> list[str]:
    query_terms = _query_terms(query)
    ranked_results = results
    if query_terms:
        ranked_results = sorted(
            results,
            key=lambda result: _result_relevance(result, query_terms),
            reverse=True,
        )
    urls: list[str] = []
    for result in ranked_results:
        haystack = " ".join(
            [
                str(result.get("title") or "").lower(),
                str(result.get("snippet") or "").lower(),
            ]
        )
        if _term_match_count(haystack, query_terms) < _min_term_matches(query_terms):
            continue
        url = str(result.get("url") or "").strip()
        if url:
            urls.append(url)
        if len(urls) >= MAX_SOURCE_URLS:
            break
    return _dedupe_preserve_order(urls)


def _should_escalate(results: list[dict], query: str) -> bool:
    return not _candidate_search_urls(results, query)


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]{3,}", query.lower())
    return [term for term in terms if term not in STOPWORDS]


def _result_relevance(result: dict, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0
    haystack = " ".join(
        [
            str(result.get("title") or "").lower(),
            str(result.get("snippet") or "").lower(),
        ]
    )
    matches = _term_match_count(haystack, query_terms)
    return matches / len(query_terms)


def _lookup_fallback_sources(
    query: str,
    *,
    seen_urls_path: Path,
    backends_used: list[str],
    backend_answers: list[str],
) -> list[_VerifiedSource]:
    collected: list[_VerifiedSource] = []
    for backend in ("grok", "gemini"):
        backends_used.append(backend)
        lookup = _run_backend_lookup(backend, query)
        if lookup.answer:
            backend_answers.append(lookup.answer)
        if not lookup.urls:
            continue
        collected = _merge_sources(
            collected,
            _extract_verified_sources(
                lookup.urls[:MAX_SOURCE_URLS],
                seen_urls_path=seen_urls_path,
                snippet_by_url={},
            ),
        )
        if collected:
            break
    return collected


def _run_backend_lookup(backend: str, query: str) -> BackendLookupResult:
    prompt = (
        "Answer this question using current web information. Then list the exact source URLs used.\n\n"
        f"QUESTION: {query}\n\n"
        "Return exactly this shape:\n"
        "ANSWER:\n"
        "<short answer or NOT_FOUND>\n\n"
        "SOURCES:\n"
        "<one URL per line>\n"
    )
    try:
        if backend == "grok":
            output = _run_grok(prompt)
        elif backend == "gemini":
            output = _run_gemini(prompt)
        else:
            return BackendLookupResult(answer="", urls=[])
    except Exception:
        return BackendLookupResult(answer="", urls=[])

    answer, urls = _parse_backend_output(output)
    return BackendLookupResult(answer=answer, urls=urls[:MAX_SOURCE_URLS])


def _run_grok(prompt: str) -> str:
    hermes_bin = Path(os.environ.get("GROUNDING_GROK_BIN", str(DEFAULT_GROK_BIN)))
    grok_model = os.environ.get("GROUNDING_GROK_MODEL") or DEFAULT_GROK_MODEL
    command = [str(hermes_bin), "-m", grok_model, "-z", prompt]
    response_text = ""
    try:
        if not _is_executable(hermes_bin):
            raise FileNotFoundError(str(hermes_bin))
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=LOOKUP_TIMEOUT_SECONDS,
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        response_text = stdout or stderr
        if completed.returncode != 0:
            raise RuntimeError(response_text)
        return stdout
    except Exception as exc:
        if not response_text:
            response_text = str(exc)
        raise
    finally:
        _append_telemetry(
            "grounding_escalation_grok",
            grok_model,
            prompt,
            response_text,
        )


def _run_gemini(prompt: str) -> str:
    agy_bin = _agy_binary()
    response_text = ""
    try:
        if agy_bin is None:
            raise FileNotFoundError("agy-cli-1")
        completed = subprocess.run(
            [agy_bin, AGY_SKIP_PERMISSIONS_FLAG, "--print", _agy_prompt_arg(prompt)],
            capture_output=True,
            text=True,
            timeout=LOOKUP_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        response_text = stdout or stderr
        if completed.returncode != 0:
            raise RuntimeError(response_text)
        return stdout
    except Exception as exc:
        if not response_text:
            response_text = str(exc)
        raise
    finally:
        _append_telemetry(
            "grounding_escalation_gemini",
            "agy-cli-1/gemini",
            prompt,
            response_text,
        )


def _agy_binary() -> str | None:
    configured = Path(os.environ.get("GROUNDING_AGY_BIN", str(DEFAULT_AGY_BIN)))
    for candidate in (configured, *DEFAULT_AGY_FALLBACKS):
        if _is_executable(candidate):
            return str(candidate)
    return None


def _agy_prompt_arg(prompt: str) -> str:
    if prompt.lstrip().startswith("-"):
        return f"Prompt:\n{prompt}"
    return prompt


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _parse_backend_output(output: str) -> tuple[str, list[str]]:
    cleaned = output.strip()
    answer = ""
    if "ANSWER:" in cleaned:
        tail = cleaned.split("ANSWER:", 1)[1]
        answer_block, _, _rest = tail.partition("SOURCES:")
        answer = answer_block.strip()
    urls = _extract_urls(cleaned)
    if answer.upper() == "NOT_FOUND":
        answer = ""
    return answer, urls


def _extract_urls(text: str) -> list[str]:
    matches = []
    for match in URL_RE.findall(text):
        url = match.rstrip(".,);]>")
        if url:
            matches.append(url)
    return _dedupe_preserve_order(matches)


def _extract_verified_sources(
    urls: list[str],
    *,
    seen_urls_path: Path,
    snippet_by_url: dict[str, str],
) -> list[_VerifiedSource]:
    verified: list[_VerifiedSource] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        extracted = extract_clean_text(
            url,
            seen_urls_path=seen_urls_path,
            tier=SourceTier.T2,
        )
        if not extracted:
            continue
        raw_text_path_value = extracted.get("raw_text_path")
        if not raw_text_path_value:
            continue
        raw_text_path = Path(str(raw_text_path_value))
        if not raw_text_path.exists():
            continue
        full_text = raw_text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not full_text:
            continue
        record = _source_record_from_extract(extracted, full_text)
        verified.append(
            _VerifiedSource(
                source=GroundSource(
                    url=record.url,
                    tier=record.tier,
                    raw_text_path=str(record.raw_text_path),
                ),
                record=record,
                full_text=full_text,
                snippet_hint=snippet_by_url.get(url, ""),
            )
        )
        if len(verified) >= MAX_SOURCE_URLS:
            break
    return verified


def _source_record_from_extract(extracted: dict, full_text: str) -> SourceRecord:
    method_value = str(extracted.get("extraction_method") or "")
    try:
        method = ExtractionMethod(method_value)
    except ValueError:
        method = ExtractionMethod.CURL
    url = str(extracted.get("url") or "")
    raw_text_path = Path(str(extracted["raw_text_path"]))
    return SourceRecord(
        url=url,
        domain=str(extracted.get("domain") or urlparse(url).netloc or "unknown"),
        title=str(extracted.get("title") or url or "Untitled"),
        author=extracted.get("author"),
        published_date=extracted.get("published_date"),
        fetched_at=datetime.now(timezone.utc),
        content_hash=SourceRecord.hash_text(full_text),
        extraction_method=method,
        raw_text_path=raw_text_path,
        char_count=int(extracted.get("char_count") or len(full_text)),
        tier=SourceTier.T2,
        topic_authority_score=0.6,
    )


def _merge_sources(
    existing: list[_VerifiedSource],
    incoming: list[_VerifiedSource],
) -> list[_VerifiedSource]:
    merged = list(existing)
    seen_urls = {source.source.url for source in merged}
    for source in incoming:
        if source.source.url in seen_urls:
            continue
        merged.append(source)
        seen_urls.add(source.source.url)
        if len(merged) >= MAX_SOURCE_URLS:
            break
    return merged


def _synthesize_from_sources(
    query: str,
    sources: list[_VerifiedSource],
    *,
    search_results: list[dict],
    backend_answers: list[str],
) -> tuple[GroundStatus, str, float]:
    if not sources:
        return "not_found", "", 0.0

    llm_result = None
    if os.environ.get("GROUNDING_DISABLE_LLM_SYNTHESIS") != "1":
        llm_result = _llm_synthesis(query, sources)
    if llm_result is not None:
        return llm_result

    answer = _heuristic_answer(query, sources, search_results, backend_answers)
    if not answer:
        return "not_found", "", 0.0

    if len(sources) >= 2:
        return "grounded", answer, 0.72
    return "partial", answer, 0.55


def _llm_synthesis(
    query: str,
    sources: list[_VerifiedSource],
) -> tuple[GroundStatus, str, float] | None:
    try:
        from research_engine import llm_call
    except Exception:
        return None

    source_blocks = []
    for index, source in enumerate(sources, start=1):
        excerpt = source.full_text[:SOURCE_EXCERPT_CHARS]
        source_blocks.append(
            f"[Source {index}] {source.source.url}\n"
            f"Tier: {source.source.tier.name}\n"
            f"{excerpt}"
        )
    prompt = (
        "Answer the question using ONLY the source text below.\n"
        "If the source text does not directly answer it, return not_found.\n"
        'Return ONLY JSON with keys "status", "answer", and "confidence".\n'
        'Valid status values: "grounded", "partial", "not_found".\n'
        "Rules:\n"
        "- grounded = direct, well-supported answer\n"
        "- partial = some support but meaningful uncertainty remains\n"
        "- not_found = do not guess; answer must be empty\n"
        "- answer must be at most two sentences\n\n"
        f"QUESTION: {query}\n\n"
        + "\n\n".join(source_blocks)
    )
    backend = "unknown"
    raw = ""
    try:
        raw, backend = llm_call.llm_complete(prompt, timeout=LOOKUP_TIMEOUT_SECONDS)
    except Exception as exc:
        _append_telemetry("grounding_synth", "error", prompt, str(exc))
        return None
    _append_telemetry("grounding_synth", backend, prompt, raw)
    return _parse_llm_json(raw)


def _append_telemetry(
    kind: str,
    model: str,
    prompt: str,
    response: str,
    *,
    session_id: str | None = None,
) -> None:
    if os.environ.get("GROUNDING_DISABLE_TELEMETRY") == "1":
        return
    payload = {
        "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "kind": kind,
        "model": model,
        "prompt_chars": len(prompt),
        "response_chars": len(response),
        "tokens_in_approx": _estimate_tokens(prompt),
        "tokens_out_approx": _estimate_tokens(response),
        "session_id": session_id or _telemetry_session_id(),
    }
    try:
        TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TELEMETRY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _telemetry_session_id() -> str:
    for key in ("SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID"):
        value = os.environ.get(key)
        if value:
            return value
    return ""


def _parse_llm_json(raw: str) -> tuple[GroundStatus, str, float] | None:
    candidates = [raw.strip()]
    match = JSON_RE.search(raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
            candidate = re.sub(r"```$", "", candidate).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        status = payload.get("status")
        answer = " ".join(str(payload.get("answer") or "").split()).strip()
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if status not in {"grounded", "partial", "not_found"}:
            continue
        confidence = max(0.0, min(1.0, confidence))
        if status == "not_found":
            return "not_found", "", 0.0
        if not answer:
            continue
        return status, answer, confidence
    return None


def _heuristic_answer(
    query: str,
    sources: list[_VerifiedSource],
    search_results: list[dict],
    backend_answers: list[str],
) -> str:
    query_terms = _query_terms(query)
    snippet_by_url = {
        str(result.get("url") or ""): str(result.get("snippet") or "").strip()
        for result in search_results
        if result.get("url")
    }
    for backend_answer in backend_answers:
        cleaned_answer = _clean_backend_answer(backend_answer)
        if cleaned_answer:
            return cleaned_answer
    for source in sources:
        snippet = source.snippet_hint or snippet_by_url.get(source.source.url, "")
        cleaned_snippet = _first_sentence(snippet)
        if cleaned_snippet and _sentence_matches_query(cleaned_snippet, query_terms):
            return cleaned_snippet
    for source in sources:
        sentence = _first_sentence(source.full_text)
        if sentence and _sentence_matches_query(sentence, query_terms):
            return sentence
    return ""


def _first_sentence(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    return sentence[:280]


def _clean_backend_answer(text: str) -> str:
    cleaned = re.sub(r"(?im)^answer:\s*", "", text).strip()
    cleaned = re.sub(r"(?im)^sources:\s*$", "", cleaned).strip()
    cleaned = URL_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned or cleaned.lower() == "not_found":
        return ""
    return _first_sentence(cleaned)


def _sentence_matches_query(sentence: str, query_terms: list[str]) -> bool:
    if not sentence:
        return False
    if not query_terms:
        return True
    lowered = sentence.lower()
    matches = _term_match_count(lowered, query_terms)
    return matches >= _min_term_matches(query_terms)


def _term_match_count(text: str, query_terms: list[str]) -> int:
    return sum(1 for term in query_terms if term in text)


def _min_term_matches(query_terms: list[str]) -> int:
    if not query_terms:
        return 0
    if len(query_terms) == 1:
        return 1
    return max(2, (len(query_terms) + 1) // 2)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Ground a web fact with verifiable sources.")
    parser.add_argument("query", help="Claim or question to verify on the web")
    parser.add_argument("--topic", dest="topic_slug")
    args = parser.parse_args()

    result = ground(args.query, topic_slug=args.topic_slug)
    if result.status == "not_found":
        print("not_found")
        return 0

    print(result.answer)
    print()
    print("SOURCES:")
    for source in result.sources:
        print(f"- {source.url} [{source.tier.name}] {source.raw_text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
