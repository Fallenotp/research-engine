"""Shared query-aware sufficiency gate for retrieved research sources."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from . import paths


SUFFICIENCY_FLASH_MODEL = os.getenv(
    "MENTOR_SUFFICIENCY_FLASH_MODEL",
    os.getenv("MENTOR_CLAIM_CHECKER_FLASH_MODEL", "gemini-3.1-flash-lite-preview"),
)
SUFFICIENCY_TIMEOUT_SECONDS = float(
    os.getenv("MENTOR_SUFFICIENCY_TIMEOUT_SECONDS", "90")
)
SUFFICIENCY_AGY_BIN = os.getenv(
    "MENTOR_SUFFICIENCY_AGY_BIN",
    "agy-cli-1",
)
SUFFICIENCY_AGY_SKIP_PERMISSIONS_FLAG = "--dangerously-skip-permissions"
SUFFICIENCY_OLLAMA_BIN = os.getenv(
    "MENTOR_SUFFICIENCY_OLLAMA_BIN",
    os.getenv("OLLAMA_BIN", "ollama"),
)
SUFFICIENCY_OLLAMA_MODEL = (
    os.getenv("MENTOR_SUFFICIENCY_OLLAMA_MODEL")
    or os.getenv("BUZZ_OLLAMA_MODEL")
    or ""
).strip()
SUFFICIENCY_SKIP_MODEL = "sufficiency-skip"
PREFERRED_OLLAMA_MODELS = (
    "qwen3.5:9b",
    "qwen2.5:7b",
    "llama3.2:3b",
    "llama3.2:1b",
)
MAX_SOURCE_CHARS = int(os.getenv("MENTOR_SUFFICIENCY_MAX_SOURCE_CHARS", "60000"))
MAX_SOURCE_CHARS_PER_SOURCE = int(
    os.getenv("MENTOR_SUFFICIENCY_MAX_SOURCE_CHARS_PER_SOURCE", "12000")
)
MAX_RECOVERY_ITERATIONS = int(os.getenv("MENTOR_SUFFICIENCY_MAX_RECOVERY_ITERATIONS", "2"))
SUFFICIENCY_PREFILTER_ENABLED = os.getenv(
    "MENTOR_SUFFICIENCY_PREFILTER", "1"
).strip().lower() not in {"0", "false", "no", "off"}

PROMPT_VERSION = "query-sufficiency-v2"
REFORMULATION_PROMPT_VERSION = "query-reformulation-v1"
RELEVANCE_PROMPT_VERSION = "query-relevance-prefilter-v1"
VALID_VERDICTS = {"sufficient", "partial", "insufficient"}
VALID_RELEVANCE_VERDICTS = {"relevant", "irrelevant"}
LOW_CONFIDENCE_PARTIAL_REASON = "source sufficiency is partial"
LOW_CONFIDENCE_EXHAUSTED_REASON = "could not substantiate from available sources"
CHECKER_UNAVAILABLE_REASON = "sufficiency checker unavailable"


class SufficiencyClient(Protocol):
    provider_label: str
    model_id: str

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Return one parsed JSON object from the checker model."""


class SufficiencyError(RuntimeError):
    """Raised when the checker transport or output cannot be used."""

    def __init__(
        self,
        message: str,
        *,
        decision: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.decision = decision or {}


@dataclass(frozen=True)
class SourceText:
    source_id: str
    title: str
    domain: str
    text: str


class CLIJsonClient:
    """Minimal JSON client over OAuth-backed local CLIs."""

    def __init__(
        self,
        *,
        provider_label: str,
        model_id: str,
        executable: str,
        timeout_seconds: float,
    ) -> None:
        self.provider_label = provider_label
        self.model_id = model_id
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.prompt_chars = 0
        self.output_chars = 0
        self.call_count = 0

    def generate_json(self, prompt: str) -> dict[str, Any]:
        self.prompt_chars += len(prompt)
        self.call_count += 1
        with tempfile.TemporaryDirectory(prefix="sufficiency-cli-") as cwd:
            completed = subprocess.run(
                self._command(prompt),
                **self._subprocess_kwargs(prompt=prompt, cwd=cwd),
            )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        self.output_chars += len(stdout)
        if completed.returncode != 0:
            raise SufficiencyError(
                f"{self.provider_label} checker failed with exit code "
                f"{completed.returncode}: {_preview(stderr) or 'no stderr'}"
            )
        try:
            return extract_json_object(stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SufficiencyError(
                f"{self.provider_label} checker returned unparseable JSON: "
                f"{_preview(stdout) or 'empty stdout'}"
            ) from exc

    def _command(self, prompt: str) -> list[str]:
        if self.provider_label == "flash":
            return [
                self.executable,
                SUFFICIENCY_AGY_SKIP_PERMISSIONS_FLAG,
                "--print",
                _agy_prompt_arg(prompt),
            ]
        if self.provider_label == "ollama-fallback":
            return [
                self.executable,
                "run",
                self.model_id,
            ]
        return [
            self.executable,
            "-p",
            "--model",
            self.model_id,
            "--output-format",
            "text",
            "--no-session-persistence",
            "--tools",
            "",
            "--disable-slash-commands",
            "--setting-sources",
            "user",
        ]

    def _subprocess_kwargs(self, *, prompt: str, cwd: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": self.timeout_seconds,
            "check": False,
            "cwd": cwd,
            "env": self._sealed_env(),
        }
        if self.provider_label == "flash":
            kwargs["stdin"] = subprocess.DEVNULL
        else:
            kwargs["input"] = prompt
        return kwargs

    def _sealed_env(self) -> dict[str, str]:
        home = str(Path.home())
        executable_dir = str(Path(self.executable).expanduser().resolve().parent)
        path_parts: list[str] = []
        for part in (
            executable_dir,
            str(Path.home() / ".bun" / "bin"),
            str(Path.home() / ".local" / "bin"),
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ):
            if part and part not in path_parts:
                path_parts.append(part)
        env = {
            "HOME": home,
            "PATH": ":".join(path_parts),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TMPDIR": tempfile.gettempdir(),
        }
        return env


class SkipSufficiencyClient:
    """Return permissive no-op verdicts when no sanctioned checker is available."""

    provider_label = "skip"
    model_id = SUFFICIENCY_SKIP_MODEL

    def __init__(self) -> None:
        self.prompt_chars = 0
        self.output_chars = 0
        self.call_count = 0

    def generate_json(self, prompt: str) -> dict[str, Any]:
        self.prompt_chars += len(prompt)
        self.call_count += 1
        if "query relevance filter" in prompt:
            payload = {
                "verdict": "relevant",
                "reason": "The sufficiency gate was skipped because no sanctioned local provider was available.",
            }
        elif "Rewrite the search query to fill the missing evidence gap." in prompt:
            payload = {"query": ""}
        else:
            payload = {
                "verdict": "sufficient",
                "reason": "The sufficiency gate was skipped because no sanctioned local provider was available.",
                "missing": [],
            }
        rendered = json.dumps(payload)
        self.output_chars += len(rendered)
        return payload


def get_sufficiency_clients() -> tuple[SufficiencyClient, SufficiencyClient]:
    skip_client = SkipSufficiencyClient()
    gemini_executable = _resolve_agy_executable()
    ollama_runtime = _resolve_ollama_runtime()
    gemini_client = (
        CLIJsonClient(
            provider_label="flash",
            model_id=SUFFICIENCY_FLASH_MODEL,
            executable=gemini_executable,
            timeout_seconds=SUFFICIENCY_TIMEOUT_SECONDS,
        )
        if gemini_executable
        else None
    )
    ollama_client = (
        CLIJsonClient(
            provider_label="ollama-fallback",
            model_id=ollama_runtime[1],
            executable=ollama_runtime[0],
            timeout_seconds=SUFFICIENCY_TIMEOUT_SECONDS,
        )
        if ollama_runtime
        else None
    )
    if gemini_client and ollama_client:
        return gemini_client, ollama_client
    if gemini_client:
        return gemini_client, skip_client
    if ollama_client:
        return ollama_client, skip_client
    return skip_client, skip_client


def run_sufficiency_loop(
    *,
    query: str,
    source_texts: Sequence[SourceText],
    retriever: Callable[[str], Sequence[SourceText]] | None = None,
    clients: tuple[SufficiencyClient, SufficiencyClient] | None = None,
    max_recovery_iterations: int = MAX_RECOVERY_ITERATIONS,
) -> dict[str, Any]:
    """Judge raw sources and optionally run a bounded recovery loop."""

    active_clients = clients or get_sufficiency_clients()
    current_sources = cap_source_texts(source_texts)
    attempts: list[dict[str, Any]] = []
    reformulations: list[str] = []
    seen_fingerprints = {source_fingerprint(current_sources)}
    stop_reason = ""

    for attempt_index in range(max(0, max_recovery_iterations) + 1):
        judge = judge_sufficiency(
            query=query,
            source_texts=current_sources,
            clients=active_clients,
        )
        attempts.append(
            {
                "iteration": attempt_index,
                "retrieval_query": reformulations[-1] if reformulations else query,
                "source_count": len(current_sources),
                "source_fingerprint": source_fingerprint(current_sources),
                "judge": judge,
            }
        )
        verdict = str(judge.get("verdict") or "insufficient")
        if verdict == "sufficient":
            return _loop_result(
                query=query,
                attempts=attempts,
                reformulations=reformulations,
                terminal_state="sufficient",
                stop_reason="sufficient",
                max_recovery_iterations=max_recovery_iterations,
            )
        if retriever is None:
            stop_reason = "no_retriever"
            break
        if attempt_index >= max_recovery_iterations:
            stop_reason = "max_recovery_iterations"
            break

        try:
            reformulation = reformulate_query(
                query=query,
                decision=judge,
                clients=active_clients,
            )
        except Exception as exc:  # noqa: BLE001 - recovery must fail closed
            attempts[-1]["reformulation_error"] = f"{type(exc).__name__}: {exc}"
            stop_reason = "reformulation_failed"
            break
        if not reformulation or reformulation in {query, *reformulations}:
            stop_reason = "reformulation_repeated"
            break
        reformulations.append(reformulation)
        recovered_sources = cap_source_texts(retriever(reformulation))
        fingerprint = source_fingerprint(recovered_sources)
        if fingerprint in seen_fingerprints or sources_substantially_same(
            current_sources,
            recovered_sources,
        ):
            attempts.append(
                {
                    "iteration": attempt_index + 1,
                    "retrieval_query": reformulation,
                    "source_count": len(recovered_sources),
                    "source_fingerprint": fingerprint,
                    "judge": None,
                    "stopped_before_judge": "same_items",
                }
            )
            stop_reason = "same_items"
            break
        seen_fingerprints.add(fingerprint)
        current_sources = recovered_sources

    final_judge = {}
    for attempt in reversed(attempts):
        if isinstance(attempt.get("judge"), dict):
            final_judge = attempt["judge"]
            break
    final_verdict = str((final_judge or {}).get("verdict") or "insufficient")
    terminal_state = "partial" if final_verdict == "partial" else "exhausted"
    return _loop_result(
        query=query,
        attempts=attempts,
        reformulations=reformulations,
        terminal_state=terminal_state,
        stop_reason=stop_reason or "insufficient",
        max_recovery_iterations=max_recovery_iterations,
    )


def judge_sufficiency(
    *,
    query: str,
    source_texts: Sequence[SourceText],
    clients: tuple[SufficiencyClient, SufficiencyClient] | None = None,
) -> dict[str, Any]:
    """Run one query-aware judge call against raw sources only."""

    primary, fallback = clients or get_sufficiency_clients()
    capped_sources = cap_source_texts(source_texts)
    if SUFFICIENCY_PREFILTER_ENABLED:
        try:
            filtered_sources, prefilter = filter_relevant_source_texts(
                query=query,
                source_texts=capped_sources,
                clients=(primary, fallback),
            )
        except Exception as prefilter_exc:  # noqa: BLE001 - fail-closed boundary
            return fail_closed_decision(
                query=query,
                source_texts=capped_sources,
                primary_error=prefilter_exc,
                fallback_error=prefilter_exc,
                fallback_model=getattr(fallback, "model_id", "unknown"),
                client=(primary, fallback),
                fail_stage="query_relevance_prefilter",
            )
    else:
        filtered_sources = capped_sources
        prefilter = {
            "prompt_version": RELEVANCE_PROMPT_VERSION,
            "input_source_count": len(capped_sources),
            "kept_source_count": len(capped_sources),
            "dropped_source_count": 0,
            "kept_source_ids": [source.source_id for source in capped_sources],
            "dropped": [],
            "skipped": True,
            "skip_reason": "disabled_by_env",
        }
    try:
        return _judge_with_client(
            query,
            filtered_sources,
            client=primary,
            checker_route=_client_route(primary),
            checker_error=None,
            prefilter=prefilter,
        )
    except Exception as primary_exc:  # noqa: BLE001 - locked fallback boundary
        try:
            return _judge_with_client(
                query,
                filtered_sources,
                client=fallback,
                checker_route=_client_route(fallback),
                checker_error=str(primary_exc),
                prefilter=prefilter,
            )
        except Exception as fallback_exc:  # noqa: BLE001 - fail-closed boundary
            return fail_closed_decision(
                query=query,
                source_texts=filtered_sources,
                primary_error=primary_exc,
                fallback_error=fallback_exc,
                fallback_model=getattr(fallback, "model_id", "unknown"),
                client=(primary, fallback),
                fail_stage="judge",
                prefilter=prefilter,
            )


def reformulate_query(
    *,
    query: str,
    decision: dict[str, Any],
    clients: tuple[SufficiencyClient, SufficiencyClient] | None = None,
) -> str:
    primary, fallback = clients or get_sufficiency_clients()
    prompt = build_reformulation_prompt(query, decision)
    try:
        payload = primary.generate_json(prompt)
    except Exception:
        payload = fallback.generate_json(prompt)
    reformulated = " ".join(str(payload.get("query") or "").split())
    return reformulated[:300]


def fail_closed_decision(
    *,
    query: str,
    source_texts: Sequence[SourceText],
    primary_error: Exception,
    fallback_error: Exception,
    fallback_model: str,
    client: SufficiencyClient | Sequence[SufficiencyClient] | None = None,
    fail_stage: str = "judge",
    prefilter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = _base_decision(query, source_texts)
    decision.update(
        {
            "verdict": "insufficient",
            "reason": "checker_failed_closed",
            "missing": ["The sufficiency checker was unavailable."],
            "checker_route": "failed-closed",
            "checker_error": f"{type(primary_error).__name__}: {primary_error}",
            "fallback_error": f"{type(fallback_error).__name__}: {fallback_error}",
            "model": None,
            "fallback_model": fallback_model,
            "checker_usage": checker_usage(client),
            "fail_closed": True,
            "fail_stage": fail_stage,
        }
    )
    if prefilter is not None:
        decision["query_relevance_prefilter"] = prefilter
    return decision


def low_confidence_reason_for_result(result: dict[str, Any]) -> str:
    final = final_judge(result)
    if final.get("checker_route") == "failed-closed":
        return CHECKER_UNAVAILABLE_REASON
    missing = final.get("missing") or []
    missing_text = "; ".join(str(item) for item in missing if str(item).strip())
    reason = str(final.get("reason") or "").strip()
    if result.get("terminal_state") == "partial":
        detail = missing_text or reason
        return f"{LOW_CONFIDENCE_PARTIAL_REASON}: {detail}" if detail else LOW_CONFIDENCE_PARTIAL_REASON
    tried = ", ".join(result.get("reformulations") or [])
    base = LOW_CONFIDENCE_EXHAUSTED_REASON
    detail = missing_text or reason
    if tried:
        return f"{base}; tried: {tried}; gap: {detail}" if detail else f"{base}; tried: {tried}"
    return f"{base}: {detail}" if detail else base


def final_judge(result: dict[str, Any]) -> dict[str, Any]:
    for attempt in reversed(result.get("attempts") or []):
        judge = attempt.get("judge")
        if isinstance(judge, dict):
            return judge
    return {}

def cap_source_texts(source_texts: Sequence[SourceText]) -> list[SourceText]:
    capped: list[SourceText] = []
    remaining_chars = MAX_SOURCE_CHARS
    for source in source_texts:
        text = " ".join((source.text or "").split())
        if not text:
            continue
        clipped = text[: min(MAX_SOURCE_CHARS_PER_SOURCE, remaining_chars)]
        if not clipped:
            break
        capped.append(
            SourceText(
                source_id=str(source.source_id),
                title=str(source.title or ""),
                domain=str(source.domain or ""),
                text=clipped,
            )
        )
        remaining_chars -= len(clipped)
        if remaining_chars <= 0:
            break
    return capped


def source_fingerprint(source_texts: Sequence[SourceText]) -> str:
    rows = []
    for source in source_texts:
        text_hash = sha256(" ".join(source.text.split()).encode("utf-8")).hexdigest()[:16]
        rows.append(f"{source.source_id}:{text_hash}")
    return sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def sources_substantially_same(
    left: Sequence[SourceText],
    right: Sequence[SourceText],
    *,
    threshold: float = 0.8,
) -> bool:
    left_keys = {_source_similarity_key(source) for source in left}
    right_keys = {_source_similarity_key(source) for source in right}
    if not left_keys and not right_keys:
        return True
    if not left_keys or not right_keys:
        return False
    overlap = len(left_keys & right_keys) / max(len(left_keys), len(right_keys))
    return overlap >= threshold


def build_judge_prompt(query: str, source_texts: Sequence[SourceText]) -> str:
    sources = "\n\n".join(
        (
            f"Source ID: {source.source_id}\n"
            f"Title: {source.title}\n"
            f"Domain: {source.domain}\n"
            f"Raw text:\n{source.text}"
        )
        for source in source_texts
    )
    return f"""You are a research sufficiency judge.

You are given the user's original query and raw retrieved item text only.
You must not use any synthesized answer, summary, or restatement.
Use only the raw titles, bodies, snippets, transcripts, comments, and excerpts below.

Verdict rules:
- sufficient: the raw source text directly and substantively answers the important parts of the query.
- partial: the raw source text answers some important parts, but important facets are missing.
- insufficient: the raw source text does not substantively answer the query.
- Keyword overlap, title-only overlap, or general topical similarity is not enough.
- If the sources are about a different meaning of the query, use insufficient.
- Return JSON only.

JSON schema:
{{
  "verdict": "sufficient|partial|insufficient",
  "reason": "short reason grounded in the raw source text",
  "missing": ["important missing facet"]
}}

Original query:
{query}

Retrieved raw source text:
{sources}
"""


def build_reformulation_prompt(query: str, decision: dict[str, Any]) -> str:
    missing = decision.get("missing") or []
    if isinstance(missing, str):
        missing_text = missing
    else:
        missing_text = "; ".join(str(item) for item in missing)
    return f"""Rewrite the search query to fill the missing evidence gap.

Use the original user query plus the judge's reason and missing facets.
Return one search query. Do not answer the query.

JSON schema:
{{"query": "reformulated search query"}}

Original user query:
{query}

Judge verdict:
{decision.get("verdict")}

Judge reason:
{decision.get("reason")}

Missing facets:
{missing_text}
"""


def build_relevance_prompt(query: str, source: SourceText) -> str:
    return f"""You are a query relevance filter for a research sufficiency gate.

You are given the user's original query and one raw retrieved item only.
Decide whether this single item materially helps answer the user's question.
Use only the raw item text below. Do not use outside context.

Verdict rules:
- relevant: the item directly addresses at least one important part of the query.
- irrelevant: the item only shares keywords, background topic, or a different meaning.
- If the match is unclear, return irrelevant.
- Return JSON only.

JSON schema:
{{
  "verdict": "relevant|irrelevant",
  "reason": "short reason grounded in the raw item text"
}}

Original query:
{query}

Single retrieved item:
Source ID: {source.source_id}
Title: {source.title}
Domain: {source.domain}
Raw text:
{source.text}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def checker_usage(
    client: SufficiencyClient | Sequence[SufficiencyClient] | None,
) -> dict[str, int]:
    if client is None:
        return {"call_count": 0, "prompt_chars": 0, "output_chars": 0}
    if isinstance(client, Sequence) and not hasattr(client, "provider_label"):
        clients = list(client)
    else:
        clients = [client]

    call_count = 0
    prompt_chars = 0
    output_chars = 0
    for current in clients:
        current_prompt_chars = int(getattr(current, "prompt_chars", 0))
        current_output_chars = int(getattr(current, "output_chars", 0))
        if current_prompt_chars == 0 and hasattr(current, "prompts"):
            current_prompt_chars = sum(len(prompt) for prompt in getattr(current, "prompts"))
        call_count += int(getattr(current, "call_count", 0) or len(getattr(current, "prompts", [])))
        prompt_chars += current_prompt_chars
        output_chars += current_output_chars
    return {
        "call_count": call_count,
        "prompt_chars": prompt_chars,
        "output_chars": output_chars,
        "estimated_prompt_tokens": max(1, round(prompt_chars / 4)) if prompt_chars else 0,
        "estimated_output_tokens": max(1, round(output_chars / 4)) if output_chars else 0,
    }


def _judge_with_client(
    query: str,
    source_texts: Sequence[SourceText],
    *,
    client: SufficiencyClient,
    checker_route: str,
    checker_error: str | None,
    prefilter: dict[str, Any],
) -> dict[str, Any]:
    capped_sources = cap_source_texts(source_texts)
    prompt = build_judge_prompt(query, capped_sources)
    payload = client.generate_json(prompt)
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        raise SufficiencyError(f"invalid sufficiency verdict: {verdict or '<empty>'}")
    reason = " ".join(str(payload.get("reason") or "").split())
    missing = _normalize_missing(payload.get("missing"))
    if verdict == "sufficient" and _ambiguous_plain_phrase_needs_confirmation(
        query,
        capped_sources,
    ):
        verdict = "insufficient"
        reason = (
            "The query is a short ambiguous phrase, and the raw results come from "
            "too narrow a source pool to establish the intended meaning."
        )
        missing = [
            "independent raw sources that establish the intended entity or meaning"
        ]

    decision = _base_decision(query, capped_sources)
    decision.update(
        {
            "verdict": verdict,
            "reason": reason,
            "missing": missing,
            "checker_route": checker_route,
            "checker_error": checker_error,
            "model": getattr(client, "model_id", "unknown"),
            "prompt_version": PROMPT_VERSION,
            "checker_usage": checker_usage(client),
            "fail_closed": False,
            "query_relevance_prefilter": prefilter,
        }
    )
    return decision


def filter_relevant_source_texts(
    *,
    query: str,
    source_texts: Sequence[SourceText],
    clients: tuple[SufficiencyClient, SufficiencyClient] | None = None,
) -> tuple[list[SourceText], dict[str, Any]]:
    primary, fallback = clients or get_sufficiency_clients()
    capped_sources = cap_source_texts(source_texts)
    kept: list[SourceText] = []
    dropped: list[dict[str, Any]] = []
    for source in capped_sources:
        decision = _check_source_relevance(
            query=query,
            source=source,
            primary=primary,
            fallback=fallback,
        )
        if decision["verdict"] == "relevant":
            kept.append(source)
            continue
        dropped.append(
            {
                "source_id": source.source_id,
                "title": source.title,
                "domain": source.domain,
                "reason": decision["reason"],
                "checker_route": decision["checker_route"],
                "model": decision["model"],
            }
        )
    return kept, {
        "prompt_version": RELEVANCE_PROMPT_VERSION,
        "input_source_count": len(capped_sources),
        "kept_source_count": len(kept),
        "dropped_source_count": len(dropped),
        "kept_source_ids": [source.source_id for source in kept],
        "dropped": dropped[:20],
        "skipped": False,
    }


def _check_source_relevance(
    *,
    query: str,
    source: SourceText,
    primary: SufficiencyClient,
    fallback: SufficiencyClient,
) -> dict[str, Any]:
    try:
        return _check_source_relevance_with_client(
            query=query,
            source=source,
            client=primary,
            checker_route=_client_route(primary),
            checker_error=None,
        )
    except Exception as primary_exc:  # noqa: BLE001 - locked fallback boundary
        return _check_source_relevance_with_client(
            query=query,
            source=source,
            client=fallback,
            checker_route=_client_route(fallback),
            checker_error=str(primary_exc),
        )


def _check_source_relevance_with_client(
    *,
    query: str,
    source: SourceText,
    client: SufficiencyClient,
    checker_route: str,
    checker_error: str | None,
) -> dict[str, Any]:
    payload = client.generate_json(build_relevance_prompt(query, source))
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in VALID_RELEVANCE_VERDICTS:
        raise SufficiencyError(f"invalid relevance verdict: {verdict or '<empty>'}")
    return {
        "verdict": verdict,
        "reason": " ".join(str(payload.get("reason") or "").split()),
        "checker_route": checker_route,
        "checker_error": checker_error,
        "model": getattr(client, "model_id", "unknown"),
    }


def _loop_result(
    *,
    query: str,
    attempts: list[dict[str, Any]],
    reformulations: list[str],
    terminal_state: str,
    stop_reason: str,
    max_recovery_iterations: int,
) -> dict[str, Any]:
    final = {}
    for attempt in reversed(attempts):
        if isinstance(attempt.get("judge"), dict):
            final = attempt["judge"]
            break
    result = {
        "schema_version": "1.0",
        "mode": "query_sufficiency_loop",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "original_query": query,
        "terminal_state": terminal_state,
        "verdict": final.get("verdict", "insufficient"),
        "reason": final.get("reason", ""),
        "missing": final.get("missing", []),
        "proceed": terminal_state in {"sufficient", "partial"},
        "low_confidence": terminal_state in {"partial", "exhausted"},
        "stop_reason": stop_reason,
        "max_recovery_iterations": max_recovery_iterations,
        "recovery_iterations": len(reformulations),
        "reformulations": list(reformulations),
        "attempts": attempts,
        "final_judge": final,
        "query_relevance_prefilter": final.get("query_relevance_prefilter", {}),
        "self_synthesized_answer_used": False,
        "input_guardrail": "original_query_and_raw_sources_only",
    }
    if terminal_state == "exhausted":
        tried = ", ".join(reformulations) if reformulations else query
        result["result_message"] = (
            "could not substantiate from available sources; "
            f"tried: {tried}"
        )
    return result


def _base_decision(query: str, source_texts: Sequence[SourceText]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "question": query,
        "mode": "query_sufficiency",
        "source_count": len(source_texts),
        "total_source_chars": sum(len(source.text) for source in source_texts),
        "source_fingerprint": source_fingerprint(source_texts),
        "self_synthesized_answer_used": False,
        "input_guardrail": "original_query_and_raw_sources_only",
    }


def _normalize_missing(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [" ".join(value.split())] if value.strip() else []
    if not isinstance(value, list):
        return [" ".join(str(value).split())]
    missing: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text:
            missing.append(text)
    return missing[:8]


def _ambiguous_plain_phrase_needs_confirmation(
    query: str,
    source_texts: Sequence[SourceText],
) -> bool:
    tokens = _plain_query_tokens(query)
    if len(tokens) == 0 or len(tokens) > 4:
        return False
    if query != query.lower():
        return False
    if any(char.isdigit() for char in query):
        return False
    first = tokens[0]
    if first in {"what", "why", "how", "when", "where", "who", "which", "compare", "latest"}:
        return False
    return _source_family_count(source_texts) < 2


def _plain_query_tokens(query: str) -> list[str]:
    stopwords = {"a", "an", "and", "for", "of", "the", "to", "with"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in stopwords
    ]


def _source_family_count(source_texts: Sequence[SourceText]) -> int:
    families = set()
    for source in source_texts:
        family = (source.domain or "").lower().strip()
        if not family:
            family = str(source.source_id).split(":", 1)[0]
        if family:
            families.add(family)
    return len(families)


def _source_similarity_key(source: SourceText) -> str:
    title = " ".join((source.title or "").lower().split())
    text = " ".join((source.text or "").lower().split())[:500]
    return sha256(f"{source.source_id}|{title}|{text}".encode("utf-8")).hexdigest()


def _preview(value: str, limit: int = 500) -> str:
    compact = " ".join(value.strip().split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _client_route(client: SufficiencyClient) -> str:
    return str(getattr(client, "provider_label", "unknown") or "unknown")


def _resolve_agy_executable() -> str | None:
    candidates = (
        SUFFICIENCY_AGY_BIN,
        str(Path.home() / "bin" / "agy-cli-1"),
        str(Path.home() / "bin" / "agy-cli-2"),
        str(Path.home() / ".local" / "bin" / "agy"),
        paths.executable(paths.AGY_BIN_ENV, "agy-cli-1", "agy-cli-2", "agy") or "",
    )
    resolved = _first_executable(candidates)
    if resolved and resolved.endswith("/agy") and SUFFICIENCY_AGY_BIN == "agy":
        return "agy"
    return resolved


def _agy_prompt_arg(prompt: str) -> str:
    if prompt.lstrip().startswith("-"):
        return f"Prompt:\n{prompt}"
    return prompt


def _resolve_ollama_runtime() -> tuple[str, str] | None:
    executable = _first_executable(
        (
            paths.executable(paths.OLLAMA_BIN_ENV, "ollama") or "",
            SUFFICIENCY_OLLAMA_BIN,
            shutil.which("ollama") or "",
        )
    )
    if not executable:
        return None
    model_id = _resolve_ollama_model(executable)
    if not model_id:
        return None
    return executable, model_id


def _resolve_ollama_model(executable: str) -> str | None:
    if SUFFICIENCY_OLLAMA_MODEL:
        return SUFFICIENCY_OLLAMA_MODEL
    installed = _list_ollama_models(executable)
    if not installed:
        return None
    for candidate in PREFERRED_OLLAMA_MODELS:
        if candidate in installed:
            return candidate
    for model_id in installed:
        lowered = model_id.lower()
        if "embed" in lowered or "minicheck" in lowered:
            continue
        return model_id
    return installed[0]


def _list_ollama_models(executable: str) -> list[str]:
    try:
        completed = subprocess.run(
            [executable, "list"],
            capture_output=True,
            text=True,
            timeout=min(SUFFICIENCY_TIMEOUT_SECONDS, 10),
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(Path.home())},
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    models: list[str] = []
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("NAME "):
            continue
        model_id = stripped.split()[0]
        if model_id:
            models.append(model_id)
    return models


def _first_executable(candidates: Sequence[str]) -> str | None:
    seen: set[str] = set()
    for candidate in candidates:
        expanded = str(candidate or "").strip()
        if not expanded or expanded in seen:
            continue
        seen.add(expanded)
        path = Path(expanded).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        # Bare command names must be resolved to an absolute path NOW: the
        # checker subprocess runs with a sealed PATH that will not contain
        # the caller's shell PATH, so returning a bare name causes
        # FileNotFoundError ([Errno 2]) at exec time.
        resolved = shutil.which(expanded)
        if resolved:
            return resolved
    return None


def _is_executable(path_or_command: str) -> bool:
    path = Path(path_or_command).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return True
    return shutil.which(path_or_command) is not None
