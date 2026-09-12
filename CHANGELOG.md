# Changelog

## 2026-09-12

Three fixes, built and reviewed one at a time. They came out of an audit in which three models read
the whole package independently and found the same defect class repeated in three places: **wherever
a signal was not measured, the code substituted a value shaped like a measurement.**

What the audit measured on disk before any change:

- 32 of 72 saved gate decisions were stamped "sufficient" by a stub that ran no checker. It was
  reproduced directly: an unrelated paragraph about bananas passed as sufficient support for a
  PostgreSQL question.
- 122 of 124 saved sessions reported `$0.00` and `0 ms`, from seven zeros typed into the source.
- The "faithfulness" score was word overlap. "The treatment does not reduce mortality" scored 1.0
  against "The treatment does reduce mortality", and `120` counted as support for `1200`.
- Test runs were writing rows into the same production telemetry the audit was reading.
- Jina without a key, Crawl4AI without its helper script and Wayback without keys had cost over
  3,400 pointless fetch attempts between them.

### Honesty

- A sufficiency verdict exists only if a checker produced it. The stub that always answered
  "sufficient" is deleted, along with the automatic local-model judge. With no checker the session
  carries `terminal_state="unchecked"` and `verdict=None`, and the evidence gate abstains with
  `gate_reason="checker_unavailable"`.
- An unmeasured number is `None`, never `0.0`. A FULL answer cannot be built without a measured
  confidence. `QueryCall` raises when a payload carries no `duration_ms`, and session totals have no
  defaults.
- Word overlap is stored as `lexical_overlap_score` under its own name. `crystal_check_passed` and
  `crystal_check_score` stay `None` until a real claim-level judge fills them.
- The number and date check runs on every save and can demote FULL to PARTIAL. Matching is
  whole-token; with no captured source text it reports `applicable: false` rather than a score.
- Saving is split: `finalize_session()` judges, `write_session()` stores. No save path skips the
  gate, and a storage failure never rewrites the answer.
- Log and cache paths resolve at write time from `RESEARCH_ENGINE_DATA_DIR` and
  `RESEARCH_ENGINE_CACHE_DIR`; a package `conftest.py` points both at temporary directories. No
  production code branches on `PYTEST_CURRENT_TEST`.

### Readers

- Each rung declares a precondition and is skipped with one logged reason when it cannot work,
  instead of being attempted and failing.
- Web rung order is now `cloudflare-markdown`, `trafilatura`, `firecrawl`, `crawl4ai`, `scrapling`,
  `crawlee`, `jina`, with `agent-browser` still double-gated for tier 3 only.
- Every page request passes one door, `fetch_gate()` — origin pages, PDFs, `robots.txt`, archive
  copies, gitingest, Docling, MarkItDown and the tier-3 browser included. It spaces requests per host
  group (`www.`, `old.`, `m.`, `amp.` collapse into one) and refuses during a cooldown.
- A rung that fetches through a service gates on **both** host groups, the service and the origin, so
  a host that returned `429` is not fetched anyway from someone else's address. A cooldown stops the
  ladder for that URL.
- Every rung that sees a `429` or `403` starts the cooldown. `Retry-After` is honoured; a value in
  the past never means retry now.
- Host grouping handles IPv6 literals, userinfo, stacked alias labels and IDNA. Cooldown and robots
  state are bounded. A blocked `HEAD` response is no longer cached for the life of the process.
- A fetched page is attached to a URL only when the returned URL matches, on the direct path as well
  as the proxy path.
- `DomainCooldown` is never swallowed by a broad `except` and never escapes `extract_clean_text()` to
  a caller.

### Pages

- The cached file holds the complete cleaned text and `char_count` equals its length. The tier-based
  storage trims are gone — tier 3 used to store the first 1,500 characters and tier 2 stitched the
  first 8,000 to the last 1,000 — and so is the 200,000-character repository cap. `excerpt` (1,500
  characters) is a display field, never what is stored.
- New `windows.py`: `tokenize`, `bm25_scores`, `rank_candidates`, `select_windows`, `render_windows`.
  Pure functions, no model call and no network. Candidates are ranked before any fetch, with one
  exploration slot per batch of five so a result the search engine ranked first cannot be dropped on
  lexical score alone.
- `PAGE_BUDGET = {"/search": 5, "/research": 6, "/deep-research": 8}` is the only page cap;
  `max_sources=2` and `max_sources=3` are gone. `extract_sources` dedupes unconditionally, so a
  duplicate URL is never fetched twice and never consumes a slot.
- Prompts carry selected windows rather than a `[:1500]` prefix, and each `EvidenceChunk` carries the
  real `char_offset` and `char_length` of its window, indexing the stored text exactly.

### Tests

519 passed, 2 skipped, 0 failed in this tree; 534 in the source package. Ruff clean. gitleaks clean.

### Known, and said out loud

- **This sends more text to a model, not less** — up to 4,000 characters of selected windows per page
  against the former 1,500-character prefix, across more pages. The budgets are provisional and fall
  back if a replay shows they do not raise the FULL-answer rate.
- Dual-group gating is carried by an optional argument, so a newly added rung can omit it.
- The listicle penalty in `rank_candidates` fires on a flag nothing currently sets.
- `test_webread_service.py` exists both at the top level and under `tests/`, which makes pytest
  refuse to collect one of them in this flattened layout. Pre-existing.
- The cooldown is per process; separate worker processes do not share a backoff.

## 2026-08-18

Fixed three path-redaction leak and corruption defects.

## 2026-08-17

Made failures visible and kept local paths out of the logs. Stopped discarding real answers; revived
the nightly key check.

## 2026-08-16

First public release under AGPL-3.0-or-later.
