# TEST-RESULTS — tel-C-search

Date: 2026-05-23
Agent: C

## Scope

Requested scope only:
- Created `/Users/cleo/lattice/research_engine/logged_search.py`
- Created `/Users/cleo/lattice/research_engine/tests/test_logged_search.py`
- Edited `/Users/cleo/.claude/skills/search/SKILL.md`
- Edited `/Users/cleo/.claude/skills/research/SKILL.md`
- Edited `/Users/cleo/.claude/skills/deep-research/SKILL.md`
- Created `/Users/cleo/lattice/research_engine/docs/TEST-RESULTS-tel-C-search.md`

Off-limits by brief:
- `/Users/cleo/lattice/research_engine/extractor.py`
- `/Users/cleo/lattice/research_engine/persistence.py`
- `/Users/cleo/lattice/research_engine/schema.py`
- `/Users/cleo/lattice/research_engine/__init__.py`
- `buzz`

## Backups

Created before editing:
- `/Users/cleo/.claude/skills/search/SKILL.md.bak-20260523-tel`
- `/Users/cleo/.claude/skills/research/SKILL.md.bak-20260523-tel`
- `/Users/cleo/.claude/skills/deep-research/SKILL.md.bak-20260523-tel`

Backup command:

```bash
cp -p /Users/cleo/.claude/skills/search/SKILL.md /Users/cleo/.claude/skills/search/SKILL.md.bak-20260523-tel && \
cp -p /Users/cleo/.claude/skills/research/SKILL.md /Users/cleo/.claude/skills/research/SKILL.md.bak-20260523-tel && \
cp -p /Users/cleo/.claude/skills/deep-research/SKILL.md /Users/cleo/.claude/skills/deep-research/SKILL.md.bak-20260523-tel
```

Result:
- PASS

## Code Changes

### `research_engine/logged_search.py`

Added a standalone stdlib-only wrapper with:
- `searxng(query, *, protocol=None, topic=None, agent=None)` for logged SearXNG GETs
- `proxy(query, *, provider=None, num_results=10, protocol=None, topic=None, agent=None)` for logged paid-proxy POSTs
- `_append_call(row)` using flocked JSONL append to `/Users/cleo/lattice/data/agent_state/research-call-log.jsonl`
- CLI entrypoint via `python3.11 -m research_engine.logged_search`

Behavior verified:
- Returns parsed endpoint JSON on success
- Returns `{"error": ...}` on failure
- Always writes one call-log row
- Logging failure never breaks the returned search data

### `research_engine/tests/test_logged_search.py`

Added focused pytest coverage for:
- SearXNG success path with `result_count == 2`
- Proxy failure path with `lane == "tavily"` and `ok == false`
- Crash-proof logging path with an unwritable call-log destination

## SKILL.md Diff Summary

### `search/SKILL.md`

Diff summary:
- Replaced the Step 2 primary SearXNG curl with `python3.11 -m research_engine.logged_search --protocol /search --topic {topic_slug} "<question>"`
- Kept the raw SearXNG curl as a one-line fallback
- Replaced the Step 3 primary paid-proxy curl with `python3.11 -m research_engine.logged_search --provider <chosen> --protocol /search --topic {topic_slug} "<question>"`
- Kept the raw paid-proxy curl as a one-line fallback
- Added a hard rule requiring `research_engine.logged_search` for SearXNG and paid-proxy calls
- Added a hard rule requiring every cited URL to be fetched via `extract_clean_text()` and recorded in `session.sources`

### `research/SKILL.md`

Diff summary:
- Replaced the Step 2 primary SearXNG curl with `python3.11 -m research_engine.logged_search --protocol /research --topic {topic_slug} "<query>"`
- Kept the raw SearXNG curl as a one-line fallback
- Replaced the Step 3 primary paid-proxy curl with `python3.11 -m research_engine.logged_search --provider <assigned> --protocol /research --topic {topic_slug} "<query>"`
- Kept the raw paid-proxy curl as a one-line fallback
- Added the same logged-search hard rule
- Added the same source-persistence hard rule

### `deep-research/SKILL.md`

Diff summary:
- Added shared Step 2 instructions that route general SearXNG confirmation through `research_engine.logged_search` for `/deep-research`
- Kept the raw SearXNG curl as a one-line fallback
- Added shared Step 3 instructions that route paid provider calls through `research_engine.logged_search --provider <assigned>`
- Kept the raw paid-proxy curl as a one-line fallback
- Preserved the existing direct-provider diversity rules and clarified that multiple assigned lanes should still run through the wrapper
- Added the same logged-search hard rule
- Added the same source-persistence hard rule

## Verification

### Ruff + pytest

Command:

```bash
cd /Users/cleo/lattice && ruff check research_engine/logged_search.py research_engine/tests/test_logged_search.py && /opt/homebrew/bin/python3.11 -m pytest research_engine/tests/test_logged_search.py -v
```

Outcome:
- PASS
- `ruff check`: `All checks passed!`
- `pytest`: `3 passed`

Observed pytest summary:

```text
research_engine/tests/test_logged_search.py::test_searxng_logs_successful_call PASSED
research_engine/tests/test_logged_search.py::test_proxy_logs_failed_call PASSED
research_engine/tests/test_logged_search.py::test_searxng_still_returns_when_log_write_fails PASSED
```

Notes:
- Pytest emitted existing environment warnings from `requests` and some Swig-backed types. Tests still passed cleanly.

### Real smoke test

Command used to capture stdout, stderr, and the new call-log row:

```bash
cd /Users/cleo/lattice && \
before=$( [ -f data/agent_state/research-call-log.jsonl ] && wc -l < data/agent_state/research-call-log.jsonl || echo 0 ) && \
/opt/homebrew/bin/python3.11 -m research_engine.logged_search "test query python" > /tmp/tel-c-logged-search-smoke.json 2> /tmp/tel-c-logged-search-smoke.stderr ; \
rc=$?; \
after=$( [ -f data/agent_state/research-call-log.jsonl ] && wc -l < data/agent_state/research-call-log.jsonl || echo 0 ); \
echo "RC=$rc"; \
echo "BEFORE=$before"; \
echo "AFTER=$after"; \
tail -n 1 data/agent_state/research-call-log.jsonl
```

Outcome:
- PASS
- `RC=0`
- `BEFORE=1`
- `AFTER=2`
- SearXNG on `:8888` was up and returned JSON
- A new call-log row was appended

Stdout check:
- Printed a JSON object beginning with `{"query": "test query python", ...}`
- Parsed result list length from the appended row: `16`

Appended call-log row:

```json
{"ts": "2026-05-24T05:54:51.710526+00:00", "protocol": null, "topic": null, "lane": "searxng_general", "ok": true, "duration_ms": 2124, "result_count": 16, "error": null, "agent": "unknown"}
```

Smoke-test stderr note:
- The package import path emitted an existing `requests` dependency warning on stderr.
- The search call still succeeded and the wrapper returned JSON to stdout.

## Scoped Status

Repo status for files created in this task:

```text
?? research_engine/logged_search.py
?? research_engine/tests/test_logged_search.py
```

Scoped status for forbidden repo paths at the time of verification:

```text
 M research_engine/schema.py
?? research_engine/__init__.py
?? research_engine/persistence.py
```

Interpretation:
- This task did not edit `extractor.py`, `persistence.py`, `schema.py`, `research_engine/__init__.py`, or `buzz`.
- The scoped status output shows pre-existing dirty-tree entries in some forbidden repo paths, but no changes from this task were made there.

## Exact Files Edited / Created

Created:
- `/Users/cleo/lattice/research_engine/logged_search.py`
- `/Users/cleo/lattice/research_engine/tests/test_logged_search.py`
- `/Users/cleo/lattice/research_engine/docs/TEST-RESULTS-tel-C-search.md`

Edited:
- `/Users/cleo/.claude/skills/search/SKILL.md`
- `/Users/cleo/.claude/skills/research/SKILL.md`
- `/Users/cleo/.claude/skills/deep-research/SKILL.md`

Created backups:
- `/Users/cleo/.claude/skills/search/SKILL.md.bak-20260523-tel`
- `/Users/cleo/.claude/skills/research/SKILL.md.bak-20260523-tel`
- `/Users/cleo/.claude/skills/deep-research/SKILL.md.bak-20260523-tel`
