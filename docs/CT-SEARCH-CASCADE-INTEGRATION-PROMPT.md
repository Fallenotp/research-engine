# Deferred Codex Prompt — CT search_cascade.py integration

**Status:** DO NOT RUN until CT Phase 1 completes.
**Phase 1 watch:** PIDs 20819 (cost_watchdog) / 52222 (phase1_wave_runner) / 53673 (phase1_supervisor) must be terminated/finished before this prompt fires.
**Verify before running:** `ps -p 20819,52222,53673 -o pid=,command=` should return nothing.

---

## Why this is deferred

CT Phase 1 is the live Stanford GLiNER Vertex Batch run. Modifying `consequence-tracker/bridge/search_cascade.py` while Phase 1 is in flight risks contract drift (Phase 1 reads search_cascade output) and could corrupt the in-flight ledger.

Once Phase 1 finishes and the manifest is closed out, this prompt is safe to fire.

---

## The prompt to give Codex 5.4

```
TASK: Update /Users/cleo/consequence-tracker/bridge/search_cascade.py to use the shared research_engine for full-text extraction instead of storing only search-result snippets.

CURRENT STATE (verify before changing):
- search_cascade.py:_normalize_hit() (around line 277-285) stores title, url, content (snippet only) from SearXNG/You.com hits.
- It does NOT fetch the full article body.
- Result: CT timeline events are grounded on 1-2 sentence previews, not full articles.

THIS IS A MODIFY STEP — Pass current file content into the prompt and ask for the COMPLETE file back. Do not append; overwrite via write_text. Run py_compile after every write.

GOAL:
After getting hits from SearXNG and You.com, run extract_clean_text on the top 5 hits per query and persist the full text. Snippets remain in the result for backward compatibility, but full text is now available downstream.

CHANGES:
1. Add at the top of search_cascade.py:
   import sys
   sys.path.insert(0, "/Users/cleo/lattice")
   from research_engine import extract_clean_text, SourceTier

2. After hits are normalized but before the function returns, add a "deep-fetch" pass:
   for hit in normalized_hits[:5]:
       record = extract_clean_text(hit["url"])
       if record:
           hit["full_text"] = record["text_preview"]   # or read raw_text_path for full
           hit["raw_text_path"] = record["raw_text_path"]
           hit["content_hash"] = record["content_hash"]
           hit["extraction_method"] = record["extraction_method"]
           hit["char_count"] = record["char_count"]
       else:
           hit["full_text"] = hit.get("content", "")  # fall back to snippet
           hit["raw_text_path"] = None

3. Update the telemetry block (around line 429-447) to ALSO log:
   - deep_fetch_attempted (count)
   - deep_fetch_success (count)
   - average char_count of successful fetches
   - method breakdown (jina/trafilatura/crawl4ai/curl/docling/pymupdf)

4. Do NOT change any return-shape contracts that downstream consumers rely on. The new keys are additive — every existing key stays untouched.

5. Be conservative on rate: cap deep-fetch at 5 URLs per query; rate-limit to 2 per second; skip URLs that already have a content_hash in the cache directory (/Users/cleo/lattice/research_engine/cache/).

CONSTRAINTS:
- Absolute paths only.
- Do NOT modify any other CT file. This is a single-file change.
- Run `python3.11 -m py_compile /Users/cleo/consequence-tracker/bridge/search_cascade.py` after every write — must pass.
- Run `ruff check /Users/cleo/consequence-tracker/bridge/search_cascade.py` — must pass.
- python3.11 compatible. Use the lattice venv for imports, but the bridge venv for running CT tests.

SUCCESS CHECKS:
- File compiles.
- Lint clean.
- Existing CT unit tests on search_cascade still pass: `cd /Users/cleo/consequence-tracker && pytest bridge/tests/test_search_cascade.py -x`
- New keys present in output: run a smoke test that calls the function with a known query and verifies the result has full_text set on at least one hit.

Return the COMPLETE updated file. Include all existing functions plus the new code.
```

---

## After Codex finishes

1. Read the actual diff with `git -C /Users/cleo/consequence-tracker diff bridge/search_cascade.py`.
2. Run the existing CT search_cascade tests.
3. Run a real query to make sure the deep-fetch path works end-to-end.
4. Send to Opus for fresh-eyes review (CT-specific risks: contract drift, Phase 2 input compatibility).
5. Only then commit.
