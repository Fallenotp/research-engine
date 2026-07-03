# TEST RESULTS - research_cli stages 2 and 3

## Scope

- Implementation edits: `research_engine/research_cli.py`
- Test edits: `research_engine/tests/test_research_cli.py`
- Required backup created: `research_engine/research_cli.py.bak-stage23`
- This results file created: `research_engine/docs/TEST-RESULTS-research-cli-stage23.md`

Implementation edits were limited to `research_cli.py` and its test file. The backup and this
results file were created as requested.

## Live CLI Safety

- PASS: No live end-to-end `research_cli.py` run was executed.
- Reason: live mode can call `codex exec` through synthesis and can hang by nesting Codex inside
  Codex.

## Commands

### Ruff

Command:

```bash
cd /Users/cleo/lattice && ruff check research_engine/research_cli.py research_engine/tests/test_research_cli.py
```

Outcome: PASS

Output:

```text
All checks passed!
```

### Pytest

Command:

```bash
cd /Users/cleo/lattice && /opt/homebrew/bin/python3.11 -m pytest research_engine/tests/test_research_cli.py -v
```

Outcome: PASS

Output summary:

```text
collected 5 items
5 passed, 6 warnings in 0.45s
```

## Result

- PASS: `--mode search` regression tests passed.
- PASS: `--mode research` builds a `/research` session with 3 territories, sources, and answer.
- PASS: `--mode deep-research` builds a `/deep-research` session and the thin-evidence iteration
  path runs without crashing.
- PASS: both new modes abstain honestly when workers yield no usable sources.
