# TEST RESULTS - research_cli Stage 2.5

Date: 2026-05-24

## Commands

- `ruff check research_engine/research_cli.py research_engine/tests/test_research_cli.py`
  - Outcome: PASS
  - Output: `All checks passed!`

- `/opt/homebrew/bin/python3.11 -m pytest research_engine/tests/test_research_cli.py -v`
  - Outcome: PASS
  - Output: `9 passed, 6 warnings in 0.86s`

## Interlock Requirement

`save_session` requires `/research` and `/deep-research` terminal sessions to include a successful `GeminiProRunRecord` with run type `scout` or `pro_synthesis_fallback` and model id `gemini-3-flash`.

## Proof Gate

- Rate: 9 pytest cases in 0.86s, about 10.5 tests/second.
- Time: ruff completed immediately; pytest reported 0.86s.
- Cost: $0 external API cost. Unit tests mocked Gemini and did not run live `research_cli.py`.

## Backup

- `research_engine/research_cli.py.bak-stage25`

## Live Run

No live `research_cli.py` run was executed.
