# Research Engine

A self-hosted web research pipeline. Give it a question or a URL and it finds sources, pulls clean text out of them (even from hard-to-scrape sites), checks the evidence, and returns a cited answer — using free/local tools first and paid services only as a last resort.

It powers three levels of research, all built on the same core:

- **search** — quick fact lookup (one worker, free sources first)
- **research** — decision-grade answer (multiple workers, counter-evidence lane)
- **deep-research** — exhaustive scour (large worker fleet, verification pass)

## How it works

Two stages, never confused with each other:

1. **SEARCH — find which URLs to look at.** The router (`router.py` + `router_config.yaml`) picks the right "lane" per topic: a local SearXNG metasearch instance first, then specialized free sources (SEC EDGAR, CourtListener, GitHub, etc.), then a paid search proxy only if the free lanes come back thin.
2. **READ — pull clean text from a URL you already have.** `extract_clean_text` in `extractor.py` walks a ladder from cheapest to heaviest and stops at the first rung that returns good text:
   - Trafilatura / plain HTTP for simple pages
   - Crawl4AI for JavaScript-heavy pages
   - Scrapling + Camoufox for bot walls (Cloudflare, Turnstile)
   - A real headed browser as the last resort for login walls
   - Special branches for PDFs, code repos, and archived pages (`wayback_fallback.py`)

The full decision rules live in [`docs/FETCH-AND-SEARCH-DECISION-GUIDE.md`](docs/FETCH-AND-SEARCH-DECISION-GUIDE.md).

## Honesty guardrails

The engine is built around a "no guessing" rule:

- `grounding.py` + `evidence_gate.py` — every claim must trace back to a fetched source.
- `verbatim_check.py` — quotes are checked word-for-word against the source text.
- `anti_hallucination_gate.py` — blocks answers that assert more than the evidence supports.
- `sufficiency.py` + `iteration_controller.py` — decides whether the evidence is enough or another search round is needed.

If a fact can't be cited to a fetched page, the engine says so instead of filling the gap.

## Main pieces

| File | What it does |
|---|---|
| `research_cli.py` | Command-line entry point (`--mode search` by default) |
| `router.py`, `router_config.yaml` | Picks search lanes per topic; cheapest first |
| `dispatcher.py` | Fans work out to LLM workers and collects results |
| `extractor.py` | The reading ladder — clean text from any URL |
| `fetch_proxy.py` | Optional rotating-proxy layer for blocked IPs |
| `schema.py` | Data shapes: sessions, sources, evidence, gaps |
| `persistence.py` | Save/load research sessions |
| `telemetry_observer.py` | Logs what each tool call cost and returned |
| `webread_service.py` | Long-running read service used by other tools |
| `docs/` | Protocols, decision guide, test results, key-health runbook |

## Setup

Requirements:

- Python 3.11+ with the scrape stack installed: `trafilatura`, `curl_cffi`, `crawl4ai`, `scrapling`
- A local [SearXNG](https://github.com/searxng/searxng) instance on `localhost:8888` (the free search backbone)
- Optional: a paid search proxy on `localhost:18791` for when free lanes are thin

Secrets are **never** stored in this repo. API keys and proxy credentials are read from environment variables and local credential files at run time. Quick check that the read stack is present:

```bash
python -c "import trafilatura, curl_cffi, crawl4ai, scrapling; print('scrape stack OK')"
```

## Usage

As a CLI:

```bash
python research_cli.py --topic "your question here" --mode search
```

As a library:

```python
from research_engine import extract_clean_text, SourceTier

text = extract_clean_text("https://example.com/article", tier=SourceTier.T2)
```

Run the tests:

```bash
pytest tests/ -q
```

## Design principles

- **Cheapest that works.** Free and local tools first; escalate one rung at a time; paid calls are the last resort and are logged.
- **Cite or admit.** A fetched, cited source or an honest "not found" — never a confident guess.
- **Stop at the first good rung.** No re-fetching a page five ways when the first way worked.

## Status

This repo is a working snapshot of the engine (2026-07-03). It runs as part of a larger local agent setup; the SearXNG instance, proxy layer, and worker models are external services you point it at.
