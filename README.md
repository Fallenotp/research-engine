# Research Engine

A self-hosted web research pipeline. Give it a question and it finds sources, pulls clean text out of them (even from hard-to-scrape sites), checks every claim against the fetched evidence, and returns a cited answer. Free and local tools always run first; paid services are a logged last resort with a hard cost cap.

It powers three levels of research, all built on the same core:

- **search** — quick fact lookup (one worker, free sources first)
- **research** — decision-grade answer (multiple workers plus a standing counter-evidence lane)
- **deep-research** — exhaustive scour (a fleet of workers across different LLM vendors, then an independent verification pass)

## How a query flows through the engine

1. **Route.** The router (`router.py` + `router_config.yaml`) reads the topic and picks which search "lanes" fit it — a legal question goes to court and government lanes, a coding question to code-search lanes, and so on. Every topic also gets the general SearXNG lane.
2. **Search.** Each lane returns candidate URLs. Free lanes run first; the paid proxy lane only fires when free results are thin.
3. **Read.** `extract_clean_text` (`extractor.py`) pulls clean text from each URL by walking a cost ladder — it starts with the cheapest tool and escalates one rung at a time, stopping at the first rung that returns good text.
4. **Check.** The grounding and evidence gates verify that every claim traces to fetched text, quotes match word-for-word, and the answer doesn't assert more than the sources support.
5. **Iterate or answer.** `sufficiency.py` scores the evidence. Strong enough → cited answer. Gaps → `iteration_controller.py` launches another round. Still weak after the budget is spent → the engine abstains and says so.

## Search sources (the lanes)

All lanes are registered in `router_config.yaml`. Grouped by what they cover:

### General web + paid fallback
| Lane | What it is |
|---|---|
| `searxng_general`, `searxng_forums` | Local SearXNG metasearch on `localhost:8888` — the free backbone, always first |
| `paid_proxy` | Semantic search proxy on `localhost:18791` — only when free lanes are thin |
| `linkup_direct`, `exa_direct`, `youcom_direct`, `tavily_direct`, `firecrawl_direct` | Direct API search lanes (Linkup, Exa, You.com, Tavily, Firecrawl) — keyed, used sparingly |
| `gemini_pro_scout`, `counter_evidence` | LLM-driven lanes: a scout that widens the query, and a standing lane that hunts for evidence AGAINST the emerging answer |

### Government, legal, and finance
`courtlistener`, `congress_gov`, `govinfo_gov`, `openstates`, `fec_gov`, `sec_edgar`, `sec_gov`, `fda_gov`, `nih_gov`, `nasa_techtransfer`, `dod_oss` — all free official APIs.

### Academic and data
`pubmed`, `arxiv`, `semantic_scholar`, `core`, `papers_with_code`, `hf_hub`, `hf_datasets`, `kaggle`.

### Code
`github_code`, `gitlab_code`, `codeberg_code`, `sourcehut_code`, `sourcegraph`, `stack_exchange`, `hn_algolia`.

### Community signal
`reddit_rss`, `reddit_json`, `x_pulse`, `bluesky_jetstream` — what people are actually saying, no API keys needed.

### Local memory
`mentor_memory`, `claude_memory_files`, `obsidian` — recall from the machine's own knowledge stores before hitting the web.

## Reading tools (the ladder)

`extractor.py` branches by content type first, then escalates by how hostile the page is:

### Normal pages
1. **Trafilatura / curl_cffi** — plain and lightly-protected pages (free, instant)
2. **Cloudflare markdown / Jina reader** — quick remote HTML-to-text converters
3. **Crawl4AI / Crawlee** — JavaScript-heavy pages, rendered in local Chromium

### Hostile pages
4. **Scrapling + Camoufox** — bot walls (Cloudflare, Turnstile) via a stealth browser
5. **Firecrawl** — API-based scraping for stubborn sites
6. **Steel + Stagehand** — a real headed browser session, the last resort for login walls
7. **Apify actors** — platform-specific scrapers (managed through `apify_accounts.py`, which rotates a pool of accounts)

### Special content
- **PDFs** — PyMuPDF, MarkItDown, Docling (never sent to a web reader)
- **Code repos** — gitingest-style repo packing
- **Dead links** — `wayback_fallback.py` tries the Wayback Machine archive
- **Blocked IPs** — `fetch_proxy.py` routes retries through a rotating NordVPN proxy

Every attempt is logged by `telemetry_observer.py` with what it cost and what it returned.

## The LLMs it uses

No metered API keys anywhere. All model calls go through subscription CLI tools installed on the machine (`llm_call.py`, backends: `codex`, `sonnet`, `opus`, `gemini`), with automatic fallback to the next backend if one fails.

### Worker models (defined in `schema.py`)
| Family | Models | Typical job |
|---|---|---|
| Anthropic | Haiku, Sonnet, Opus | Cheap search workers (Haiku) up to synthesis and oversight (Opus) |
| OpenAI Codex | codex-mini, 5.3, 5.4, 5.5 | Default code-oriented worker is 5.4; 5.5 reserved for genuinely hard reasoning |
| Google | Gemini Flash, Gemini Pro | High-volume search workers and the query scout |
| Others | Mistral (free-tier keys), Grok | Extra lanes; Grok is the standing counter-evidence / real-time voice |

### Worker roles (how a fleet divides the work)
- **keyword** — classic search-term worker
- **semantic** — meaning-based search worker
- **counter_evidence** — actively tries to disprove the emerging answer
- **domain_specialist** — deep-dives one lane (legal, academic, code…)

Deep-research runs workers from *different vendors* on the same question on purpose — agreement across unrelated models is treated as a stronger signal than agreement within one family.

## Honesty guardrails

- `grounding.py` + `evidence_gate.py` — every claim must trace to a fetched source
- `verbatim_check.py` — quotes checked word-for-word against source text
- `anti_hallucination_gate.py` — blocks answers that outrun the evidence
- `sufficiency.py` — confidence scoring; below 0.35 the engine **abstains** ("Insufficient evidence — refine query or expand search"), 0.70+ is required for a full-confidence answer
- Cost control — paid calls are logged and capped (default **$1.50 per session**)

## Main pieces

| File | What it does |
|---|---|
| `research_cli.py` | Command-line entry point (`--mode search` by default) |
| `router.py`, `router_config.yaml` | Topic → lanes; cheapest first; all lanes registered in one file |
| `dispatcher.py` | Fans work out to LLM workers and collects results |
| `llm_call.py` | One interface to all model backends, with fallback order |
| `extractor.py` | The reading ladder — clean text from any URL |
| `fetch_proxy.py`, `apify_accounts.py`, `politeness.py` | Proxy rotation, account pooling, rate-limit manners |
| `schema.py` | Data shapes: sessions, sources, evidence, gaps, worker models |
| `persistence.py` | Save/load research sessions |
| `telemetry_observer.py`, `telemetry_to_csv.py` | Per-call cost and result logging |
| `webread_service.py`, `l3_guard.py`, `l3_reaper.py` | Long-running read service + safety rails for the headed-browser tier |
| `docs/` | Protocols, the fetch/search decision guide, test results, key-health runbook |

## Setup

Requirements:

- Python 3.11+ with the scrape stack: `trafilatura`, `curl_cffi`, `crawl4ai`, `scrapling`
- A local [SearXNG](https://github.com/searxng/searxng) instance on `localhost:8888`
- The subscription CLIs for whichever LLM backends you want (`codex`, `claude`, `agy`/Gemini)
- Optional: paid search proxy on `localhost:18791`, NordVPN credentials for the rotating proxy, Apify accounts for platform scrapers

Secrets are **never** stored in this repo. Keys and credentials are read from environment variables and local credential files at run time. Quick check that the read stack is present:

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

- **Cheapest that works.** Free and local tools first; escalate one rung at a time; every paid call logged and capped.
- **Cite or admit.** A fetched, cited source or an honest "not found" — never a confident guess.
- **Disagreement is data.** A dedicated lane hunts for counter-evidence, and cross-vendor model agreement counts for more than same-family agreement.
- **Stop at the first good rung.** No re-fetching a page five ways when the first way worked.

## Status

This repo is a working snapshot of the engine (2026-07-03). It runs as part of a larger local agent setup; the SearXNG instance, proxy layer, LLM CLIs, and credential files are external things you point it at.
