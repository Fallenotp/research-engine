# Research Engine

A self-hosted web research pipeline. Give it a question and it finds sources, pulls clean text out of them (even from hard-to-scrape sites), grounds every claim in the fetched evidence, and returns a cited answer — or honestly abstains when the evidence is too thin.

> **Read this first:** this is a working system, not a polished library. Every machine-specific path is now an environment variable with a sensible default (see [Configuration](#configuration)), so it runs off your machine — but it still expects several local services and third-party API keys you supply yourself. There is no packaging file yet; see [Making imports work](#making-imports-work). Treat it as a system to stand up, not a `pip install`-and-go package.

It powers three research modes, all in `research_cli.py`:

- **search** — quick fact lookup: free sources first, paid proxy only when free results are thin
- **research** — decision-grade answer: several LLM workers in parallel plus a standing counter-evidence lane
- **deep-research** — the big version: 16 worker "territories" (10 Haiku, 5 Codex, 1 Grok), cross-vendor on purpose, with one optional follow-up pass to fill gaps

## How a query flows through the engine

1. **Route.** The router (`router.py` + `router_config.yaml`) matches the topic to search "lanes" — legal questions to court/government lanes, code questions to code-search lanes, and so on. In **search** mode these lanes are really used: SearXNG first, then the topic's free API lanes if results are thin, then the paid proxy. In the multi-worker modes each worker searches SearXNG plus the semantic proxy directly.
2. **Read.** `extract_clean_text` (`extractor.py`) turns each URL into clean text by trying tools in a fixed order (see the ladder below) and stopping at the first that returns good text.
3. **Check.** Grounding gates verify claims trace to fetched text, quotes match word-for-word, and the answer doesn't outrun the evidence.
4. **Answer or abstain.** Enough evidence → cited answer. Not enough → the engine says "Insufficient evidence — refine query or expand search" instead of guessing. Deep-research gets at most one extra follow-up round for gaps.

## Search sources (the lanes)

All lanes are registered in `router_config.yaml`. Grouped by what they cover:

### General web + paid fallback
| Lane | What it is |
|---|---|
| `searxng_general`, `searxng_forums` | Local SearXNG metasearch on `localhost:8888` — the free backbone |
| `paid_proxy` | Semantic search proxy on `localhost:18791` |
| `linkup_direct`, `exa_direct`, `youcom_direct`, `tavily_direct`, `firecrawl_direct` | Direct API search lanes (keyed) |
| `gemini_pro_scout`, `counter_evidence` | LLM lanes: a scout that widens the query, and a lane that hunts for evidence AGAINST the emerging answer |

### Government, legal, and finance
`courtlistener`, `congress_gov`, `govinfo_gov`, `openstates`, `fec_gov`, `sec_edgar`, `sec_gov`, `fda_gov`, `nih_gov`, `nasa_techtransfer`, `dod_oss` — free official APIs (several need free API keys, listed under Setup).

### Academic and data
`pubmed`, `arxiv`, `semantic_scholar`, `core`, `papers_with_code`, `hf_hub`, `hf_datasets`, `kaggle`.

### Code
`github_code`, `gitlab_code`, `codeberg_code`, `sourcehut_code`, `sourcegraph`, `stack_exchange`, `hn_algolia`.

### Community signal
`reddit_rss`, `reddit_json`, `x_pulse`, `hn_algolia`, plus a Bluesky lane (`bluesky_jetstream` — despite the name it now shells out to an HTTP Bluesky search; the original websocket path is dead).

### Local memory
`mentor_memory`, `claude_memory_files`, `obsidian` — recall from the machine's own knowledge stores before hitting the web.

## Reading tools (the ladder)

For a normal web URL, `extract_clean_text()` tries, in order:

1. **Repo ingest** — GitHub/GitLab-style repo URLs get packed into one text file first
2. **Apify actors** — platform-specific scrapers for routed platforms (X, Instagram…), via an account pool (`apify_accounts.py`) and a local Apify proxy
3. **agent-browser** — a real browser session driven by the `agent-browser` CLI (only when explicitly enabled)
4. **Crawl4AI** — JavaScript pages rendered in local Chromium (shells out to a helper script)
5. **Crawlee** — a second crawler engine as backup
6. **Scrapling + Camoufox** — stealth browser for bot walls (Cloudflare, Turnstile)
7. **Firecrawl** — API-based scraping as the final web rung

Special content branches:

- **Local files and `file://` links** work too — not just web URLs
- **Documents and media** — Word, PowerPoint, Excel, PDF, images, and audio (`.mp3/.m4a/.wav`) go through MarkItDown; PDFs can also use PyMuPDF/Docling
- **Blocked IPs** — `fetch_proxy.py` can route retries through a rotating VPN proxy
- **Dead links** — `wayback_fallback.py` is wired into the ladder in `extractor.py` and retries via the Wayback Machine

Every attempt is logged by `telemetry_observer.py` with what it returned.

## The LLMs it uses

Model calls run through two paths:

- **Subscription CLI tools** via `llm_call.py` — backends `codex`, `sonnet`, `opus`, `gemini` (Gemini through the `agy` CLI), with automatic fallback to the next backend if one fails.
- **Direct calls outside `llm_call.py`** — Mistral is called over its HTTP API using a rotating pool of free-tier keys, and Grok runs through a local `hermes` command.

**Gemini is not optional** for `research` and `deep-research`: the query scout and final synthesis depend on it, and a finished session is rejected by `persistence.py` if the required Gemini records are missing. No Gemini CLI → those modes fail closed or abstain.

### Worker models (defined in `schema.py`)
| Family | Models | Typical job |
|---|---|---|
| Anthropic | Haiku, Sonnet, Opus | Cheap search workers (Haiku) up to synthesis and oversight |
| OpenAI Codex | codex-mini, 5.3, 5.4, 5.5 | Code-oriented and hard-reasoning workers |
| Google | Gemini Flash, Gemini Pro | Query scout, high-volume workers, final synthesis |
| Others | Mistral (free-tier HTTP keys), Grok (via hermes) | Extra lanes; Grok is the counter-evidence / real-time voice |

### Worker roles
- **keyword** — classic search-term worker
- **semantic** — meaning-based search worker
- **counter_evidence** — actively tries to disprove the emerging answer
- **domain_specialist** — deep-dives one lane (legal, academic, code…)

Deep-research deliberately mixes vendors — agreement across unrelated model families counts for more than agreement within one.

## Honesty guardrails

- `grounding.py` + `evidence_gate.py` — every claim must trace to a fetched source
- `verbatim_check.py` — quotes checked word-for-word against source text
- `anti_hallucination_gate.py` — blocks answers that outrun the evidence
- Abstention — thin evidence gets an explicit "insufficient evidence" answer, never a guess
- `sufficiency.py` + `iteration_controller.py` — a fuller evidence-scoring loop (confidence thresholds, multi-round iteration). Note: in this snapshot the CLI does not call this loop; it stamps completed sessions with a fixed confidence and deep-research gets one follow-up pass. The modules are here for library use.

## Main pieces

| File | What it does |
|---|---|
| `research_cli.py` | Command-line entry point for all three modes |
| `router.py`, `router_config.yaml` | Topic → lanes; all lanes registered in one file |
| `dispatcher.py` | Fans work out to LLM workers, writes worker briefs, collects results |
| `llm_call.py` | CLI-backend interface (codex/sonnet/opus/gemini) with fallback order |
| `extractor.py` | The reading ladder — clean text from any URL or local file |
| `fetch_proxy.py`, `apify_accounts.py`, `politeness.py` | Proxy rotation, account pooling, rate-limit manners |
| `schema.py` | Data shapes: sessions, sources, evidence, gaps, worker models |
| `persistence.py` | Save/load research sessions (with completeness checks) |
| `sufficiency.py`, `iteration_controller.py` | Evidence-scoring loop (library use; not called by the CLI in this snapshot) |
| `telemetry_observer.py`, `telemetry_to_csv.py` | Per-call logging |
| `webread_service.py`, `l3_guard.py`, `l3_reaper.py` | Long-running read service + safety rails for the browser tier |
| `docs/` | Protocols, the fetch/search decision guide, test results, key-health runbook |

## Setup

### What it expects on the machine
- Python 3.11+ with the scrape stack: `trafilatura`, `curl_cffi`, `crawl4ai`, `scrapling`
- A local [SearXNG](https://github.com/searxng/searxng) instance on `localhost:8888`
- The subscription LLM CLIs: `codex`, `claude`, and `agy` (Gemini — required for research/deep-research modes)
- The `agent-browser` CLI, if you want the real-browser reading rung
- Optional services on localhost: semantic search proxy (`:18791`), Apify proxy (`:18793`), local read service (`:8077`)

### Keys and credential files (never stored in this repo)
- Free API keys read from the environment for some lanes: `GITHUB_TOKEN`, `GITLAB_TOKEN`, `DATA_GOV_API_KEY`, `FEC_API_KEY`, `OPENSTATES_API_KEY`, `FDA_API_KEY`, plus `~/.kaggle/kaggle.json` for Kaggle
- A Mistral free-key pool file, pointed at by `RESEARCH_ENGINE_MISTRAL_KEYS_FILE`
- Optional VPN credentials for the rotating fetch proxy

## Configuration

Nothing is hard-coded to one machine. Every path and binary is resolved through `paths.py`, which reads
environment variables and falls back to sensible defaults. Copy `.env.example`, fill in what you need, and
point `RESEARCH_ENGINE_ENV_FILE` at it — or just export the variables.

The ones most people will set:

| Variable | Default | What it does |
| --- | --- | --- |
| `RESEARCH_ENGINE_DATA_DIR` | `~/.research_engine` | Where telemetry, logs and the session store are written |
| `RESEARCH_ENGINE_CONTACT_EMAIL` | *(unset)* | Your contact address for outbound `User-Agent` headers. Crossref and the Wayback Machine give better rate limits when you supply one |
| `RESEARCH_ENGINE_ENV_FILE` | *(unset)* | Optional `.env` file to load API keys from |
| `RESEARCH_ENGINE_AGY_BIN` | found on `PATH` | Path to the `agy` (Gemini) CLI |
| `RESEARCH_ENGINE_GROK_BIN` | found on `PATH` | Path to the `grok` CLI |

Binaries are located with `shutil.which()` first, so if the CLIs are on your `PATH` you do not need to set
anything. When a required binary is genuinely missing, the error names the exact variable to set.

Integrations with the author's other tools (Obsidian, a local buzz script, a semantic-search proxy) are
**off unless you set their variable**. See `.env.example` for the full list of 24 variables.

### Making imports work
The code imports `from research_engine ...`, and this repo has no packaging file (`pyproject.toml`/`setup.py`). Clone it into a folder **named `research_engine`** and run from the parent directory, or add the parent directory to `PYTHONPATH`.

Quick check that the read stack is present:

```bash
python -c "import trafilatura, curl_cffi, crawl4ai, scrapling; print('scrape stack OK')"
```

## Usage

The CLI takes the question as a direct argument (`--topic` is an optional topic label, not the question):

```bash
python research_cli.py "your question here" --mode search
python research_cli.py "your question here" --mode research
python research_cli.py "your question here" --mode deep-research
```

Other flags: `--agent` and `--llm`. Output is the cited answer (or the abstain reason), followed by `session: <path>` and a `logged: <n> sources, <n> api calls` line. Sessions are saved under `~/.claude/research-sessions/`; deep-research also writes per-worker briefs and results under `/tmp/deep-research-*`.

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

- **Cheapest that works.** In search mode, free lanes run first and paid search only fires when free results are thin; the reading ladder always escalates one rung at a time.
- **Cite or admit.** A fetched, cited source or an honest "insufficient evidence" — never a confident guess.
- **Disagreement is data.** A dedicated lane hunts for counter-evidence, and deep-research spreads the same question across different model vendors.
- **Stop at the first good rung.** No re-fetching a page five ways when the first way worked.

## Status

Working snapshot, 2026-08-16, of a live personal system that runs daily. Running it elsewhere means
standing up the services listed under Setup and supplying your own API keys — but it no longer means
editing paths in the source.

**What changed since the 2026-07-03 snapshot**

- Every `/Users/...` path replaced by an environment variable with a default (`paths.py`).
- Contact email in outbound `User-Agent` headers is now yours to set, not baked in.
- The dead Steel browser backend was removed; `agent-browser` covers that rung.
- A wiring audit found that several quality scores — source authority, rerank score, answer confidence —
  were hardcoded constants that measured nothing. They now derive from real signals, and confidence
  fails *down* to an abstain rather than up to a confident-looking answer when config cannot be read.
- Router `fleets:` config is now actually read by the research path instead of being ignored.
- Test suite: 362 passing.

**Known gaps, stated plainly**

- No `pyproject.toml`/`setup.py` yet — see [Making imports work](#making-imports-work).
- One test (`test_proxy_json_enforce`) fails without a local proxy service running.
- The internal `docs/` folder is not published; it is working notes, not documentation.

## Licence

MIT. See [LICENSE](LICENSE).
