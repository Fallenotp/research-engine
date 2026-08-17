# Research Engine

A self-hosted web research pipeline. Ask it a question and it finds sources, pulls clean text out of
them (including from sites that fight scrapers), grounds every claim in the text it actually fetched,
and returns a cited answer — or says the evidence is too thin and abstains.

> **Read this first.** This is a working personal system, published because it may be useful, not a
> product. It runs daily on one machine. Running it elsewhere means standing up several local services
> and supplying your own API keys. Machine-specific paths are all environment variables, so nothing
> needs editing in the source, but plenty still needs configuring. Known gaps are listed at the bottom
> rather than hidden.

## Contents

- [How a query flows](#how-a-query-flows)
- [The three modes](#the-three-modes) — start here if you are deciding which to run
- [Search lanes](#search-lanes)
- [The reader ladder](#the-reader-ladder)
- [The LLMs it uses](#the-llms-it-uses)
- [Honesty guardrails](#honesty-guardrails)
- [Install](#install)
- [Configuration](#configuration)
- [Usage](#usage)
- [Known gaps](#known-gaps)

## How a query flows

1. The **router** (`router.py` + `router_config.yaml`) reads the question and scores 18 rules to pick
   which of the 46 search lanes to run. Multiple rules can match; it picks a primary and merges the
   rest, capped at 10 lanes.
2. Chosen lanes run. Free lanes first.
3. Each promising URL goes through the **reader ladder** (`extractor.py`), which escalates one rung at
   a time until something returns usable text.
4. **Grounding** (`grounding.py`) matches claims against the fetched text and scores confidence from
   real signals.
5. The answer comes back with citations, or the run abstains.

## The three modes

The CLI takes `--mode search|research|deep-research`. Default is `search`. There is no other mode.

| | `search` | `research` | `deep-research` |
|---|---|---|---|
| What it is | A fast fact lookup | A decision-grade answer | An exhaustive sweep |
| Worker LLMs | **None** | 3 | 16 territories |
| Extra rounds | None | None | One follow-up if gaps remain |
| Cost ceiling | Free-first | $0.50 | $1.50 |
| Use it for | "What is X's current pricing?" | "Should we pick A or B?" | An irreversible decision, or new territory |

**`search`** runs no worker LLMs at all. It queries SearXNG first, adds free API lanes only if fewer
than 5 results come back, and reaches for the paid proxy last. Cheapest by a wide margin.

**`research`** adds an optional Gemini Flash scout and a fixed fleet of 3 workers: one Codex
(`gpt-5.4-mini`), one Mistral (`mistral-small-latest`), one Grok (`grok-4.5`). **The Mistral worker is
inert unless you configure a key file** (see [Configuration](#configuration)), so out of the box this
runs on 2 live workers.

**`deep-research`** plans 16 territories across a larger fleet and runs one focused follow-up pass if
the session still has gaps.

## Search lanes

46 lanes are defined in `router_config.yaml`. Which ones fire depends on the question.

**Free, no key needed:** SearXNG (general web, your own instance), `reddit_rss`, `reddit_failures`,
`hackernews`, `arxiv`, `crossref`, `openalex`, `wikipedia`, `grok_x_search` (X/Twitter via the Grok
CLI), and others.

**Free but keyed — inert until you supply the key:**

| Lane | Needs |
|---|---|
| `github_code`, `dod_oss` | `GITHUB_TOKEN` |
| `gitlab_code` | `GITLAB_TOKEN` |
| `congress_gov`, `govinfo_gov` | `DATA_GOV_API_KEY` |
| `fec_gov` | `FEC_API_KEY` |
| `openstates` | `OPENSTATES_API_KEY` |
| `fda_gov` | `FDA_API_KEY` |
| `kaggle` | `~/.kaggle/kaggle.json` |

**Paid / proxied:** `paid_proxy`, `linkup_direct`, `tavily_direct`, `youcom_direct`,
`firecrawl_direct`, `exa_direct`. These route through a semantic-search proxy on `localhost:18791`
that is not part of this repo.

**Not callable from the CLI:** `hf_hub` and `hf_datasets` are declared as MCP lanes, and the
dispatcher ignores the `mcp_server` field, so they never run. `reddit_json` is marked deprecated in
the config (its endpoint returns 403); `reddit_rss` replaces it. `x_pulse` expects an
`x-pulse-research` script that is not in this repo.

## The reader ladder

`extract_clean_text()` in `extractor.py`. For a normal web URL it tries these in order, stopping at
the first rung that returns usable text:

1. **gitingest** — GitHub repo URLs only (not GitLab) get packed into one text file
2. **Apify actors** — only for routed platforms (X, Instagram and similar), via an account pool
3. **trafilatura** — plain HTTP article extraction. The workhorse; most pages stop here
4. **Crawl4AI** — JavaScript pages in local Chromium. ⚠️ Shells out to a helper script that is **not
   in this repo** — see [Known gaps](#known-gaps)
5. **Jina Reader** — hosted text extraction
6. **Crawlee** — a second crawler engine
7. **Scrapling** — stealth browser for bot walls (Cloudflare, Turnstile); brings in Camoufox
8. **agent-browser** — a real browser session. Runs by default, not behind a toggle
9. **Firecrawl** — API-based scraping, the last web rung
10. **Publisher fallback** — open-access publisher lookup, for paper-like URLs only
11. **Wayback Machine** — for dead links

Other content routes:

- **Local files and `file://` links** work, not just web URLs
- **Documents and media** — Word, PowerPoint, Excel, PDF, images and audio go through MarkItDown;
  PDFs can also use PyMuPDF or Docling
- **Blocked IPs** — `fetch_proxy.py` can retry through a rotating VPN proxy. If both the VPN and its
  backup fail it browses unproxied and labels the result `no_proxy:...` rather than implying a proxy
  was in use

Every rung attempt is logged by `extractor.py` to `research-reader-telemetry.jsonl` in your data
directory. `telemetry_observer.py` is a separate tool that post-processes saved sessions.

## The LLMs it uses

Everything runs through CLIs you are already signed into. No API keys are read for the LLMs
themselves.

| Role | What actually runs |
|---|---|
| Final synthesis | Opus → Codex `gpt-5.5` → Sonnet, tried in that order |
| Scout (research, deep-research) | Gemini 3.7 Flash via `agy`. **Optional** — if it fails the run continues without it |
| Worker: Codex | `gpt-5.4-mini` (territories are labelled `codex-5.4`) |
| Worker: Grok | `grok-4.5` via the Grok CLI (`grok --single`). Fallback is `cursor-agent --model cursor-grok-4.5-high`. **Not** Hermes — its xAI auth is dead |
| Worker: Mistral | `mistral-small-latest` over HTTP. Needs a key file |

Three things worth knowing:

- **Gemini Pro is never used.** The scout is explicitly restricted to Flash.
- **Territories labelled `haiku` do not run Haiku.** `llm_call.py` supports exactly four backends —
  codex, sonnet, opus, gemini. A `haiku` territory falls through to a generic summariser that uses
  Codex or Sonnet. The label is a leftover.
- There is a **Gemini daily budget of 300 calls** (`RESEARCH_ENGINE_GEMINI_DAILY_BUDGET`). Past it, or
  when `RESEARCH_ENGINE_UNATTENDED` is set, Gemini calls are swapped to `GPT-OSS 120B (Medium)`.

## Honesty guardrails

The point of the project. Several were added after an audit found the engine was reporting quality
scores that were constants typed into the source — a content-farm blog scored identically to a
manufacturer.

- **Confidence is measured, not asserted.** Source authority and count feed a weighted score, mapped
  through thresholds in `router_config.yaml`.
- **Failures fail down, not up.** If the confidence config cannot be read, the run abstains and logs
  why. It does not fall back to a confident-looking default.
- **Evidence scoring is containment, not Jaccard**, calibrated on real claim and paragraph pairs.
- **Counter-evidence lane.** One Grok worker is pointed at a mutated query specifically to find
  reasons the obvious answer is wrong.
- **Cite or admit.** A fetched, cited source, or an honest "insufficient evidence".

## Install

```bash
git clone https://github.com/Fallenotp/research-engine.git
cd research-engine
pip install -e .                                          # core
pip install -e '.[extractors,scrapling,pdf,markitdown]'   # the full read stack
```

Python 3.10 or newer. It is only actually run and tested on 3.11.

### Services it expects

| Thing | Needed for | Where |
|---|---|---|
| SearXNG on `localhost:8888` | All modes. The primary search lane | https://github.com/searxng/searxng |
| `agy` (Gemini CLI wrapper) | The optional scout | Local wrapper; must be on `PATH` |
| `grok` CLI | The Grok worker and `grok_x_search` | https://grok.com |
| `codex` CLI | Default LLM backend | OpenAI Codex CLI |
| `claude` CLI | Sonnet and Opus backends | Anthropic Claude CLI |
| Semantic proxy on `localhost:18791` | Paid search lanes only | Not in this repo |

### Reader tools

Every one of these has a live call site in `extractor.py`. Install only what you need — a missing tool
skips its rung rather than failing the run.

| Tool | Install | Source |
|---|---|---|
| trafilatura | `pip install trafilatura` (core dep) | https://github.com/adbar/trafilatura |
| gitingest | `pip install gitingest` | https://github.com/cyclotruc/gitingest |
| Crawl4AI | `pip install crawl4ai` **plus your own helper script** | https://github.com/unclecode/crawl4ai |
| Jina Reader | No install; optional `JINA_API_KEY` | https://jina.ai/reader |
| Crawlee | `pip install crawlee` | https://github.com/apify/crawlee-python |
| Scrapling | `pip install scrapling && scrapling install` | https://github.com/D4Vinci/Scrapling |
| Camoufox | Comes in with Scrapling | https://github.com/daijro/camoufox |
| agent-browser | `npm install -g agent-browser` | https://github.com/vercel-labs/agent-browser |
| Firecrawl | No package; needs API keys | https://github.com/firecrawl/firecrawl |
| Apify | No package; needs account tokens | https://docs.apify.com |
| MarkItDown | `pip install 'markitdown[all]'` | https://github.com/microsoft/markitdown |
| PyMuPDF | `pip install PyMuPDF` | https://github.com/pymupdf/PyMuPDF |
| Docling | `pip install docling` | https://github.com/docling-project/docling |
| Wayback | No install | https://archive.org/help/wayback_api.php |

## Configuration

Nothing is hardcoded to one machine. Paths and binaries resolve through `paths.py`, which reads
environment variables and falls back to sensible defaults. Copy `.env.example`, fill in what you need,
and point `RESEARCH_ENGINE_ENV_FILE` at it — or export the variables directly.

The ones most people set:

| Variable | Default | What it does |
|---|---|---|
| `RESEARCH_ENGINE_DATA_DIR` | `~/.research_engine` | Telemetry, logs, session store, memory DB |
| `RESEARCH_ENGINE_CONTACT_EMAIL` | *(unset)* | Your address in outbound `User-Agent` headers. Crossref and the Wayback Machine give better rate limits when you supply one |
| `RESEARCH_ENGINE_ENV_FILE` | *(unset)* | An `.env` file to load API keys from |
| `RESEARCH_ENGINE_MISTRAL_KEYS_FILE` | *(unset)* | Without this the Mistral worker is inert |
| `RESEARCH_ENGINE_CRAWL4AI_SCRIPT` | *(unset)* | Path to your Crawl4AI helper — that rung is dead without it |
| `RESEARCH_ENGINE_GEMINI_DAILY_BUDGET` | `300` | Gemini calls per day before swapping to GPT-OSS |

Binaries are found with `shutil.which()` first, so if the CLIs are on your `PATH` you need not set
anything. When a required binary is genuinely missing, the error names the exact variable to set.

Integrations with the author's other tools (Obsidian, a local buzz script, a semantic-search proxy)
are **off unless you set their variable**. See `.env.example` for the full list.

## Usage

```bash
python research_cli.py --mode search        "current pricing for X"
python research_cli.py --mode research      "should we choose A or B"
python research_cli.py --mode deep-research "everything known about X"
```

Sessions are saved under `$RESEARCH_ENGINE_DATA_DIR/research-sessions/`, which defaults to
`~/.research_engine/research-sessions/`. Override with `RESEARCH_ENGINE_RESEARCH_SESSIONS_DIR`.

Run the tests from the repo root, not from `tests/` — twelve test files live at the top level:

```bash
pytest . -q
ruff check .
```

## Design principles

- **Cheapest that works.** In `search` mode free lanes run first and paid search fires only when free
  results are thin. This applies to `search` only; `research` and `deep-research` use the paid proxy
  as part of their normal flow.
- **Cite or admit.** A cited source or an honest abstention, never a confident guess.
- **Disagreement is data.** A counter-evidence worker hunts for reasons the obvious answer is wrong.
- **Stop at the first good rung.** No fetching a page five ways when the first way worked.

## Known gaps

Stated plainly rather than discovered later.

- **Crawl4AI needs a helper script that is not in this repo.** Set `RESEARCH_ENGINE_CRAWL4AI_SCRIPT`
  to your own, or that rung returns nothing.
- **Some lanes cannot run at all.** `hf_hub` and `hf_datasets` are MCP lanes the dispatcher ignores;
  `x_pulse` needs an external script that is not included; `reddit_json` is deprecated upstream.
- **`haiku` worker labels are cosmetic** — those territories execute on Codex or Sonnet.
- **The Mistral worker is inert** until you supply a key file, so `research` mode runs 2 of 3 workers
  out of the box.
- **No console entry point.** `research_cli.py` has a `main()`, but the module imports heavy optional
  extractors at load time, so a `research-engine` command would break on a minimal install.
- **The full read stack is heavy.** The core install is light; the extras pull in browsers.
- Several MEDIUM and LOW findings from an internal code audit remain open — long functions, duplicate
  helpers, style. None affect correctness.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues go through the Security tab, not a public
issue — see [SECURITY.md](SECURITY.md).

## Licence

GNU AGPL-3.0-or-later. See [LICENSE](LICENSE).

Use it, change it, run it, charge for it. The one condition: if you distribute it, or run a modified
version as a network service other people can reach, you have to make your source available under the
same licence. You cannot take this, close it, and ship it as a private product.

If that does not suit your situation, ask.
