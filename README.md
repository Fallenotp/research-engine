# Research Engine

A self-hosted web research pipeline. Ask it a question and it finds sources, pulls clean text out of
them (including from sites that fight scrapers), grounds every claim in the text it actually fetched,
and returns a cited answer — or says the evidence is too thin and abstains.

> **Read this first.** This is a working personal system, published because it may be useful, not a
> product. It runs daily on one machine. Running it elsewhere means standing up several local services
> and supplying your own API keys. Known gaps are listed at the bottom rather than hidden.

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
2. Chosen lanes run. **In `search` mode**, free lanes run first and the paid proxy fires only if the
   free ones come back thin. In `research` and `deep-research` the workers always hit SearXNG and the
   paid proxy — lane names there shape the worker's brief rather than selecting the APIs that run.
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
| Worker fleet | **none** | 3 workers | 16 territories |
| Still uses LLMs? | Yes — see below | Yes | Yes |
| Extra rounds | None | None | One follow-up if gaps remain |
| Use it for | "What is X's current pricing?" | "Should we pick A or B?" | An irreversible decision, or new territory |

**`search`** runs no worker *fleet*, but it is not LLM-free: every web search fires one Grok CLI
X-search pass, and the final answer is synthesised by an LLM (Codex, falling back to Sonnet). It is
much the cheapest mode, not a zero-cost one.

**`research`** adds an optional Gemini Flash scout and a fixed fleet of 3 workers: one Codex
(`gpt-5.4-mini`), one Mistral (`mistral-small-latest`), one Grok (`grok-4.5`). **The Mistral worker is
inert unless you configure a key file** (see [Configuration](#configuration)), so out of the box this
runs on 2 live workers.

**`deep-research`** plans 16 territories across a larger fleet and runs one focused follow-up pass if
the session still has gaps.

### How many pages each mode reads

`research_cli.PAGE_BUDGET` is the single table, and it is the only page cap in the codebase:

| Mode | Pages read |
|---|---|
| `search` | 5 |
| `research` | 6 **per territory** |
| `deep-research` | 8 **per territory** |

A budget counts *unique successful* pages: a failed fetch, a skipped rung or a duplicate URL never
consumes a slot. Candidates are ranked before any fetch (see below), so the budget is spent on the
results most likely to answer the question rather than on whichever arrived first.

**What the model actually sees.** The full cleaned page is always stored — `char_count` equals the
length of the file on disk — but a prompt carries only the best-matching windows, up to 4,000
characters per page. Windows are chosen by BM25 over the question, aligned to paragraph boundaries,
returned in document order, and carry exact offsets into the stored text. Ranking and windowing are
plain arithmetic in `windows.py`: no model call, no network, same answer every run.

One slot per batch of five candidates is an **exploration slot**, so a page the search engine ranked
first is never dropped purely because it shares few words with your question.

⚠️ These budgets are provisional. They read more pages than the previous hard-coded 2 or 3, and they
send more text to the model, not less: up to 4,000 characters of selected windows per page against a
former 1,500-character prefix. They stand or fall on a replay measurement — if a larger budget does
not raise the FULL-answer rate, the numbers come back down.

`router_config.yaml` carries `budget_ceiling_usd` values of 0.50 and 1.50 for these modes. **They are
not enforced.** The only code that reads them lives in `iteration_controller.py`, which the CLI does
not call. Treat them as intent, not a spend limit.

## Search lanes

46 lanes are defined in `router_config.yaml`. Which ones fire depends on the question.

**Free, no key needed:** SearXNG (general web, your own instance), `hn_algolia`, `reddit_rss`,
`arxiv`, `pubmed`, `semantic_scholar`, `core`, `papers_with_code`, `sec_edgar`, `courtlistener`,
`sourcegraph`, `codeberg_code`, `grok_x_search` (X/Twitter via the Grok CLI), and others.

**Free but keyed — that lane fails with a `requires env var` error until you supply the key:**

| Lane | Needs |
|---|---|
| `github_code`, `dod_oss` | `GITHUB_TOKEN` |
| `gitlab_code` | `GITLAB_TOKEN` |
| `congress_gov`, `govinfo_gov` | `DATA_GOV_API_KEY` |
| `fec_gov` | `FEC_API_KEY` |
| `openstates` | `OPENSTATES_API_KEY` |
| `fda_gov` | `FDA_API_KEY` |
| `kaggle` | `~/.kaggle/kaggle.json` |

**Proxied** through a semantic-search proxy on `localhost:18791` that is not part of this repo:
`paid_proxy`, `linkup_direct`, `tavily_direct`, `youcom_direct`, `firecrawl_direct` (paid), and
`exa_direct` (proxied but **free** — cost 0.0 in the config).

**Local lanes** search your own files and are off unless you point their variable at real data. If the
path does not exist the lane reports `not configured: missing <path>. Set <ENV_VAR>.` rather than
returning an empty result set that reads like "nothing found".

**Cannot run at all:** `hf_hub` and `hf_datasets` are declared as MCP lanes and the dispatcher ignores
the `mcp_server` field. `x_pulse` and `bluesky_jetstream` are `type: cli` lanes whose `command` field
the dispatcher also ignores. `reddit_json` is marked deprecated (its endpoint returns 403); note that
`reddit_failures` still calls that same dead endpoint.

## The reader ladder

`extract_clean_text()` in `extractor.py`. It is not one linear ladder — three branches are tried
first, each returning on its own, and only an ordinary web URL reaches the 7-rung ladder.

**Branches, in order:**

1. **Documents, PDFs and local files** — PDFs try PyMuPDF (default), then Docling, then MarkItDown.
   Word, PowerPoint, Excel, images and audio go through MarkItDown. `file://` and plain local paths
   work. A failed document extraction stops here; it does not fall through to the web.
2. **gitingest** — **GitHub** repo URLs only (not GitLab) get packed into one text file.
3. **Apify actors** — **Instagram and TikTok only**. For those hosts it is exclusive: if Apify fails,
   extraction ends rather than continuing down the ladder.

**Then the web ladder proper, 7 rungs:**

`cloudflare-markdown` → `trafilatura` → `firecrawl` → `crawl4ai` → `scrapling` (stealth browser,
brings in Camoufox) → `crawlee` → `Jina Reader`, and then `agent-browser` (a real browser session)
for tier-3 requests only.

**A rung is attempted only when its precondition holds.** Each rung declares what it needs — an API
key, a helper script, an installed package. When that is missing the rung is skipped with a single
`status=skipped` telemetry row naming the reason, instead of being attempted and failing. Before
this, Jina without a key, Crawl4AI without its helper script and Wayback without keys cost over
3,400 pointless fetch attempts between them. `tools/reader_ladder_check.py` prints which rungs can
actually run on your machine, using the same predicate the ladder uses.

**Every page request passes one door, `fetch_gate()`.** That includes the origin page, PDFs,
`robots.txt`, archive copies, gitingest, Docling, MarkItDown and the tier-3 browser. The door spaces
requests per host *group* — `www.`, `old.`, `m.` and `amp.` collapse into one, so Reddit's aliases
share a single backoff — and refuses while that group is cooling down after a `429` or `403`.
`Retry-After` is honoured; a header in the past never means "retry now".

**A blocked host stays blocked, including through a third party.** Firecrawl calls
`api.firecrawl.dev` and Jina calls `r.jina.ai`, but the page they hand back is the origin's. Both
gate on *both* host groups, so a host that returned `429` is not then fetched from someone else's
address. A cooldown stops the ladder for that URL rather than working around it.

**A fetched page is attached to a URL only when the returned URL matches.** The Firecrawl paths used
to fall back to the first item in a result list, which could file the wrong page under a real
citation.

The `agent-browser` rung is gated twice, because a real browser is the one rung that can take a
machine down. It runs only for a tier-3 request, and only after `l3_guard.preflight()` checks free
RAM, swap pressure, the killswitch and the failure circuit-breaker. A blocked preflight is recorded
and the rung returns nothing; it never falls back to launching the browser anyway.

**Then, if nothing worked:** an open-access **publisher fallback** for paper-like URLs (Crossref and
OpenAlex live here, not as search lanes), and finally the **Wayback Machine** for dead links.

⚠️ The **crawl4ai** rung shells out to a helper script that is **not shipped in this repo**. Set
`RESEARCH_ENGINE_CRAWL4AI_SCRIPT` or that rung is skipped with a logged reason.

**Blocked IPs** — `fetch_proxy.py` can retry through a rotating VPN proxy. If both the VPN and its
backup fail it browses unproxied and labels the result `no_proxy:...` rather than implying a proxy was
in use.

Every rung attempt is logged by `extractor.py` to `research-reader-telemetry.jsonl` in your data
directory. `telemetry_observer.py` is a separate tool that post-processes saved sessions.

## The LLMs it uses

Codex, Claude, Grok and `agy` run as CLIs you are already signed into. The one exception is Mistral,
which is an HTTP API and needs a key file.

| Role | What actually runs |
|---|---|
| Final synthesis (`research`, `deep-research`) | Opus → Codex `gpt-5.5` → Sonnet, in that order |
| Final synthesis (`search`) | Codex → Sonnet |
| Scout (`research`, `deep-research`) | Gemini 3.7 Flash via `agy`. **Optional** — if it fails the run continues without it |
| Worker: Codex | `gpt-5.4-mini` (territories are labelled `codex-5.4`) |
| Worker: Grok | The Grok CLI, invoked as `grok --single`. **Not** Hermes — its xAI auth is dead |
| Worker: Mistral | `mistral-small-latest` over HTTPS with a Bearer key. Needs a key file |

Four things worth knowing:

- **Gemini Pro is never used.** The scout is explicitly restricted to Flash.
- **Territories labelled `haiku` do not run Haiku.** `llm_call.py` supports exactly four backends —
  codex, sonnet, opus, gemini. A `haiku` territory falls through to a generic summariser that uses
  Codex or Sonnet. The label is a leftover.
- **There is no automatic fallback if the Grok CLI fails.** That worker's pass is recorded as an
  error. (A `cursor-agent` fallback is mentioned in a source comment but no code path invokes it.)
- There is a **Gemini daily budget of 300 calls** (`RESEARCH_ENGINE_GEMINI_DAILY_BUDGET`). Past it, or
  when `RESEARCH_ENGINE_UNATTENDED` is set, Gemini calls are swapped to `GPT-OSS 120B (Medium)`.

## Honesty guardrails

The point of the project. Several were added after an audit found the engine was reporting quality
scores that were constants typed into the source — a content-farm blog scored identically to a
manufacturer.

- **A quality, cost or provenance field is measured, or it is absent — and absent never passes a
  gate.** An unmeasured number is `None`, never `0.0`. A FULL answer cannot be constructed without a
  measured confidence, and `QueryCall` raises rather than defaulting when a payload carries no
  `duration_ms`. Session totals are wall clock and summed cost, with no defaults.
- **A sufficiency verdict exists only if a checker produced it.** The stub that always answered
  "sufficient" is deleted, and so is the automatic local-model judge. With no checker the session
  gets `terminal_state="unchecked"` and `verdict=None`, and the evidence gate abstains with
  `gate_reason="checker_unavailable"`. Before this, an unrelated paragraph about bananas passed as
  sufficient support for a PostgreSQL question.
- **Token overlap is not faithfulness.** Word overlap is stored under its own name,
  `lexical_overlap_score`. `crystal_check_passed` and `crystal_check_score` stay `None` until a real
  claim-level judge fills them. Overlap rates "the treatment does not reduce mortality" at 1.0
  against its own opposite.
- **The number and date check runs on every save** and can demote FULL to PARTIAL. Matching is
  whole-token, so `120` no longer counts as support for `1200`, and it reports
  `applicable: false` rather than a score when there is no captured source text to check against.
- **No save path skips the gate.** Judgement happens once in `finalize_session()`, then
  `write_session()` only stores. A storage failure reports itself; it never rewrites the answer.
- **Evidence offsets are real.** Each `EvidenceChunk` carries the `char_offset` and `char_length` of
  the window it came from, indexing the stored full text exactly, rather than a literal `0`.
- **Tests cannot touch production state.** Log and cache locations resolve at write time from
  `RESEARCH_ENGINE_DATA_DIR` and `RESEARCH_ENGINE_CACHE_DIR`, and a package `conftest.py` points both
  at temporary directories. No production code branches on `PYTEST_CURRENT_TEST`.
- **Confidence is measured, not asserted.** Source authority and count feed a weighted score, mapped
  through thresholds in `router_config.yaml`. If that config cannot be read, the engine uses documented
  fail-closed defaults (0.70 / 0.35) and still grades from the measured score, rather than inventing a
  confident result without measuring.
- **A measured zero stays zero.** The weak-sources gate caps confidence at 0.5; it does not raise an
  unmeasured or zero score up to it.
- **Grounding abstains** when its thresholds cannot be loaded, and logs why.
- **Evidence scoring is containment, not Jaccard**, calibrated on real claim and paragraph pairs.
- **Counter-evidence lane.** One Grok worker is pointed at a mutated query specifically to find
  reasons the obvious answer is wrong.
- **Missing data announces itself.** A local lane whose source is absent says "not configured" instead
  of returning zero results.

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
| `grok` CLI | The Grok worker and `grok_x_search` — used in every mode | https://grok.com |
| `codex` CLI | Default LLM backend | OpenAI Codex CLI |
| `claude` CLI | Sonnet and Opus backends | Anthropic Claude CLI |
| `agy` (Gemini CLI wrapper) | The optional scout | Local wrapper; must be on `PATH` |
| Semantic proxy on `localhost:18791` | Proxied search lanes only | Not in this repo |

### Reader tools

Every one of these has a live call site in `extractor.py`. Install only what you need — a missing tool
fails its own rung and the ladder continues. (PDFs are the exception: PyMuPDF, Docling and MarkItDown
are tried in turn, and only if all three are unavailable does the PDF path raise.)

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

Copy `.env.example`, fill in what you need, and point `RESEARCH_ENGINE_ENV_FILE` at it — or export the
variables directly.

The ones most people set:

| Variable | Default | What it does |
|---|---|---|
| `RESEARCH_ENGINE_DATA_DIR` | `~/.research_engine` | Telemetry, logs, session store, memory DB |
| `RESEARCH_ENGINE_CONTACT_EMAIL` | *(unset)* | Your address in outbound `User-Agent` headers. Crossref and the Wayback Machine give better rate limits when you supply one |
| `RESEARCH_ENGINE_ENV_FILE` | *(unset)* | An `.env` file to load API keys from |
| `RESEARCH_ENGINE_MISTRAL_KEYS_FILE` | *(unset)* | Without this the Mistral worker is inert |
| `RESEARCH_ENGINE_CRAWL4AI_SCRIPT` | *(unset)* | Path to your Crawl4AI helper — that rung is dead without it |
| `RESEARCH_ENGINE_GEMINI_DAILY_BUDGET` | `300` | Gemini calls per day before swapping to GPT-OSS |

**Precedence:** if a `RESEARCH_ENGINE_*_BIN` variable is set it wins. Only when it is unset does the
code fall back to `shutil.which()` on your `PATH`. So if the CLIs are on your `PATH` you need not set
anything, and when a required binary is genuinely missing the error names the exact variable to set.

Local-file lanes (Obsidian, a Claude memory glob) and integrations with the author's other tools are
**off unless you set their variable**. See `.env.example` for the full list.

⚠️ A few lane integrations in `router_config.yaml` still assume fixed home-directory locations —
`~/.kaggle/kaggle.json` for `kaggle` and `~/bin/x-pulse-research` for `x_pulse`. Those are not
env-var driven.

## Usage

```bash
python research_cli.py --mode search        "current pricing for X"
python research_cli.py --mode research      "should we choose A or B"
python research_cli.py --mode deep-research "everything known about X"
```

Sessions are saved to `~/.claude/research-sessions/YYYY-MM-DD/<session_id>.json`. Set
`RESEARCH_ENGINE_RESEARCH_SESSIONS_DIR` to put them somewhere else. The directory is created on the
first write, and a storage failure never rewrites the answer — the session is preserved and the
failure is logged separately.

Run the tests from the repo root, not from `tests/` — thirteen test modules live at the top level:

```bash
pytest . -q
ruff check .
```

## Design principles

- **Cheapest that works** — in `search` mode. Free lanes run first and paid search fires only when
  free results are thin. `research` and `deep-research` use the paid proxy as part of their normal
  flow, so this is a `search`-mode property, not a global one.
- **Cite or admit.** A cited source or an honest abstention, never a confident guess.
- **Disagreement is data.** A counter-evidence worker hunts for reasons the obvious answer is wrong.
- **Stop at the first good rung.** No fetching a page five ways when the first way worked.

## Known gaps

Stated plainly rather than discovered later.

- **Crawl4AI needs a helper script that is not in this repo.** Set `RESEARCH_ENGINE_CRAWL4AI_SCRIPT`
  to your own, or that rung is skipped with a logged reason.
- **The cooldown is per process, not shared.** Two workers in separate processes do not honour each
  other's `429` backoff.
- **Gating on the origin host is an optional argument.** `fetch_gate(url, group_url=...)` refuses on
  either host group, but a newly added rung that simply omits `group_url` would gate only the service
  it calls. It is a convention, not something the structure enforces.
- **The listicle penalty never fires.** `rank_candidates` halves the score of a result flagged
  `listicle_flagged`, and nothing in the pipeline currently sets that flag.
- **`test_webread_service.py` exists twice**, at the top level and under `tests/`. In this flattened
  layout pytest refuses to collect one of them; run it on its own, or ignore that path in a full run.
- **The page budgets are unproven.** See the mode table above.
- **Some lanes cannot run at all.** `hf_hub` and `hf_datasets` are MCP lanes the dispatcher ignores;
  `x_pulse` and `bluesky_jetstream` are CLI lanes whose command field is also ignored; `reddit_json`
  is deprecated upstream and `reddit_failures` still uses that same dead endpoint.
- **The `budget_ceiling_usd` values are not enforced.** Nothing in the CLI reads them.
- **`haiku` worker labels are cosmetic** — those territories execute on Codex or Sonnet.
- **The Mistral worker is inert** until you supply a key file, so `research` mode runs 2 of 3 workers
  out of the box.
- **No console entry point.** `pyproject.toml` declares no `[project.scripts]`. The package itself
  imports fine on a minimal install; this is a packaging gap, not an import problem.
- **Apify is exclusive for Instagram and TikTok.** If it fails on those hosts, extraction stops rather
  than falling back to the generic ladder.
- Several MEDIUM and LOW findings from an internal code audit remain open — long functions, duplicate
  helpers, style. None affect correctness.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues go through the Security tab, not a public
issue — see [SECURITY.md](SECURITY.md).

## Changes

[CHANGELOG.md](CHANGELOG.md) — what changed in each release, including what is known to be wrong.

## Licence

GNU AGPL-3.0-or-later. See [LICENSE](LICENSE).

Use it, change it, run it, charge for it. The one condition: if you distribute it, or run a modified
version as a network service other people can reach, you have to make your source available under the
same licence. You cannot take this, close it, and ship it as a private product.

If that does not suit your situation, ask.
