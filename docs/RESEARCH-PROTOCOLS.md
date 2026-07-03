# Research Protocols — Master Reference

**The four tools: `buzz`, `search`, `research`, `deep-research`.**
Written 2026-05-23. Source of truth: the four `SKILL.md` files, plus the engine at
`/Users/cleo/lattice/research_engine/` (`router_config.yaml`, `schema.py`, `dispatcher.py`).

This document has two layers:
- **Plain layer** (for Ian): what each tool is, when to use which, what it costs.
- **Operating layer** (for the LLMs): connections, setups, schemas, briefs, and the exact
  rules for choosing one API or one tool over another.

---

## PART A — PLAIN OVERVIEW (read this first)

### The four tools are one system, on one ladder

They are not four separate apps. They are one research system that gets more thorough as you
climb. Same idea as asking a question of: a quick glance, a sharp assistant, a small team, or
a full research department.

| Tool | What it is | Helpers | Typical cost | Use it for |
|---|---|---|---|---|
| **buzz** | Community pulse sweep | engine, no LLM team | free | "What are people saying / is X still working / recent chatter" |
| **search** | One quick, sourced fact | 1 | near-free | "What year / latest version / is Y maintained" |
| **research** | A decision between options | scout + 3 | a few cents | "Should I use X or Y", "compare A vs B" |
| **deep-research** | Exhaustive scour | scout + 5 + self-loop | $0.30–$1.50 | "Cover everything", new territory, irreversible call |

**Rule of thumb:** buzz for mood, search for a fact, research for a decision, deep-research
for "I can't afford to miss anything."

### Two things they all share

1. **One shared brain** (the engine at `/Users/cleo/lattice/research_engine`). It picks which
   search service to use, reads web pages cleanly, and runs the honesty check that every answer
   is actually backed by real sources.
2. **buzz** is the "what's the community saying" step *inside* search, research, and
   deep-research, as well as a standalone tool.

### What has to be running

- **SearXNG** (your private free search) at `localhost:8888`.
- **A paid-search proxy** at `localhost:18791` that fronts the paid services and rotates keys.
- **Gemini and Codex** command tools, used as the planner ("scout") and as extra helpers.

The brains live **outside** any plugin. A plugin would only be thin buttons pointing at them.

---

## PART B — OPERATING LAYER (for the LLMs)

### B0. The shared engine — what it guarantees

`import sys; sys.path.insert(0, "/Users/cleo/lattice"); from research_engine import ...`

The engine enforces, so individual protocols do not re-implement:

- **Provider routing** — rule-based first (deterministic, free), Haiku fallback only when no
  rule matches (`router.py` + `router_config.yaml`).
- **Clean extraction** — `extract_clean_text(url, seen_urls_path=, tier=)` cascades
  Jina → trafilatura → Crawl4AI → curl+readability → Wayback snapshot; PDFs route to
  Docling/pymupdf. URL dedup via the shared `seen_urls_path`.
- **Sufficiency gate** — judges the *original query* against *raw source text only* (never a
  self-written answer), after a per-item relevance prefilter. Verdicts: `sufficient` /
  `partial` / `insufficient`. Up to 2 reformulate-and-retry passes when a retriever is wired.
  Checker chain: Gemini Flash → Claude Haiku fallback; if both fail, fails closed as
  `insufficient`.
- **Schema discipline** — every claim must be grounded by an `EvidenceChunk` that points to a
  real `SourceRecord` with a URL. Orphan claims are physically unstorable (see B6).
- **Reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, blend alpha 0.7. **Locked**
  2026-04-26; do not swap without a new bake-off.
- **Abstention fail-safe** — if fewer than 25% of candidate chunks clear the calibrated gate,
  or fewer than 2 distinct sources pass, the protocol **must abstain** rather than synthesize a
  weak answer.

### B1. CONNECTIONS — the full lane registry

Every "lane" is one place the engine can fetch from. Source: `router_config.yaml`. Costs are
per-call estimates. "Setup" = what must exist for the lane to work.

#### Free web + local (use first, always)

| Lane | Endpoint / command | Auth / setup | Cost |
|---|---|---|---|
| `searxng_general` | `http://localhost:8888/search?q={q}&format=json` | none; SearXNG must be up | $0 |
| `searxng_forums` | same + `&engines=reddit,hackernews,lemmy,stackoverflow,discourse` | none | $0 |
| `mentor_memory` | `lattice/data/agent_state/memory.db` (local) | none; never call retired Brain API :18790 | $0 |
| `claude_memory_files` | `~/.claude/projects/-Users-cleo/memory/*.md` | none | $0 |
| `obsidian` | `~/ObsidianVault/**/*.md` | none | $0 |

#### Paid search (all via the one proxy at `localhost:18791`, key rotation handled there)

| Lane | Provider | When to pick it | Sustainability (as recorded — verify before bulk use) |
|---|---|---|---|
| `tavily_direct` | Tavily | **Default paid.** Quick fact, news, general web | Monthly refill (~6k/mo). Use freely. |
| `linkup_direct` | Linkup | "What did X say", official numbers, citation precision | Likely monthly (~8k). Use moderately. |
| `exa_direct` | Exa | "Find similar / related / like X", semantic + academic neighbors | One-time pool, ~766 calls left, 1 of 4 keys live. Sparingly. |
| `youcom_direct` | You.com | Comprehensive multi-aspect "deep dive", social-aware | Config: 4 keys alive, under-used. Skills: RESERVE finite pool. Treat as reserve until reconciled. |
| `firecrawl_direct` | Firecrawl | Extract clean text from a JS-heavy URL | Monthly, ~500 credits/key, 3 keys. Use freely (1 credit/page). |
| `paid_proxy` | rotates Linkup→Tavily→You.com→Exa | Fallback only when no specific provider chosen | Mixed. |

> **Hard rule:** NEVER call `mcp__exa__*` directly — a hook blocks it. Use `provider:"exa"`
> through the proxy.

#### Community / real-time (free)

Reddit MUST be read via RSS (`search.rss`, `/r/<sub>/.rss`, or `<thread>/.rss`) with
User-Agent `IanResearch/1.0 (123icpe@gmail.com)`. The JSON API and HTML scraping are
blocked (403) — do not use them.

| Lane | Endpoint / command | Setup |
|---|---|---|
| `buzz` (the engine) | `python ~/buzz/buzz.py "<q>" --emit=json [--quick|--deep]` | Reddit/HN/Polymarket/GitHub free; X+YouTube need cookies/`yt-dlp` |
| `reddit_rss` | `reddit.com/search.rss?q={q}&sort=relevance&limit=25` | UA header `IanResearch/1.0 (123icpe@gmail.com)` |
| `reddit_json` | `reddit.com/search.json?q={q}&t=week` | deprecated 2026-06-19; blocked with 403 |
| `reddit_failures` | `…?q={q}+failed+OR+broken+OR+banned&t=month` | same UA — counter-evidence lane |
| `x_pulse` | `~/bin/x-pulse-research "{q}"` | Gemini-summarised, ~last hour, ~$0.01 |
| `bluesky_jetstream` | `wss://jetstream2.us-east.bsky.network/subscribe` | 60s sample window |
| `hn_algolia` | `hn.algolia.com/api/v1/search?query={q}&hitsPerPage=30` | none |
| (X verified raw) | `~/bin/_x_fetch.py "<q>"` | run separately from buzz for the X four-source check |

#### Academic (free)

`arxiv`, `semantic_scholar`, `core`, `papers_with_code` — all keyless or optional-key. PDFs via
`extract_clean_text` (Docling/pymupdf auto).

#### Government / legal / finance (free, mostly keyless or `api.data.gov` key)

`courtlistener`, `nasa_techtransfer`, `congress_gov` (`DATA_GOV_API_KEY`), `sec_edgar`,
`sec_gov`, `fec_gov` (`FEC_API_KEY`), `openstates` (`OPENSTATES_API_KEY`), `govinfo_gov`
(`DATA_GOV_API_KEY`), `fda_gov` (`FDA_API_KEY`), `nih_gov`, `pubmed`.

#### Code (free)

`github_code` (`GITHUB_TOKEN`, strict 1 qps), `sourcegraph` (use when GitHub rate-limits),
`stack_exchange`.

#### AI hub / data

`hf_hub` + `hf_datasets` (HuggingFace MCP, already configured), `kaggle` (`~/.kaggle/kaggle.json`).

#### The scout (planner) lane

`gemini_pro_scout` — Gemini 3 Flash through `agy-cli-1`, OAuth subscription, $0.
**AGY RULE: never call the retired `gemini` CLI. Pass the brief as the `--print` argument,
never pass `--model`/`-m`, and use `agy-cli-1`'s configured default only.** If the default
fails, HALT and ask Ian — do not downgrade or substitute.

---

### B2. WHEN ONE API VS ANOTHER — the routing rules

The engine matches the query against these rules **top-down, first match wins** (most specific
first). Each rule names the lanes to use and the worker LLM. If nothing matches, the
`general_web` fallback runs and Haiku decides.

| Rule (query shape) | Lanes it fires | Worker | Notes |
|---|---|---|---|
| gov tech transfer (nasa, patent, spinoff) | nasa_techtransfer, github_code, searxng | haiku | |
| federal court ruling (scotus, opinion) | courtlistener, congress_gov, searxng, paid_proxy | haiku | requires Tier-1 |
| sec filing (10-k, edgar) | sec_edgar, sec_gov, searxng | haiku | requires Tier-1 |
| campaign finance (fec, super pac) | fec_gov, openstates, congress_gov, searxng | haiku | |
| legislation (bill, congress, h.r.) | congress_gov, openstates, govinfo_gov, searxng | haiku | |
| state legislation | openstates, searxng, paid_proxy | haiku | |
| academic paper (arxiv, doi, et al) | arxiv, semantic_scholar, core, papers_with_code, searxng, exa | haiku | requires Tier-1 |
| medical research (clinical trial, pubmed) | pubmed, semantic_scholar, fda_gov, nih_gov, searxng, exa, linkup | haiku | requires Tier-1 |
| ml sota (benchmark, leaderboard) | papers_with_code, hn_algolia, github_code, hf_hub, semantic_scholar | haiku | |
| code pattern (how to, sdk, api) | github_code, sourcegraph, stack_exchange, hn_algolia, searxng, firecrawl | **codex 5.4** | |
| error diagnosis (traceback, bug) | stack_exchange, github_code, sourcegraph, searxng | **codex 5.4** | |
| breaking news (today, just announced) | x_pulse, bluesky, hn_algolia, searxng, tavily, firecrawl, paid_proxy | haiku | max age 24h |
| social sentiment (reddit, what people think) | reddit_rss, x_pulse, bluesky, hn_algolia, searxng_forums, youcom | haiku | |
| counter-evidence (failures, banned, debunked) | searxng_forums, reddit_failures, hn_algolia, paid_proxy, courtlistener | grok | standing counter-evidence lane |
| dataset lookup (kaggle, parquet) | kaggle, hf_datasets, searxng | haiku | |
| model lookup (gguf, checkpoint, mlx) | hf_hub, papers_with_code, github_code, searxng | haiku | |
| memory recall (did we discuss, what do I know) | mentor_memory, claude_memory_files, obsidian | haiku | skips web |
| **general (anything else)** | searxng, linkup, firecrawl, paid_proxy | haiku | fallback |

**Free-first sustainability order (most → least preferred):**
SearXNG + free APIs (unlimited) → Tavily / Firecrawl (monthly, use freely) → Linkup
(moderate) → Exa (semantic-only, finite) → You.com (reserve).

---

### B3. WHEN ONE TOOL VS ANOTHER — pick the protocol

```
Need recent community mood only?           → buzz
One fact, want it fast and sourced?         → search
A decision between options, need a "why"
   and the counter-argument?                → research
Exhaustive, new territory, can't miss
   anything, irreversible call?             → deep-research
Cross-LLM debate of an idea?               → think-tank   (sibling, not covered here)
Check an existing answer for gaps?         → research-check (gap detector; deep-research
                                              calls it on its iteration loop)
Just recall what we already know?          → recall        (memory only, skips web)
```

These five live in the `Protocol` enum: `/search`, `/research`, `/deep-research`,
`/research-check`, `/recall`.

---

### B4. THE FOUR PROTOCOLS — one breakdown each

#### B4.1 — buzz

- **Purpose:** recent multi-source community signal. Engine: `~/buzz/buzz.py` (v3.0.0, MIT
  extraction of `mvanhorn/last30days-skill`).
- **Run:** `python ~/buzz/buzz.py "<topic>" --emit=json [--quick|--deep] [--competitors]
  [--github-user=<h>] [--watchlist ...]`. Import: `from buzz import gather_signals, ...`.
- **Sources (default free):** Reddit, Hacker News, Polymarket, GitHub (if `gh`/`GITHUB_TOKEN`),
  X + YouTube (if cookies/`yt-dlp`). Paid-optional, off by default: ScrapeCreators, Brave,
  Perplexity Sonar.
- **Research-engine handoff:** Buzz remains the community-signal feeder. The standing Grok
  worker now covers the research counter-evidence/real-time lane through Hermes:
  `hermes -m grok-4.20-0309-reasoning -z "$(cat <brief_path>)"`.
- **Four-source community check (required when claiming community coverage):** BlueSky, X,
  Reddit, YouTube must all return non-empty. Use
  `buzz.py "<q>" --emit=json --search reddit,bluesky,youtube` plus `_x_fetch.py "<q>"`.
- **Role in the system:** this IS "Step 1 — community input" inside search/research/deep-research.
- **Output:** ranked, deduped JSON of signals. No source-grounding gate (it is signal, not a
  verified answer). Write the brief in plain English; no badge line, no laws appendix.

#### B4.2 — search

- **Trigger:** a fast fact ("what year", "latest version", "is Y maintained", "what did X say").
- **Team:** orchestrator (Sonnet) classifies, then spawns **1 Haiku worker**. Orchestrator does
  not search itself. Grok is not part of this single-worker quick path unless the dispatcher is
  explicitly given a counter-evidence territory.
- **Flow (worker stops early at any clean answer):**
  1. Community (buzz) — for recent-discussion questions.
  2. Confirm + currency (SearXNG general/forums).
  3. ONE specialized paid API (chosen by orchestrator per B2).
  4. Extract clean text (Firecrawl/engine) on top 1–2 URLs.
- **Cost:** near-free; one specialized call at most.
- **Output:** one-paragraph sourced answer; `answer_kind` = full / partial / abstain.

#### B4.3 — research

- **Trigger:** a decision with options ("X or Y", "compare A vs B"). Not exhaustive scour, not
  cross-LLM debate.
- **Team:** Gemini **scout** runs FIRST (plans territories; HALT on `GeminiProScoutError` — do
  not proceed scout-less). Then **3 workers**: Agent A keyword breadth, Agent B semantic depth,
  Agent C counter-evidence (standing Grok API wrapper, live web/X + reasoning).
- **Recursion tiers** (`select_recursion_tier`): **deep** if ≥3 distinct LLMs answered (depth 2,
  3 sub-Qs/node, 12-leaf cap, $0.50); **one_llm** otherwise (flat, 4-leaf cap, $0.15).
- **Each worker** runs the same 4-step flow as search (community → SearXNG → assigned API →
  extract), plus counter-evidence searches the negative slice.
- **Judge + verbatim check** before synthesis. Surfaces disagreements (does not auto-resolve).
- **Output:** Answer (labelled FULL/PARTIAL/ABSTAIN) · Why (3 cited bullets) · What could be
  wrong · Disagreements · Decision point · Verbatim check · Session file · Worker/API breakdown.

#### B4.4 — deep-research

- **Trigger:** full coverage on new territory / irreversible decision. **Cost $0.30–$1.50.**
- **Team:** scout FIRST (HALT on error), then **5 workers across different AIs**:
  A keyword (Codex 5.4, code lanes), B semantic (Haiku, academic + Exa & Linkup), C
  counter-evidence (Grok API wrapper, live web/X + reasoning), D domain/social (Gemini Flash
  + You.com), E cross-model verifier (Sonnet, reads others' output, 3-lens check on top claims).
- **Recursion tiers:** deep (depth 3, 4 sub-Qs, 25-leaf cap, $1.50) / one_llm (depth 1, 6-leaf,
  $0.30).
- **Provider diversity is mandatory:** each run must hit ≥2 direct paid lanes in parallel, not
  just SearXNG.
- **Auto-iteration:** `detect_gaps` → `decide_iteration`; loops on HIGH/MEDIUM gaps, **max 3
  loops, 2 lanes/iteration, $1.50 cap**.
- **Output:** as research, plus Neural triangulation (Agent E) · What we don't know ·
  Single-source flags · Search-provider hit counts.

---

### B5. WHICH WORKER LLM — dispatch routing

The dispatcher (`dispatch()`) chooses the worker model; protocols must not hardcode Haiku.
From `worker_routing` in config (Model Law):

| Job | Model |
|---|---|
| search / read / fetch | **haiku** (mandatory) |
| short plain-English synthesis | sonnet |
| dense synthesis, find holes, redesign | opus |
| 1M-context long document pile | codex 5.3 |
| code-oriented research | codex 5.4 (default code worker) |
| genuinely hard architecture | codex 5.5 |
| trivial mechanical | codex-mini |
| counter-evidence / real-time | grok |

How a worker is actually spawned:
- `anthropic_subagent` → Task tool, `subagent_type="general-purpose"`, `model="haiku"`.
- `codex_cli` → `codex exec -m gpt-5.4-mini --skip-git-repo-check -s danger-full-access < brief > out`.
- `agy_cli` → `agy-cli-1 --dangerously-skip-permissions --print "$(cat <brief_path>)" > <output_path> 2>&1` (no `-m` / `--model`; the brief is the `--print` argument).
- `grok_cli` → `hermes -m grok-4.20-0309-reasoning -z "$(cat <brief_path>)" > <output_path> 2>&1`.
- `sonnet_inline` / `opus_subagent` → orchestrator handles / Task tool.

---

### B6. SCHEMAS — the data contract (`schema.py`)

Every protocol produces one **`ResearchSession`** (persisted to
`~/.claude/research-sessions/{date}/{id}.json`). The schema is the bouncer: it refuses to
construct objects that lack evidence, which makes hallucinated results physically unstorable.

**Enums (locked vocabularies):**
- `Protocol`: /search, /research, /deep-research, /research-check, /recall
- `SourceTier`: T1 (official/peer-reviewed/primary), T2 (reputable press/forum), T3
  (anonymous/single/undated)
- `AnswerKind`: full / partial / abstain · `FinalStatus`: complete / weak_sources /
  insufficient_evidence / in_progress / failed
- `AgentRole`: keyword / semantic / counter_evidence / domain_specialist / cross_model_verifier
- `WorkerModel`: haiku / sonnet / opus / codex-mini/5.3/5.4/5.5 / gemini-pro/flash / grok
- `GapSeverity`: high / medium / low

**Key records:**
- `SourceRecord` (frozen): url, domain, title, `content_hash` (sha256), `extraction_method`,
  `raw_text_path`, `tier`, `topic_authority_score` (0–1), `counter_evidence_flagged`,
  optional `archive_url`. Cannot exist without an extraction method and a content hash.
- `EvidenceChunk` (frozen): FK `source_id` → SourceRecord, `paragraph_text`, `rerank_score`,
  `supports_claim`, `crystal_check_passed` + score (faithfulness). No chunk without a real source.
- `Territory`: a non-overlapping search area with `assigned_agent_role`, `assigned_lanes`,
  `assigned_worker_model`.
- `QueryCall`: one lane invocation (lane id, worker_model, duration, result_count, cost, error).
- `Disagreement`: two agents conflict on one fact — surfaced, never auto-resolved.
- `Gap`: detected by /research-check (severity + reason + recommended lane).
- `CrossModelVerification`: deep-research Agent E's analytical/creative/skeptical 3-lens check.
- `GeminiProRunRecord`: records each scout/synthesis run for the persistence interlock.

**Invariants the schema enforces (cannot be saved if violated):**
- Every `EvidenceChunk.source_id` must match a real `SourceRecord` — no orphan claims.
- Disagreements must reference real chunk ids.
- `final_status=COMPLETE` requires a non-empty answer and `answer_kind != abstain`.
- `answer_kind=full` ⇒ COMPLETE + non-empty answer. `partial` ⇒ answer + confidence +
  open_questions. `abstain` ⇒ weak/insufficient/failed status + open_questions with concrete
  next steps (no synthesized answer allowed).
- Graduated-answer thresholds: full ≥0.70 confidence, partial ≥0.35, abstain below 0.35.

**Optional integration contexts:** `CTContext` (Consequence-Tracker: 5-state resolution,
Brier, accountability baseline) and `MentorContext` (MENTOR 4-rule ingest gate). Present only
when the session feeds those systems.

---

### B7. LLM OPERATING INSTRUCTIONS (condensed worker brief)

Every worker brief MUST contain:

```
You are Agent <X> for territory: <description>.
Execute these 4 steps IN ORDER. Stop early at any step that gives a clean answer.

SHARED SEEN-URLS PATH: /tmp/<protocol>-briefs-{slug}/shared-seen-urls.txt
SPECIALIZED API ASSIGNED: <exa|linkup|youcom|tavily|firecrawl|none>
PARENT SUMMARY: <scout or parent-node summary — do not re-cover this ground>

Step 1 — COMMUNITY (free): buzz; required four-source check (BlueSky, X, Reddit, YouTube).
Step 2 — CONFIRM + CURRENCY (free): SearXNG general/forums; academic APIs if academic.
Step 3 — SPECIALIZED API (only the one assigned): POST localhost:18791/search provider=<...>.
         Counter-evidence role also searches the negative slice (failed/broken/banned).
Step 4 — EXTRACT: extract_clean_text(url, seen_urls_path=, tier=) on top 2–3 URLs.

Tier guidance: T1 if authority_score ≥0.85, T2 if 0.5–0.85, T3 if <0.5.
Use compact_search_results() to strip junk before reading.
Append to session.queries_run / .sources / .evidence_chunks. Mirror notes to
/tmp/<protocol>-{slug}-agent-<X>.md.
```

**Hard rules (all protocols):**
- Free signals (community + SearXNG) FIRST; paid API only at Step 3, only if needed.
- One specialized API per worker — the orchestrator chose it; don't hit multiple.
- Never `mcp__exa__*` directly. Always `extract_clean_text()` for fetching.
- Scout is mandatory for research/deep-research; HALT on `GeminiProScoutError`.
- Briefs written BEFORE workers spawn. Sessions saved with `save_session()`.
- Single-source = untrustworthy until confirmed. Surface disagreements; don't auto-resolve.
- Source requirement: ≥1 Tier-1 OR ≥2 Tier-2 per claim; never Tier-3 alone.

---

### B8. FOR HERMES / OTHER AGENTS — what "access" requires

Hermes wants its own research-protocol access. There are two honest paths:

**Path 1 — thin wrappers (like transcribe).** Hermes skills that shell out to the engine. Works
for `buzz` immediately (`buzz.py` is a clean CLI). For search/research/deep-research it is
harder, because the orchestrator is itself an LLM that spawns sub-workers — Hermes would have to
play orchestrator and spawn its own workers.

**Path 2 — Hermes follows the SKILL instructions directly.** Drop adapted SKILL.md copies into
Hermes' skill tree (`~/.hermes/hermes-agent/skills/research/<name>/`), pointed at this engine,
and let Hermes act as the orchestrator following B4 + B7.

**Either way, Hermes still needs the services up** (SearXNG :8888, proxy :18791, Gemini/Codex
CLIs) and the same model-routing + hard rules. The engine is the shared brain; agents are
interchangeable orchestrators in front of it.

**Cross-agent status as of 2026-05-23:** Claude = full. Codex = has `buzz` + `search-order`.
Gemini = neither. Hermes = a different "research" folder (arxiv/polymarket), not this engine.

---

### Verification / open items
- You.com sustainability conflicts between skills ("RESERVE finite") and config ("4 keys alive,
  under-used"). Reconcile before relying on it.
- Scout model id is intentionally unspecified here (CLI invoked with no `-m`); config
  `model_candidates` is for health-check logging only.
- API key/quota counts are as-recorded snapshots — verify live before bulk paid runs.
