# Research Engine — Fetch & Search Decision Guide

> The guardrail the orchestrator and workers follow to pick the RIGHT tool for any
> site or situation. Built 2026-06-09 from Haiku worker profiles that read each
> tool's official GitHub/docs. Grounding files: `/tmp/tool-guide-cluster{1..5}-*.md`.
>
> **This is meant to be wired into the `/search`, `/research`, and `/deep-research`
> skill briefs** so agents are actually told the ladder (the current gap).

---

## The one rule: two stages, never confuse them

1. **SEARCH** = find which URLs to look at. (SearXNG, Linkup, Exa, You.com, Tavily; Firecrawl can also search.)
2. **READ** = pull clean text out of a URL you already have. (Cloudflare-markdown, Jina, Trafilatura, Crawl4AI, Firecrawl, Scrapling, PDF/repo/archive readers.)

Scrapling, Apify, browser engines, and the proxy layer are all **READ**. Cheapest-that-works always wins — escalate only when the cheaper tool returns nothing usable.

---

## A. THE READING LADDER (what to use for a given page)

The engine's `extract_clean_text` walks this top-to-bottom and stops at the first that returns good text. **Branch by what the page IS first, then escalate by how hard it is.**

### Step 0 — Branch by content type
| The URL is… | Use | Why |
|---|---|---|
| a **PDF, paper, or office doc** | **Docling** (complex layout, tables, formulas, papers) or **PyMuPDF** (fast bulk, plain text) | purpose-built doc parsers; don't send these to web readers |
| a **GitHub repo** | **gitingest** | turns the whole repo into LLM-readable text in one shot |
| a normal **web page** | continue down the ladder ↓ | |

### Steps 1-5 — Escalate by difficulty (web pages)
| Step | Tool | Fires when | Cost | Notes |
|---|---|---|---|---|
| 1 | **Cloudflare-markdown preflight** | site already serves a markdown view | free | cheapest possible |
| 2 | **Trafilatura** (+ readability/curl) | static HTML, no JS needed | **free, local** | fast, batch-friendly; **fails on JS/SPA** |
| 3 | **Jina Reader** (r.jina.ai) | page needs JS rendering | cheap API (rate-limited) | renders JS via Puppeteer; handles PDFs/office too |
| 4 | **Firecrawl** | JS-heavy + you want clean markdown back | **1,000 pages/mo free** | best free managed tier; markdown-native |
| 5 | **Crawl4AI** | need a real local browser (Playwright) | free (local compute) | most production-ready local browser reader |
| 6 | **Crawlee HttpCrawler** **through the shared proxy/session pool** | HTTP page is blocked or flaky but does not need browser stealth | free engine + proxy cost | cheaper than browser stealth; rotates proxy/session without installing Playwright browsers |
| 7 | **Scrapling StealthyFetcher (+ Camoufox)** **through a proxy** | site is actively **bot-blocking** (Cloudflare Turnstile, Akamai) | free engine + proxy cost | **the ONLY tool that explicitly defeats Turnstile**; heavy/slow — last resort |
| 8 | **Wayback Machine** | live page is dead or still blocked | free | archived snapshot; no JS, may be stale |

**Why this order:** Steps 1-2 are free and instant. 3-4 cost a little but are cheap/free-tier. 5 spends local compute. 6 (Scrapling) is the heavy artillery — slow, resource-hungry, and the only one that beats hard anti-bot — so it fires only when everything cheaper failed. Camoufox does engine-level fingerprint masking but the maintainers admit it "doesn't always succeed," and it has **no documented WAF bypass on its own** — which is exactly why it's paired with Scrapling + a residential proxy, not used alone.

---

## B. THE PROXY / IP LAYER (orthogonal — protects your home IP)

Applies to the browser/stealth steps (5-6). The home IP is never used for a bot-protected target.

| Decision | Rule |
|---|---|
| **Which backend?** | Config switch: `apify` (residential, 5 shared free accounts) · `own` (your proxies/Tor/ScraperAPI) · `none` (direct — explicit opt-out). One setting, not a per-request choice. |
| **Sticky or rotate?** | **Sticky session** for multi-step flows (login, cart, paginated journey). **Rotate** for independent one-off public pages. |
| **Budget guard** | Apify: rotate the 5 tokens, poll `GET /v2/users/me`, stop each before the ~$5 cap, fall back to `own`/`none` when all are exhausted. |
| **Politeness** | One shared rate-limit + robots.txt cache across all accounts so five accounts can't gang up on one site. |

**Backend cost notes:** Apify residential = traffic-based, Camoufox actor is "resource heavy." ScraperAPI = cheap base credit but multipliers stack fast (JS +10, premium +10, anti-bot +10 → ~31 credits/request). Firecrawl's flat 1-credit-per-page is the most predictable managed option.

---

## C. THE SEARCH LANES (which provider for which query)

Free signals first (SearXNG + community Buzz/Reddit/X), then ONE paid lane per territory only if needed.

Reddit MUST be read via RSS (`search.rss`, `/r/<sub>/.rss`, or `<thread>/.rss`) with
User-Agent `IanResearch/1.0 (123icpe@gmail.com)`. The JSON API and HTML scraping are
blocked (403) — do not use them.

| Query shape | Provider | Why |
|---|---|---|
| general / breaking news / default | **Tavily** | cheap, 1,000 free credits/mo, toolkit (search+extract+crawl) |
| "find similar / related / conceptually like X" | **Exa** | the only true semantic/neural search; 1,000 free/mo |
| "what did X say" / verified citation / official numbers | **Linkup** | #1 on SimpleQA accuracy; returns inline citations |
| comprehensive multi-aspect "deep dive" | **You.com Research** | #1 on DeepSearchQA — but **RESERVE**, expensive ($50-300/1k) |
| privacy / free first-pass keyword sweep | **SearXNG** | free, self-hosted, multi-engine; keyword only (no semantics) |
| "extract clean content from this URL" | **Firecrawl** | JS render + clean markdown (also lives in the read ladder, step 4) |

**Sustainability:** SearXNG + community = unlimited. Tavily/Firecrawl = monthly refill, use freely. Linkup = moderate. Exa = small pool, semantic discovery only. You.com = reserve for irreplaceable deep-synthesis cases.

---

## D. Quick "any situation" cheat answers

- **Plain news article** → Trafilatura; if it's JS-heavy → Jina or Firecrawl.
- **Cloudflare/"are you human" wall** → Scrapling StealthyFetcher + residential proxy (sticky if login).
- **PDF research paper** → Docling (or PyMuPDF for speed/bulk).
- **GitHub repo** → gitingest.
- **Page is 404/dead/taken down** → Wayback.
- **Need to FIND sources, not read one** → SearXNG + community first, then Exa (semantic) / Linkup (citations) / Tavily (news).
- **Multi-step login-gated flow** → sticky proxy session + Scrapling.
- **Budget nearly gone on Apify** → controller falls back to `own` proxies or `none`; never silently overspend.

---

## E. What still needs to happen (so this guide is enforced, not just written)

1. **Add the read-side tools that aren't in the ladder yet** — Firecrawl and Scrapling must be wired into `extract_clean_text` (see the Scrapling plan). Today Firecrawl only exists on the search side.
2. **Paste sections A-D into the skill briefs** (`/search`, `/research`, `/deep-research`) so the orchestrator and workers actually receive these rules. The skill files are passphrase-gated, so a code worker makes that edit, not Claude.
