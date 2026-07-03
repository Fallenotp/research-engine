# webread — how any agent reads a web page

One command. Any agent (Claude, Codex, Gemini) uses the same path to read any website.

## Use it

```
webread <url>                 # read a page, get clean text back
webread <url> --ask "<task>"  # interactive read (login/click) — needs the heavy L3b path
```

Or hit the warm service directly:

```
GET http://127.0.0.1:8077/read?url=<url>&max_layer=2          # cheap→medium read
GET http://127.0.0.1:8077/read?url=<url>&ask=<task>&max_layer=3   # allow the L3 browser
```

You get back JSON: `{ text, receipt }`. The receipt says which rung answered (`method`), the `layer` (1/2/3), timing, and any `error`.

## The ladder (cheap → expensive)

- **L1 / L2** — fast HTTP + light scrapers (trafilatura, crawl4ai, scrapling, firecrawl). Most pages. No browser.
- **L3a** — a real browser (Steel) renders the page and returns its text/markdown. **No AI model.** Use for hard/JS pages that the cheap rungs return empty. Proven live on react.dev, BBC, Wikipedia, Hacker News.
- **L3b** — the browser *plus* a local AI model (Stagehand + MLX) for pages needing interaction (login, clicking). Heaviest. Currently deferred — needs free RAM.

`max_layer` is a **hard cap**: pass `max_layer=3` to allow the browser; `--ask` requires layer 3.

## The safety guard (why L3 sometimes refuses)

Before the browser/model start, a preflight checks the Mac has room. If not, L3 **refuses** and returns a `L3_BLOCKED_*` receipt; you still get the best lower-rung text. Blocks: low RAM (<5 GB free), memory-pressure not normal, swap over cap, low disk, the kill switch, or the circuit breaker (3 fails → 1 h off).

- Force L3 off entirely: `WEBREAD_L3_KILL=1` or `touch ~/.cache/webread/l3.off`.
- An idle janitor frees the model/sidecar after 5 min unused (keeps the browser warm).
- Every L3 attempt is logged to `~/.cache/webread/l3.log`.

## Live e2e (2026-06-22, L3a)

| URL | chars | note |
|---|---|---|
| example.com | 165 | baseline |
| news.ycombinator.com | 6,589 | |
| en.wikipedia.org/wiki/Web_scraping | 40,217 | large page |
| react.dev | 4,764 | **JS app — proves JS rendering** |
| bbc.com/news | 2,524 | |

5/5 returned real content through the real Steel browser.
