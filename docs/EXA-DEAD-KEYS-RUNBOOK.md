# Exa Dead-Keys Runbook

**As of 2026-05-10:** 3 of Ian's 4 Exa accounts are out of credits. The keys still exist (commented out in `.env`) but were removed from the proxy's `KEY_SEQUENCE` so they don't waste rotation slots.

## What's drained

| Env var | Account | Key (last 12) | Status |
|---|---|---|---|
| `EXA_API_KEY` | ianabugayda | `...6bf65ae920e2` | 402 NO_MORE_CREDITS |
| `EXA_API_KEY_2` | (unknown) | `...1ee698bd33d5` | 402 NO_MORE_CREDITS |
| `EXA_API_KEY_3` | fallenotp | `...0410a9ba5b47` | 402 NO_MORE_CREDITS |
| `EXA_API_KEY_4` | (only alive) | `...62eb4de1b53a` | active in queue (~766 calls left at last check) |

## Why they're drained

Exa's free tier appears to be a **one-time $20 signup credit** (~2,857 searches at $7/1k), NOT a monthly auto-refill. Once the $20 is gone, the account is billed monthly — but these accounts have no payment method on file, so every call returns 402.

This was verified on 2026-05-10 by:
1. Direct API hit returning `{"error":"You have exceeded your credits limit. Please top up to keep using Exa at dashboard.exa.ai","tag":"NO_MORE_CREDITS"}`
2. Logging into 3 of Ian's Exa account dashboards — 2 showed $0 balance, 1 had an unpaid invoice for $0.73
3. Haiku research run found no documented monthly-refill policy on Exa's site

## How to revive a key (if Ian ever tops up)

For each account Ian wants to revive:

1. Log into `dashboard.exa.ai` with that account's email
2. Click **Billing → Top up balance** → add at least $5 (or whatever)
3. (Optional) Add a payment method so it doesn't drain again
4. Test the key directly:
   ```bash
   /opt/homebrew/bin/python3.11 -c "
   import http.client, json, ssl
   key = 'PASTE_KEY_HERE'
   ctx = ssl.create_default_context()
   conn = http.client.HTTPSConnection('api.exa.ai', timeout=20, context=ctx)
   conn.request('POST', '/search',
                body=json.dumps({'query':'test','numResults':1}),
                headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
   r = conn.getresponse(); print('HTTP', r.status, r.read()[:120])
   "
   ```
   If it prints `HTTP 200`, the key is alive.

5. **Uncomment the env line** in `/Users/cleo/consequence-tracker/.env` (lines 10-12).

6. **Add the key back to the proxy `KEY_SEQUENCE`** in `/Users/cleo/semantic-search-proxy.py`. Insert in the appropriate slot (interleave Linkup + Exa per the comment block):
   ```python
   ("EXA_API_KEY",   "exa", "exa_1"),   # add at the right position to interleave
   ```

7. **Restart the proxy:**
   ```bash
   launchctl kickstart -k gui/$UID/com.semantic-search-proxy
   ```

8. **Verify** via `/status`:
   ```bash
   curl -s http://localhost:18791/status | python3 -m json.tool | head -25
   ```

## How the daily key-health check finds revived keys automatically

The launchd job at `~/Library/LaunchAgents/com.cleo.daily-key-health.plist` runs daily at 04:30. The script `/Users/cleo/lattice/research_engine/scripts/daily_key_health.py` tests every key in `.env` (commented or not — depends on its parsing).

**Currently the daily check ONLY tests keys that are uncommented in `.env`.** So commented-out keys won't be re-tested automatically. If you want auto-revival to fire when Ian tops up, leave the keys uncommented and let the proxy mark them dead — the daily script will then notice when they revive and clear the dead flag in `~/.ct-api-keys.json`.

**Trade-off** (decision Ian made on 2026-05-10):
- Commented out → no proxy noise, no auto-revival → manual intervention required to re-add
- Uncommented + dead-flagged → daily check auto-revives, but proxy queue has dead slots wasting one position each → small overhead

Currently Ian chose commented-out (cleaner queue). Revival is manual.

## Other files touched when the keys were removed

- `/Users/cleo/semantic-search-proxy.py` — KEY_SEQUENCE shrunk from 16 → 15 entries; comment block updated
- `/Users/cleo/.ct-api-keys.json` — proxy state file; current_index reset because old indices became stale

## When Ian asks "why is exa not in rotation"

Answer: 3 of 4 keys drained, no auto-refill, removed 2026-05-10 to keep the queue clean. Only `exa_4` is still active. To get more Exa coverage, either top up an existing account (per this runbook) or create a new account and add the key as `EXA_API_KEY_5` and append to `KEY_SEQUENCE`.
