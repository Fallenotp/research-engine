# TEST RESULTS — tel-B hook

## Scope

- Edited only `/Users/cleo/.claude/hooks/save-research-outputs.sh`
- Created this proof file

## Backup

- Backup path: `/Users/cleo/.claude/hooks/save-research-outputs.sh.bak-20260523-tel`

## Exact added lines

```diff
+# --- research telemetry observer (read-only; must never block or fail this hook) ---
+( /opt/homebrew/bin/python3.11 /Users/cleo/lattice/research_engine/telemetry_observer.py >/dev/null 2>&1 & ) || true
```

## Syntax check

- Command: `bash -n /Users/cleo/.claude/hooks/save-research-outputs.sh`
- Result: PASSED
- Exit code: `0`
- Output: none

## Hook execution

- Executed hook: no
- Why: not clearly side-effect-free. The hook reads Stop-hook stdin, copies matching files into `~/research-archive/`, and writes `_session-info.txt`, so running it would perform real writes outside the requested additive edit.

## Change confirmation

- Confirmed: no existing lines were removed or altered
- Confirmed: only the hook was edited
