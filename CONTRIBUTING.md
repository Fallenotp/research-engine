# Contributing

Thanks for looking. Read this first, because this repo is probably not what you expect.

## What this is

A personal system that runs daily on one machine, published because it might be useful
to someone else. It is not a product and there is no roadmap. I built it to answer my own
research questions.

That has consequences. There is no packaging file yet, several features expect local
services you have to stand up yourself, and one test fails unless a local proxy is
running. The README says which. I would rather tell you that up front than have you find
out at line 200.

## Before you open a pull request

Run the tests:

```bash
python -m pytest . -q
```

Expect `1 failed, 362 passed`. The failure is `test_proxy_json_enforce`, which needs a
local model proxy on port 8084. If anything else fails, that is either your environment
or a real bug, and it is worth saying which you think it is.

Run the linter:

```bash
ruff check .
```

Expect 69 errors, all of them in `vendor/mlx-token-proxy.py`. That file predates the
linter and is on the list. Anything outside it should be clean.

## Things I care about

**No hardcoded paths.** Every machine-specific path goes through `paths.py` as an
environment variable with a default. This repo used to be full of `/Users/...` strings
and cleaning them out was tedious. Please do not add more.

**Fail down, never up.** If a quality signal cannot be measured, return the low-confidence
answer and log why. Do not substitute a plausible-looking default. This codebase shipped
for months reporting hardcoded confidence scores that measured nothing, and it looked
completely fine from the outside. That is the failure mode to design against.

**Absolute assertions in tests.** Assert the value you expect, not that one number is
bigger than another. A relative assertion passes happily on a broken scale. That has
bitten this project more than once.

**Say when something does not work.** An honest "I could not get this to pass" is more
useful than a green test that skips the hard case.

## Things I will probably say no to

- Reformatting sweeps, dependency bumps with no reason given, or renaming things to match
  a style guide.
- New search lanes or reader rungs without evidence they return better results than what
  is already there. There are 46 lanes and 17 rungs; the problem is not coverage.
- Anything that makes a failure quieter.

## Reporting bugs

Open an issue with the question or URL that triggered it, what you expected, what you got,
and the relevant lines from the telemetry log in your data directory (default
`~/.research_engine`). Strip your API keys out of anything you paste.

Security problems go through the Security tab instead. See [SECURITY.md](SECURITY.md).

## Licence

Contributions are accepted under the MIT licence, same as the rest of the repo.
