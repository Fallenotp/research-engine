# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: go to the **Security** tab of this
repository and click **Report a vulnerability**. That opens a private thread visible
only to the maintainer.

Please do not open a public issue for a security problem. A public issue tells everyone
about the hole at the same time it tells me.

I maintain this in my spare time, so expect a first reply within about a week rather
than within hours.

## What this project touches

Worth knowing before you run it, because the risk surface is wider than a typical library.

It fetches arbitrary web pages, runs headless browsers against them, shells out to local
CLI tools, and sends page text to language models. Treat any content it retrieves as
untrusted input, because it is.

Specifically:

- **Prompt injection is a real risk here.** Fetched page text goes into LLM prompts. A
  hostile page can contain instructions aimed at the model. Nothing in this repo fully
  solves that. The grounding and abstain logic reduce the blast radius; they do not
  eliminate it.
- **Headless browsers execute JavaScript** from pages you did not choose. Run the browser
  rungs in a sandbox or a VM if the URLs come from somewhere you do not control.
- **Shelling out.** The engine invokes local binaries (`agy`, `grok`, `agent-browser`).
  Their paths come from environment variables. Do not point those variables at anything
  you would not run yourself.
- **API keys** are read from the environment or from an env file you point at with
  `RESEARCH_ENGINE_ENV_FILE`. No key is stored in this repository. Do not commit yours;
  `.gitignore` blocks `.env` files, but that is a safety net, not a guarantee.

## Supported versions

There is no release process and no version numbers. Only the current `master` branch
gets fixes. If you are running an older checkout, pull before reporting anything.

## Out of scope

- Findings from an automated scanner with no working proof of exploit.
- Rate limits, quotas, or terms-of-service issues with third-party APIs the engine calls.
- The fact that scraping some sites may breach their terms. That is your call to make,
  not a defect in this code.
