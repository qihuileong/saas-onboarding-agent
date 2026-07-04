# SaaS Onboarding Agent

An agent that reads a SaaS product's API documentation and bootstraps an integration:
it explores the docs, extracts a **structured API spec** (base URL, auth method, endpoints),
and can make **real, grounded API calls** on the user's behalf. Runs fully local via Ollama.

## How to run
- `uv run python main.py --url <docs-url>` — one-shot prose summary of an API's docs.
- `uv run python agent.py` — the interactive agent (the main program). Type a docs URL, ask it
  to read/summarize/test endpoints. Commands: `extract` (dump structured spec), `end session` (quit).
- Deps are managed with **uv** (`pyproject.toml` + `uv.lock`), Python >= 3.14.
- Requires a local **Ollama** server with the models pulled (see below).

## Key files
- **agent.py** — the heart of the project. A LangGraph ReAct agent (`create_react_agent`) with
  three tools: `fetch_url` (read a docs page), `make_api_call` (real HTTP request),
  `authorize` (human-in-the-loop credential prompt). Also holds the structured-extraction pipeline
  and the OpenAPI-first onboarding flow (`onboard` -> `find_openapi_spec`/`parse_openapi`, else scrape).
- **extract.py** — simplest extractor: one LLM call over the first 4000 chars of a page -> prose summary.
- **scraper.py** — `fetch_text` (requests + BeautifulSoup, strips nav/footer boilerplate, returns
  text **and** same-host links) and `crawl` (BFS over same-section doc pages).
- **llm.py** — a tiny ChatOllama smoke-test; not part of the main flow.
- **main.py** — CLI entry point for the one-shot `extract.py` path (argparse `--url`).

## Models (Ollama)
- **agent.py** uses `qwen2.5:7b` (temperature 0, num_ctx 8192) — this is the model that matters.
- `extract.py` / `llm.py` use `qwen2.5-coder:7b`. (Inconsistent on purpose-of-history; the agent path
  is the current one.)
- Do **not** pull the 14b models — the machine has 8GB VRAM and won't fit them.

## Patterns that matter here
- **Structured output beats fabrication.** Extraction uses a Pydantic schema (`ApiSpec`/`Endpoint`)
  via `llm.with_structured_output(...)`. This is what stopped the model inventing endpoints — keep it.
- **Grounding guards.** `make_api_call` refuses paths whose segments never appeared in fetched docs
  (`DOC_CORPUS`); the system prompt forbids inventing endpoints or fabricating tool results.
- **Secrets never reach the model.** Tokens live in the module-level `SECRETS` dict (host -> token),
  never in `messages`. The model uses the literal placeholder `{{TOKEN}}`; real values are substituted
  only inside `make_api_call`. `authorize` obtains tokens via LangGraph `interrupt` (human in the loop).
- **OpenAPI-first.** If a machine-readable spec exists, parse it deterministically (no LLM);
  only fall back to LLM scraping when there's no spec.

## Session continuity
**At the start of a session:** if a `./sessions/` folder exists, read the most recent file in it
to see where the last session left off before starting new work. Files are named by date-time.
(The SessionStart hook also auto-injects the newest one.)

**At the end of a session:** when the user says **"wrap up and save the summary"**, write a new
file `./sessions/session_<YYYY-MM-DD_HHmmss>.md` yourself (use the Write tool). Cover: what we
worked on, decisions made, current state, and concrete next steps. Use ONLY what actually happened
this session — do not invent files, commands, or results. A Claude-written summary is preferred
over the hook's automatic raw-excerpt fallback.
