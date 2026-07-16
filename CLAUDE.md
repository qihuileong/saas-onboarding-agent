# SaaS Onboarding Agent

An agent that reads a SaaS product's API documentation and bootstraps an integration:
it explores the docs, extracts a **structured API spec** (base URL, auth method, endpoints),
and can make **real, grounded API calls** on the user's behalf. Runs fully local via Ollama.

## How to run
- `uv run python main.py --url <docs-url>` — one-shot prose summary of an API's docs.
- `uv run python agent.py` — the interactive agent (the main program). Type a docs URL **or a local
  spec path** (`C:\...\openapi.json`, `./spec.yaml`, `file://...`; JSON or YAML), ask it to
  read/describe/find/test endpoints. Commands: `extract` (dump structured spec), `end session` (quit).
- Deps are managed with **uv** (`pyproject.toml` + `uv.lock`), Python >= 3.14.
- Requires a local **Ollama** server with the models pulled (see below).

## Key files
- **agent.py** — the heart of the project. A LangGraph ReAct agent (`create_react_agent`) with
  three tools: `fetch_url` (read a docs page), `make_api_call` (real HTTP request),
  `authorize` (human-in-the-loop credential prompt). Also holds the structured-extraction pipeline
  and the OpenAPI-first onboarding flow (`onboard` -> `find_openapi_spec`/`parse_openapi`, else scrape).
  The REPL loop routes each turn by **intent** (see Patterns): describe -> spec lookup, discovery ->
  RAG search, test/call -> the tool-agent. The whole per-turn body is wrapped in a safety net so no
  single turn (bad input, model/network error, Ctrl+C) can kill the session.
- **extract.py** — simplest extractor: one LLM call over the first 4000 chars of a page -> prose summary.
- **scraper.py** — `fetch_text` (requests + BeautifulSoup, strips nav/footer boilerplate, returns
  text **and** same-host links) and `crawl` (BFS over same-section doc pages).
- **llm.py** — a tiny ChatOllama smoke-test; not part of the main flow.
- **main.py** — CLI entry point for the one-shot `extract.py` path (argparse `--url`).

## Models (Ollama)
- **agent.py** uses `qwen2.5:7b` (temperature 0, num_ctx 8192) — this is the model that matters.
- **agent.py** also uses `nomic-embed-text` for endpoint-discovery RAG (see Patterns). Pull it with
  `ollama pull nomic-embed-text` — without it, discovery degrades to keyword matching (no crash).
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
  only fall back to LLM scraping when there's no spec. `parse_openapi`/`_load_spec` accept an
  http(s) URL **or a local file** (`file://` or a bare path), JSON **or YAML** (`yaml.safe_load`).
- **Intent routing keeps decisions out of the weak model.** The 7B is unreliable at choosing tools
  and at grounding discipline, so the REPL decides per turn instead of trusting it:
  - **list/page** (`_listing_request`) -> deterministic slice of the spec ("show me the next 10
    endpoints"), tracked by a `list_cursor`. Kept narrow so a discovery ask ("show me the endpoint
    that creates a meeting") does NOT match and still reaches RAG.
  - **describe** (`DESCRIBE_INTENT`, "what does X do") -> `_find_endpoints` keyword lookup over the
    full spec, phrased by the **bare `llm`** (no tools bound, so it *can't* hallucinate a URL or fire
    a call).
  - **example/payload** (`EXAMPLE_INTENT`) -> synthesize a grounded request/response example from the
    endpoint's stored schema. Honors **direction**: "payload"/"send" -> request body only,
    "response"/"returns" -> response only.
  - **discovery** (`SEARCH_INTENT`, "which endpoint lets me…") -> **RAG**: `nomic-embed-text`
    embeddings of every endpoint (lazy-built on first search via `ensure_endpoint_index`, ~47s once
    per spec, in-memory cosine, no vector DB), retrieve top-k, bare `llm` phrases from ONLY those.
    There's no relevance threshold — an off-topic ask still returns the k nearest, and the prompt
    tells the model to say plainly when none fit (so "create an invoice" vs Zoom -> "no such
    endpoint"). Degrades to keyword search if the embed model is missing.
  - **test/call** -> the real tool-agent (`make_api_call`) — that genuinely needs a live request.
  Two routing invariants make this reliable:
  - **Method-aware matching.** `_query_methods` maps an explicit token (`patch`) or a CRUD verb
    (`update`->PATCH/PUT, `create`->POST, `delete`->DELETE) to method(s); `_find_endpoints` uses it to
    disambiguate endpoints sharing a path and to break tied segment scores (REST puts the verb in the
    method, not the path). A specificity tiebreak favors the more general path on ties.
  - **`named` is exact-only** (`_named_endpoint`) — true ONLY when the user typed a real path, never a
    fuzzy keyword hit. A fuzzy hit must not masquerade as "user named this endpoint" and pre-empt RAG.
  - **Follow-ups** ("how does the response schema look like?") resolve to `last_endpoint` (the endpoint
    the previous turn settled on) when the message names no path but refers back anaphorically.
  This routing is a *reliability substitute* for a weak local model; with a frontier model you'd
  delete most of it and just give the agent a `search_endpoints` tool.

## Session continuity
**At the start of a session:** if a `./sessions/` folder exists, read the most recent file in it
to see where the last session left off before starting new work. Files are named by date-time.
(The SessionStart hook also auto-injects the newest one.)

**At the end of a session:** when the user says **"wrap up and save the summary"**, write a new
file `./sessions/session_<YYYY-MM-DD_HHmmss>.md` yourself (use the Write tool). Cover: what we
worked on, decisions made, current state, and concrete next steps. Use ONLY what actually happened
this session — do not invent files, commands, or results. A Claude-written summary is preferred
over the hook's automatic raw-excerpt fallback.
