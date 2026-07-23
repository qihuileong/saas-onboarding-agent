# SaaS Onboarding Agent

An agent that reads a SaaS product's API documentation and bootstraps an integration:
it explores the docs, extracts a **structured API spec** (base URL, auth method, endpoints),
and can make **real, grounded API calls** on the user's behalf. Runs fully local via Ollama.

## How to run
- `uv run python main.py --url <docs-url>` — one-shot prose summary of an API's docs.
- `uv run python tests/test_call_fidelity.py [spec ...]` — audit that the parser keeps everything a
  real HTTP call needs (auth placement, bodies, media types, constraints). Defaults to
  `tests/fixtures/`; non-zero exit on a HIGH finding.
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
- **Report coverage, because extraction fails silently.** Every extraction bug here has failed
  without raising: an unresolved `$ref` renders as `{}`, a `$ref` parameter gets skipped, scopes in
  prose come back `[]`. Nothing errors — the agent just describes a smaller, emptier API than the
  real one, and you find out several turns later via a confidently wrong answer. `spec_coverage()`
  prints what the parse actually YIELDED right after onboarding (endpoints, how many carry
  auth/params/schemas, and a `$ref` audit: resolved / broken / remote), and flags an anomalous zero
  with the usual cause. Treat a zero in that report as a parser bug until proven otherwise.
- **Structured output beats fabrication.** Extraction uses a Pydantic schema (`ApiSpec`/`Endpoint`)
  via `llm.with_structured_output(...)`. This is what stopped the model inventing endpoints — keep it.
- **Grounding guards.** `make_api_call` refuses paths whose segments never appeared in fetched docs
  (`DOC_CORPUS`); the system prompt forbids inventing endpoints or fabricating tool results.
- **Check the model's OUTPUT, not just its input.** The grounding prompts forbid inventing
  endpoints, and the 7B complied for four turns and then didn't: handed a candidate list containing
  `GET /scheduled_events`, it produced **`POST /scheduled_events`** — inventing a verb because the
  collection existed — and attached `POST /invitees`' required fields to it. Retrieval was correct;
  the model added a fact that was never in context, so no amount of prompt wording fixes it.
  `verify_endpoint_mentions()` regexes every `METHOD /path` out of the answer, normalises it
  (host and base path stripped, `{uuid}`/`{id}` collapsed to `{}`, markdown punctuation trimmed —
  collapse templates BEFORE trimming, or a trailing `}` is eaten and nothing matches) and checks it
  against the spec, printing what's real on that path when it isn't. Every spec-answering route
  goes through `grounded_answer()` so the check can't be forgotten at one of the three call sites.
- **Secrets never reach the model.** Tokens live in the module-level `SECRETS` dict (host -> token),
  never in `messages`. The model uses the literal placeholder `{{TOKEN}}`; real values are substituted
  only inside `make_api_call`. `authorize` obtains tokens via LangGraph `interrupt` (human in the loop).
- **OpenAPI-first.** If a machine-readable spec exists, parse it deterministically (no LLM);
  only fall back to LLM scraping when there's no spec. `parse_openapi`/`_load_spec` accept an
  http(s) URL **or a local file** (`file://` or a bare path), JSON **or YAML** (`yaml.safe_load`).
- **`$ref` resolution is not optional.** Specs vary wildly in how much they inline: Zoom's file is
  fully expanded, Calendly's has ~500 `$ref`s. `_deref` resolves same-document pointers lazily
  against `_SPEC_ROOT` (set by `parse_openapi`), and every schema walker (`_example_from_schema`,
  `_object_properties`, `_object_required`, `_params_list`, `response_fields`) derefs on entry.
  Skipping it doesn't error — it silently degrades: nested arrays render `[]`, shared parameters
  vanish, and the agent confidently reports an emptier API than the one that exists.
- **Expand nested schemas ALL the way, and announce every cut.** A top-level-only view of a response
  (`collection (array)`) made the agent answer "the exact fields are not specified" about an item
  schema sitting right there in the spec — the same silent-starvation shape as the dropped param
  descriptions. `_schema_tree` walks every level; `endpoint_listing` embeds the full tree
  (`TREE_BUDGET_SINGLE` when one endpoint is in focus, `TREE_BUDGET_MULTI` per candidate otherwise).
  Two guards, and **both report rather than apply silently**: a char budget, and a cycle guard
  keyed on `$ref` NAMES along the current branch (`_deref` only breaks ref→ref chains within one
  call, so `User -> Organization -> User` would otherwise recurse forever). When either fires,
  `_tree_block` appends a `[!] TRUNCATED` marker phrased for the model to relay, and every grounding
  prompt instructs it to. `_fit_prompt` is the last line of defence before each `llm.invoke`: it
  measures the assembled prompt against `MODEL_PROMPT_CHARS` and, if over, cuts it, tells the model
  to say so, **and prints a warning itself** — never rely on the 7B to volunteer that its context was
  clipped. This is the whole project's rule applied to context: a partial answer is fine, a partial
  answer presented as complete is not.
- **Keep everything a real CALL depends on, and audit that you did.** `tests/test_call_fidelity.py`
  walks each fact an HTTP request needs and asserts the parser CAPTURED it, comparing the raw
  document against `ENDPOINT_SCHEMAS`/`AUTH_SCHEMES` — not against the parser's own opinion. Run it
  against any spec (`uv run python tests/test_call_fidelity.py [spec ...]`); it defaults to the
  pinned specs in `tests/fixtures/` and exits non-zero on a HIGH finding. Pin fixtures rather than
  auditing a live URL: Calendly's hosted spec grew 56→61 endpoints between two sessions, and a
  moving target makes a vendor's feature launch look like a parser regression.
  It found two bugs that produced confident falsehoods:
  - **Swagger 2.0 layouts differ, and missing them reads as an assertion.** In 2.0 the request body
    is a `parameters` entry (`in: body` / `in: formData`), and the response schema hangs straight
    off the response (`responses.200.schema`), not under `content`. `_op_request_schema` /
    `_op_response_schema` handle both. Before: every Petstore 2 body was `None`, which by the
    `request_required()` contract *asserts* "this endpoint takes no body" — so the agent said
    `POST /pet` needs no payload; and all 20 endpoints appeared to return nothing.
  - **The credential's type is not its location.** `apiKey` is unsendable until you also know
    `in: header` and `name: api_key`. `AUTH_SCHEMES` + `_credential_placement` parse that, and
    `run_grounded_call` builds the header (or query param) from it instead of hardcoding
    `Authorization: Bearer`. A query-placed key is substituted in `_do_http_call`, which matches the
    percent-encoded placeholder too because `urlencode` mangles it on the way in.
  Also carried now, each because its absence turns a documented rejection into a mystery: request
  media type (a JSON body sent to an `application/octet-stream` endpoint fails on content type, so
  `run_grounded_call` refuses rather than firing), `deprecated`, param constraints (`min`/`max`/
  `pattern` — what a 422 is made of), array `style`/`explode` (`?t=a,b` vs `?t=a&t=b`), `readOnly`
  (must NOT be sent), and **nested** required fields — `request_required()` is top-level by
  contract, so `_schema_tree` marks `required` per object, which is the only place
  `invitee.email` shows up.
- **Distinguish "absent" from "empty".** `request_required` returns `None` (no body) vs `[]` (body,
  nothing required); `_op_security` returns scopes *and* scheme names, so a spec that names
  `{oauth2: []}` can still answer "oauth2 authenticates this" instead of "no auth documented".
  Collapsing either pair produces an answer that is wrong in a way that reads as authoritative.
- **An empty structured field does not mean undocumented — check the prose.** Calendly's `security[]`
  is a bare `{oauth2: []}`, yet all 56 endpoints state their scope inside the operation
  `description`: `> #### Required scopes: \`availability:read\``. `_doc_scopes` regexes that out
  (also matching Zoom's `**Scope:** \`...\`` form) and `_op_security` uses it **only as a fallback**
  when `security[]` yields nothing — so a spec that declares scopes properly is never polluted by a
  loose match on the word "scope". Same root cause: `description` was being dropped whenever
  `summary` existed, so per-endpoint constraints ("date range can be no greater than 1 week",
  "does not support keyset pagination") were invisible. They're kept as `notes` and shown in
  single-endpoint (`param_detail=True`) listings only, to keep them out of the RAG index.
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
  - **Follow-ups** (`_is_followup`) resolve to `last_endpoint` (the endpoint the previous turn settled
    on) when the message names no path. Three ways to qualify: an explicit back-reference ("the response
    schema of **this endpoint**"); a short subjectless **attribute** question ("what scope is
    needed") — gated on an attribute noun from `ENDPOINT_ATTRS`, because a tersely typed *new* search
    ("cancel an event") carries none and must still reach RAG; or a **nested-field** ask ("what's
    inside `collection`") — a `NESTING_PHRASES` word **plus** a whole-token match against
    `_schema_field_names(focus)`. Both halves are required there: field names are ordinary words
    (`user`, `email`, `status`), so matching the name alone would hijack "which endpoint returns a
    user email" away from discovery. `FRESH_SEARCH` phrasings override all three.
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
