# SaaS Onboarding Agent

An agent that reads a SaaS product's API documentation and bootstraps an integration:
it explores the docs, extracts a **structured API spec** (base URL, auth method, endpoints),
and can make **real, grounded API calls** on the user's behalf. Runs fully local via Ollama.

## How to run
- `uv run streamlit run streamlit_app.py` — the browser UI: onboard with staged progress, chat
  against the spec, inspect endpoints, prepare/confirm real calls, and view/download redacted logs.
- `uv run python main.py --url <docs-url>` — one-shot prose summary of an API's docs.
- `uv run python tests/test_call_fidelity.py [spec ...]` — audit that the parser keeps everything a
  real HTTP call needs (auth placement, bodies, media types, constraints). Defaults to
  `tests/fixtures/`; non-zero exit on a HIGH finding.
- `uv run python tests/test_onboard_evidence.py [--live]` — the onboarding guards: the evidence gate,
  spec-ancestor walk-up, and scraped-vs-parsed coverage wording. Network-free without `--live`.
- `uv run python tests/test_ui_service.py` and
  `uv run python tests/test_streamlit_workflows.py` — network-free service and browser-journey
  regressions, including auth variants, credential lifecycle/redaction, reruns, and call handoff.
- `uv run python tests/test_live_agent_journeys.py` — local Ollama embeddings + model answers against
  the pinned Calendly spec; no third-party API call or credential.
- `uv run python agent.py` — the interactive agent (the main program). Type a docs URL **or a local
  spec path** (`C:\...\openapi.json`, `./spec.yaml`, `file://...`; JSON or YAML), ask it to
  read/describe/find/test endpoints. Commands: `extract` (dump structured spec), `end session` (quit).
- Deps are managed with **uv** (`pyproject.toml` + `uv.lock`), Python >= 3.14.
- Requires a local **Ollama** server with the models pulled (see below).

## Key files
- **streamlit_app.py** — browser UI and Streamlit session state. It deliberately uses a two-step
  prepare/confirm flow for real calls; credentials are session-scoped by API host + auth scheme, so
  they follow endpoint changes without being shared across browser sessions or written to disk.
  The sidebar credential manager is the single edit/replace/clear surface; Chat and API Call only
  check that shared store and direct the user there when the required scheme is missing.
  Calls selected from Chat append their results to Chat. Failed responses in that linked flow are
  reviewed against the selected endpoint's schema and masked request, and the local model drafts a
  corrected next request; the user must still prepare, inspect, and explicitly send it. Calls
  prepared directly in API Call remain standalone unless the user explicitly selects **Ask agent
  to diagnose this response** after a failure.
  The Debug & Traces tab separates transcript, detailed traces, and event log, with per-trace and
  filtered-trace downloads in addition to the complete local debug bundle.
- **ui_service.py** — UI-safe reusable workflows: deterministic chat routing, call preparation and
  execution, Langfuse-shaped trace spans, credential-safe previews, and redaction. It contains no
  Streamlit imports.
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
- **Event logs and detailed traces are different layers.** Event logs record concise
  INFO/WARNING/ERROR events and exclude response bodies. Local traces capture full chat turns,
  routing, retrieval candidates, LLM prompt/output, timings, masked HTTP requests, and API results;
  the UI can copy/download a single trace, export only the current filtered trace set, or export the
  entire session with its transcript. Credentials are redacted from both, but trace bundles can
  contain API/user data and must be reviewed before sharing. Langfuse can later persist/evaluate the
  same spans; it does not replace application progress/error logs.
- **Response truncation is a model-context boundary, not a browser boundary.** Tool/CLI calls keep
  `_do_http_call`'s 500-character default so a response cannot flood the local model context. The
  Streamlit call builder passes `response_limit=None`: its result stays local, must remain valid JSON
  for pretty rendering, and may be inspected in full. A failed-call review sends only the masked
  request plus a redacted, bounded response excerpt to the local model; the full result remains in
  Pretty/Raw response and the local trace. Do not reapply the tool excerpt limit to UI calls.
- **UI progress must reflect real stages.** `onboard(..., progress=callback)` and
  `find_openapi_spec(..., progress=callback)` report discovery, validation, parsing/scraping, and
  completion. Do not fake a timer-based progress bar; long embedding/LLM stages report before and
  after the blocking operation.
- **A Streamlit rerun must restore the whole parsed runtime, not only `ApiSpec`.** Endpoint names live
  in `ApiSpec`, but request/response schemas, auth placement, `$ref` roots, and the RAG index live in
  `agent.py` globals. Hot reload can preserve `st.session_state.spec` while reinitializing those
  globals, producing plausible falsehoods such as `POST /one_off_event_types [no request body]`.
  `agent_runtime` snapshots and `_restore_agent()` rehydrate them before every UI render.
- **Streamlit can retain stale imported modules during a hot rerun.** `ui_service.API_VERSION` and
  the guarded reload at the top of `streamlit_app.py` keep a newer page from calling an older
  service signature. API-call preparation catches unexpected exceptions locally so one broken tab
  never replaces the entire app with Streamlit's traceback page.
- **Spec-grounded is weaker than retrieval-grounded.** `verify_endpoint_mentions()` proves that a
  named endpoint exists somewhere in the full spec. During RAG discovery, the model must also be
  checked against the candidate block it actually received; otherwise a real but unretrieved
  endpoint is prior-knowledge leakage. `ui_service.verify_candidate_mentions()` enforces that
  narrower evidence boundary.
- **Report coverage, because extraction fails silently.** Every extraction bug here has failed
  without raising: an unresolved `$ref` renders as `{}`, a `$ref` parameter gets skipped, scopes in
  prose come back `[]`. Nothing errors — the agent just describes a smaller, emptier API than the
  real one, and you find out several turns later via a confidently wrong answer. `spec_coverage()`
  prints what the parse actually YIELDED right after onboarding (endpoints, how many carry
  auth/params/schemas, and a `$ref` audit: resolved / broken / remote), and flags an anomalous zero
  with the usual cause. Treat a zero in that report as a parser bug until proven otherwise —
  **unless the spec was scraped**, where the zeros are structural and mean something else entirely
  (next bullet but two).
- **Structured output beats fabrication — of FIELDS, not of EXISTENCE.** Extraction uses a Pydantic
  schema (`ApiSpec`/`Endpoint`) via `llm.with_structured_output(...)`; keep it. But know its limit:
  it constrains the shape of the answer, never whether there was anything to answer. Handed a
  Stoplight *model export* (`.../openapi.yaml/components/schemas/Event` — a bare JSON-Schema
  fragment, 24KB with no `paths:` and **zero** HTTP verbs), it returned 12 confident endpoints built
  out of nouns in the schema: `POST /google_conference`, `GET /conferences/gotomeeting/{conference_id}`.
  Every noun was real (`gotomeeting` ×7 in the doc); every **method** and every **path parameter**
  was invented. `_clean` can't help — the output is schema-valid. So the guard has to sit upstream.
- **Check the EVIDENCE before the model sees it.** `_has_endpoint_evidence` requires a literal
  `METHOD /path` pair (`_METHOD_PATH_RE`; tolerates markdown between the two — `**POST** \`/invitees\``
  is one pair — but bounded and word-char-free, so "the POST body described in /docs" can't bridge
  it). `extract_from_pages` skips chunks without it and **raises** if none qualify, rather than
  returning a spec made of guesses. Conservative in the safe direction on purpose: a miss refuses to
  onboard (loud, recoverable), a false pass fabricates silently and reads as authoritative for the
  rest of the session.
- **A scraped spec must announce that it is scraped.** `SPEC_PROVENANCE` ("openapi" | "scraped") is
  set by `onboard`. It matters because the scrape path *never* populates params/schemas/scopes, so
  it always reports `0/N` across the board — and the parsed-path hint ("the spec may document them
  in prose this parser doesn't recognise") then sends you hunting a parser bug that doesn't exist,
  while implying the endpoints themselves are solid and merely under-annotated. They aren't.
  `spec_coverage` says `SCRAPED, NOT PARSED`, that the zeros are expected, and that nothing is
  corroborated; it also drops the `$ref` tally, which is vacuous when `_SPEC_ROOT` is `{}`.
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
  **Its blind spot:** it validates answers *against the spec*, so it is useless when the spec itself
  is the fabrication — a scraped `POST /scheduled_events` gets certified as real. Output-side
  grounding assumes the spec is ground truth; that assumption is what the evidence gate protects.
- **Secrets never reach the model.** Tokens live in the module-level `SECRETS` dict (host -> token),
  never in `messages`. The model uses the literal placeholder `{{TOKEN}}`; real values are substituted
  only inside `make_api_call`. `authorize` obtains tokens via LangGraph `interrupt` (human in the loop).
- **OpenAPI-first.** If a machine-readable spec exists, parse it deterministically (no LLM);
  only fall back to LLM scraping when there's no spec. `parse_openapi`/`_load_spec` accept an
  http(s) URL **or a local file** (`file://` or a bare path), JSON **or YAML** (`yaml.safe_load`).
  A URL may also point *into* a spec rather than at one: doc portals export a piece by appending a
  pointer to the spec's own path (`.../calendly-api/openapi.yaml/components/schemas/Event`).
  `_spec_ancestors` truncates at any `.json`/`.yaml`/`.yml` path segment that has more path after it,
  innermost first, and `find_openapi_spec` tries those at step 0.5 — purely structural, so it covers
  any portal with that shape, and `_is_openapi` still validates every candidate. On the URL above
  that's the difference between 12 fabricated endpoints and the real 61.
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
    `run_grounded_call` builds the header, query parameter, or cookie from it instead of hardcoding
    `Authorization: Bearer`. `credential_options` drives scheme-specific UI labels for PAT/bearer,
    OAuth/OpenID access tokens, API keys, and Basic credentials. Missing or unsupported placement
    must stop request preparation; never guess Bearer. A query-placed key is substituted in
    `_do_http_call`, which matches the percent-encoded placeholder too because `urlencode` mangles
    it on the way in.
  Also carried now, each because its absence turns a documented rejection into a mystery: request
  media type (a JSON body sent to an `application/octet-stream` endpoint fails on content type, so
  `run_grounded_call` refuses rather than firing), `deprecated`, param constraints (`min`/`max`/
  `pattern` — what a 422 is made of), array `style`/`explode` (`?t=a,b` vs `?t=a&t=b`), `readOnly`
  (must NOT be sent), prose-only cross-parameter requirements such as "either `organization` or
  `user` is required", and **nested** required fields — `request_required()` is top-level by
  contract, so `_schema_tree` marks `required` per object, which is the only place `invitee.email`
  shows up.
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
