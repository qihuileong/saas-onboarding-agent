from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
import math
from langgraph.prebuilt import create_react_agent   
from scraper import fetch_text
import requests
from langgraph.checkpoint.memory import MemorySaver
import getpass
from urllib.parse import urlparse
from langgraph.types import interrupt, Command
import re
import os
import json
import yaml
import textwrap
from typing import Callable
from pydantic import BaseModel, Field
from collections import Counter
from scraper import crawl
from urllib.parse import urlparse, urljoin, urlunparse

API_VERSION = 4

SECRETS: dict[str, str] = {}   # host -> token. Never enters `messages`, so the model never sees it.
DOC_CORPUS: list[str] = []           # all doc text fetched so far (grounding evidence)
# (METHOD, path) -> {"request": schema|None, "response": schema|None}. Populated from the OpenAPI
# spec at parse time so we can synthesize grounded example payloads on demand. Kept OUT of the model's
# context and the handoff — schemas run 40-80 KB each and would blow num_ctx instantly.
ENDPOINT_SCHEMAS: dict[tuple[str, str], dict] = {}
# The whole parsed spec document, kept so $ref pointers stay resolvable AFTER parsing. The stored
# schemas above are the spec's own (unmodified) sub-objects, so a nested "#/components/schemas/Foo"
# is only meaningful while its root is around. Replaced wholesale on each parse_openapi().
_SPEC_ROOT: dict = {}
# name -> {"type", "scheme", "in", "param", "token_url", "authorization_url"} from securitySchemes.
# The TYPE alone is not enough to make a call: a spec that says `apiKey, in: header, name: api_key`
# needs `api_key: <token>`, NOT `Authorization: Bearer <token>`. Sending the wrong one is a
# guaranteed 401 that looks like a bad credential rather than a bad request.
AUTH_SCHEMES: dict[str, dict] = {}
# How the current spec was obtained: "openapi" (deterministic parse of a spec document) or "scraped"
# (an LLM reading prose). It changes what a zero in the coverage report MEANS - the scrape path never
# populates params/schemas/scopes at all, so reporting "0/12 parameters" there as a possible parser
# gap points at the wrong thing entirely. See spec_coverage().
SPEC_PROVENANCE: str = ""
SKIP_SEGMENTS = {"v1", "v2", "v3", "api", "rest"}   # version/prefix noise, not "real" path words

class Endpoint(BaseModel):
    method: str = Field(description="HTTP method, e.g. GET or POST")
    path: str = Field(description="Path or full URL of the endpoint")
    description: str = Field(description="What this endpoint does")

class ApiSpec(BaseModel):
    base_url: str = Field(description="Base URL of the API")
    auth_method: str = Field(description="Auth scheme, e.g. 'Bearer token in Authorization header'")
    endpoints: list[Endpoint] = Field(description="Documented endpoints found in the docs")

def _looks_like_id(segment: str) -> bool:
    # UUIDs, hex blobs, or numeric ids won't appear verbatim in docs — don't ground on them
    return bool(re.fullmatch(r"[0-9a-fA-F-]{8,}", segment)) or segment.isdigit()

def _truncate(text: str, limit: int, what: str = "output") -> str:
    """Cap text at `limit` chars. If it was cut, append a LABELED marker — so the model can
    tell a truncated result from a genuinely short one, and knows not to retry to get the rest
    (which would just loop, since the tool always truncates the same way)."""
    if len(text) <= limit:
        return text
    return (f"{text[:limit]}\n...[truncated: showing {limit} of {len(text)} chars of {what}. "
            f"This IS the full result the tool returns — do NOT retry the same call to get more; "
            f"if you need different data, ask the user to narrow the request.]")

@tool
def fetch_url(url: str) -> str:
    """Fetch a documentation page. Returns readable text PLUS links found on the page."""
    try:
        text, links = fetch_text(url)
    except requests.RequestException as e:        # 401, timeout, DNS, etc.
        return f"Could not fetch {url}: {e}"
    DOC_CORPUS.append(text)
    link_block = "\n".join(links[:40])
    return f"{_truncate(text, 4000, 'page text')}\n\n--- LINKS ON THIS PAGE ---\n{link_block}"

def _do_http_call(url: str, method: str = "GET", headers: dict | None = None,
                  body: dict | None = None, check_grounding: bool = True,
                  response_limit: int | None = 500) -> str:
    """Shared real-HTTP execution: grounding guard + token substitution + request. Used by the
    make_api_call tool AND the deterministic grounded-call flow (run_grounded_call). Secrets are
    substituted HERE, at HTTP time, so the real token never lives in headers the caller handled.
    `check_grounding` is skipped by run_grounded_call, whose path comes straight from the parsed
    spec (inherently grounded) and whose only variable segments are USER-supplied param values.
    Tool/CLI callers keep the small default body limit for model context; the browser UI passes
    ``None`` because its response is rendered locally and never enters the model prompt."""
    if check_grounding:
        # Grounding guard (B): refuse paths whose documented segments we haven't actually read.
        corpus = " ".join(DOC_CORPUS).lower()
        segments = [
            s for s in urlparse(url).path.split("/")
            if s and not _looks_like_id(s) and s.lower() not in SKIP_SEGMENTS
        ]
        ungrounded = [s for s in segments if s.lower() not in corpus]
        if ungrounded:
            return (f"Refusing to call {url}: path segment(s) {ungrounded} were not found in any "
                    f"documentation fetched so far. Read the relevant reference page first.")

    headers = dict(headers or {})
    host = urlparse(url).netloc
    token = SECRETS.get(host)

    # A credential can be placed in the QUERY STRING, not just a header (OpenAPI `apiKey, in: query`).
    # urlencode() percent-encodes the placeholder on the way in, so match both forms — otherwise the
    # literal text "{{TOKEN}}" is sent as the API key and the call fails as an auth error.
    PLACEHOLDERS = ("{{TOKEN}}", "%7B%7BTOKEN%7D%7D", "%7b%7bTOKEN%7d%7d")
    in_url = any(ph in url for ph in PLACEHOLDERS)

    # Guard: if a secret is referenced but we don't have one yet, force authorization first.
    needs_secret = any("{{TOKEN}}" in str(v) for v in headers.values()) or in_url
    if needs_secret and not token:
        return (f"No credential stored for {host}. "
                f"Call authorize('{host}') to obtain one, then retry this request.")

    real_headers = {
        k: (v.replace("{{TOKEN}}", token) if token else v)
        for k, v in headers.items()
    }
    if in_url and token:
        from urllib.parse import quote as _q
        for ph in PLACEHOLDERS:
            url = url.replace(ph, _q(token, safe=""))

    try:
        response = requests.request(method, url, headers=real_headers,
                                    json=body if body is not None else None, timeout=15)
    except requests.RequestException as e:
        return f"Request failed: {e}"
    response_body = (
        response.text
        if response_limit is None
        else _truncate(response.text, response_limit, "response body")
    )
    return f"Status: {response.status_code}\nBody: {response_body}"

@tool
def make_api_call(url: str, method: str = "GET", headers: dict | None = None,
                  body: dict | None = None) -> str:
    """Make a real HTTP request and return status + body. Pass a JSON `body` for write calls. Use the
    literal placeholder {{TOKEN}} wherever a secret belongs (e.g. Authorization: 'Bearer {{TOKEN}}')."""
    return _do_http_call(url, method, headers, body)


@tool
def authorize(host: str) -> str:
    """Request a credential for an API host. You will NEVER see the token value."""
    token = interrupt(f"Provide token for {host}")
    if token == "__CANCEL__":
        return (f"Authorization cancelled for {host}. Do NOT retry the call. "
                f"Report that a valid token is required.")
    SECRETS[host] = token
    return (f"Credential stored for {host}. "
            "Retry the request with the Authorization header set to 'Bearer {{TOKEN}}'.")


SYSTEM = """You are an API onboarding assistant.
- When given a documentation URL, EXPLORE it: read the page, follow links to the API Reference
  (e.g. /reference/...) to find concrete, documented endpoints. Check /llms.txt if it exists.
- Only make real API calls (make_api_call) or call `authorize` when the user EXPLICITLY asks you
  to test or call an endpoint. If they only ask you to read, summarize, or understand the docs,
  DO NOT call make_api_call or authorize — just read and report. A token is only needed for real calls.
- Before testing an endpoint, OPEN its API Reference page and identify the exact HTTP method,
  the full path, and ALL required headers (including any version header, e.g. Notion-Version).
- For a first test, PREFER an endpoint that needs no path parameters or IDs (e.g. a 'list' endpoint).
- State the method, path, and required headers you found (and the page URL) BEFORE you call.
- For secrets use the literal placeholder {{TOKEN}} (e.g. Authorization: "Bearer {{TOKEN}}").
- If a call returns 401/403, call `authorize` with the host, then retry.
- Do NOT invent endpoints, paths, or headers. If you can't find it documented, keep reading.
- NEVER fabricate, simulate, or write example API responses, tokens, or tool results.
- The ONLY valid source of a result is an actual tool call's returned Tool Message.
- To authorize or call an API you MUST emit a real tool call (authorize / make_api_call).
  Writing the call as text does NOT execute it. If you have not received a real Tool Message,
  you have no result — do not claim success, and do not invent a response body.
  - If a call returns 401 AFTER you authorized, the token is invalid. Report that clearly and STOP.
  Do NOT call authorize again unless the user explicitly asks you to retry.
  - "Test", "call", or "hit" an endpoint means use make_api_call (a real HTTP request).
  Use fetch_url ONLY for reading documentation pages, never to test an API endpoint."""

llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=8192)
extractor = llm.with_structured_output(ApiSpec)

# --- RAG index for endpoint discovery ------------------------------------------------------------
# A full 184-endpoint spec can't fit the model's context, so discovery questions ("which endpoint
# lets me create a meeting") are answered by RETRIEVAL: embed every endpoint once at onboard time,
# then embed the query and hand only the top-k nearest endpoints to the LLM. The model never sees
# the whole list — Python does the search, the model just phrases the grounded answer. 184 vectors
# is tiny, so a plain in-memory cosine scan beats it; no vector DB needed.
embedder = OllamaEmbeddings(model="nomic-embed-text")
ENDPOINT_INDEX: list[Endpoint] = []          # endpoints in the current spec, parallel to ENDPOINT_VECS
ENDPOINT_VECS: list[list[float]] = []        # their embedding vectors
_INDEXED_FOR: ApiSpec | None = None          # which spec the current index was built for (lazy build)

def _endpoint_text(e: Endpoint) -> str:
    return f"{e.method} {e.path} {e.description}".strip()

def build_endpoint_index(spec: ApiSpec) -> None:
    """Embed every endpoint once (one batched call) so discovery can retrieve instead of dumping
    the whole spec into context. Rebuilt on each onboard. Raises if the embed model is unavailable;
    the REPL catches that and falls back to keyword search."""
    global ENDPOINT_INDEX, ENDPOINT_VECS
    ENDPOINT_INDEX, ENDPOINT_VECS = [], []
    if not spec.endpoints:
        return
    ENDPOINT_VECS = embedder.embed_documents([_endpoint_text(e) for e in spec.endpoints])
    ENDPOINT_INDEX = list(spec.endpoints)

def endpoint_index_ready(spec: ApiSpec) -> bool:
    """True if the RAG index is already built for this exact spec (so we can skip the ~47s rebuild)."""
    return _INDEXED_FOR is spec and bool(ENDPOINT_VECS)

def ensure_endpoint_index(spec: ApiSpec) -> None:
    """Lazily build the index on first use, and rebuild only when the spec changes (new onboard)."""
    global _INDEXED_FOR
    if endpoint_index_ready(spec):
        return
    build_endpoint_index(spec)
    _INDEXED_FOR = spec

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def search_endpoints_semantic(query: str, k: int = 5) -> list[Endpoint]:
    """Top-k endpoints by embedding similarity to the query. Returns [] if the index is empty or the
    query embedding fails, so the caller can fall back to keyword matching (_find_endpoints)."""
    if not ENDPOINT_VECS:
        return []
    try:
        qv = embedder.embed_query(query)
    except Exception:
        return []
    ranked = sorted(zip(ENDPOINT_VECS, ENDPOINT_INDEX),
                    key=lambda pair: _cosine(qv, pair[0]), reverse=True)
    return [e for _, e in ranked[:k]]

_CLAUSE_SPLIT = re.compile(r"\b(?:then|and then|and|after that|afterwards|next|followed by)\b|[,;]", re.I)
_QUERY_ALIASES = {
    # Calendly and several other scheduling APIs call a human attendee an "invitee". Without this
    # deterministic bridge, keyword fallback ranks event-type endpoints for "add attendees" and
    # misses the endpoint that actually creates the booking.
    "attendee": ("invitee",),
    "attendees": ("invitee", "invitees", "guests"),
    "guest": ("invitee",),
    "guests": ("invitee", "invitees"),
}

def _expand_query_aliases(query: str) -> str:
    words = set(re.findall(r"[a-z_]+", query.lower()))
    additions = [alias for word in words for alias in _QUERY_ALIASES.get(word, ())]
    return query + (" " + " ".join(additions) if additions else "")

def search_endpoints(query: str, k: int = 8) -> list[Endpoint]:
    """Discovery retrieval. Splits a multi-intent query ('create a meeting THEN download the
    recording') into clauses and unions each clause's top matches — a compound query's single
    embedding is diluted and neither intent ranks. Single-intent queries pass straight through."""
    query = _expand_query_aliases(query)
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(query) if len(c.strip()) >= 8]
    if len(clauses) <= 1:
        return search_endpoints_semantic(query, k)
    per = max(3, k // len(clauses) + 1)
    seen, out = set(), []
    for clause in clauses:
        for e in search_endpoints_semantic(clause, per):
            if (e.method, e.path) not in seen:
                seen.add((e.method, e.path))
                out.append(e)
    return out[:k] or search_endpoints_semantic(query, k)

def extract_api_spec() -> ApiSpec:
    """One constrained call over everything fetched so far. Can only return schema-valid JSON."""
    if not DOC_CORPUS:
        raise ValueError("No docs fetched yet — read a docs URL first.")
    docs = "\n\n".join(DOC_CORPUS)[:12000]   # stay inside the context window
    prompt = (
        "Extract the API structure from the documentation below. "
        "Use ONLY information present in the text — do not invent endpoints.\n\n"
        f"{docs}"
    )
    return extractor.invoke(prompt)

checkpointer = MemorySaver()
agent = create_react_agent(
    llm,
    tools=[fetch_url, make_api_call, authorize],
    checkpointer=checkpointer,
    prompt=SYSTEM,
)

def _chunks(text, size=7000, overlap=1500):
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += size - overlap

VALID_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
def _clean(spec: ApiSpec) -> ApiSpec:
    seen, kept = set(), []
    for ep in spec.endpoints:
        method = ep.method.upper()
        if method not in VALID_METHODS:          # drops GIT, SSH
            continue
        path = ep.path.rstrip("/")
        key = (method, path)
        if key in seen:                          # drops exact dupes
            continue
        seen.add(key)
        kept.append(Endpoint(method=method, path=path, description=ep.description))
    spec.endpoints = kept
    return spec

# A block of documentation containing no `METHOD /path` pair anywhere cannot honestly yield an
# endpoint - whatever the extractor returns for it is invented from the nouns on the page. Check the
# EVIDENCE before the LLM ever sees the text: a Stoplight `components/schemas/Event` export (a data
# model - no `paths:`, not one HTTP verb in 24KB) produced 12 confident endpoints, because the schema
# happened to mention `google_conference`, `gotomeeting`, `webex_conference`. Every verb and every
# path parameter was fabricated. Structured output constrains the SHAPE of the answer, never whether
# there was anything to answer - so the guard has to sit upstream of the model, not on its output.
_METHOD_PATH_RE = re.compile(
    r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"         # the verb
    r"(?:[\s*`_|:.\-]){0,6}"                                  # markup/punctuation between the two -
                                                              # `**POST** \`/invitees\`` is one pair.
                                                              # Bounded, and no word chars, so "the
                                                              # POST body described in /docs" can't
                                                              # bridge the gap and match.
    r"(?:https?://[^\s\"'<>]+)?"                              # an optional absolute host
    r"/[A-Za-z0-9_\-{]",                                      # ...followed by an actual path
    re.I)

def _has_endpoint_evidence(text: str) -> bool:
    """True if `text` actually SHOWS a method+path pair. Conservative in the safe direction: a miss
    refuses to onboard (loud, recoverable, the user can point at a real reference page), whereas a
    false pass invents endpoints silently and they read as authoritative for the rest of the session."""
    return bool(_METHOD_PATH_RE.search(text))

def extract_from_pages(pages) -> ApiSpec:
    """pages: [(url, text)]. Chunk each page, extract per chunk, merge & dedup endpoints.
    Chunks showing no method+path evidence are skipped before the LLM sees them; if NONE of them
    qualify, raise instead of returning a spec built entirely out of guesses."""
    base_urls, auth_methods, seen, endpoints = [], [], set(), []
    total = skipped = 0
    for url, text in pages:
        for chunk in _chunks(text):
            total += 1
            if not _has_endpoint_evidence(chunk):
                skipped += 1
                continue
            spec = extractor.invoke(
                "From the API reference text below, extract ONLY documented request endpoints — "
"each must be a callable operation with an HTTP method and a path, like a section "
"titled 'List X' or 'Create Y'. IGNORE URLs that appear inside example JSON response "
"bodies (fields such as url, html_url, git_url, forks_url, languages_url), ignore "
"example values and sample curl commands. Use ONLY information present in the text; "
"do not invent endpoints.\n\n" + chunk
            )
            if spec.base_url.startswith("http"):     # a base_url must be absolute, not a path
                base_urls.append(spec.base_url)
            if spec.auth_method:
                auth_methods.append(spec.auth_method)
            for ep in spec.endpoints:
                key = (ep.method.upper(), ep.path)
                if key not in seen:
                    seen.add(key)
                    endpoints.append(ep)
    if total and skipped == total:
        raise ValueError(
            f"none of the {total} text block(s) fetched contain a `METHOD /path` pair - this page "
            f"documents something other than callable endpoints (a data model, a changelog, a "
            f"landing page). Refusing to extract, because anything returned would be invented from "
            f"the nouns on the page. Point me at the API reference itself, or at its OpenAPI spec")
    if skipped:
        print(f"[--] skipped {skipped}/{total} text block(s) with no METHOD /path evidence "
              f"- extracted from the remaining {total - skipped}.")
    base = Counter(base_urls).most_common(1)[0][0] if base_urls else ""
    auth = Counter(auth_methods).most_common(1)[0][0] if auth_methods else ""
    return _clean(ApiSpec(base_url=base, auth_method=auth, endpoints=endpoints))

def _load_spec(source: str) -> dict:
    """Load a spec from an http(s) URL or a local file. JSON or YAML — yaml.safe_load parses
    both (YAML is a JSON superset), so no extension sniffing needed. Returns the parsed mapping."""
    if source.startswith("file://"):
        source = source[7:]
        if re.match(r"/[A-Za-z]:", source):   # file:///C:/... — drop the leading slash on Windows
            source = source[1:]
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        raw = resp.text
    else:
        with open(source, "r", encoding="utf-8") as f:
            raw = f.read()
    return yaml.safe_load(raw)

def _deref(node, _depth: int = 0):
    """Resolve local `$ref` pointers ("#/components/schemas/Foo") against the spec being onboarded.
    Without this, a spec that defines its payloads by reference collapses to nothing useful:
    Calendly's `collection.items` is a bare $ref, so the response example rendered as `[]` and the
    whole object shape was lost. Only same-document refs are followed - a remote ref resolves to {}
    rather than firing a network fetch mid-answer. Sibling keys alongside a $ref (legal in OpenAPI
    3.1) override the target's, and a ref chain that loops back on itself stops instead of hanging."""
    seen = set()
    while isinstance(node, dict) and isinstance(node.get("$ref"), str) and _depth < 20:
        ref = node["$ref"]
        if not ref.startswith("#/") or ref in seen:
            return {}
        seen.add(ref)
        target = _SPEC_ROOT
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")   # JSON-Pointer escapes
            target = target.get(part) if isinstance(target, dict) else None
            if target is None:
                return {}
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        node = {**target, **siblings} if siblings else target
        _depth += 1
    return node

def _json_schema(content: dict) -> dict | None:
    """The application/json schema out of an OpenAPI content map (falls back to any media type)."""
    content = content or {}
    media = content.get("application/json") or (next(iter(content.values()), {}) if content else {})
    return media.get("schema") if isinstance(media, dict) else None

def _request_media_type(op: dict) -> str:
    """Which media type the request body is actually sent as. We render JSON examples and
    `requests(json=...)` sends JSON, so an endpoint expecting multipart/form-data or
    application/octet-stream needs to SAY so — otherwise the agent hands the user a JSON body for
    an upload endpoint and the call fails on content type, not on content."""
    body = _deref(op.get("requestBody") or {}) or {}
    content = body.get("content") or {}
    if content:
        return "application/json" if "application/json" in content else next(iter(content))
    # Swagger 2.0: `consumes` at operation level (falls back to the document's, applied by caller)
    consumes = op.get("consumes") or []
    if consumes:
        return "application/json" if "application/json" in consumes else consumes[0]
    if any((_deref(p) or {}).get("in") == "formData" for p in (op.get("parameters") or [])):
        return "application/x-www-form-urlencoded"
    return ""

def _op_request_schema(op: dict) -> dict | None:
    """The request-body schema, from EITHER OpenAPI 3 (`requestBody.content`) or Swagger 2.0, where
    the body is a `parameters` entry with `in: body` (a schema) or a set of `in: formData` fields.
    Missing the 2.0 forms doesn't degrade quietly: `request_required()` would return None, which by
    this module's convention ASSERTS 'this endpoint takes no body at all' - so the agent would tell
    the user POST /pet needs no payload while the spec defines a whole Pet schema."""
    body = _deref(op.get("requestBody") or {})
    schema = _json_schema((body or {}).get("content", {}))
    if schema is not None:
        return schema
    params = [_deref(p) or {} for p in (op.get("parameters") or [])]
    for p in params:                                   # Swagger 2.0 `in: body`
        if p.get("in") == "body" and p.get("schema"):
            return p["schema"]
    form = [p for p in params if p.get("in") == "formData"]
    if form:                                           # Swagger 2.0 `in: formData` -> object schema
        return {
            "type": "object",
            "properties": {p["name"]: {k: v for k, v in p.items()
                                       if k in ("type", "format", "enum", "default", "description",
                                                "items", "example")}
                           for p in form if p.get("name")},
            "required": [p["name"] for p in form if p.get("required") and p.get("name")],
        }
    return None

def _op_response_schema(op: dict) -> dict | None:
    """Schema of the first 2xx response (that's the success shape the user wants an example of).
    The response object itself may be a $ref (#/components/responses/...), so deref before reading.
    Handles both layouts: OpenAPI 3 nests the schema under `content.<media>.schema`, Swagger 2.0
    hangs it straight off the response as `schema`. Missing the 2.0 form isn't visibly broken - it
    just yields an API whose every endpoint appears to return nothing."""
    for code, r in (op.get("responses") or {}).items():
        r = _deref(r)
        if str(code).startswith("2") and isinstance(r, dict):
            s = _json_schema(r.get("content", {})) or r.get("schema")
            if s:
                return s
    return None

def _plain(text: str, limit: int = 200) -> str:
    """Collapse whitespace and flatten markdown links ([label](url) -> label) so a doc description
    reaches a prompt as readable prose. The flattening matters as much as the length cap: truncating
    raw doc text tends to slice through a link and leave a dangling URL fragment as the 'answer'."""
    text = re.sub(r"<!--.*?-->", " ", text or "", flags=re.S)   # doc-tool directives, not content
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = " ".join(text.split())
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")

def _params_list(raw) -> list[dict]:
    """Compact path/query parameters from an OpenAPI `parameters` array. Each: name, in
    (path/query/header), type, required, default, enum, desc, example. The description and example
    are the parts that make a param actually usable ("comma-separated list of occurrence IDs",
    `1648194360000,...`) - carry them, don't drop them. Entries are dereferenced first: specs
    routinely share parameters via #/components/parameters, and skipping those silently drops a
    chunk of the real input surface (25 of Calendly's 106)."""
    out = []
    for p in raw or []:
        p = _deref(p)
        if not isinstance(p, dict) or not p.get("name"):
            continue
        if p.get("in") in ("body", "formData"):
            continue          # Swagger 2.0 request body — carried as a schema by _op_request_schema
        sch = _deref(p.get("schema") or {})
        t = sch.get("type") or p.get("type") or ""
        items = _deref(sch.get("items") or p.get("items") or {})
        if t == "array" and items.get("type"):
            t = f"array of {items['type']}"       # "array" alone doesn't say what to put in it
        out.append({
            "name": p.get("name", ""),
            "in": p.get("in", ""),
            "type": t,
            "format": sch.get("format") or p.get("format") or "",
            "required": bool(p.get("required", False)),
            "default": sch.get("default", p.get("default")),
            "enum": sch.get("enum") or p.get("enum"),
            "desc": _plain(p.get("description") or ""),
            "example": sch.get("example", p.get("example")),
            # Constraints are what a 422 is made of - a spec that says maximum: 100 will reject 500.
            "limits": {k: (sch.get(k) if k in sch else p.get(k))
                       for k in ("minimum", "maximum", "minLength", "maxLength", "pattern",
                                 "minItems", "maxItems")
                       if k in sch or k in p},
            "deprecated": bool(p.get("deprecated", False)),
            # How a MULTI-VALUE param is written on the wire. OpenAPI's default (style=form,
            # explode=true) means `?tags=a&tags=b`; explode=false means `?tags=a,b`. Same data,
            # different string - and the user is the one typing it, so they need to be told which.
            "explode": p.get("explode", True) if t.startswith("array") else None,
            "style": p.get("style", "form") if t.startswith("array") else None,
        })
    return out


def parameter_format(param: dict) -> str:
    """Return a parameter's documented format, including a safe legacy-snapshot fallback."""
    documented = param.get("format") or ""
    if documented:
        return documented
    # Older Streamlit schema snapshots predate the explicit `format` field but retain descriptions
    # such as "the user's URI". Preserve the same validation without forcing a new onboarding run.
    if re.search(r"\bURI\b", param.get("desc") or "", re.I):
        return "uri"
    return ""

def _parse_security_schemes(spec: dict) -> dict[str, dict]:
    """securitySchemes (OpenAPI 3) / securityDefinitions (Swagger 2) -> how to actually SEND the
    credential. The type alone can't build a request: `apiKey` says nothing until you also know
    `in: header` and `name: api_key`. Also keeps the OAuth token/authorization URLs, which are the
    only documented answer to "where do I get a token"."""
    raw = ((spec.get("components") or {}).get("securitySchemes")
           or spec.get("securityDefinitions") or {})
    out: dict[str, dict] = {}
    for name, s in (raw or {}).items():
        s = _deref(s) or {}
        if not isinstance(s, dict):
            continue
        flows = s.get("flows") or {}
        token_url = next((f.get("tokenUrl") for f in flows.values()
                          if isinstance(f, dict) and f.get("tokenUrl")), s.get("tokenUrl", ""))
        auth_url = next((f.get("authorizationUrl") for f in flows.values()
                         if isinstance(f, dict) and f.get("authorizationUrl")),
                        s.get("authorizationUrl", ""))
        out[name] = {
            "type": s.get("type", ""),
            "scheme": (s.get("scheme") or "").lower(),   # http type: bearer / basic
            "in": s.get("in", ""),                       # apiKey: header / query / cookie
            "param": s.get("name", ""),                  # apiKey: the header/query name
            "token_url": token_url or "",
            "authorization_url": auth_url or "",
            "desc": _plain(s.get("description") or "", 160),
        }
    return out

def credential_options(scheme_names: list[str]) -> list[dict]:
    """Describe each documented authentication alternative without guessing.

    The returned placement is directly usable by request builders. Unsupported schemes remain in
    the result with ``supported=False`` so the UI can explain why it refuses to send.
    """
    options = []
    for name in scheme_names:
        scheme = AUTH_SCHEMES.get(name) or {}
        auth_type = scheme.get("type", "")
        http_scheme = scheme.get("scheme", "")
        where = scheme.get("in", "")
        param = scheme.get("param", "")
        lowered_name = name.lower()
        option = {
            "name": name,
            "supported": True,
            "kind": "",
            "parameter": "",
            "template": "",
            "label": name,
            "input_label": "API credential",
            "help": "",
            "token_url": scheme.get("token_url", ""),
            "authorization_url": scheme.get("authorization_url", ""),
        }
        if auth_type == "apiKey" and where in ("header", "query", "cookie") and param:
            option.update(
                kind=where,
                parameter=param,
                template="{{TOKEN}}",
                label=f"API key ({param} in {where})",
                input_label=f"API key for `{param}`",
                help=f"The spec sends this value in the `{param}` {where}.",
            )
        elif auth_type == "oauth2":
            option.update(
                kind="header",
                parameter="Authorization",
                template="Bearer {{TOKEN}}",
                label="OAuth 2.0 access token",
                input_label="OAuth 2.0 access token",
                help="Paste an issued access token, not a client secret or authorization code.",
            )
        elif auth_type == "openIdConnect":
            option.update(
                kind="header",
                parameter="Authorization",
                template="Bearer {{TOKEN}}",
                label="OpenID Connect access token",
                input_label="OpenID Connect access token",
                help="Paste an issued bearer access token.",
            )
        elif auth_type == "http" and http_scheme == "bearer":
            is_pat = "personal" in lowered_name and "token" in lowered_name
            option.update(
                kind="header",
                parameter="Authorization",
                template="Bearer {{TOKEN}}",
                label="Personal access token" if is_pat else "Bearer access token",
                input_label="Personal access token (PAT)" if is_pat else "Bearer access token",
                help="The value is sent in the Authorization header as a Bearer token.",
            )
        elif auth_type == "http" and http_scheme == "basic":
            option.update(
                kind="header",
                parameter="Authorization",
                template="Basic {{TOKEN}}",
                label="HTTP Basic credential",
                input_label="HTTP Basic credential",
                help="Paste the Base64-encoded `username:password` value.",
            )
        else:
            detail = f"{auth_type or 'unknown'}"
            if http_scheme:
                detail += f"/{http_scheme}"
            option.update(
                supported=False,
                label=f"{name} (unsupported: {detail})",
                help="This authentication scheme cannot be safely constructed by the call builder.",
            )
        options.append(option)
    return options


def _credential_placement(scheme_names: list[str]) -> tuple[str, str, str]:
    """Return the first supported documented placement; never invent a fallback."""
    option = next(
        (item for item in credential_options(scheme_names) if item["supported"]),
        None,
    )
    if option is None:
        named = ", ".join(scheme_names) or "none"
        raise ValueError(
            f"No supported credential placement is documented (schemes: {named})."
        )
    return option["kind"], option["parameter"], option["template"]

def auth_instructions(method: str, path: str) -> str:
    """One line telling the user exactly how to authenticate THIS endpoint, and where to get the
    token. Built from securitySchemes, not from a hardcoded assumption about Bearer."""
    names = endpoint_auth_schemes(method, path)
    if not names:
        return ""
    rendered = []
    for option in credential_options(names):
        if not option["supported"]:
            rendered.append(option["label"])
            continue
        shown = option["template"].replace("{{TOKEN}}", "<credential>")
        if option["kind"] == "header":
            placement = f"header `{option['parameter']}: {shown}`"
        else:
            placement = f"{option['kind']} `{option['parameter']}=<credential>`"
        rendered.append(f"{option['label']} via {placement}")
    return "Accepted authentication: " + "; or ".join(rendered) + "."

def _doc_scopes(text: str) -> list[str]:
    """Scopes a spec documents in PROSE instead of in `security[]`. Calendly writes
    "> #### Required scopes: `availability:read`" into every operation's description while leaving
    security[] as a bare {oauth2: []} - the machine-readable field says nothing, the human-readable
    one says everything, and all 56 of its endpoints do it this way. Also matches Zoom's
    "**Scope:** `meeting:write:admin`" form. Only ever used as a FALLBACK (see _op_security), which
    is what keeps a loose match on the generic word "scope" from polluting a spec that declares
    its scopes properly."""
    out: list[str] = []
    for m in re.finditer(r"(?:required\s+)?scopes?\s*:?\*{0,2}\s*((?:\s*`[^`]+`[,/&+]?\s*)+)",
                         text or "", re.I):
        for s in re.findall(r"`([^`]+)`", m.group(1)):
            s = s.strip()
            if s and s not in out:
                out.append(s)
    return out

def _op_security(op: dict, root_security=None) -> tuple[list[str], list[str]]:
    """(granular scopes, scheme names) an operation requires, from its OpenAPI `security`
    requirements (each maps a scheme name -> list of scopes), falling back to the document-level
    `security` when the operation declares none. Deterministic; answers 'what does this endpoint
    need to authenticate' from real spec data instead of the model guessing.

    Both halves matter because specs differ: Zoom lists real per-operation scopes, while Calendly
    lists `{oauth2: []}` - a scheme with no granular scopes. Returning only scopes would report
    Calendly's endpoints as 'no auth documented', which is worse than wrong: it's plausible."""
    reqs = op.get("security")
    if reqs is None:
        reqs = root_security or []
    scopes: list[str] = []
    schemes: list[str] = []
    for req in reqs or []:
        for name, sc in (req or {}).items():
            if name not in schemes:
                schemes.append(name)
            for s in sc or []:
                if s not in scopes:
                    scopes.append(s)
    if not scopes:   # fall back to prose only when the structured field gave us nothing
        scopes = _doc_scopes(op.get("description") or "")
    return scopes, schemes


def _parameter_any_of_groups(description: str, params: list[dict]) -> list[dict]:
    """Extract simple prose-only "either X or Y is required" parameter constraints.

    OpenAPI marks each member optional because neither one is individually mandatory, leaving the
    cross-field rule in operation prose. Keep only names that are real parameters on this operation.
    """
    known = {param["name"]: param.get("in", "") for param in params}
    groups = []
    pattern = re.compile(
        r"\bEither\s+[`'\"]?([A-Za-z_][\w.-]*)[`'\"]?\s+or\s+"
        r"[`'\"]?([A-Za-z_][\w.-]*)[`'\"]?\s+(?:is|are)\s+required\b",
        re.I,
    )
    for match in pattern.finditer(description or ""):
        names = [match.group(1), match.group(2)]
        if not all(name in known for name in names):
            continue
        locations = {known[name] for name in names}
        group = {
            "names": names,
            "in": locations.pop() if len(locations) == 1 else "",
        }
        if group not in groups:
            groups.append(group)
    return groups

def _example_from_schema(schema: dict, _depth: int = 0) -> object:
    """Synthesize a grounded example value from a JSON Schema, preferring the spec's own per-field
    `example`/`default`/`enum` values and falling back to type placeholders. Deterministic, no LLM."""
    schema = _deref(schema)
    if not isinstance(schema, dict) or _depth > 6:
        return None
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    if schema.get("allOf"):                       # merge composed object fragments
        merged: dict = {}
        for sub in schema["allOf"]:
            val = _example_from_schema(sub, _depth + 1)
            if isinstance(val, dict):
                merged.update(val)
        if merged:
            return merged
    for key in ("oneOf", "anyOf"):                # pick the first alternative
        if schema.get(key):
            return _example_from_schema(schema[key][0], _depth + 1)
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        return {name: _example_from_schema(sub, _depth + 1)
                for name, sub in (schema.get("properties") or {}).items()}
    if t == "array":
        item = _example_from_schema(schema.get("items", {}), _depth + 1)
        return [item] if item is not None else []
    if schema.get("enum"):
        return schema["enum"][0]
    return {"string": "string", "integer": 0, "number": 0, "boolean": False}.get(t)

# Char budgets for a rendered schema tree. Sized against real specs: across Calendly's 61 endpoints
# a FULL expansion is 595 chars at the median and 2406 at the worst, so a single-endpoint answer
# almost never truncates; the per-candidate cap is what keeps a 5-hit discovery listing bounded.
TREE_BUDGET_SINGLE = 4000    # one endpoint in focus — room for the whole tree
TREE_BUDGET_MULTI = 900      # per endpoint in a multi-candidate listing
TREE_MAX_DEPTH = 12          # deepest real nesting seen is 6; this only stops pathological specs

def _schema_tree(schema: dict, budget: int, max_depth: int = TREE_MAX_DEPTH) -> tuple[list[str], dict]:
    """Render a schema as an indented `field (type)` tree, ALL levels deep — not one layer.

    A one-level view is what made the agent answer "the exact fields are not specified" about a
    `collection` array whose item schema was sitting right there in the spec. Nested data that
    exists must be shown, so this recurses until the schema ends.

    Two things stop it running away, and BOTH are reported rather than applied silently (returned in
    `meta`, rendered by the caller): a char budget, and a cycle guard. The cycle guard tracks $ref
    NAMES along the current branch — `_deref` only breaks ref->ref chains within a single call, so a
    schema that legitimately contains itself (User -> Organization -> User) would otherwise recurse
    forever. Depth is capped far above any real spec purely as a backstop."""
    lines: list[str] = []
    meta = {"cut_budget": False, "cut_depth": False, "cycles": [], "shown": 0, "total": 0}

    def walk(node, depth: int, label: str, chain: tuple, required: bool = False):
        ref = node.get("$ref", "") if isinstance(node, dict) else ""
        name = ref.rsplit("/", 1)[-1] if ref.startswith("#/") else ""
        node = _deref(node)
        if not isinstance(node, dict):
            return
        if depth > max_depth:
            meta["cut_depth"] = True
            return
        if name and name in chain:          # self-referential schema — show it, don't follow it
            meta["total"] += 1
            if name not in meta["cycles"]:
                meta["cycles"].append(name)
            _emit(depth, f"{label} (recurses into {name} — not expanded again)")
            return
        chain = chain + (name,) if name else chain

        for key in ("allOf", "oneOf", "anyOf"):     # composed schemas: walk the merged/first shape
            if node.get(key):
                if key == "allOf":
                    merged: dict = {"type": "object", "properties": {}}
                    for sub in node[key]:
                        merged["properties"].update(_object_properties(sub))
                    node = merged
                else:
                    node = _deref(node[key][0]) or {}
                break

        t = node.get("type") or ("object" if node.get("properties") else "?")
        if label:
            meta["total"] += 1
            bits = []
            if node.get("enum"):
                bits.append("one of: " + ", ".join(str(v) for v in node["enum"][:4]))
            elif node.get("format"):
                bits.append(f"[{node['format']}]")
            if node.get("readOnly"):
                bits.append("read-only, do NOT send in a request")
            if node.get("writeOnly"):
                bits.append("write-only, never returned")
            extra = (" " + " ".join(bits)) if bits else ""
            _emit(depth, f"{label} ({t}{', required' if required else ''}){extra}")

        if t == "array":
            items = node.get("items") or {}
            if items:
                walk(items, depth, f"{label}[]" if label else "[]", chain)
        # Required is per-OBJECT, so a nested object's own required list only shows up here.
        # request_required() reports the TOP level only; without this, "invitee is required" hides
        # that `invitee` itself requires `email` and `timezone`, and the call 422s.
        req_here = set(_object_required(node))
        for prop, sub in (node.get("properties") or {}).items():
            walk(sub, depth + 1, prop, chain, prop in req_here)

    def _emit(depth: int, text: str):
        if meta["cut_budget"]:
            return
        line = "  " * max(depth, 0) + text
        if sum(len(x) + 1 for x in lines) + len(line) > budget:
            meta["cut_budget"] = True
            return
        lines.append(line)
        meta["shown"] += 1

    walk(schema or {}, 0, "", ())
    return lines, meta

def _tree_block(schema: dict, budget: int, what: str) -> str:
    """A rendered tree plus, when anything was withheld, a LOUD marker saying so. Silent truncation
    is the failure mode this whole project keeps tripping over: the model reads a short tree as a
    complete one and reports a smaller API than exists. The marker is phrased for the model to
    RELAY — the user needs to know to go read the vendor's docs for the rest."""
    lines, meta = _schema_tree(schema, budget)
    if not lines:
        return ""
    out = "\n".join(lines)
    if meta["cut_budget"] or meta["cut_depth"]:
        hidden = max(meta["total"] - meta["shown"], 0)
        out += (f"\n[!] TRUNCATED {what}: showing {meta['shown']} of {meta['total']}+ fields"
                f"{f' ({hidden}+ hidden)' if hidden else ''} — this is NOT the full schema. "
                f"Tell the user it was cut to fit the model's context and that the remaining fields "
                f"are in the API's own documentation. Do NOT claim these are all the fields.")
    if meta["cycles"]:
        out += (f"\n[note] self-referential schema(s) not re-expanded: {', '.join(meta['cycles'])}")
    return out

# qwen2.5:7b runs at num_ctx=8192 TOKENS, covering prompt + generation. At ~3.5 chars/token this
# leaves room for a ~5k-token prompt and still ~3k tokens to answer in. Anything past this is
# silently dropped by the runtime — the model would answer from a prompt whose tail it never saw.
MODEL_PROMPT_CHARS = 18000

def _fit_prompt(prompt: str, limit: int = MODEL_PROMPT_CHARS) -> str:
    """Last line of defence before any llm.invoke: measure the assembled prompt and, if it exceeds
    what the context window can hold, cut it and SAY SO — in the prompt (so the model reports it)
    and on stdout (so the user learns it even if the 7B forgets to mention it). Overflow that isn't
    announced is the worst case: the model answers confidently from a prompt it only half received."""
    if len(prompt) <= limit:
        return prompt
    print(f"  [!] context limit: the grounded prompt was {len(prompt)} chars, cut to {limit}. "
          f"The answer below is based on PARTIAL spec data — check the API's own docs for the rest.")
    return (prompt[:limit] +
            "\n\n[!] TRUNCATED CONTEXT: the endpoint data above was cut off to fit the model's "
            "context window, so it is INCOMPLETE. Answer only from what survived, state clearly "
            "that the spec data was truncated, and tell the user to consult the API documentation "
            "for the remaining fields.")

def response_tree(method: str, path: str, budget: int = TREE_BUDGET_SINGLE) -> str:
    """Fully-nested response shape for an endpoint, budget-capped and labeled if cut."""
    info = ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}
    return _tree_block(info.get("response") or {}, budget, "response schema")

def request_tree(method: str, path: str, budget: int = TREE_BUDGET_SINGLE) -> str:
    """Fully-nested request-body shape for an endpoint, budget-capped and labeled if cut."""
    info = ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}
    return _tree_block(info.get("request") or {}, budget, "request schema")

def endpoint_examples(method: str, path: str) -> dict:
    """Synthesized request/response examples for an endpoint from its stored schemas, or None each
    if the spec documented no schema. Read from ENDPOINT_SCHEMAS (populated by parse_openapi)."""
    info = ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}
    req = _example_from_schema(info["request"]) if info.get("request") else None
    resp = _example_from_schema(info["response"]) if info.get("response") else None
    return {"request": req, "response": resp}

# Fresh-discovery phrasings. These always win over any follow-up cue below, so a genuine new search
# is never captured as a question about the endpoint that happens to be in focus.
FRESH_SEARCH = ("which endpoint", "what endpoint", "an endpoint", "any endpoint", "is there",
                "are there", "endpoint that", "endpoint to", "endpoint for", "how do i",
                "how can i", "how would i", "i want to", "i need to", "i'd like to")
# Explicit back-references. Deliberately endpoint-REFERENTIAL — bare "this"/"that" also modify
# ordinary nouns ("that recording"), which is why they aren't here.
BACK_REFERENCE = ("this endpoint", "that endpoint", "the endpoint", "this action", "that action",
                  "this one", "that one", "this call", "that call", "same endpoint",
                  "the response", "the payload", "the request body", "the schema", "the fields",
                  "look like", "of it", "for it", "its ")
# Metadata an endpoint HAS. A short subjectless question about one of these can only be about the
# endpoint already in focus — there is nothing else in the conversation for it to attach to.
ENDPOINT_ATTRS = ("scope", "auth", "token", "permission", "credential", "rate limit", "parameter",
                  "param", "required", "optional", "response", "payload", "request body",
                  "schema", "header", "query string")
# Asks about a field NESTED inside the focus endpoint's schema ("what's inside `collection`").
# Paired with an exact field-name match below — never used alone, because these words also appear
# in ordinary searches ("endpoints under /users").
NESTING_PHRASES = ("inside", "within", "in the", "under", "nested", "sub-field", "subfield",
                   "sub field", "expand", "drill", "break down", "contains", "contain",
                   "made up of", "what is in", "what's in", "fields of", "elements of")

def _schema_field_names(method: str, path: str) -> set[str]:
    """Every field name at ANY depth in an endpoint's request/response schema. Used to tell a
    question about a nested field of the endpoint in focus ("what's inside collection") from a
    fresh search that happens to share a word. Derived from the same walker that renders the tree,
    so it inherits the cycle guard rather than re-implementing it."""
    info = ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}
    names: set[str] = set()
    for key in ("response", "request"):
        lines, _ = _schema_tree(info.get(key) or {}, budget=200_000)
        for line in lines:
            label = line.strip().split(" (")[0].replace("[]", "")
            if label:
                names.add(label.lower())
    return names

def _is_followup(user_text: str, focus: "Endpoint | None" = None) -> bool:
    """True when a message that names no endpoint is asking about the one already in focus, so the
    REPL answers from `last_endpoint` instead of running a fresh (probably irrelevant) search.

    Two ways to qualify. An explicit back-reference ("payload for THIS endpoint"), or — the case
    that bit us — a short subjectless ATTRIBUTE question with no back-reference at all: "what scope
    is needed", asked right after settling on an endpoint. That used to fall through to RAG, which
    matched the word "scope" against endpoints taking a `scope` QUERY PARAM and answered about four
    unrelated endpoints.

    The attribute route is gated on an attribute noun rather than on mere subjectlessness: a tersely
    typed NEW search ("cancel an event") carries no attribute noun and must still reach discovery.
    The length cap and the plural-"endpoints" exclusion keep listing asks ("show me endpoints with a
    required param") out too. Fresh-discovery phrasing overrides both routes.

    A third route, when `focus` is given: a NESTING phrase plus a word that is an actual field name
    in the focus endpoint's schema ("what's inside `collection`"). Both conditions are required.
    Field names are ordinary words - `user`, `email`, `status` - so matching on the name alone would
    hijack "which endpoint returns a user email" away from discovery. Matching is on whole tokens,
    never substrings, for the same reason `_named_endpoint` is exact-only: a fuzzy hit must not
    masquerade as the user naming something."""
    lower = user_text.lower()
    if any(s in lower for s in FRESH_SEARCH):
        return False
    if any(p in lower for p in BACK_REFERENCE):
        return True
    if focus is not None and any(p in lower for p in NESTING_PHRASES):
        tokens = {t for t in re.split(r"[^a-z0-9_]+", lower) if t}
        if tokens & _schema_field_names(focus.method, focus.path):
            return True
    return (len(user_text.split()) <= 12 and "endpoints" not in lower
            and any(a in lower for a in ENDPOINT_ATTRS))

def format_example(e: Endpoint, query: str) -> str:
    """Return a grounded request/response example for endpoint `e` from its stored schema, honoring
    which DIRECTION the query asked about ("payload"/"send" -> request only, "response"/"returns" ->
    response only, otherwise both). If the spec has no schema for the asked direction (e.g. a GET has
    no request body), say so and point at the direction it DOES document. Shared by the named-endpoint
    example route and the anaphoric follow-up route, including the Streamlit UI."""
    lower = query.lower()
    ex = endpoint_examples(e.method, e.path)
    params = endpoint_params(e.method, e.path)
    want_req = any(k in lower for k in ("payload", "request body", "request", "send", "post body",
                                        "what to send", "param", "input", "query string"))
    want_resp = any(k in lower for k in ("response", "returns", "return", "comes back",
                                         "get back", "output", "result"))
    if want_req and not want_resp:
        show_req, show_resp = True, False
    elif want_resp and not want_req:
        show_req, show_resp = False, True
    else:
        show_req, show_resp = True, True
    # "request input" = parameters (query/path) OR a request body; a GET carries params, not a body.
    req_has = ex["request"] is not None or bool(params)
    if (show_req and req_has) or (show_resp and ex["response"] is not None):
        lines = ["From the onboarded spec's schema (no live call needed):", "",
                 f"{e.method} {e.path}"]
        if e.description:
            lines.append(f"Documented purpose: {e.description}")
        notes = (ENDPOINT_SCHEMAS.get((e.method.upper(), e.path)) or {}).get("notes")
        if notes:
            lines.append(f"Documented behavior: {notes}")
        if show_req and params:
            lines.append("parameters:")
            for p in params:
                lines.append(_param_line(p))   # same rendering the grounded prompts see
        if show_req and ex["request"] is not None:
            lines += ["request body (example):", "```json",
                      _truncate(json.dumps(ex["request"], indent=2), 4000, "example"), "```"]
        if show_resp and ex["response"] is not None:
            lines += ["response (example):", "```json",
                      _truncate(json.dumps(ex["response"], indent=2), 4000, "example"), "```"]
        return "\n".join(lines)
    want = ("request input" if show_req and not show_resp
            else "response schema" if show_resp and not show_req else "example/schema")
    lines = [f"The spec documents no {want} for {e.method} {e.path}."]
    other = ("response schema" if ex["response"] is not None
             else "request input" if (ex["request"] is not None or params) else None)
    if other:
        lines.append(f"It does document a {other}. Ask for that, or open the API Call tab "
                     "to make a live request.")
    else:
        lines.append("Use the API Call tab if you want to make a live request.")
    return "\n".join(lines)

def emit_example(e: Endpoint, query: str) -> None:
    """Terminal renderer for :func:`format_example`."""
    print(format_example(e, query))

def _object_properties(schema: dict, _depth: int = 0) -> dict:
    """The top-level property map of an object schema, merging allOf fragments and unwrapping the
    first oneOf/anyOf. Returns {} if the schema isn't object-shaped."""
    schema = _deref(schema)
    if not isinstance(schema, dict) or _depth > 5:
        return {}
    if "properties" in schema:
        return schema["properties"]
    if schema.get("allOf"):
        merged: dict = {}
        for sub in schema["allOf"]:
            merged.update(_object_properties(sub, _depth + 1))
        return merged
    for key in ("oneOf", "anyOf"):
        if schema.get(key):
            return _object_properties(schema[key][0], _depth + 1)
    return {}

def _object_required(schema: dict, _depth: int = 0) -> list[str]:
    """Top-level required field names of an object schema, merging allOf and the first oneOf/anyOf
    (mirrors _object_properties). [] if the schema marks nothing required."""
    schema = _deref(schema)
    if not isinstance(schema, dict) or _depth > 5:
        return []
    req = list(schema.get("required") or [])
    for sub in schema.get("allOf") or []:
        req += _object_required(sub, _depth + 1)
    for key in ("oneOf", "anyOf"):
        if schema.get(key):
            req += _object_required(schema[key][0], _depth + 1)
    seen, out = set(), []
    for r in req:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

def _coerce(val: str, t: str):
    """Coerce a user's string input to the JSON type the schema declares, so a request body carries
    a real int/number/bool rather than a quoted string. Falls back to the raw string on any mismatch."""
    try:
        if t == "integer":
            return int(val)
        if t == "number":
            return float(val)
        if t == "boolean":
            return val.strip().lower() in ("true", "1", "yes", "y")
    except ValueError:
        pass
    return val

def endpoint_scopes(method: str, path: str) -> list[str]:
    """OAuth scopes an endpoint requires (from the spec's `security`), or [] if none documented."""
    return (ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}).get("scopes") or []

def endpoint_auth_schemes(method: str, path: str) -> list[str]:
    """Security scheme names an endpoint accepts (oauth2, personal_access_token, ...). Any ONE of
    them authenticates the call - OpenAPI treats the `security` array as alternatives. Useful when
    the spec names schemes but no granular scopes, where endpoint_scopes() alone returns []."""
    return (ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}).get("auth_schemes") or []

def endpoint_params(method: str, path: str) -> list[dict]:
    """Path/query/header parameters an endpoint accepts (from the spec), or [] if none. This is the
    input GET endpoints take instead of a request body (page_size, from/to, filters, ...)."""
    return (ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}).get("params") or []

def response_fields(method: str, path: str, limit: int = 20) -> list[str]:
    """Compact 'name (type)' list of an endpoint's top-level response fields — enough to ground the
    model on what a call actually returns, without dumping the (huge) full schema into context."""
    info = ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}
    props = _object_properties(info.get("response") or {})
    out = []
    for name, sub in list(props.items())[:limit]:
        sub = _deref(sub) or {}   # a property can itself be a $ref; else every type reads as '?'
        t = sub.get("type") or ("object" if sub.get("properties") else "?")
        out.append(f"{name} ({t})")
    return out

def request_required(method: str, path: str) -> list[str] | None:
    """Top-level request-body fields the spec marks required. `None` means the endpoint takes no
    body at all — deliberately distinct from `[]` ("takes a body, but marks nothing required", very
    common in this spec). Collapsing the two would let the model report an absent body as
    'nothing is required', or invent requirements for one that documents none."""
    schema = (ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}).get("request")
    return _object_required(schema) if schema else None

def _param_line(p: dict) -> str:
    """One fully-described parameter: name, location, type, whether it's required, the spec's own
    description and example. Every optional param says 'optional' out loud — encoding optional as
    the *absence* of a marker reads to a small model as 'listed, therefore needed'."""
    bits = [p["in"], p["type"] or "?", "required" if p["required"] else "optional"]
    if parameter_format(p):
        bits.append(f"format={parameter_format(p)}")
    if p.get("deprecated"):
        bits.append("DEPRECATED")
    if p.get("default") is not None:
        bits.append(f"default={p['default']}")
    if p.get("enum"):
        bits.append("one of: " + ", ".join(str(v) for v in p["enum"][:6]))
    # Constraints the server enforces; omitting them turns a documented 422 into a surprise.
    for k, v in (p.get("limits") or {}).items():
        bits.append(f"{k}={v}")
    if p.get("explode") is not None:
        bits.append(f"repeat the param: ?{p['name']}=a&{p['name']}=b" if p["explode"]
                    else f"comma-separated: ?{p['name']}=a,b")
    head = f"  - {p['name']} ({', '.join(bits)})"
    desc = f": {p['desc']}" if p.get("desc") else ""
    ex = f" [example: {p['example']}]" if p.get("example") is not None else ""
    return head + desc + ex

def endpoint_listing(
    endpoints,
    field_limit: int = 15,
    param_detail: bool = False,
    include_notes: bool = False,
    include_response_tree: bool = True,
    include_request_tree: bool = False,
) -> str:
    """One grounded block per endpoint: method, path, description, documented top-level response
    fields, parameters, required body fields and OAuth scopes — so the describe/discovery routes can
    answer 'what does it return' / 'is this required' / 'what scopes' from real spec data instead of
    the model inventing them. `param_detail` expands each parameter onto its own line with the
    spec's description and example; use it when the answer is about ONE endpoint, and leave it off
    for multi-endpoint listings where that much text would crowd the 8k context."""
    lines = []
    for e in endpoints:
        rf = response_fields(e.method, e.path, limit=field_limit)
        tail = f"  [response fields: {', '.join(rf)}]" if rf else ""
        # The flat line above is only the TOP level. Nested objects/arrays carry the fields users
        # actually ask about ("what's inside `collection`"), so the full tree goes in too — a
        # bigger budget when one endpoint is in focus, a tight per-endpoint one when several are.
        tree = (response_tree(e.method, e.path,
                              TREE_BUDGET_SINGLE if param_detail else TREE_BUDGET_MULTI)
                if include_response_tree else "")
        params = endpoint_params(e.method, e.path)
        pnames = [f"{p['name']} ({p['in']}, {'required' if p['required'] else 'optional'})"
                  for p in params]
        param_tail = f"  [params: {', '.join(pnames)}]" if pnames else ""
        info = ENDPOINT_SCHEMAS.get((e.method.upper(), e.path)) or {}
        br = request_required(e.method, e.path)
        ctype = info.get("content_type") or ""
        # A non-JSON body is a call-breaker the schema alone doesn't reveal: we render JSON examples
        # and send with `json=`, so an upload endpoint must say what it really expects.
        ctype_tail = (f" as {ctype}" if ctype and ctype != "application/json"
                      else "")
        mand = "mandatory" if info.get("body_required") else "optional"
        if br:
            body_tail = f"  [required body fields ({mand} body{ctype_tail}): {', '.join(br)}]"
        elif br is not None:
            body_tail = (f"  [request body: {mand}{ctype_tail}, but the spec marks no field "
                         f"required]")
        else:
            body_tail = "  [no request body]"
        if ctype and ctype != "application/json":
            body_tail += (f"  [!] this endpoint expects {ctype}, NOT JSON - a JSON body will be "
                          f"rejected on content type")
        if info.get("deprecated"):
            body_tail = "  [!] DEPRECATED endpoint - the spec marks it for removal" + body_tail
        sc = endpoint_scopes(e.method, e.path)
        auth = endpoint_auth_schemes(e.method, e.path)
        how = auth_instructions(e.method, e.path)
        auth_tail = f"  [how to authenticate: {how}]" if how else ""
        if sc:
            scope_tail = f"  [oauth scopes: {', '.join(sc)}]{auth_tail}"
        elif auth:
            # Scheme names but no granular scopes - say which auth works rather than "none documented".
            scope_tail = (f"  [auth: {' or '.join(auth)} (the spec lists no granular OAuth scopes "
                          f"for this endpoint)]")
        else:
            scope_tail = ""
        lines.append(f"{e.method} {e.path} - {e.description}{tail}{param_tail}{body_tail}{scope_tail}")
        if tree:
            lines.append("full response shape (all nested levels):")
            lines.append(tree)
        if param_detail or include_request_tree:
            rtree = request_tree(
                e.method,
                e.path,
                TREE_BUDGET_SINGLE if param_detail else TREE_BUDGET_MULTI,
            )
            if rtree:
                lines.append("full request-body shape (all nested levels):")
                lines.append(rtree)
        if param_detail or include_notes:
            notes = (ENDPOINT_SCHEMAS.get((e.method.upper(), e.path)) or {}).get("notes")
            if notes:
                # Multi-candidate discovery needs operation semantics ("creates a new booking") but
                # cannot afford eight full prose blocks in an 8k context. Single-endpoint detail keeps
                # the existing full captured note; discovery gets a compact, explicitly documented one.
                shown_notes = notes if param_detail or len(notes) <= 260 else (
                    notes[:260].rstrip() + " ... [endpoint notes truncated]"
                )
                lines.append(f"notes from the docs: {shown_notes}")
            if params:
                lines.append("parameters:")
                lines.extend(_param_line(p) for p in params)
    return "\n".join(lines)

# "POST /scheduled_events", "**GET** `/users/me`", "POST https://api.calendly.com/invitees" - the
# forms a model actually writes an endpoint in. Method and path must be adjacent; prose like
# "use GET on the events endpoint" deliberately does not match.
_MENTION_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b[\s`*_]{1,4}"
    r"((?:https?://[^\s`*)\],]+)?/[A-Za-z0-9_\-./{}]*)"
)

def _norm_path(p: str, base_url: str = "") -> str:
    """A path in the form used for comparison: no host, no base path, parameter names collapsed
    ({uuid} and {id} are the same slot), no trailing slash or markdown punctuation."""
    p = (p or "").strip().strip("`*_\"'")
    if p.startswith("http"):
        p = urlparse(p).path
    if base_url:
        bp = urlparse(base_url).path.rstrip("/")
        if bp and p.startswith(bp):
            p = p[len(bp):]
    # Collapse templates BEFORE trimming punctuation: a path legitimately ends in '}', so stripping
    # first turns "/event_types/{id}" into "/event_types/{id" and it stops matching "{uuid}".
    p = re.sub(r"\{[^}]*\}", "{}", p)
    p = p.rstrip(".,;:!?)]`*_\"'")
    return p.rstrip("/") or "/"

def _path_template_matches(template: str, candidate: str) -> bool:
    """True when a concrete candidate path fits a documented template path.

    `GET /event_types/123e...` is a real mention of `GET /event_types/{uuid}`. The output checker
    should flag invented endpoints, not legitimate examples where the model filled a path param.
    """
    template = _norm_path(template)
    candidate = _norm_path(candidate)
    if template == candidate:
        return True
    tparts = [s for s in template.strip("/").split("/") if s]
    cparts = [s for s in candidate.strip("/").split("/") if s]
    if len(tparts) != len(cparts):
        return False
    return all(
        t == c or (
            t == "{}"
            and (_looks_like_id(c) or bool(re.fullmatch(r"(?=.*\d)[A-Za-z0-9_-]{8,}", c)))
        )
        for t, c in zip(tparts, cparts)
    )

def verify_endpoint_mentions(answer: str, spec: "ApiSpec") -> str:
    """Check every endpoint the model NAMED against the spec, and report any that don't exist.

    The grounding prompts forbid inventing endpoints, and a 7B obeys that until it doesn't: given a
    candidate list containing `GET /scheduled_events`, it produced `POST /scheduled_events` - a
    textbook REST completion, inventing a verb because the collection existed - and then attached
    another endpoint's required body fields to it. Retrieval was correct; the model added a fact
    that was never in its context.

    So this is a deterministic check on the OUTPUT, the same shape as make_api_call's DOC_CORPUS
    guard but for prose: no prompt wording can talk it out of the answer, because it isn't asking
    the model. Returns a correction block, or "" when every mention is real."""
    real: dict[str, set[str]] = {}
    for e in spec.endpoints:
        real.setdefault(_norm_path(e.path), set()).add(e.method.upper())

    bad, seen = [], set()
    for method, raw in _MENTION_RE.findall(answer or ""):
        np = _norm_path(raw, spec.base_url or "")
        if np == "/" or (method.upper(), np) in seen:
            continue
        seen.add((method.upper(), np))
        matching_paths = [path for path in real if _path_template_matches(path, np)]
        methods = set().union(*(real[path] for path in matching_paths)) if matching_paths else None
        if methods is None or method.upper() not in methods:
            bad.append((method.upper(), np, methods))
    if not bad:
        return ""

    lines = ["  [!] grounding check: the answer above names endpoint(s) that are NOT in this spec."]
    for method, np, methods in bad:
        if methods:
            lines.append(f"      {method} {np} does not exist - the spec documents only "
                         f"{', '.join(sorted(methods))} on that path.")
        else:
            head = np.split("/")[1] if "/" in np.strip("/") + "/" else ""
            near = sorted({f"{m} {e.path}" for e in spec.endpoints
                           for m in [e.method.upper()]
                           if head and _norm_path(e.path).lstrip("/").startswith(head)})
            lines.append(f"      {method} {np} does not exist." +
                         (f" Real endpoints under /{head}: {', '.join(near[:6])}"
                          if near else " No endpoint with that path is documented."))
    lines.append("      Treat those parts of the answer as unreliable; the rest came from the spec.")
    return "\n".join(lines)

def grounded_answer_text(prompt: str, spec: "ApiSpec") -> tuple[str, str]:
    """Return a grounded answer and any endpoint-verification warning.

    Keeping this separate from terminal rendering lets the Streamlit UI use the exact same
    grounding check without redirecting stdout or scraping console text.
    """
    answer = llm.invoke(_fit_prompt(prompt)).content
    return answer, verify_endpoint_mentions(answer, spec)

def grounded_answer(prompt: str, spec: "ApiSpec") -> None:
    """Run a grounded prompt and print the answer, then verify the endpoints it named. Every
    spec-answering route goes through here so the check can't be forgotten at one of them."""
    answer, warning = grounded_answer_text(prompt, spec)
    print(answer)
    if warning:
        print(warning)

def parse_openapi(spec_source: str) -> ApiSpec:
    """Deterministically turn an OpenAPI/Swagger spec into an ApiSpec. No LLM involved.
    `spec_source` is an http(s) URL or a local file path (JSON or YAML). Side effect: repopulates
    ENDPOINT_SCHEMAS with each endpoint's request/response schema for on-demand example synthesis."""
    global _SPEC_ROOT
    spec = _load_spec(spec_source)
    ENDPOINT_SCHEMAS.clear()
    _SPEC_ROOT = spec if isinstance(spec, dict) else {}   # must be set before any _deref() below

    # base_url — OpenAPI 3 uses servers[]; Swagger 2 uses host + basePath
    servers = spec.get("servers")
    if servers:
        base_url = servers[0].get("url", "")
    else:
        host, base_path = spec.get("host", ""), spec.get("basePath", "")
        scheme = (spec.get("schemes") or ["https"])[0]
        base_url = f"{scheme}://{host}{base_path}" if host else ""

    # servers[].url may be relative (e.g. "/api/v3") — resolve against the spec's own URL.
    # Only meaningful when the spec came from a URL; a local file has no host to resolve against.
    if base_url and not base_url.startswith("http") and spec_source.startswith(("http://", "https://")):
        base_url = urljoin(spec_source, base_url)

    # auth — securitySchemes (3.0) or securityDefinitions (2.0). Keep WHERE the credential goes,
    # not just the type: "apiKey" alone can't be sent, "apiKey in header api_key" can.
    global AUTH_SCHEMES
    AUTH_SCHEMES = _parse_security_schemes(spec)
    described = []
    for name, s in AUTH_SCHEMES.items():
        label = s["scheme"] or s["type"] or name
        if s["type"] == "apiKey" and s["in"] and s["param"]:
            label = f"apiKey in {s['in']} '{s['param']}'"
        described.append(label)
    auth_method = ", ".join(dict.fromkeys(described)) or "None documented"

    # endpoints — paths -> {method: operation}
    HTTP = {"get", "post", "put", "patch", "delete", "head", "options"}
    endpoints = []
    for path, ops in spec.get("paths", {}).items():
        # Path-level parameters apply to every operation on the path; merge them with each op's own.
        shared = ops.get("parameters") if isinstance(ops, dict) else None
        for method, op in ops.items():
            if method.lower() not in HTTP or not isinstance(op, dict):
                continue   # skip 'parameters', '$ref', etc. at path level
            desc = (op.get("summary") or op.get("description") or "").strip()
            endpoints.append(Endpoint(method=method.upper(), path=path, description=desc))
            # When a spec has BOTH, `summary` is the one-liner and `description` is where the real
            # constraints live ("date range can be no greater than 1 week", "does not support
            # keyset pagination"). Endpoint.description keeps the one-liner - it feeds listings and
            # the RAG index, where prose would bloat both - and the prose is kept here for
            # single-endpoint answers.
            prose = (op.get("description") or "").strip()
            notes = _plain(prose, 400) if prose and prose != desc else ""
            params, seen = [], set()
            for p in _params_list(shared) + _params_list(op.get("parameters")):
                key = (p["name"], p["in"])
                if key not in seen:
                    seen.add(key)
                    params.append(p)
            scopes, auth_schemes = _op_security(op, spec.get("security"))
            media = _request_media_type(op)
            if not media and _op_request_schema(op) is not None:
                # Swagger 2.0 falls back to the document-level `consumes`
                doc_consumes = spec.get("consumes") or []
                media = ("application/json" if "application/json" in doc_consumes
                         else (doc_consumes[0] if doc_consumes else ""))
            ENDPOINT_SCHEMAS[(method.upper(), path)] = {
                "request": _op_request_schema(op),
                "response": _op_response_schema(op),
                "scopes": scopes,
                "auth_schemes": auth_schemes,
                # An explicit empty security array means this operation is intentionally public,
                # even when the rest of the API has document-level authentication.
                "security_explicit_none": op.get("security") == [],
                "params": params,
                "param_any_of": _parameter_any_of_groups(prose, params),
                "notes": notes,
                "content_type": media,
                "deprecated": bool(op.get("deprecated", False)),
                # Whether the body itself is mandatory, as opposed to which FIELDS in it are.
                # A body can be required while marking no field required, and vice versa.
                "body_required": bool((_deref(op.get("requestBody") or {}) or {}).get("required")
                                      or any((_deref(p) or {}).get("in") == "body"
                                             and (_deref(p) or {}).get("required")
                                             for p in (op.get("parameters") or []))),
            }

    return ApiSpec(base_url=base_url, auth_method=auth_method, endpoints=endpoints)

def _ref_audit(root) -> Counter:
    """Classify every `$ref` in the spec document: resolved / broken / remote. Cheap single walk of
    the raw document — no schema synthesis — so it can run at onboard time."""
    tally = Counter()
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if not ref.startswith("#/"):
                    tally["remote"] += 1
                elif _deref({"$ref": ref}) == {}:
                    tally["broken"] += 1
                else:
                    tally["resolved"] += 1
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return tally

def spec_coverage(spec: ApiSpec) -> str:
    """Report what the parse actually YIELDED, not just that it succeeded.

    This exists because every extraction bug this project has hit failed *silently*: unresolved
    $refs render as empty objects, $ref parameters get skipped, scopes documented in prose come back
    as []. Nothing raises - the agent just describes a smaller, emptier API than the real one, and
    you only find out several turns later when an answer is confidently wrong. A count printed at
    onboard time makes an anomalous zero visible immediately, and the hints name the usual cause."""
    n = len(spec.endpoints)
    if not n:
        return "  (no endpoints parsed)"
    have = lambda f: sum(1 for e in spec.endpoints if f(e))
    scoped = have(lambda e: endpoint_scopes(e.method, e.path))
    authed = have(lambda e: endpoint_scopes(e.method, e.path) or endpoint_auth_schemes(e.method, e.path))
    parms = have(lambda e: endpoint_params(e.method, e.path))
    resp = have(lambda e: (ENDPOINT_SCHEMAS.get((e.method, e.path)) or {}).get("response"))
    body = have(lambda e: (ENDPOINT_SCHEMAS.get((e.method, e.path)) or {}).get("request"))
    refs = _ref_audit(_SPEC_ROOT)

    scraped = SPEC_PROVENANCE == "scraped"
    lines = [f"  endpoints        {n}" + ("   (scraped from prose - see below)" if scraped else ""),
             f"  auth documented  {authed}/{n}   (granular scopes: {scoped}/{n})",
             f"  parameters       {parms}/{n}",
             f"  request schema   {body}/{n}",
             f"  response schema  {resp}/{n}"]
    if not scraped:   # a scraped spec has no document behind it, so a $ref tally of all zeros is
        lines.append(f"  $refs            {refs['resolved']} resolved, {refs['broken']} broken, "
                     f"{refs['remote']} remote (not followed)")   # vacuous, not reassuring
    warn = []
    if scraped:
        # The zeros above are STRUCTURAL on this path, not a parser gap - saying "the parser may not
        # recognise this prose" would send you hunting for a bug that isn't there, and worse, implies
        # the endpoints themselves are solid and merely under-annotated. They aren't: they're an LLM's
        # reading of a page, with nothing to check them against.
        warn.append("SCRAPED, NOT PARSED - no machine-readable spec was found, so these endpoints "
                    "are an LLM's reading of prose. The scrape path does not extract params, "
                    "schemas or scopes at all, so the zeros above are expected and say nothing "
                    "about coverage. Nothing here is corroborated by a spec document - treat every "
                    "endpoint as a claim to verify before calling it.")
    if not authed and not scraped:
        warn.append("no auth/scopes found - the spec may document them in prose this parser "
                    "doesn't recognise (see _doc_scopes)")
    if not resp and not scraped:
        warn.append("no response schemas - answers about return values will be thin")
    if refs["broken"]:
        warn.append(f"{refs['broken']} $ref(s) point at nothing - those schemas render empty")
    if refs["remote"]:
        warn.append(f"{refs['remote']} $ref(s) point at another file - not followed, so those "
                    "schemas render empty")
    for w in warn:   # wrap — the scraped-spec warning is the most important line in the report and
        wrapped = textwrap.wrap(w, 84)   # must not scroll off the right edge of a terminal
        lines.append(f"  [!] {wrapped[0]}")
        lines += [f"      {cont}" for cont in wrapped[1:]]
    return "\n".join(lines)

def _is_openapi(source: str) -> bool:
    """True if `source` (URL or local file) resolves to a parseable OpenAPI/Swagger spec."""
    try:
        data = _load_spec(source)
    except (requests.RequestException, ValueError, OSError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and ("openapi" in data or "swagger" in data)

def _spec_ancestors(url: str) -> list[str]:
    """Candidate spec URLs formed by truncating `url` at a spec-file segment that has more path
    hanging off it. Doc portals export a PIECE of a spec by appending a pointer to the spec's own
    path - Stoplight's model export is
    `.../calendly-api/openapi.yaml/components/schemas/Event?snapshotType=model`, which serves a bare
    JSON-Schema fragment: no `paths:`, no HTTP verbs, nothing callable. Truncating at `openapi.yaml`
    yields the real 47-path Calendly document. Purely structural, so it covers any portal using the
    same shape, and every candidate is still validated by `_is_openapi` before it's used."""
    parts = urlparse(url)
    segs = parts.path.split("/")
    out = []
    for i, seg in enumerate(segs[:-1]):                       # :-1 — a trailing spec file is not a
        if seg.lower().endswith((".json", ".yaml", ".yml")):  # fragment URL, step 0 already tried it
            trimmed = parts._replace(path="/".join(segs[:i + 1]), query="", fragment="")
            out.append(urlunparse(trimmed))
    return list(reversed(out))                                # innermost (most specific) spec first

def find_openapi_spec(
    docs_url: str,
    progress: Callable[[int, str], None] | None = None,
) -> str | None:
    """Locate an OpenAPI/Swagger spec for a docs site. Returns its URL, or None."""
    report = progress or (lambda _percent, _message: None)
    # 0. The URL itself might already be a spec
    report(10, "Checking whether the supplied source is an OpenAPI document")
    if _is_openapi(docs_url):
        report(55, "The supplied source is a valid OpenAPI document")
        return docs_url

    # 0.5. ...or it may point INTO one (a portal's "export this model" link). Walk up to the spec.
    ancestors = _spec_ancestors(docs_url)
    for index, ancestor in enumerate(ancestors):
        report(15 + int(15 * (index + 1) / max(len(ancestors), 1)),
               "Checking whether the source points inside an OpenAPI document")
        if _is_openapi(ancestor):
            print(f"[ok] That URL points into a spec, not at an endpoint reference - walked up to it.")
            report(55, "Found the parent OpenAPI document")
            return ancestor

    # 1. Links on the docs page that look like a spec
    report(32, "Inspecting documentation links for an OpenAPI document")
    try:
        _, links = fetch_text(docs_url)
    except requests.RequestException:
        links = []
    candidates = [l for l in links
                  if l.lower().endswith((".json", ".yaml", ".yml"))
                  or "openapi" in l.lower() or "swagger" in l.lower()]

    # 2. Common well-known locations on the host
    root = f"{urlparse(docs_url).scheme}://{urlparse(docs_url).netloc}"
    well_known = ["/openapi.json", "/swagger.json", "/v3/api-docs",
                  "/api-docs", "/.well-known/openapi.json"]
    candidates += [root + p for p in well_known]

    # 3. First candidate that actually parses as a spec wins
    candidates = list(dict.fromkeys(candidates))  # dedupe, keep order
    for index, url in enumerate(candidates):
        report(35 + int(20 * (index + 1) / max(len(candidates), 1)),
               f"Validating OpenAPI candidate {index + 1} of {len(candidates)}")
        if _is_openapi(url):
            report(55, "Found a valid OpenAPI document")
            return url
    report(55, "No machine-readable OpenAPI document was found")
    return None

def onboard(
    url: str,
    progress: Callable[[int, str], None] | None = None,
) -> ApiSpec:
    """OpenAPI-first: clean parse if a spec exists, else fall back to the scraper."""
    global _SPEC_ROOT, SPEC_PROVENANCE
    report = progress or (lambda _percent, _message: None)
    report(2, "Starting API onboarding")
    ENDPOINT_SCHEMAS.clear()   # drop any previous spec's schemas; parse_openapi repopulates
    _SPEC_ROOT = {}            # ...and its $ref root, so a scraped fallback can't audit stale refs
    SPEC_PROVENANCE = ""
    spec_url = find_openapi_spec(url, progress=report)
    if spec_url:
        print(f"[ok] Found OpenAPI spec: {spec_url} - using clean parse.")
        SPEC_PROVENANCE = "openapi"
        report(65, "Parsing paths, authentication, parameters, and schemas")
        spec = parse_openapi(spec_url)
        report(92, f"Parsed {len(spec.endpoints)} documented endpoints")
        report(100, "Onboarding complete")
        return spec
    print("[--] No OpenAPI spec found - falling back to doc scraping.")
    SPEC_PROVENANCE = "scraped"
    report(62, "Crawling documentation pages")
    pages = crawl(url)
    report(78, f"Read {len(pages)} documentation page(s)")
    report(84, "Extracting endpoints from documented method/path evidence")
    spec = extract_from_pages(pages)
    report(96, f"Extracted {len(spec.endpoints)} endpoint claim(s) from prose")
    report(100, "Onboarding complete")
    return spec

def _onboard_target(text: str) -> str | None:
    """Deterministic routing: return an onboarding source (URL or local spec path) if the input
    contains one, else None. Purely structural — a token either matches a rule or it doesn't:
      - starts with http:// / https:// / file://           -> that token
      - ends with .json/.yaml/.yml AND contains a path sep  -> that token (a local/remote path,
        not a bare word like 'response.json' sitting in prose)"""
    for tok in text.split():
        t = tok.strip("\"'").rstrip(".,);")   # trailing prose punctuation only — keep leading ./ ../
        if t.startswith(("http://", "https://", "file://")):
            return t
        if (t.lower().endswith((".json", ".yaml", ".yml"))
                and ("/" in t or "\\" in t or re.match(r"[A-Za-z]:", t))):
            return t
    return None

def _points_into_loaded_api(target: str, spec: "ApiSpec | None") -> bool:
    """True if `target` is a URL under the already-loaded spec's base_url — i.e. a reference to an
    existing endpoint (as in 'how does calling https://api.zoom.us/v2/... work'), NOT a new spec to
    onboard. Guards against a question that merely mentions an endpoint URL re-triggering onboarding."""
    base = (spec.base_url or "").rstrip("/") if spec else ""
    return bool(base) and target.rstrip("/").startswith(base)

def _spec_summary(spec: ApiSpec, show: int = 10) -> str:
    """A compact, scroll-friendly view of a spec: base/auth + the first `show` endpoints,
    with an 'and N more' pointer instead of dumping hundreds of lines."""
    lines = [
        f"base_url:  {spec.base_url or '(none found)'}",
        f"auth:      {spec.auth_method or '(none)'}",
        f"endpoints: {len(spec.endpoints)} found"
        + (f" - showing first {min(show, len(spec.endpoints))}:" if spec.endpoints else ""),
    ]
    for ep in spec.endpoints[:show]:
        lines.append(f"  {ep.method:<6} {ep.path}")
    extra = len(spec.endpoints) - show
    if extra > 0:
        lines.append(f"  ... and {extra} more. Type 'extract' to dump the full spec.")
    return "\n".join(lines)

# Enumerate-the-spec requests ("show me the next 10 endpoints", "list all endpoints", "endpoints
# 20-40") — a STRUCTURAL page through the list, NOT a capability search. Kept deliberately narrow so
# a discovery ask ("show me the endpoint that creates a meeting") does NOT match and still hits RAG:
# the list/show branch must END right at "endpoints", so a trailing capability clause ("...that
# creates X") fails to match.
_LIST_RE = re.compile(
    r"\b(?:list|show|see|display|view|give\s+me|what\s+are)\b"
    r"(?:\s+(?:me|all|the|of|other|remaining|rest|available|first|last|\d+))*"
    r"\s+endpoints?\b\s*[?.!]*\s*$"
    r"|\b(?:next|more|other|remaining|another|additional|rest\s+of\s+the)\b(?:\s+\d+)?\s+endpoints?\b"
    r"|\bendpoints?\s+\d+\s*(?:-|to|through)\s*\d+\b",
    re.I)

def _listing_request(text: str, total: int, cursor: int) -> "tuple[int, int] | None":
    """If `text` asks to ENUMERATE the spec's endpoints (not discover one by capability), return the
    (start, end) slice to show; else None. Handles 'next N', 'more', 'first N', 'all', 'endpoints
    20-40'. Discovery asks return None -> they fall through to RAG."""
    if not _LIST_RE.search(text):
        return None
    rng = re.search(r"endpoints?\s+(\d+)\s*(?:-|to|through)\s*(\d+)", text, re.I)
    if rng:
        return (max(0, int(rng.group(1)) - 1), min(total, int(rng.group(2))))
    if re.search(r"\ball\b", text, re.I):
        return (0, total)
    num = re.search(r"\b(\d+)\b", text)
    count = int(num.group(1)) if num else 10
    if re.search(r"\b(next|more|other|remaining|another|additional|rest)\b", text, re.I):
        return (cursor, min(total, cursor + count))
    return (0, min(total, count))        # "first N" / bare "list [the] endpoints" -> from the top

# Verb -> HTTP method(s) for CRUD-style natural language ("update a meeting" -> PATCH/PUT). REST puts
# the verb in the METHOD, not the path, so segment matching alone can't tell create/read/update/delete
# apart; this lets us disambiguate endpoints that share a path and break otherwise-tied keyword scores.
_VERB_METHODS = {
    "create": ["POST"], "add": ["POST"], "new": ["POST"], "schedule": ["POST"], "register": ["POST"],
    "update": ["PATCH", "PUT"], "change": ["PATCH", "PUT"], "modify": ["PATCH", "PUT"],
    "edit": ["PATCH", "PUT"], "set": ["PATCH", "PUT"], "replace": ["PUT"],
    "delete": ["DELETE"], "remove": ["DELETE"], "cancel": ["DELETE"],
    "get": ["GET"], "list": ["GET"], "show": ["GET"], "fetch": ["GET"], "retrieve": ["GET"],
    "read": ["GET"], "view": ["GET"],
}
_CRUD_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")

def _query_methods(query: str) -> tuple[list[str], bool]:
    """Infer the HTTP method(s) a query implies. Returns (methods, explicit): `explicit` is True when
    the user literally typed a method token (GET/POST/...), which callers may HARD-filter on; a
    verb-implied method (from _VERB_METHODS) is only a soft hint and must not hard-exclude."""
    q = _expand_query_aliases(query).lower()
    explicit = [m for m in _CRUD_METHODS if re.search(rf"\b{m.lower()}\b", q)]
    if explicit:
        return explicit, True
    implied: list[str] = []
    for verb, methods in _VERB_METHODS.items():
        if re.search(rf"\b{verb}\b", q):
            implied += [m for m in methods if m not in implied]
    return implied, False

def _exact_path_matches(spec: ApiSpec, query: str) -> list[Endpoint]:
    """Endpoints whose path the user literally TYPED, narrowed by any named/implied method and to the
    most specific path. [] if no path was named. This is the high-confidence 'the user named THIS
    endpoint' signal, shared by _find_endpoints and _named_endpoint so a fuzzy keyword hit can never
    masquerade as an explicit name (which would wrongly pre-empt semantic discovery)."""
    q = _expand_query_aliases(query).lower()
    # Exact path named in the text, e.g. "GET /meetings/{meetingId}/recordings/analytics_summary".
    exact = [e for e in spec.endpoints if e.path.lower() in q]
    if not exact:
        return []
    methods, explicit = _query_methods(query)
    if methods:
        preferred = [e for e in exact if e.method.upper() in methods]
        if preferred:
            exact = preferred          # "patch /meetings/{id}" -> the PATCH, not the GET on that path
        elif explicit:
            return []                  # a method was typed but no matched endpoint has it: not confident
    # Keep only the most specific: drop any matched path that is a prefix of another matched one
    # (naming the long path also substring-matches its parents).
    matched = {e.path.lower() for e in exact}
    specific = [e for e in exact if not any(
        other != e.path.lower() and other.startswith(e.path.lower().rstrip("/") + "/")
        for other in matched)]
    specific.sort(key=lambda e: len(e.path), reverse=True)
    return specific or exact

def _find_endpoints(spec: ApiSpec, query: str, limit: int = 5) -> list[Endpoint]:
    """Look up endpoints in an onboarded spec by the user's text. Deterministic — the source of
    truth is last_spec (all endpoints), NOT the model's context (which only saw the capped 40)."""
    # 1. An exact path named in the text (optionally disambiguated by method) wins outright.
    exact = _exact_path_matches(spec, query)
    if exact:
        return exact[:limit]
    # 2. Otherwise score by significant path segments (ignore {params} and version noise), matching
    #    on substrings so "recording"/"analytics" hit "recordings"/"analytics_summary". A method the
    #    text implies (create->POST, "update"->PATCH/PUT) boosts matching endpoints so the CRUD verb —
    #    which lives in the method, not the path — breaks otherwise-tied segment scores.
    q = _expand_query_aliases(query).lower()
    methods, _ = _query_methods(query)
    qwords = [w for w in re.findall(r"[a-z_]+", q) if len(w) >= 4]
    def sig(path: str) -> list[str]:
        return [s.lower() for s in path.split("/")
                if s and not s.startswith("{") and s.lower() not in SKIP_SEGMENTS]
    scored = []
    for e in spec.endpoints:
        segs = sig(e.path)
        score = sum(1 for w in qwords if any(w in s or s in w for s in segs))
        if score and methods and e.method.upper() in methods:
            score += 2                 # method match breaks segment ties toward the right CRUD verb
        if score:
            # Tiebreak on specificity: among equal scores prefer the MORE GENERAL path (fewer
            # significant segments), so "update a meeting" ranks PATCH /meetings/{id} above a deep
            # PATCH /meetings/{id}/recordings/registrants/questions that also happens to match.
            scored.append((score, -len(segs), e))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [e for _, _, e in scored[:limit]]

def _named_endpoint(spec: ApiSpec, query: str) -> "Endpoint | None":
    """The single endpoint the user explicitly NAMED by path, or None — never a fuzzy keyword guess.
    Routing uses this (not _find_endpoints) to decide 'did the user point at a specific endpoint?',
    so a query that merely mentions a common word ('meeting') doesn't hijack the describe route and
    pre-empt semantic discovery."""
    matches = _exact_path_matches(spec, query)
    return matches[0] if matches else None

def _print_step(msg) -> None:
    """Render one streamed agent step as a progress line."""
    for tc in getattr(msg, "tool_calls", None) or []:
        args = ", ".join(f"{k}={v!r}" for k, v in (tc.get("args") or {}).items())
        print(f"  -> {tc['name']}({args})")
    if getattr(msg, "type", None) == "tool":
        first = (msg.content or "").splitlines()[0] if msg.content else ""
        print(f"  <- {msg.name}: {first[:120]}")

def run_turn(payload, config):
    """Stream the agent (the LangGraph way) so each tool step prints as live progress instead
    of the terminal sitting silent. Returns (final_state, interrupt_value | None)."""
    print("  -> thinking...")
    interrupt_value = None
    for update in agent.stream(payload, config, stream_mode="updates"):
        if "__interrupt__" in update:
            interrupt_value = update["__interrupt__"][0].value
            continue
        for _node, data in update.items():
            if isinstance(data, dict):
                for m in data.get("messages", []):
                    _print_step(m)
    final = agent.get_state(config).values
    return final, interrupt_value

def run_grounded_call(e: Endpoint, base_url: str) -> None:
    """Execute a REAL call to endpoint `e`, collecting every value from the USER, never the model.
    The model's only job was picking the endpoint; here Python asks for each required input (path +
    query params, then required body fields), lets the user add optional body fields as JSON, shows
    the fully assembled request, and fires only on explicit confirmation. This is the grounding
    guarantee for writes - the agent can't invent a meetingId or a required field, because it never
    supplies one. (Cancel any prompt with a blank line.)"""
    from urllib.parse import quote, urlencode
    print(f"  -> preparing a real call to {e.method} {e.path}. I'll ask for the required values.")
    params = endpoint_params(e.method, e.path)
    by_name = {p["name"]: p for p in params}

    # 1. Path params: every {placeholder} in the path is required.
    path = e.path
    for name in re.findall(r"{([^}]+)}", e.path):
        meta = by_name.get(name, {})
        if meta.get("desc"):
            print(f"     ({name}: {meta['desc']})")
        val = input(f"     {name} (path param, {meta.get('type') or 'string'}): ").strip()
        if not val:
            print("     (no value - call cancelled)")
            return
        path = path.replace("{" + name + "}", quote(val, safe=""))

    # 2. Query params: required must be filled; optional can be skipped with a blank line.
    query = {}
    for p in params:
        if p["in"] != "query":
            continue
        tag = "required" if p["required"] else "optional, Enter to skip"
        extra = f", default={p['default']}" if p.get("default") is not None else ""
        if p.get("desc"):
            print(f"     ({p['name']}: {p['desc']})")
        val = input(f"     {p['name']} (query, {p['type'] or '?'}, {tag}{extra}): ").strip()
        if val:
            query[p["name"]] = val
        elif p["required"]:
            print("     (that one is required - call cancelled)")
            return

    # 3. Body: prompt each required top-level field; let the user add optional fields as JSON.
    body = None
    req_schema = (ENDPOINT_SCHEMAS.get((e.method, e.path)) or {}).get("request")
    if req_schema:
        props = _object_properties(req_schema)
        required = _object_required(req_schema)
        body = {}
        for fname in required:
            sub = props.get(fname, {})
            t = sub.get("type", "?")
            if t in ("object", "array") or "properties" in sub:
                print(f"     {fname} (required, {t}) - example: {json.dumps(_example_from_schema(sub))}")
                raw = input(f"     paste JSON for {fname} (Enter to use the example): ").strip()
                try:
                    body[fname] = json.loads(raw) if raw else _example_from_schema(sub)
                except json.JSONDecodeError as ex:
                    print(f"     (invalid JSON: {ex} - call cancelled)")
                    return
            else:
                val = input(f"     {fname} (required, {t}): ").strip()
                if not val:
                    print("     (that one is required - call cancelled)")
                    return
                body[fname] = _coerce(val, t)
        optional = [k for k in props if k not in required]
        if optional:
            preview = ", ".join(optional[:20]) + (" ..." if len(optional) > 20 else "")
            print(f"     optional fields available: {preview}")
            raw = input('     paste a JSON object of optional fields to add (e.g. {"start_time": "..."}), '
                        "or Enter for none: ").strip()
            if raw:
                try:
                    body.update(json.loads(raw))
                except json.JSONDecodeError as ex:
                    print(f"     (ignored - invalid JSON: {ex})")
        if not body:
            body = None   # nothing supplied -> send no body at all

    # 4. Auth placement comes from the SPEC, not from a guess. An `apiKey in query` scheme has to go
    #    into the URL, so this is resolved before the URL is assembled.
    schemes = endpoint_auth_schemes(e.method, e.path)
    kind = cred_name = template = ""
    if schemes:
        print(f"     auth: {auth_instructions(e.method, e.path)}")
        try:
            kind, cred_name, template = _credential_placement(schemes)
        except ValueError as exc:
            print(f"  [!] {exc} Refusing to guess; call cancelled.")
            return
    else:
        print("     (the spec documents no authentication for this endpoint)")

    # 5. Get a token for the host if we don't already have one (never shown to the model).
    probe_host = urlparse(base_url).netloc
    if schemes and probe_host and probe_host not in SECRETS:
        label = (f"{cred_name} header" if kind == "header" and cred_name != "Authorization"
                 else f"{template.split()[0] if ' ' in template else 'API'} token")
        try:
            tok = getpass.getpass(f"     {label} for {probe_host} (Enter to send without auth): ")
        except (EOFError, KeyboardInterrupt):
            print("\n     (call cancelled)")
            return
        if tok.strip():
            SECRETS[probe_host] = tok

    headers = {}
    cookies = {}
    if schemes and probe_host in SECRETS:
        # Placeholder only; the real token is substituted inside _do_http_call, at HTTP time.
        if kind == "query":
            query[cred_name] = "{{TOKEN}}"
        elif kind == "cookie":
            cookies[cred_name] = "{{TOKEN}}"
        else:
            headers[cred_name] = template
    if cookies:
        headers["Cookie"] = "; ".join(f"{key}={value}" for key, value in cookies.items())

    # 6. Assemble the full URL (after any query-placed credential is in `query`).
    if not base_url:
        print("  (no base_url on this spec - can't build an absolute URL; cancelling)")
        return
    url = base_url.rstrip("/") + path + ("?" + urlencode(query) if query else "")
    host = urlparse(url).netloc

    ctype = (ENDPOINT_SCHEMAS.get((e.method, e.path)) or {}).get("content_type") or ""
    if body is not None:
        if ctype and ctype != "application/json":
            # We can only send JSON here. Say so instead of firing a request the API will reject.
            print(f"  [!] this endpoint expects {ctype}, but this flow can only send JSON. "
                  f"The call would be rejected on content type - use the API's own client or curl "
                  f"for {ctype} requests. Cancelling.")
            return
        headers["Content-Type"] = "application/json"

    # 6. Show the assembled request (token stays a placeholder) and require explicit confirmation.
    print("\n  Ready to send this real request:")
    print(f"     {e.method} {url}")
    if headers:
        print(f"     headers: {headers}")
    if body is not None:
        print(f"     body: {json.dumps(body, indent=2)}")
    if input("  Fire it? (y/N): ").strip().lower() not in ("y", "yes"):
        print("  (not sent)")
        return

    # 7. Fire. Grounding is guaranteed (path is spec-derived), so skip the corpus check.
    print(_do_http_call(url, e.method, headers, body, check_grounding=False))

if __name__ == "__main__":
    import sys
    # Docs pages and API response bodies can contain non-ASCII; the default Windows console
    # codec (cp1252) would crash pretty_print/print on them. Force UTF-8 output.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    config = {"configurable": {"thread_id": "session-1"}}   # session identity — reused every turn
    last_spec: ApiSpec | None = None   # most recent onboarding result, for the `extract` command
    last_endpoint: Endpoint | None = None   # endpoint the previous turn settled on, for follow-ups
    list_cursor = 0                    # how far the user has paged through the endpoint list

    print("Agent ready. Type 'end session' to quit.")
    while True:
        try:
            user_text = input("you> ")
        except EOFError:                 # EOF (Ctrl+Z<Enter> on Windows, Ctrl+D on Unix): exit clean
            print()
            break
        except KeyboardInterrupt:        # Ctrl+C at the prompt: clear the line, keep the session
            print("\n(cancelled - back to prompt)")
            continue

        if user_text.strip() == "end session":
            break

        # Per-turn safety net: no single turn (bad input, model/network error, Ctrl+C) should be
        # able to kill the session. onboard() keeps its own inner try/except for a specific message.
        try:
            if user_text.strip() == "extract":
                # Prefer the structured spec from onboarding; fall back to an LLM extraction.
                try:
                    spec = last_spec or extract_api_spec()
                except ValueError:       # nothing fetched yet -> extract_api_spec() raises
                    print("(nothing to extract yet - onboard a docs URL or spec first)")
                    continue
                print(spec.model_dump_json(indent=2))
                continue

            # New onboarding target -> run the OpenAPI-first router (clean parse if a spec exists,
            # else scrape+LLM). Deterministic gate: a URL (http/file) or a local spec path.
            target = _onboard_target(user_text)
            # Bug A: a URL under the already-loaded API's base is a REFERENCE to an existing endpoint
            # (e.g. "how does calling https://api.zoom.us/v2/... work"), not a new spec. Don't onboard
            # it -> fall through to the describe/discovery routes with the loaded spec intact.
            if target and _points_into_loaded_api(target, last_spec):
                target = None
            if target:
                # Bug B: snapshot session state so a failed / empty onboard can't wipe a good spec.
                # onboard() + the clear below mutate globals; restore them if nothing usable comes back.
                _prev = (last_spec, list(DOC_CORPUS), dict(ENDPOINT_SCHEMAS))
                def _restore():
                    DOC_CORPUS[:] = _prev[1]
                    ENDPOINT_SCHEMAS.clear(); ENDPOINT_SCHEMAS.update(_prev[2])
                DOC_CORPUS.clear()   # fresh evidence corpus so an earlier API can't leak in
                try:
                    new_spec = onboard(target)
                except Exception as e:
                    print(f"(onboarding failed: {e} - keeping the current spec)")
                    last_spec = _prev[0]; _restore()
                    continue
                if not new_spec.endpoints:
                    print(f"(no endpoints found at {target} - keeping the current spec)")
                    last_spec = _prev[0]; _restore()
                    continue
                last_spec = new_spec
                last_endpoint = None                     # new API -> old follow-up target is stale
                print("\n=== STRUCTURED API SPEC ===")
                print(_spec_summary(last_spec))          # capped view; 'extract' dumps the full spec
                # What the parse actually yielded. A zero here is the earliest possible warning that
                # the agent is about to describe an emptier API than the one that exists.
                print("\n=== COVERAGE ===")
                print(spec_coverage(last_spec))
                # The summary just showed the first 10; page 'next 10' onward from there.
                list_cursor = min(10, len(last_spec.endpoints))
                # Seed grounding evidence: make_api_call validates real calls against the docs
                # corpus, but onboard() doesn't fetch through fetch_url, so feed it the spec.
                DOC_CORPUS.append(last_spec.model_dump_json())
                # NB: the RAG index is built lazily (on the first discovery question), not here, so
                # onboarding stays instant — see ensure_endpoint_index() and the discovery route.
                # Hand the spec to the agent as ground truth so later "test X" turns can reason over
                # the real endpoints. CAP it: a full 900-endpoint list overflows num_ctx=8192, so the
                # model would only see a truncated fragment and behave erratically. Send a bounded
                # slice plus an explicit note so it doesn't loop trying to fetch the rest.
                HANDOFF_CAP = 40
                shown = last_spec.endpoints[:HANDOFF_CAP]
                ep_lines = "\n".join(f"{e.method} {e.path} - {e.description}" for e in shown)
                more = len(last_spec.endpoints) - len(shown)
                more_note = "" if more <= 0 else (
                    f"\n\nShowing {len(shown)} of {len(last_spec.endpoints)} endpoints - the full list "
                    f"is too large to include. If the user asks about an endpoint not listed here, ask "
                    f"them to name it; do NOT ask to re-send the full list, it will not fit in context."
                )
                handoff = (
                    f"Structured onboarding of {target} is complete. base_url={last_spec.base_url}; "
                    f"auth={last_spec.auth_method}. Treat these endpoints as ground truth - do NOT "
                    f"rediscover them by reading docs:\n{ep_lines}{more_note}\n"
                    f"Reply with a one-line confirmation and wait for my next instruction."
                )
                state, _ = run_turn({"messages": [("user", handoff)]}, config)
                state["messages"][-1].pretty_print()
                continue

            # ---- Grounded routing (spec loaded) ----------------------------------------------------
            # Keep decisions out of the weak tool-agent (bug C). Precedence:
            #   - explicit "test/call" -> the tool-agent (real HTTP), handled by falling through below.
            #   - a specific endpoint named/referenced in the message -> describe it (NO keyword needed;
            #     a path or an under-base_url URL is enough).
            #   - example/payload ask for that endpoint -> synthesize from schema.
            #   - any other request/question -> RAG discovery over the FULL spec, instead of dropping
            #     to the tool-agent with its capped 40-endpoint context and hallucinating.
            if last_spec and last_spec.endpoints:
                lower = user_text.lower()

                # 0. Enumerate/page the spec ("show me the next 10 endpoints") -> deterministic slice,
                #    no LLM. Checked first so a structural ask never leaks into semantic discovery.
                page = _listing_request(user_text, len(last_spec.endpoints), list_cursor)
                if page is not None:
                    start, end = page
                    if start >= len(last_spec.endpoints):
                        print(f"  -> that's past the end - the spec has {len(last_spec.endpoints)} "
                              f"endpoints. Say 'list endpoints' to start over.")
                        continue
                    print(f"  -> endpoints {start + 1}-{end} of {len(last_spec.endpoints)}:")
                    for e in last_spec.endpoints[start:end]:
                        print(f"  {e.method:<6} {e.path}")
                    list_cursor = end
                    if end < len(last_spec.endpoints):
                        print(f"  ... {len(last_spec.endpoints) - end} more. Say 'next 10 endpoints' "
                              f"to continue, or 'extract' for the full spec.")
                    continue

                TEST_INTENT = ("test ", "test the", "call it", "make a call", "make a request",
                               "make the call", "hit the", "hit it", "try it", "run it",
                               "actually call", "live call", "send a request", "fire the")
                EXAMPLE_INTENT = ("example", "sample", "payload", "request body", "response body",
                                  "response look like", "what does the response", "show me the response",
                                  "what fields", "json response", "shape of", "schema")
                DESCRIBE_INTENT = ("describe", "what does", "what is", "what's", "tell me about",
                                   "more about", "explain", "purpose of", "used for", "what are",
                                   "what do you know")
                # Broad "this is a request" signal so natural phrasings ("how would i", "how does")
                # reach RAG instead of the tool-agent; also gates the ~47s index build off chit-chat.
                QUESTION_SIGNALS = ("how ", "what ", "which ", "where ", "who ", "why ", "can i",
                                    "could i", "is there", "are there", "do i", "does ", "i want",
                                    "i need", "i'd like", "find ", "get ", "list ", "show ", "give me",
                                    "help me", "need to", "want to", "?")
                # Metadata an endpoint HAS. A subjectless question about one of these can only be
                # exact-only: a real named path, NOT a fuzzy keyword hit (which would pre-empt RAG).
                named = _named_endpoint(last_spec, user_text)
                anaphoric = last_endpoint is not None and _is_followup(user_text, last_endpoint)

                # test/execute -> deterministic grounded call: collect required inputs FROM THE USER
                # (path/query params + required body fields), confirm, then fire. The model never
                # supplies values, so it can't invent a meetingId or a required field. Needs a concrete
                # endpoint (named, or the one in focus); otherwise fall through to the exploratory agent.
                if any(p in lower for p in TEST_INTENT):
                    target = named or last_endpoint
                    if target is not None:
                        try:
                            run_grounded_call(target, last_spec.base_url or "")
                        except (EOFError, KeyboardInterrupt):
                            print("\n  (call cancelled - back to prompt)")
                        last_endpoint = target
                        continue

                if not any(p in lower for p in TEST_INTENT):
                    # 0. Anaphoric follow-up about the last endpoint ("payload for THIS endpoint",
                    #    "what scopes does THIS action need") -> answer about last_endpoint, resolved
                    #    HERE before any fresh search. This is what stops a follow-up from (a) running
                    #    an irrelevant semantic search and (b) overwriting the focus endpoint with that
                    #    search's top off-topic hit. Crucially, it does NOT reassign last_endpoint - a
                    #    follow-up asks *about* the current endpoint, it doesn't move the focus.
                    if anaphoric and not named:
                        e = last_endpoint
                        if any(p in lower for p in EXAMPLE_INTENT):
                            emit_example(e, user_text)
                        else:
                            grounded = (
                                "Answer the user's question about THIS API endpoint using ONLY the "
                                "info below. Everything documented is there: the method, path, "
                                "description, response fields, every parameter (with its location, "
                                "type, description, example, and whether it is required or "
                                "optional), the required request-body fields, and the OAuth scopes. "
                                "Report required vs optional EXACTLY as marked - never call an "
                                "optional parameter required. The response/request shapes are given "
                                "in full, ALL nested levels - answer questions about nested fields "
                                "(what is inside an array or object) from that tree. If any block is "
                                "marked [!] TRUNCATED, say so in your answer and tell the user the "
                                "rest is in the API's own docs. If the answer genuinely isn't below - "
                                "e.g. rate limits or error codes - say so plainly and do NOT guess. "
                                "Do NOT invent endpoints, paths, parameters, or fields.\n\n"
                                f"Endpoint:\n{endpoint_listing([e], param_detail=True)}"
                                f"\n\nUser question: {user_text}"
                            )
                            print(f"  -> answering about {e.method} {e.path} from the onboarded spec...")
                            grounded_answer(grounded, last_spec)
                        continue

                    # 1. Example/payload for a NAMED endpoint -> synthesize from schema (no live call),
                    #    honoring which DIRECTION the user asked about. (Anaphoric example asks were
                    #    already handled above.)
                    if named and any(p in lower for p in EXAMPLE_INTENT):
                        emit_example(named, user_text)
                        last_endpoint = named
                        continue

                    # 2. A named endpoint, or a describe-style question -> describe it, grounded.
                    if named or any(w in lower for w in DESCRIBE_INTENT):
                        matches = _find_endpoints(last_spec, user_text)
                        if matches:
                            listing = endpoint_listing(matches, param_detail=len(matches) == 1)
                            grounded = (
                                "Answer the user's question about these API endpoints using ONLY the "
                                "info below - the method, path, description, response fields, "
                                "parameters, required body fields and OAuth scopes listed there are "
                                "the ONLY documented facts. Report required vs optional EXACTLY as "
                                "marked - never call an optional parameter required. Where a full "
                                "nested response shape is given, use it to answer questions about "
                                "fields inside an array or object. If any block is marked "
                                "[!] TRUNCATED, say so and point the user at the API's own docs. "
                                "Do NOT invent "
                                "endpoints, paths, parameters, or response fields, and do NOT "
                                "suggest a live call.\n\n"
                                f"Endpoints:\n{listing}\n\nUser question: {user_text}"
                            )
                            print("  -> answering from the onboarded spec (no tool call needed)...")
                            grounded_answer(grounded, last_spec)
                            last_endpoint = matches[0]
                            continue

                    # 3. Any other request/question -> RAG discovery over the FULL spec, grounded.
                    if any(s in lower for s in QUESTION_SIGNALS):
                        if not endpoint_index_ready(last_spec):
                            print(f"  -> indexing {len(last_spec.endpoints)} endpoints for semantic "
                                  f"search (one time, ~a minute)...")
                        try:
                            ensure_endpoint_index(last_spec)
                        except Exception as exc:
                            print(f"  (semantic index unavailable: {exc} - using keyword match)")
                        # search_endpoints() splits compound asks ("create a meeting THEN download the
                        # recording") so each intent retrieves; falls back to keyword if no index.
                        matches = search_endpoints(user_text) or _find_endpoints(last_spec, user_text)
                        if matches:
                            listing = endpoint_listing(matches)
                            grounded = (
                                f"The user is looking for an endpoint in the {last_spec.base_url or 'onboarded'} "
                                "API. Below are the closest-matching candidate endpoints (nearest matches, "
                                "NOT guaranteed relevant), each with its documented response fields. Tell "
                                "the user which one(s) do what they asked - and if they're after a specific "
                                "value, which endpoint's response field holds it - using ONLY these "
                                "candidates and their listed fields. If none fit, say so plainly and note "
                                "what these cover. Do not mention, imply, or recommend a plausible "
                                "endpoint that is not in the candidate list; if the API would need a "
                                "missing endpoint such as bookings, scheduling, checkout, or creation "
                                "of another resource, say that this spec does not document one. Nested "
                                "response shapes are given in full where "
                                "available - use them when the user asks for a field inside an "
                                "object or array. If any block is marked [!] TRUNCATED, say so and "
                                "point the user at the API's own docs. "
                                "Do NOT invent endpoints, paths, parameters, or fields, "
                                "and do NOT make a live call.\n\n"
                                f"Candidates:\n{listing}\n\nUser request: {user_text}"
                            )
                            print(f"  -> semantic search over {len(ENDPOINT_INDEX) or len(last_spec.endpoints)} "
                                  f"endpoints (no tool call needed)...")
                            grounded_answer(grounded, last_spec)
                            last_endpoint = matches[0]
                            continue
                # nothing matched -> fall through to the normal tool-agent

            state, interrupt_value = run_turn({"messages": [("user", user_text)]}, config)

            attempts = 0
            while interrupt_value is not None:
                for m in state["messages"][-3:]:        # SHOW why it's asking (e.g. the 401)
                    m.pretty_print()

                attempts += 1
                if attempts > 3:                        # safety net against infinite re-auth
                    print(">>> Too many auth attempts - cancelling. Is the token valid?")
                    state, interrupt_value = run_turn(Command(resume="__CANCEL__"), config)
                    break

                try:
                    token = getpass.getpass(f"{interrupt_value} (blank to cancel): ")
                except (KeyboardInterrupt, EOFError):
                    # Ctrl+C or EOF at the auth prompt: abort the input, but RESOLVE the pending
                    # tool call first. Bailing out here would leave an authorize() call with no
                    # ToolMessage, wedging the thread (every later turn -> INVALID_CHAT_HISTORY).
                    # Cancelling feeds authorize() its __CANCEL__ result, closing the call cleanly.
                    print("\n(cancelled - clearing pending authorization)")
                    state, interrupt_value = run_turn(Command(resume="__CANCEL__"), config)
                    continue
                if not token.strip():                   # escape hatch
                    state, interrupt_value = run_turn(Command(resume="__CANCEL__"), config)
                    continue
                state, interrupt_value = run_turn(Command(resume=token), config)

            for m in state["messages"][-3:]:            # final result
                m.pretty_print()

            # --- auto-extract: if this turn was an "understand the API" request, return a structured spec
            EXTRACT_INTENT = ("summarize", "summarise", "endpoints", "understand", "structure", "spec", "extract")
            if any(w in user_text.lower() for w in EXTRACT_INTENT) and DOC_CORPUS:
                try:
                    spec = last_spec or extract_api_spec()   # reuse onboarding spec if we have one
                    print("\n=== STRUCTURED API SPEC ===")
                    print(spec.model_dump_json(indent=2))
                except Exception as e:
                    print(f"(extraction skipped: {e})")

        except KeyboardInterrupt:        # Ctrl+C mid-turn: abort this turn, keep the session alive
            print("\n(cancelled - back to prompt)")
            continue
        except Exception as e:           # model/network/other failure: report and stay alive
            hint = " - is the Ollama server running?" if "connect" in str(e).lower() else ""
            print(f"(turn failed: {e}{hint})")
            continue

    
