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
from pydantic import BaseModel, Field
from collections import Counter
from scraper import crawl
from urllib.parse import urlparse, urljoin

SECRETS: dict[str, str] = {}   # host -> token. Never enters `messages`, so the model never sees it.
DOC_CORPUS: list[str] = []           # all doc text fetched so far (grounding evidence)
# (METHOD, path) -> {"request": schema|None, "response": schema|None}. Populated from the OpenAPI
# spec at parse time so we can synthesize grounded example payloads on demand. Kept OUT of the model's
# context and the handoff — schemas run 40-80 KB each and would blow num_ctx instantly.
ENDPOINT_SCHEMAS: dict[tuple[str, str], dict] = {}
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

@tool
def make_api_call(url: str, method: str = "GET", headers: dict | None = None) -> str:
    """Make a real HTTP request and return status + body. Use the literal placeholder
    {{TOKEN}} wherever a secret belongs (e.g. Authorization: 'Bearer {{TOKEN}}')."""
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

    headers = headers or {}
    host = urlparse(url).netloc
    token = SECRETS.get(host)

    # Guard: if a secret is referenced but we don't have one yet, force authorization first.
    needs_secret = any("{{TOKEN}}" in str(v) for v in headers.values())
    if needs_secret and not token:
        return (f"No credential stored for {host}. "
                f"Call authorize('{host}') to obtain one, then retry this request.")

    real_headers = {
        k: (v.replace("{{TOKEN}}", token) if token else v)
        for k, v in headers.items()
    }

    try:
        response = requests.request(method, url, headers=real_headers, timeout=15)
    except requests.RequestException as e:
        return f"Request failed: {e}"
    return f"Status: {response.status_code}\nBody: {_truncate(response.text, 500, 'response body')}"


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

def search_endpoints(query: str, k: int = 8) -> list[Endpoint]:
    """Discovery retrieval. Splits a multi-intent query ('create a meeting THEN download the
    recording') into clauses and unions each clause's top matches — a compound query's single
    embedding is diluted and neither intent ranks. Single-intent queries pass straight through."""
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

def extract_from_pages(pages) -> ApiSpec:
    """pages: [(url, text)]. Chunk each page, extract per chunk, merge & dedup endpoints."""
    base_urls, auth_methods, seen, endpoints = [], [], set(), []
    for url, text in pages:
        for chunk in _chunks(text):
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

def _json_schema(content: dict) -> dict | None:
    """The application/json schema out of an OpenAPI content map (falls back to any media type)."""
    content = content or {}
    media = content.get("application/json") or (next(iter(content.values()), {}) if content else {})
    return media.get("schema") if isinstance(media, dict) else None

def _op_request_schema(op: dict) -> dict | None:
    return _json_schema(op.get("requestBody", {}).get("content", {}))

def _op_response_schema(op: dict) -> dict | None:
    """Schema of the first 2xx response (that's the success shape the user wants an example of)."""
    for code, r in (op.get("responses") or {}).items():
        if str(code).startswith("2") and isinstance(r, dict):
            s = _json_schema(r.get("content", {}))
            if s:
                return s
    return None

def _example_from_schema(schema: dict, _depth: int = 0) -> object:
    """Synthesize a grounded example value from a JSON Schema, preferring the spec's own per-field
    `example`/`default`/`enum` values and falling back to type placeholders. Deterministic, no LLM."""
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

def endpoint_examples(method: str, path: str) -> dict:
    """Synthesized request/response examples for an endpoint from its stored schemas, or None each
    if the spec documented no schema. Read from ENDPOINT_SCHEMAS (populated by parse_openapi)."""
    info = ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}
    req = _example_from_schema(info["request"]) if info.get("request") else None
    resp = _example_from_schema(info["response"]) if info.get("response") else None
    return {"request": req, "response": resp}

def _object_properties(schema: dict, _depth: int = 0) -> dict:
    """The top-level property map of an object schema, merging allOf fragments and unwrapping the
    first oneOf/anyOf. Returns {} if the schema isn't object-shaped."""
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

def response_fields(method: str, path: str, limit: int = 20) -> list[str]:
    """Compact 'name (type)' list of an endpoint's top-level response fields — enough to ground the
    model on what a call actually returns, without dumping the (huge) full schema into context."""
    info = ENDPOINT_SCHEMAS.get((method.upper(), path)) or {}
    props = _object_properties(info.get("response") or {})
    return [f"{name} ({(sub or {}).get('type', '?')})" for name, sub in list(props.items())[:limit]]

def endpoint_listing(endpoints, field_limit: int = 15) -> str:
    """One grounded line per endpoint: method, path, description, and its documented top-level
    response fields — so the describe/discovery routes can answer 'what does it return' / 'which
    endpoint returns X' from real field names instead of the model inventing them."""
    lines = []
    for e in endpoints:
        rf = response_fields(e.method, e.path, limit=field_limit)
        tail = f"  [response fields: {', '.join(rf)}]" if rf else ""
        lines.append(f"{e.method} {e.path} - {e.description}{tail}")
    return "\n".join(lines)

def parse_openapi(spec_source: str) -> ApiSpec:
    """Deterministically turn an OpenAPI/Swagger spec into an ApiSpec. No LLM involved.
    `spec_source` is an http(s) URL or a local file path (JSON or YAML). Side effect: repopulates
    ENDPOINT_SCHEMAS with each endpoint's request/response schema for on-demand example synthesis."""
    spec = _load_spec(spec_source)
    ENDPOINT_SCHEMAS.clear()

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

    # auth — securitySchemes (3.0) or securityDefinitions (2.0)
    schemes = (spec.get("components", {}).get("securitySchemes")
               or spec.get("securityDefinitions") or {})
    auth_method = ", ".join(
        s.get("scheme") or s.get("type", "") for s in schemes.values()
    ) or "None documented"

    # endpoints — paths -> {method: operation}
    HTTP = {"get", "post", "put", "patch", "delete", "head", "options"}
    endpoints = []
    for path, ops in spec.get("paths", {}).items():
        for method, op in ops.items():
            if method.lower() not in HTTP or not isinstance(op, dict):
                continue   # skip 'parameters', '$ref', etc. at path level
            desc = (op.get("summary") or op.get("description") or "").strip()
            endpoints.append(Endpoint(method=method.upper(), path=path, description=desc))
            ENDPOINT_SCHEMAS[(method.upper(), path)] = {
                "request": _op_request_schema(op),
                "response": _op_response_schema(op),
            }

    return ApiSpec(base_url=base_url, auth_method=auth_method, endpoints=endpoints)

def _is_openapi(source: str) -> bool:
    """True if `source` (URL or local file) resolves to a parseable OpenAPI/Swagger spec."""
    try:
        data = _load_spec(source)
    except (requests.RequestException, ValueError, OSError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and ("openapi" in data or "swagger" in data)

def find_openapi_spec(docs_url: str) -> str | None:
    """Locate an OpenAPI/Swagger spec for a docs site. Returns its URL, or None."""
    # 0. The URL itself might already be a spec
    if _is_openapi(docs_url):
        return docs_url

    # 1. Links on the docs page that look like a spec
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
    for url in dict.fromkeys(candidates):        # dedupe, keep order
        if _is_openapi(url):
            return url
    return None

def onboard(url: str) -> ApiSpec:
    """OpenAPI-first: clean parse if a spec exists, else fall back to the scraper."""
    ENDPOINT_SCHEMAS.clear()   # drop any previous spec's schemas; parse_openapi repopulates
    spec_url = find_openapi_spec(url)
    if spec_url:
        print(f"[ok] Found OpenAPI spec: {spec_url} - using clean parse.")
        return parse_openapi(spec_url)
    print("[--] No OpenAPI spec found - falling back to doc scraping.")
    return extract_from_pages(crawl(url))

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
    q = query.lower()
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
    q = query.lower()
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
    q = query.lower()
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
                # exact-only: a real named path, NOT a fuzzy keyword hit (which would pre-empt RAG).
                named = _named_endpoint(last_spec, user_text)
                # A follow-up that names no endpoint but refers back to the last one ("how does the
                # response schema look like?") -> resolve to last_endpoint instead of falling to RAG.
                anaphoric = last_endpoint is not None and (
                    any(w in lower.split() for w in ("it", "its", "that", "this", "same", "these"))
                    or any(p in lower for p in ("look like", "the response", "the schema",
                                                "the payload", "the request", "the fields",
                                                "of it", "for it")))

                if not any(p in lower for p in TEST_INTENT):
                    # 1. Example/payload for the named (or referred-back) endpoint -> synthesize from
                    #    schema (no live call), honoring which DIRECTION the user asked about.
                    if any(p in lower for p in EXAMPLE_INTENT):
                        target = named or (last_endpoint if anaphoric else None)
                        if target is not None:
                            e = target
                            ex = endpoint_examples(e.method, e.path)
                            want_req = any(k in lower for k in ("payload", "request body", "request",
                                                                "send", "post body", "what to send"))
                            want_resp = any(k in lower for k in ("response", "returns", "return",
                                                                 "comes back", "get back", "output",
                                                                 "result"))
                            if want_req and not want_resp:
                                show_req, show_resp = True, False
                            elif want_resp and not want_req:
                                show_req, show_resp = False, True
                            else:
                                show_req, show_resp = True, True
                            has = ((show_req and ex["request"] is not None)
                                   or (show_resp and ex["response"] is not None))
                            if has:
                                print("  -> from the onboarded spec's schema (no live call needed):\n")
                                print(f"{e.method} {e.path}")
                                if show_req and ex["request"] is not None:
                                    print("request body (example):")
                                    print(_truncate(json.dumps(ex["request"], indent=2), 4000, "example"))
                                if show_resp and ex["response"] is not None:
                                    print("response (example):")
                                    print(_truncate(json.dumps(ex["response"], indent=2), 4000, "example"))
                            else:
                                # The spec has no schema for the direction asked (e.g. a GET has no
                                # request body). Say so, and point at the direction it DOES document.
                                want = ("request body" if show_req and not show_resp
                                        else "response schema" if show_resp and not show_req
                                        else "example/schema")
                                print(f"  -> the spec documents no {want} for {e.method} {e.path}.")
                                other = ("response schema" if ex["response"] is not None
                                         else "request body" if ex["request"] is not None else None)
                                if other:
                                    print(f"     It does document a {other} - ask for that, or say "
                                          f"'test {e.method} {e.path}' for a live call.")
                                else:
                                    print(f"     Want the real thing? Say 'test {e.method} {e.path}' "
                                          f"and I'll make a live call to show the actual response.")
                            last_endpoint = e
                            continue

                    # 2. A named endpoint, or a describe-style question -> describe it, grounded.
                    if named or any(w in lower for w in DESCRIBE_INTENT):
                        matches = _find_endpoints(last_spec, user_text)
                        if matches:
                            listing = endpoint_listing(matches)
                            grounded = (
                                "Answer the user's question about these API endpoints using ONLY the "
                                "info below - the method, path, description, and listed response fields "
                                "are the ONLY documented facts. Do NOT invent endpoints, paths, "
                                "parameters, or response fields, and do NOT suggest a live call.\n\n"
                                f"Endpoints:\n{listing}\n\nUser question: {user_text}"
                            )
                            print("  -> answering from the onboarded spec (no tool call needed)...")
                            print(llm.invoke(grounded).content)
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
                                "what these cover. Do NOT invent endpoints, paths, parameters, or fields, "
                                "and do NOT make a live call.\n\n"
                                f"Candidates:\n{listing}\n\nUser request: {user_text}"
                            )
                            print(f"  -> semantic search over {len(ENDPOINT_INDEX) or len(last_spec.endpoints)} "
                                  f"endpoints (no tool call needed)...")
                            print(llm.invoke(grounded).content)
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

    
