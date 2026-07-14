from langchain_core.tools import tool          
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent   
from scraper import fetch_text
import requests
from langgraph.checkpoint.memory import MemorySaver
import getpass
from urllib.parse import urlparse
from langgraph.types import interrupt, Command
import re
from pydantic import BaseModel, Field
from collections import Counter
from scraper import crawl
from urllib.parse import urlparse, urljoin

SECRETS: dict[str, str] = {}   # host -> token. Never enters `messages`, so the model never sees it.
DOC_CORPUS: list[str] = []           # all doc text fetched so far (grounding evidence)
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

@tool
def fetch_url(url: str) -> str:
    """Fetch a documentation page. Returns readable text PLUS links found on the page."""
    try:
        text, links = fetch_text(url)
    except requests.RequestException as e:        # 401, timeout, DNS, etc.
        return f"Could not fetch {url}: {e}"
    DOC_CORPUS.append(text)
    link_block = "\n".join(links[:40])
    return f"{text[:4000]}\n\n--- LINKS ON THIS PAGE ---\n{link_block}"

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
    return f"Status: {response.status_code}\nBody: {response.text[:500]}"


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

def parse_openapi(spec_url: str) -> ApiSpec:
    """Deterministically turn an OpenAPI/Swagger spec into an ApiSpec. No LLM involved."""
    resp = requests.get(spec_url, timeout=15)
    resp.raise_for_status()
    spec = resp.json()   # JSON specs for now; YAML handled later

    # base_url — OpenAPI 3 uses servers[]; Swagger 2 uses host + basePath
    servers = spec.get("servers")
    if servers:
        base_url = servers[0].get("url", "")
    else:
        host, base_path = spec.get("host", ""), spec.get("basePath", "")
        scheme = (spec.get("schemes") or ["https"])[0]
        base_url = f"{scheme}://{host}{base_path}" if host else ""

    # servers[].url may be relative (e.g. "/api/v3") — resolve against the spec's own URL
    if base_url and not base_url.startswith("http"):
        base_url = urljoin(spec_url, base_url)

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

    return ApiSpec(base_url=base_url, auth_method=auth_method, endpoints=endpoints)

def _is_openapi(url: str) -> bool:
    """True if the URL returns a parseable OpenAPI/Swagger spec."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
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
    spec_url = find_openapi_spec(url)
    if spec_url:
        print(f"[ok] Found OpenAPI spec: {spec_url} - using clean parse.")
        return parse_openapi(spec_url)
    print("[--] No OpenAPI spec found - falling back to doc scraping.")
    return extract_from_pages(crawl(url))

def _first_url(text: str) -> str:
    """Pull the first URL out of a user message (onboard() needs a bare URL)."""
    m = re.search(r"https?://\S+", text)
    return m.group(0).rstrip(".,);") if m else text.strip()

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

    print("Agent ready. Type 'end session' to quit.")
    while True:
        user_text = input("you> ")
        if user_text.strip() == "end session":
            break
        
        if user_text.strip() == "extract":
            # Prefer the structured spec from onboarding; fall back to an LLM extraction.
            spec = last_spec or extract_api_spec()
            print(spec.model_dump_json(indent=2))
            continue

        # New onboarding target → run the OpenAPI-first router (clean parse if a spec
        # exists, else scrape+LLM). This is the default path on any URL submit.
        if "http" in user_text.lower():
            url = _first_url(user_text)
            DOC_CORPUS.clear()   # fresh evidence corpus so an earlier API can't leak in
            try:
                last_spec = onboard(url)
            except Exception as e:
                print(f"(onboarding failed: {e})")
                continue
            print("\n=== STRUCTURED API SPEC ===")
            print(last_spec.model_dump_json(indent=2))
            # Seed grounding evidence: make_api_call validates real calls against the docs
            # corpus, but onboard() doesn't fetch through fetch_url, so feed it the spec.
            DOC_CORPUS.append(last_spec.model_dump_json())
            # Hand the spec to the agent as ground truth so later "test X" turns can reason
            # over the real endpoints without having to rediscover them by reading docs.
            handoff = (
                f"Structured onboarding of {url} is complete. Treat this API spec as ground "
                f"truth for what endpoints exist — do NOT rediscover them by reading docs:\n"
                f"{last_spec.model_dump_json(indent=2)}\n"
                f"Reply with a one-line confirmation and wait for my next instruction."
            )
            state = agent.invoke({"messages": [("user", handoff)]}, config)
            state["messages"][-1].pretty_print()
            continue

        state = agent.invoke({"messages": [("user", user_text)]}, config)

        attempts = 0
        while "__interrupt__" in state:
            for m in state["messages"][-3:]:        # SHOW why it's asking (e.g. the 401)
                m.pretty_print()

            attempts += 1
            if attempts > 3:                        # safety net against infinite re-auth
                print(">>> Too many auth attempts — cancelling. Is the token valid?")
                state = agent.invoke(Command(resume="__CANCEL__"), config)
                break

            request = state["__interrupt__"][0].value
            token = getpass.getpass(f"{request} (blank to cancel): ")
            if not token.strip():                   # escape hatch
                state = agent.invoke(Command(resume="__CANCEL__"), config)
                continue
            state = agent.invoke(Command(resume=token), config)

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

    
