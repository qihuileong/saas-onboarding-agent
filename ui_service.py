"""Reusable UI-facing workflows for the SaaS onboarding agent.

The terminal REPL in ``agent.py`` is intentionally interactive. This module exposes the same
grounded spec lookup, examples, semantic discovery, and real-call preparation as ordinary
functions so Streamlit reruns never have to fake ``input()`` or scrape console output.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Callable
from urllib.parse import quote, urlencode, urlparse

import agent as A

API_VERSION = 11

Progress = Callable[[int, str], None]
Trace = Callable[[str, dict[str, Any]], None]

TEST_INTENT = (
    "test ", "test the", "call it", "make a call", "make a request", "make the call",
    "hit the", "hit it", "try it", "run it", "actually call", "live call",
    "send a request", "fire the", "call this", "call that", "execute this", "execute it",
    "send this", "use this endpoint",
)
EXAMPLE_INTENT = (
    "example", "sample", "payload", "request body", "response body",
    "response look like", "what does the response", "show me the response",
    "what fields", "json response", "shape of", "schema",
)
DESCRIBE_INTENT = (
    "describe", "what does", "what is", "what's", "tell me about", "more about",
    "explain", "purpose of", "used for", "what are", "what do you know",
)
API_ERROR_FOLLOWUP_INTENT = (
    "error", "failed", "failure", "wrong", "fix", "correct", "retry",
    "try again", "why did", "what happened", "change the request",
    "draft again", "next request",
)


@dataclass
class ChatReply:
    text: str
    focus: A.Endpoint | None
    list_cursor: int
    route: str


@dataclass
class PreparedCall:
    endpoint: A.Endpoint
    url: str
    headers: dict[str, str]
    body: dict[str, Any] | list[Any] | None
    host: str
    needs_credential: bool
    auth_scheme: str = ""

    def preview(self) -> str:
        """A copy/paste-friendly preview that can never contain the actual credential."""
        url = _mask_placeholders(self.url)
        headers = {key: _mask_placeholders(value) for key, value in self.headers.items()}
        lines = [f"{self.endpoint.method} {url}"]
        if headers:
            lines.append("headers:")
            lines.append(json.dumps(headers, indent=2))
        if self.body is not None:
            lines.append("body:")
            lines.append(json.dumps(self.body, indent=2))
        return "\n".join(lines)


def _report(progress: Progress | None, percent: int, message: str) -> None:
    if progress:
        progress(percent, message)


def _trace(trace: Trace | None, name: str, **payload: Any) -> None:
    if trace:
        trace(name, payload)


def _with_warning(answer: str, warning: str) -> str:
    if not warning:
        return answer
    return f"{answer}\n\n---\n\n{warning}"


def verify_candidate_mentions(
    answer: str,
    candidates: list[A.Endpoint],
    spec: A.ApiSpec,
) -> str:
    """Flag real endpoints that the model named without receiving them as retrieval evidence.

    ``agent.verify_endpoint_mentions`` answers "does this exist anywhere in the spec?" Discovery
    needs the stricter question "was this endpoint in the candidate block for this turn?" Otherwise
    a real-but-unretrieved endpoint such as ``POST /invitees`` silently passes even though the model
    introduced it from prior knowledge rather than the supplied evidence.
    """
    outside, seen = [], set()
    for method, raw in A._MENTION_RE.findall(answer or ""):
        method = method.upper()
        path = A._norm_path(raw, spec.base_url or "")
        key = (method, path)
        if path == "/" or key in seen:
            continue
        seen.add(key)
        allowed = any(
            endpoint.method.upper() == method
            and A._path_template_matches(endpoint.path, path)
            for endpoint in candidates
        )
        if allowed:
            continue
        exists = any(
            endpoint.method.upper() == method
            and A._path_template_matches(endpoint.path, path)
            for endpoint in spec.endpoints
        )
        if exists:
            outside.append(f"{method} {path}")
    if not outside:
        return ""
    listed = ", ".join(f"{endpoint.method} {endpoint.path}" for endpoint in candidates)
    return (
        "  [!] retrieval grounding: the answer names documented endpoint(s) that were NOT in "
        f"this turn's retrieved evidence: {', '.join(outside)}.\n"
        f"      Retrieved candidates were: {listed or '(none)'}.\n"
        "      Treat recommendations involving those unsupported endpoints as unreliable."
    )


def focus_from_answer(
    answer: str,
    candidates: list[A.Endpoint],
    spec: A.ApiSpec,
) -> A.Endpoint | None:
    """The first retrieved endpoint the answer actually selected/named, preserving follow-up focus."""
    for method, raw in A._MENTION_RE.findall(answer or ""):
        path = A._norm_path(raw, spec.base_url or "")
        for endpoint in candidates:
            if (endpoint.method.upper() == method.upper()
                    and A._path_template_matches(endpoint.path, path)):
                return endpoint
    return candidates[0] if candidates else None


def _ask_grounded(
    prompt: str,
    spec: A.ApiSpec,
    progress: Progress | None,
    trace: Trace | None,
    evidence: list[A.Endpoint],
) -> str:
    _report(progress, 78, "Generating an answer from the selected spec data")
    started = time.perf_counter()
    try:
        answer, warning = A.grounded_answer_text(prompt, spec)
    except Exception as exc:
        _trace(
            trace,
            "llm",
            input=prompt,
            error=str(exc),
            status="error",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise
    candidate_warning = verify_candidate_mentions(answer, evidence, spec)
    combined_warning = "\n".join(
        warning for warning in (warning, candidate_warning) if warning
    )
    _trace(
        trace,
        "llm",
        input=prompt,
        output=answer,
        grounding_warning=combined_warning,
        status="ok",
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    _report(progress, 95, "Verifying every endpoint mentioned in the answer")
    return _with_warning(answer, combined_warning)


def answer_question(
    spec: A.ApiSpec,
    user_text: str,
    *,
    last_endpoint: A.Endpoint | None = None,
    last_api_exchange: dict[str, Any] | None = None,
    list_cursor: int = 0,
    progress: Progress | None = None,
    trace: Trace | None = None,
) -> ChatReply:
    """Answer one UI chat turn from an onboarded spec.

    The route is deterministic for the same reason as the terminal REPL: the local 7B model phrases
    answers, while Python decides whether this is listing, endpoint detail, schema example, semantic
    discovery, or a request to open the real-call builder.
    """
    text = user_text.strip()
    if not text:
        raise ValueError("Enter a question first.")
    lower = text.lower()
    _report(progress, 5, "Reading the request")
    _trace(
        trace,
        "routing_input",
        input=text,
        focus=_endpoint_name(last_endpoint),
        list_cursor=list_cursor,
    )

    page = A._listing_request(text, len(spec.endpoints), list_cursor)
    if page is not None:
        start, end = page
        if start >= len(spec.endpoints):
            answer = (f"That is past the end of the spec. It contains {len(spec.endpoints)} "
                      "endpoints; ask to list endpoints to start over.")
            return ChatReply(answer, last_endpoint, list_cursor, "listing")
        lines = [f"Endpoints {start + 1}-{end} of {len(spec.endpoints)}:"]
        lines += [f"- `{endpoint.method} {endpoint.path}`" for endpoint in spec.endpoints[start:end]]
        if end < len(spec.endpoints):
            lines.append(f"\n{len(spec.endpoints) - end} remain. Ask for the next endpoints to continue.")
        _report(progress, 100, "Endpoint list ready")
        _trace(trace, "route", route="listing", start=start, end=end)
        return ChatReply("\n".join(lines), last_endpoint, end, "listing")

    named = A._named_endpoint(spec, text)
    anaphoric = last_endpoint is not None and A._is_followup(text, last_endpoint)

    exchange_status = (
        last_api_exchange.get("status_code")
        if isinstance(last_api_exchange, dict)
        else None
    )
    exchange_endpoint = (
        str(last_api_exchange.get("endpoint") or "")
        if isinstance(last_api_exchange, dict)
        else ""
    )
    exchange_available = (
        bool(last_api_exchange.get("available_to_chat", True))
        if isinstance(last_api_exchange, dict)
        else False
    )
    exchange_focus = named or last_endpoint
    exchange_matches_focus = bool(
        exchange_focus
        and exchange_endpoint == _endpoint_name(exchange_focus)
    )
    if (
        exchange_available
        and
        exchange_matches_focus
        and (exchange_status is None or exchange_status >= 400)
        and any(phrase in lower for phrase in API_ERROR_FOLLOWUP_INTENT)
    ):
        evidence = A.endpoint_listing([exchange_focus], param_detail=True)
        masked_request = redact_log_text(
            str(last_api_exchange.get("masked_request_preview") or "")
        )
        safe_response = redact_log_text(
            str(last_api_exchange.get("response") or "")
        )
        prompt = (
            "Answer this follow-up about the most recent failed API call. Use ONLY the documented "
            "endpoint evidence, masked request, and actual server response below. Explain concrete "
            "causes and draft a corrected next request when requested. Never claim it was sent. "
            "Treat constraints stated only by the response as server-reported rules, not OpenAPI "
            "facts. Never ask for or expose a credential; use `<credential>` in any request draft. "
            "If a required value is unknown, tell the user exactly what to supply.\n\n"
            f"Documented endpoint:\n{evidence}\n\n"
            f"Most recent masked request:\n{masked_request}\n\n"
            f"Actual server response:\n{safe_response[:6000]}\n\n"
            f"User follow-up: {text}"
        )
        _trace(
            trace,
            "route",
            route="api-error-follow-up",
            endpoint=_endpoint_name(exchange_focus),
        )
        answer = _ask_grounded(
            prompt,
            spec,
            progress,
            trace,
            [exchange_focus],
        )
        _report(progress, 100, "Failed-call guidance ready")
        return ChatReply(
            answer,
            exchange_focus,
            list_cursor,
            "api-error-follow-up",
        )

    explicit_named_call = named is not None and bool(
        re.search(r"\b(?:call|execute|send)\b", lower)
    )
    if any(phrase in lower for phrase in TEST_INTENT) or explicit_named_call:
        target = named or last_endpoint
        if target is None:
            answer = ("Name an endpoint first, for example `GET /users/me`, then open the "
                      "**API Call** tab to prepare and confirm the real request.")
        else:
            answer = (f"I selected `{target.method} {target.path}`. Open the **API Call** tab to "
                      "fill its documented inputs, inspect the credential-safe preview, and explicitly "
                      "confirm before anything is sent.")
        _report(progress, 100, "Real-call request routed to the API Call tab")
        _trace(trace, "route", route="call", endpoint=_endpoint_name(target))
        return ChatReply(answer, target or last_endpoint, list_cursor, "call")

    focus = last_endpoint if anaphoric and named is None else named
    if focus and any(phrase in lower for phrase in EXAMPLE_INTENT):
        _report(progress, 60, "Building an example deterministically from the stored schema")
        answer = A.format_example(focus, text)
        _report(progress, 100, "Schema example ready")
        _trace(trace, "route", route="example", endpoint=_endpoint_name(focus), output=answer)
        return ChatReply(answer, focus, list_cursor, "example")

    if focus and anaphoric:
        listing = A.endpoint_listing([focus], param_detail=True)
        prompt = (
            "Answer the user's question about this API endpoint using ONLY the documented block "
            "below. Required and optional markers are authoritative. Use the complete nested schema "
            "when present. If data is marked [!] TRUNCATED, say so. If the answer is absent, say so; "
            "do not guess or invent endpoints, parameters, fields, scopes, or results.\n\n"
            f"Endpoint:\n{listing}\n\nUser question: {text}"
        )
        _trace(trace, "route", route="follow-up", endpoint=_endpoint_name(focus))
        answer = _ask_grounded(prompt, spec, progress, trace, [focus])
        _report(progress, 100, "Grounded answer ready")
        return ChatReply(answer, focus, list_cursor, "follow-up")

    if named or any(phrase in lower for phrase in DESCRIBE_INTENT):
        matches = A._find_endpoints(spec, text)
        if matches:
            listing = A.endpoint_listing(matches, param_detail=len(matches) == 1)
            prompt = (
                "Answer using ONLY these documented API endpoints. Required and optional markers "
                "are authoritative. Use nested request and response fields when present. If data is "
                "marked [!] TRUNCATED, say so. Do not invent endpoints, parameters, fields, scopes, "
                "headers, or live-call results.\n\n"
                f"Endpoints:\n{listing}\n\nUser question: {text}"
            )
            _trace(
                trace,
                "route",
                route="describe",
                candidates=[_endpoint_name(endpoint) for endpoint in matches],
            )
            answer = _ask_grounded(prompt, spec, progress, trace, matches)
            _report(progress, 100, "Grounded answer ready")
            return ChatReply(answer, matches[0], list_cursor, "describe")

    if not A.endpoint_index_ready(spec):
        _report(progress, 20, f"Indexing {len(spec.endpoints)} endpoints for semantic search")
    index_started = time.perf_counter()
    try:
        A.ensure_endpoint_index(spec)
        _report(progress, 58, "Semantic endpoint index ready")
        _trace(
            trace,
            "semantic_index",
            status="ok",
            endpoints=len(spec.endpoints),
            duration_ms=round((time.perf_counter() - index_started) * 1000, 2),
        )
    except Exception as exc:
        _report(progress, 58, f"Semantic index unavailable; using keyword matching ({exc})")
        _trace(
            trace,
            "semantic_index",
            status="fallback",
            error=str(exc),
            duration_ms=round((time.perf_counter() - index_started) * 1000, 2),
        )

    retrieval_started = time.perf_counter()
    # Four candidates keep the weak local model focused and still retain both halves of compound
    # requests (event type + invitees). Larger sets made it ignore decisive operation notes.
    matches = A.search_endpoints(text, k=4) or A._find_endpoints(spec, text, limit=4)
    _trace(
        trace,
        "retrieval",
        input=text,
        candidates=[{
            "method": endpoint.method,
            "path": endpoint.path,
            "description": endpoint.description,
        } for endpoint in matches],
        duration_ms=round((time.perf_counter() - retrieval_started) * 1000, 2),
    )
    if not matches:
        _report(progress, 100, "No documented endpoint matched")
        _trace(trace, "route", route="no-match")
        return ChatReply(
            "I could not find a documented endpoint matching that request. Try naming a resource, "
            "an HTTP method, or an exact path from the Endpoint Explorer.",
            last_endpoint,
            list_cursor,
            "no-match",
        )

    _report(progress, 68, f"Retrieved {len(matches)} candidate endpoint(s)")
    needs_response_detail = any(word in lower for word in (
        "response", "return", "field", "value", "inside", "contains", "schema",
    ))
    needs_workflow_detail = bool(re.search(
        r"\b(?:then|after|before|workflow|first|next|followed by|and then)\b",
        lower,
    ))
    if needs_workflow_detail:
        implied_methods, _ = A._query_methods(text)
        method_matches = [
            endpoint for endpoint in matches
            if endpoint.method.upper() in implied_methods
        ]
        if method_matches:
            matches = method_matches
        answer = _workflow_candidate_brief(matches)
        selected_focus = matches[0]
        _trace(
            trace,
            "route",
            route="workflow-brief",
            candidates=[_endpoint_name(endpoint) for endpoint in matches],
        )
        _trace(trace, "focus", endpoint=_endpoint_name(selected_focus))
        _report(progress, 100, "Documented operations ready; unverified handoffs withheld")
        return ChatReply(answer, selected_focus, list_cursor, "workflow-brief")

    listing = A.endpoint_listing(
        matches,
        field_limit=8,
        include_notes=True,
        include_response_tree=needs_response_detail,
    )
    prompt = (
        f"The user is looking for an endpoint in the {spec.base_url or 'onboarded'} API. "
        "The blocks below are nearest candidates, not guaranteed matches. Explain which candidates "
        "fit using ONLY their documented facts. If none fit, say that plainly. Never mention or "
        "recommend a plausible endpoint outside this candidate list. If the capability would require "
        "an endpoint the spec does not document, say so. If data is marked [!] TRUNCATED, say so. "
        "Multiple candidates may cover separate parts of the request, so discuss each relevant "
        "candidate independently. This candidate list does NOT certify cross-endpoint handoffs. "
        "Do not turn it into an ordered workflow or say one candidate feeds another. If the user "
        "asks for a sequence, state that the relationship and order are not established by the "
        "retrieved evidence, then explain the independently documented operations. Similar "
        "resource words alone do not prove a handoff. "
        "Treat operation notes as decisive: when an "
        "endpoint says it creates a booking, that is the operation that schedules the actual event "
        "for its invitee. Distinguish an event type (the scheduling configuration) from a booking or "
        "scheduled event (the actual occurrence with invitees). "
        "Do not make a live call.\n\n"
        # Put the request before the potentially large evidence block. If a pathological spec still
        # reaches _fit_prompt's hard cap, the model must never lose the question it is answering.
        f"User request: {text}\n\nCandidates:\n{listing}"
    )
    _trace(
        trace,
        "route",
        route="discovery",
        candidates=[_endpoint_name(endpoint) for endpoint in matches],
    )
    answer = _ask_grounded(prompt, spec, progress, trace, matches)
    selected_focus = focus_from_answer(answer, matches, spec)
    _trace(trace, "focus", endpoint=_endpoint_name(selected_focus))
    _report(progress, 100, "Grounded answer ready")
    return ChatReply(answer, selected_focus, list_cursor, "discovery")


def _endpoint_name(endpoint: A.Endpoint | None) -> str:
    return f"{endpoint.method} {endpoint.path}" if endpoint else ""


def _workflow_candidate_brief(candidates: list[A.Endpoint]) -> str:
    """Describe multi-step candidates without letting the model invent cross-endpoint handoffs."""
    lines = [
        "I found operations relevant to separate parts of that sequence:",
        "",
    ]
    for endpoint in candidates:
        required = A.request_required(endpoint.method, endpoint.path)
        response = A.response_fields(endpoint.method, endpoint.path, limit=8)
        lines.append(f"- **`{endpoint.method} {endpoint.path}`** — {endpoint.description}")
        if required is None:
            lines.append("  - Documented request body: none")
        elif required:
            lines.append("  - Required body fields: " + ", ".join(f"`{name}`" for name in required))
        else:
            lines.append("  - Documented request body: no individually required fields")
        if response:
            lines.append("  - Top-level response fields: " + ", ".join(f"`{name}`" for name in response))
        notes = (
            A.ENDPOINT_SCHEMAS.get((endpoint.method.upper(), endpoint.path)) or {}
        ).get("notes")
        if notes:
            compact = notes if len(notes) <= 220 else notes[:220].rstrip() + "…"
            lines.append(f"  - Documented behavior: {compact}")
    lines += [
        "",
        "The retrieved OpenAPI evidence does **not** establish an execution order or prove that an "
        "output from one operation is accepted as an input by another. I won’t invent that handoff. "
        "Inspect the exact request/response schemas for the operations you intend to connect, or ask "
        "about one endpoint at a time.",
    ]
    return "\n".join(lines)


def prepare_call(
    endpoint: A.Endpoint,
    base_url: str,
    *,
    path_values: dict[str, Any] | None = None,
    query_values: dict[str, Any] | None = None,
    header_values: dict[str, Any] | None = None,
    cookie_values: dict[str, Any] | None = None,
    body: dict[str, Any] | list[Any] | None = None,
    include_auth: bool = True,
    auth_required: bool | None = None,
    auth_scheme: str = "",
) -> PreparedCall:
    """Validate and assemble a grounded call without sending it.

    All variable values come from UI fields. The credential remains ``{{TOKEN}}`` in the prepared
    request and is substituted only inside ``agent._do_http_call`` at execution time.
    """
    if not base_url:
        raise ValueError("This spec has no base URL, so an absolute request URL cannot be built.")

    path_values = path_values or {}
    query_values = query_values or {}
    header_values = header_values or {}
    cookie_values = cookie_values or {}
    params = A.endpoint_params(endpoint.method, endpoint.path)
    by_location = {
        location: {param["name"]: param for param in params if param["in"] == location}
        for location in ("path", "query", "header", "cookie")
    }

    path = endpoint.path
    for name in re.findall(r"{([^}]+)}", endpoint.path):
        value = path_values.get(name)
        if value is None or str(value).strip() == "":
            raise ValueError(f"Path parameter `{name}` is required.")
        path = path.replace("{" + name + "}", quote(str(value), safe=""))

    query: dict[str, Any] = {}
    headers: dict[str, str] = {}
    cookies: dict[str, str] = {}
    supplied = {
        "query": query_values,
        "header": header_values,
        "cookie": cookie_values,
    }
    destinations = {"query": query, "header": headers, "cookie": cookies}
    for location in ("query", "header", "cookie"):
        for name, param in by_location[location].items():
            value = supplied[location].get(name)
            missing = value is None or (isinstance(value, str) and not value.strip())
            if missing and param["required"]:
                raise ValueError(f"{location.title()} parameter `{name}` is required.")
            if not missing:
                param_format = A.parameter_format(param)
                if param_format in {"uri", "url"}:
                    parsed_value = urlparse(str(value).strip())
                    if not parsed_value.scheme:
                        raise ValueError(
                            f"{location.title()} parameter `{name}` must be a complete URI, "
                            "including its scheme (for example, `https://...`)."
                        )
                if isinstance(value, (list, tuple)):
                    is_array = str(param.get("type") or "").startswith("array")
                    if is_array and location == "query" and param.get("explode", True):
                        value = list(value)
                    else:
                        value = ",".join(str(item) for item in value)
                destinations[location][name] = value

    info = A.ENDPOINT_SCHEMAS.get((endpoint.method.upper(), endpoint.path)) or {}
    for group in info.get("param_any_of") or []:
        location = group.get("in", "")
        names = group.get("names") or []
        values = path_values if location == "path" else supplied.get(location, {})
        if names and not any(
            values.get(name) is not None and str(values.get(name)).strip()
            for name in names
        ):
            rendered = ", ".join(f"`{name}`" for name in names)
            where = f" {location}" if location else ""
            raise ValueError(
                f"At least one of the{where} parameters {rendered} is required by the "
                "endpoint documentation."
            )

    request_schema = info.get("request")
    if body is not None and request_schema is None:
        raise ValueError("The spec does not document a request body for this endpoint.")
    if body is None and info.get("body_required"):
        raise ValueError("The request body is required.")
    if body is not None:
        issues = _request_body_issues(request_schema, body)
        if issues:
            raise ValueError("; ".join(issues))

    content_type = info.get("content_type") or ""
    if body is not None:
        if content_type and content_type != "application/json":
            raise ValueError(
                f"This endpoint expects {content_type}; the UI currently sends JSON only."
            )
        headers["Content-Type"] = "application/json"

    schemes = A.endpoint_auth_schemes(endpoint.method, endpoint.path)
    if auth_scheme and auth_scheme not in schemes:
        raise ValueError(
            f"Authentication scheme `{auth_scheme}` is not documented for this endpoint."
        )
    selected_schemes = [auth_scheme] if auth_scheme else schemes
    needs_credential = include_auth and (
        bool(schemes) if auth_required is None else auth_required
    )
    if needs_credential:
        if not selected_schemes:
            raise ValueError(
                "Authentication is required, but the spec does not document how to place the "
                "credential. The call builder will not guess."
            )
        kind, credential_name, template = A._credential_placement(selected_schemes)
        if kind == "query":
            query[credential_name] = "{{TOKEN}}"
        elif kind == "cookie":
            cookies[credential_name] = "{{TOKEN}}"
        else:
            headers[credential_name] = template

    if cookies:
        headers["Cookie"] = "; ".join(f"{key}={value}" for key, value in cookies.items())
    url = base_url.rstrip("/") + path
    if query:
        url += "?" + urlencode(query, doseq=True)
    return PreparedCall(
        endpoint=endpoint,
        url=url,
        headers=headers,
        body=body,
        host=urlparse(url).netloc,
        needs_credential=needs_credential,
        auth_scheme=selected_schemes[0] if needs_credential else "",
    )


def execute_call(prepared: PreparedCall, credential: str = "") -> str:
    """Execute a previously prepared request after the UI's explicit confirmation."""
    token = credential.strip()
    if prepared.needs_credential and not token:
        raise ValueError("This endpoint requires a credential. Enter one before sending.")

    # `_do_http_call` performs last-moment placeholder substitution through the legacy module cache.
    # Keep that cache populated only for the duration of this request; Streamlit session state owns
    # persistence so one browser session cannot donate a credential to another.
    missing = object()
    previous = A.SECRETS.get(prepared.host, missing)
    if token:
        A.SECRETS[prepared.host] = token
    try:
        return A._do_http_call(
            prepared.url,
            prepared.endpoint.method,
            prepared.headers,
            prepared.body,
            check_grounding=False,
            response_limit=None,
        )
    finally:
        if previous is missing:
            A.SECRETS.pop(prepared.host, None)
        else:
            A.SECRETS[prepared.host] = previous


def parse_http_result(result: str) -> dict[str, Any]:
    """Split the agent's ``Status: N\nBody: ...`` response for friendly UI rendering."""
    # Spaces/tabs only: `\s` also consumes the newline and made the status banner swallow the
    # complete `Body: {...}` line, duplicating the response above the Pretty/Raw tabs.
    status_match = re.match(
        r"Status:[ \t]*(\d+)(?:[ \t]+([^\r\n]*))?",
        result or "",
    )
    status_code = int(status_match.group(1)) if status_match else None
    status_text = status_match.group(2).strip() if status_match and status_match.group(2) else ""
    if "\nBody:" in result:
        body_text = result.split("\nBody:", 1)[1].lstrip()
    else:
        body_text = result
    try:
        json_body = json.loads(body_text)
        is_json = True
    except (json.JSONDecodeError, TypeError):
        json_body = None
        is_json = False
    return {
        "status_code": status_code,
        "status_text": status_text,
        "body_text": body_text,
        "truncated": "\n...[truncated:" in body_text,
        "is_json": is_json,
        "json_body": json_body,
    }


def api_result_chat_message(prepared: PreparedCall, result: str) -> str:
    """Deterministic fallback/summary for placing an HTTP result back into Chat."""
    parsed = parse_http_result(result)
    status = parsed["status_code"]
    endpoint_name = f"{prepared.endpoint.method} {prepared.endpoint.path}"
    if status is not None and 200 <= status < 300:
        return (
            f"The call to `{endpoint_name}` completed with **HTTP {status}**. "
            "The full Pretty and Raw response is available in **API Call**."
        )

    lines = [
        f"The call to `{endpoint_name}` returned "
        f"**HTTP {status if status is not None else 'request error'}**."
    ]
    body = parsed.get("json_body")
    if isinstance(body, dict):
        if body.get("message"):
            lines.append(str(body["message"]))
        details = body.get("details")
        if isinstance(details, list):
            for detail in details[:8]:
                if not isinstance(detail, dict):
                    continue
                parameter = detail.get("parameter")
                message = detail.get("message")
                if message:
                    prefix = f"`{parameter}`: " if parameter else ""
                    lines.append(f"- {prefix}{message}")
    lines.append(
        "The masked request and response are attached to the Chat context for the next correction."
    )
    return "\n\n".join(lines)


def analyze_api_result(
    spec: A.ApiSpec,
    prepared: PreparedCall,
    result: str,
    trace: Trace | None = None,
) -> str:
    """Ask the grounded local model to diagnose a failed call and draft the next request."""
    endpoint = prepared.endpoint
    evidence = A.endpoint_listing([endpoint], param_detail=True)
    masked_request = prepared.preview()
    safe_result = redact_log_text(result)
    prompt = (
        "Review this failed API call using ONLY the documented endpoint evidence and the actual "
        "server response below. Explain the concrete cause. Then draft the corrected next request "
        "for the same endpoint, including exact query/path values or a JSON body when the evidence "
        "supports them. Never invent a field or claim the corrected request was sent. If the server "
        "error reveals a rule missing from the OpenAPI schema, label it as a server-reported rule. "
        "Use `<credential>` only; never request that a secret be pasted into Chat. If a necessary "
        "value is unknown, name the value the user must supply instead of fabricating it.\n\n"
        f"Documented endpoint:\n{evidence}\n\n"
        f"Failed masked request:\n{masked_request}\n\n"
        f"Actual server result:\n{safe_result[:6000]}"
    )
    _trace(
        trace,
        "api_error_review_input",
        endpoint=_endpoint_name(endpoint),
        masked_request=masked_request,
        response=safe_result[:6000],
    )
    answer, warning = A.grounded_answer_text(prompt, spec)
    candidate_warning = verify_candidate_mentions(answer, [endpoint], spec)
    guidance = answer
    for note in (warning, candidate_warning):
        if note and note not in guidance:
            guidance += f"\n\n{note}"
    _trace(trace, "api_error_review_output", guidance=guidance)
    return guidance


def request_body_example(endpoint: A.Endpoint) -> Any:
    """A conservative spec-derived request starter containing required fields only."""
    schema = (A.ENDPOINT_SCHEMAS.get((endpoint.method.upper(), endpoint.path)) or {}).get("request")
    if not schema:
        return None
    return _strip_read_only(schema, _minimal_request_example(schema))


def _minimal_request_example(schema: dict, depth: int = 0) -> Any:
    """Build the smallest schema-valid draft instead of sending every optional example field."""
    schema = A._deref(schema) or {}
    if depth > 8:
        return None
    if schema.get("allOf"):
        properties = A._object_properties(schema)
        required = A._object_required(schema)
        return {
            name: _minimal_request_example(properties.get(name) or {}, depth + 1)
            for name in required
            if name in properties
        }
    for key in ("oneOf", "anyOf"):
        if schema.get(key):
            return _minimal_request_example(schema[key][0], depth + 1)
    properties = A._object_properties(schema)
    if properties:
        return {
            name: _minimal_request_example(properties.get(name) or {}, depth + 1)
            for name in A._object_required(schema)
            if name in properties
        }
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    if schema.get("enum"):
        return schema["enum"][0]
    if schema.get("type") == "array":
        item = _minimal_request_example(schema.get("items") or {}, depth + 1)
        return [item] if item is not None else []
    return {
        "string": "string",
        "integer": 0,
        "number": 0,
        "boolean": False,
    }.get(schema.get("type"))


def _strip_read_only(schema: dict, value: Any) -> Any:
    schema = A._deref(schema) or {}
    if isinstance(value, dict):
        properties = A._object_properties(schema)
        out = {}
        for key, item in value.items():
            subschema = A._deref(properties.get(key) or {}) or {}
            if subschema.get("readOnly"):
                continue
            out[key] = _strip_read_only(subschema, item)
        return out
    if isinstance(value, list):
        item_schema = schema.get("items") or {}
        return [_strip_read_only(item_schema, item) for item in value]
    return value


def _request_body_issues(schema: dict, value: Any, path: str = "$") -> list[str]:
    """Validate nested required/read-only request facts that the spec states unambiguously."""
    schema = A._deref(schema) or {}
    for keyword in ("oneOf", "anyOf"):
        alternatives = schema.get(keyword) or []
        if alternatives:
            base = {
                key: item for key, item in schema.items()
                if key not in {"oneOf", "anyOf"}
            }
            branch_results = [
                _request_body_issues(
                    {"allOf": [base, branch]} if base else branch,
                    value,
                    path,
                )
                for branch in alternatives
            ]
            if any(not issues for issues in branch_results):
                return []
            closest = min(branch_results, key=len)
            return [
                f"{path} does not match any documented `{keyword}` request shape",
                *closest,
            ]
    properties = A._object_properties(schema)
    if properties:
        if not isinstance(value, dict):
            return [f"{path} must be a JSON object"]
        issues = []
        for name in A._object_required(schema):
            if name not in value or value[name] in (None, ""):
                issues.append(f"Missing required body field `{path}.{name}`")
        for name, item in value.items():
            subschema = A._deref(properties.get(name) or {}) or {}
            if subschema.get("readOnly"):
                issues.append(f"Body field `{path}.{name}` is read-only and must not be sent")
            elif name in properties:
                issues += _request_body_issues(subschema, item, f"{path}.{name}")
        return issues
    if schema.get("type") == "array":
        if not isinstance(value, list):
            return [f"{path} must be a JSON array"]
        issues = []
        for index, item in enumerate(value):
            issues += _request_body_issues(schema.get("items") or {}, item, f"{path}[{index}]")
        return issues
    return []


def _mask_placeholders(value: str) -> str:
    return (str(value)
            .replace("{{TOKEN}}", "<credential>")
            .replace("%7B%7BTOKEN%7D%7D", "<credential>")
            .replace("%7b%7bTOKEN%7d%7d", "<credential>"))


def redact_log_text(value: str) -> str:
    """Best-effort redaction for operational logs. UI credential fields are never logged at all."""
    text = _mask_placeholders(str(value))
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/\-=]+", r"\1 <redacted>", text)
    text = re.sub(
        r"(?i)(authorization|api[_-]?key|access[_-]?token|token)"
        r"([\"']?\s*[:=]\s*[\"']?)([^,\s}\"']+)",
        r"\1\2<redacted>",
        text,
    )
    return text
