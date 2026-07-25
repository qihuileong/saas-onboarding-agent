"""Network-free checks for the Streamlit-facing workflows."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as A
import ui_service as UI


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "fixtures" / "calendly.yaml"
AUTH_VARIANTS_SPEC = ROOT / "fixtures" / "auth_variants.yaml"
failures = 0


def check(condition, label):
    global failures
    if condition:
        print(f"  [PASS] {label}")
    else:
        failures += 1
        print(f"  [FAIL] {label}")


print("=== 1. onboarding progress is real staged progress ===")
events = []
spec = A.onboard(str(SPEC), progress=lambda percent, message: events.append((percent, message)))
check(bool(spec.endpoints), "the local OpenAPI fixture onboards")
check(events[0][0] == 2, "progress begins at the onboarding stage")
check(events[-1][0] == 100, "progress reaches 100 only after parsing")
check(len({percent for percent, _ in events}) >= 4, "progress contains multiple real stages")


print("\n=== 2. terminal-free chat routes ===")
trace_spans = []
listing = UI.answer_question(
    spec,
    "list 3 endpoints",
    trace=lambda name, payload: trace_spans.append((name, payload)),
)
check(listing.route == "listing", "endpoint listing is deterministic")
check("Endpoints 1-3" in listing.text, "the requested page is returned")
check(any(name == "routing_input" for name, _ in trace_spans),
      "the trace records the complete routing input")
check(any(name == "route" and payload.get("route") == "listing"
          for name, payload in trace_spans),
      "the trace records the selected route")

compound_matches = A._find_endpoints(spec, "create an event and add attendees", limit=8)
compound_names = {(endpoint.method, endpoint.path) for endpoint in compound_matches}
check(("POST", "/invitees") in compound_names,
      "attendee language deterministically retrieves Calendly's invitees endpoint")
check(any(method == "POST" and "event_type" in path for method, path in compound_names),
      "the same compound request also retrieves an event-type creation endpoint")
workflow_candidates = [
    next(endpoint for endpoint in spec.endpoints if endpoint.method == method and endpoint.path == path)
    for method, path in (
        ("POST", "/scheduling_links"),
        ("GET", "/user_availability_schedules"),
        ("POST", "/one_off_event_types"),
        ("POST", "/invitees"),
    )
]
original_ensure_index = A.ensure_endpoint_index
original_search_endpoints = A.search_endpoints
original_live_answer = A.grounded_answer_text
A.ensure_endpoint_index = lambda current_spec: None
A.search_endpoints = lambda query, k=4: workflow_candidates
A.grounded_answer_text = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("workflow briefing must not ask the model to invent handoffs")
)
try:
    workflow_reply = UI.answer_question(
        spec,
        "create a one-off scheduling link and then book an attendee",
    )
finally:
    A.ensure_endpoint_index = original_ensure_index
    A.search_endpoints = original_search_endpoints
    A.grounded_answer_text = original_live_answer
check(
    workflow_reply.route == "workflow-brief"
    and "does **not** establish an execution order" in workflow_reply.text,
    "explicit sequence questions withhold cross-endpoint handoffs the spec does not prove",
)
check(
    "GET /user_availability_schedules" not in workflow_reply.text
    and all(
        path in workflow_reply.text
        for path in ("/scheduling_links", "/one_off_event_types", "/invitees")
    ),
    "the workflow brief removes method-incompatible retrieval noise",
)
selected_focus = UI.focus_from_answer(
    "First use POST /one_off_event_types, then POST /invitees.",
    compound_matches,
    spec,
)
check(selected_focus is not None and selected_focus.path == "/one_off_event_types",
      "follow-up focus uses the endpoint selected in the answer, not retrieval rank one")
discovery_evidence = A.endpoint_listing(compound_matches, include_notes=True)
check("Creates a new booking for an event invitee" in discovery_evidence,
      "discovery evidence includes the documented invitee booking semantics")

event_types = next(
    endpoint for endpoint in spec.endpoints
    if endpoint.method == "GET" and endpoint.path == "/event_types/{uuid}"
)
example = UI.answer_question(
    spec,
    "show the response schema for this endpoint",
    last_endpoint=event_types,
)
check(example.route == "example", "schema follow-up uses the deterministic example route")
check("GET /event_types/{uuid}" in example.text, "the example stays on the focus endpoint")

one_off = next(
    endpoint for endpoint in spec.endpoints
    if endpoint.method == "POST" and endpoint.path == "/one_off_event_types"
)
call_followup = UI.answer_question(
    spec,
    "ok, i want to call this",
    last_endpoint=one_off,
)
check(call_followup.route == "call", "'call this' routes to the real-call builder")
check(call_followup.focus == one_off, "the call follow-up preserves the selected endpoint")
named_call = UI.answer_question(spec, "call GET /users/me")
check(named_call.route == "call", "'call METHOD /path' routes directly to the call builder")

candidate_warning = UI.verify_candidate_mentions(
    "You could then use POST /invitees.",
    [one_off],
    spec,
)
check("POST /invitees" in candidate_warning and "NOT in" in candidate_warning,
      "a real endpoint outside the retrieved candidates is flagged")

invitees = next(
    endpoint for endpoint in spec.endpoints
    if endpoint.method == "POST" and endpoint.path == "/invitees"
)
invitee_example = A.format_example(invitees, "what is the payload?")
check("Creates a new booking" in invitee_example,
      "payload output explains what the invitees endpoint actually does")
create_event_type = next(
    endpoint for endpoint in spec.endpoints
    if endpoint.method == "POST" and endpoint.path == "/event_types"
)
create_event_type_body = UI.request_body_example(create_event_type)
check(
    set(create_event_type_body) == {"owner", "name"},
    "call-builder starters include required request fields without unsafe optional combinations",
)


print("\n=== 3. safe real-call preparation ===")
prepared = UI.prepare_call(
    event_types,
    spec.base_url,
    path_values={"uuid": "123e4567-e89b-12d3-a456-426614174000"},
)
check("{uuid}" not in prepared.url, "the user-supplied path value replaces the template")
check("123e4567-e89b-12d3-a456-426614174000" in prepared.url,
      "the concrete UUID appears in the request URL")
check("{{TOKEN}}" in str(prepared.headers) or "{{TOKEN}}" in prepared.url,
      "the prepared request contains only a credential placeholder")
check("{{TOKEN}}" not in prepared.preview() and "<credential>" in prepared.preview(),
      "the displayed preview masks the placeholder")

try:
    UI.prepare_call(event_types, spec.base_url)
except ValueError as exc:
    missing_path_rejected = "uuid" in str(exc)
else:
    missing_path_rejected = False
check(missing_path_rejected, "a missing required path value is rejected before sending")

list_event_types = next(
    endpoint for endpoint in spec.endpoints
    if endpoint.method == "GET" and endpoint.path == "/event_types"
)
any_of_groups = (
    A.ENDPOINT_SCHEMAS[("GET", "/event_types")].get("param_any_of") or []
)
check(
    any(group["names"] == ["organization", "user"] for group in any_of_groups),
    "the prose-only organization-or-user requirement is captured from the spec",
)
try:
    UI.prepare_call(
        list_event_types,
        spec.base_url,
        query_values={"sort": "name:asc", "count": "20"},
        auth_scheme="personal_access_token",
    )
except ValueError as exc:
    missing_required_choice_rejected = (
        "organization" in str(exc) and "user" in str(exc)
    )
else:
    missing_required_choice_rejected = False
check(
    missing_required_choice_rejected,
    "GET /event_types is blocked unless organization or user is supplied",
)
try:
    UI.prepare_call(
        list_event_types,
        spec.base_url,
        query_values={"user": "CCDGSPAYTLACARJL"},
        auth_scheme="personal_access_token",
    )
except ValueError as exc:
    bare_user_id_rejected = "complete URI" in str(exc)
else:
    bare_user_id_rejected = False
check(
    bare_user_id_rejected,
    "a bare Calendly user ID is rejected when the spec requires a URI",
)
event_type_list_call = UI.prepare_call(
    list_event_types,
    spec.base_url,
    query_values={
        "user": "https://api.calendly.com/users/AAAAAAAAAAAAAAAA",
        "sort": "name:asc",
        "count": "20",
    },
    auth_scheme="personal_access_token",
)
check(
    "user=https%3A%2F%2Fapi.calendly.com%2Fusers%2F" in event_type_list_call.url,
    "supplying the documented user URI satisfies the required choice",
)

calendly_auth = A.credential_options(
    A.endpoint_auth_schemes(event_types.method, event_types.path)
)
check(
    {option["label"] for option in calendly_auth}
    == {"OAuth 2.0 access token", "Personal access token"},
    "credential labels come from Calendly's documented OAuth and PAT schemes",
)
pat_call = UI.prepare_call(
    event_types,
    spec.base_url,
    path_values={"uuid": "123e4567-e89b-12d3-a456-426614174000"},
    auth_scheme="personal_access_token",
)
check(
    pat_call.auth_scheme == "personal_access_token"
    and pat_call.headers.get("Authorization") == "Bearer {{TOKEN}}",
    "the selected documented auth alternative controls credential placement",
)
try:
    UI.execute_call(pat_call)
except ValueError as exc:
    missing_selected_auth_rejected = "requires a credential" in str(exc)
else:
    missing_selected_auth_rejected = False
check(
    missing_selected_auth_rejected,
    "a documented authenticated request cannot execute without its credential",
)
original_http_call = A._do_http_call
observed_request_token = {"value": ""}


def fake_http_call(
    url,
    method="GET",
    headers=None,
    body=None,
    check_grounding=True,
    response_limit=500,
):
    observed_request_token["value"] = A.SECRETS.get("api.calendly.com", "")
    observed_request_token["response_limit"] = response_limit
    return 'Status: 200\nBody: {"ok":true}'


A.SECRETS.pop("api.calendly.com", None)
A._do_http_call = fake_http_call
try:
    UI.execute_call(pat_call, "session-only-test-token")
finally:
    A._do_http_call = original_http_call
check(
    observed_request_token["value"] == "session-only-test-token",
    "the UI credential is available during the HTTP request",
)
check(
    observed_request_token["response_limit"] is None,
    "browser calls request the full response body instead of the model-context excerpt",
)
check(
    "api.calendly.com" not in A.SECRETS,
    "the UI credential is removed from the process-global cache after the request",
)
try:
    UI.prepare_call(
        A.Endpoint(method="GET", path="/protected", description="Protected endpoint"),
        "https://example.test",
        auth_required=True,
    )
except ValueError as exc:
    missing_auth_metadata_rejected = "will not guess" in str(exc)
else:
    missing_auth_metadata_rejected = False
check(
    missing_auth_metadata_rejected,
    "missing auth placement is rejected instead of guessed as Bearer",
)

synthetic = A.Endpoint(method="POST", path="/widgets", description="Create a widget")
A.ENDPOINT_SCHEMAS[("POST", "/widgets")] = {
    "request": {
        "type": "object",
        "required": ["name", "settings"],
        "properties": {
            "id": {"type": "string", "readOnly": True, "example": "server-id"},
            "name": {"type": "string", "example": "demo"},
            "settings": {
                "type": "object",
                "required": ["enabled"],
                "properties": {"enabled": {"type": "boolean", "example": True}},
            },
        },
    },
    "response": None,
    "params": [],
    "auth_schemes": [],
    "body_required": True,
    "content_type": "application/json",
}
safe_example = UI.request_body_example(synthetic)
check("id" not in safe_example, "read-only fields are removed from the call-builder example")
try:
    UI.prepare_call(
        synthetic,
        "https://example.test",
        body={"id": "should-not-send", "name": "demo", "settings": {"enabled": True}},
    )
except ValueError as exc:
    read_only_rejected = "read-only" in str(exc)
else:
    read_only_rejected = False
check(read_only_rejected, "a user-added read-only body field is rejected before sending")

try:
    UI.prepare_call(
        synthetic,
        "https://example.test",
        body={"name": "demo", "settings": {}},
    )
except ValueError as exc:
    nested_required_rejected = "$.settings.enabled" in str(exc)
else:
    nested_required_rejected = False
check(nested_required_rejected, "nested required body fields are validated before sending")
A.ENDPOINT_SCHEMAS.pop(("POST", "/widgets"))


print("\n=== 4. log redaction ===")
redacted = UI.redact_log_text("Authorization: Bearer secret-value token=abc123")
check("secret-value" not in redacted and "abc123" not in redacted,
      "bearer tokens and token fields are redacted")

pretty_result = UI.parse_http_result(
    'Status: 400\nBody: {"message":"Invalid input","details":[{"parameter":"user"}]}'
)
check(
    pretty_result["status_code"] == 400
    and pretty_result["status_text"] == ""
    and pretty_result["is_json"]
    and pretty_result["json_body"]["details"][0]["parameter"] == "user",
    "HTTP results are split into status and structured JSON for pretty rendering",
)
text_result = UI.parse_http_result("Status: 502\nBody: upstream service unavailable")
check(
    text_result["status_code"] == 502
    and not text_result["is_json"]
    and text_result["body_text"] == "upstream service unavailable",
    "non-JSON response bodies remain available as wrapped text",
)
old_truncated_result = UI.parse_http_result(
    'Status: 200\nBody: {"collection":[{"name":"partial"}]\n'
    "...[truncated: showing 30 of 300 chars of response body.]"
)
check(
    old_truncated_result["truncated"] and not old_truncated_result["is_json"],
    "older truncated results are identified instead of presented as valid JSON",
)
failed_chat_update = UI.api_result_chat_message(
    event_type_list_call,
    'Status: 400\nBody: {"message":"Invalid input","details":['
    '{"parameter":"user","message":"must be a complete URI"}]}',
)
check(
    "HTTP 400" in failed_chat_update
    and "`user`: must be a complete URI" in failed_chat_update
    and "Chat context" in failed_chat_update,
    "failed HTTP results produce a useful credential-safe Chat summary",
)
successful_chat_update = UI.api_result_chat_message(
    event_type_list_call,
    'Status: 200\nBody: {"collection":[]}',
)
check(
    "HTTP 200" in successful_chat_update
    and "Pretty and Raw response" in successful_chat_update,
    "successful HTTP results are also returned to Chat",
)
original_grounded_answer = A.grounded_answer_text
captured_error_prompt = {"value": ""}


def fake_grounded_answer(prompt, current_spec):
    captured_error_prompt["value"] = prompt
    return (
        "Use the complete documented user URI and prepare the request again.",
        "",
    )


A.grounded_answer_text = fake_grounded_answer
try:
    error_followup = UI.answer_question(
        spec,
        "why did that fail and how do I fix it?",
        last_endpoint=list_event_types,
        last_api_exchange={
            "endpoint": "GET /event_types",
            "status_code": 400,
            "masked_request_preview": (
                "GET https://api.calendly.com/event_types?user=bad-id\n"
                'headers:\n{"Authorization":"Bearer <credential>"}'
            ),
            "response": (
                'Status: 400\nBody: {"details":['
                '{"parameter":"user","message":"invalid"}]}'
            ),
        },
    )
    error_followup_prompt = captured_error_prompt["value"]
    captured_error_prompt["value"] = ""
    manual_call_followup = UI.answer_question(
        spec,
        "why did that fail and how do I fix it?",
        last_endpoint=list_event_types,
        last_api_exchange={
            "endpoint": "GET /event_types",
            "status_code": 400,
            "masked_request_preview": "GET /event_types?user=manual-bad-id",
            "response": "Status: 400\nBody: manual-only-error",
            "origin": "manual",
            "available_to_chat": False,
        },
    )
    manual_followup_prompt = captured_error_prompt["value"]
finally:
    A.grounded_answer_text = original_grounded_answer
check(
    error_followup.route == "api-error-follow-up"
    and error_followup.focus == list_event_types,
    "a later Chat question stays attached to the most recent failed endpoint",
)
check(
    "user=bad-id" in error_followup_prompt
    and '"message":"invalid"' in error_followup_prompt
    and "Bearer <credential>" in error_followup_prompt,
    "the next Chat turn receives the masked request and actual error response",
)
check(
    manual_call_followup.route != "api-error-follow-up"
    and "manual-only-error" not in manual_followup_prompt,
    "a standalone API Call does not silently enter the Chat context",
)


print("\n=== 5. authentication and request-shape variants ===")
variant_spec = A.parse_openapi(str(AUTH_VARIANTS_SPEC))


def variant_endpoint(method, path):
    return next(
        endpoint for endpoint in variant_spec.endpoints
        if endpoint.method == method and endpoint.path == path
    )


inherited = variant_endpoint("GET", "/inherited")
public = variant_endpoint("GET", "/public")
query_key = variant_endpoint("GET", "/query-key")
cookie_key = variant_endpoint("GET", "/cookie-key")
basic = variant_endpoint("GET", "/basic")
unsupported = variant_endpoint("GET", "/unsupported")
choice = variant_endpoint("POST", "/choice")

check(
    A.endpoint_auth_schemes("GET", "/inherited") == ["HeaderKey"],
    "document-level authentication is inherited by operations",
)
check(
    A.endpoint_auth_schemes("GET", "/public") == []
    and A.ENDPOINT_SCHEMAS[("GET", "/public")]["security_explicit_none"],
    "an explicit empty operation security list remains public",
)
public_call = UI.prepare_call(public, variant_spec.base_url)
check(
    not public_call.needs_credential and "<credential>" not in public_call.preview(),
    "a public operation prepares without inventing authentication",
)

header_call = UI.prepare_call(
    inherited,
    variant_spec.base_url,
    auth_scheme="HeaderKey",
)
check(
    header_call.headers.get("X-API-Key") == "{{TOKEN}}"
    and "X-API-Key" in header_call.preview()
    and "{{TOKEN}}" not in header_call.preview(),
    "a header API key uses the documented header and a masked preview",
)
query_call = UI.prepare_call(
    query_key,
    variant_spec.base_url,
    query_values={"tags": ["red", "blue"]},
    auth_scheme="QueryKey",
)
check(
    "tags=red%2Cblue" in query_call.url
    and "api_key=%7B%7BTOKEN%7D%7D" in query_call.url,
    "an explode=false query array is comma-serialized beside the query API key",
)
check(
    "api_key=<credential>" in query_call.preview()
    and "{{TOKEN}}" not in query_call.preview(),
    "a query-placed credential is masked after URL encoding",
)
cookie_call = UI.prepare_call(
    cookie_key,
    variant_spec.base_url,
    auth_scheme="CookieKey",
)
check(
    cookie_call.headers.get("Cookie") == "session_key={{TOKEN}}"
    and "session_key=<credential>" in cookie_call.preview(),
    "a cookie API key uses the documented cookie name and a masked preview",
)
basic_call = UI.prepare_call(
    basic,
    variant_spec.base_url,
    auth_scheme="BasicAuth",
)
check(
    basic_call.headers.get("Authorization") == "Basic {{TOKEN}}",
    "HTTP Basic uses the documented Authorization template",
)
try:
    UI.prepare_call(
        unsupported,
        variant_spec.base_url,
        auth_scheme="DigestAuth",
    )
except ValueError as exc:
    unsupported_rejected = "No supported credential placement" in str(exc)
else:
    unsupported_rejected = False
check(
    unsupported_rejected,
    "an unsupported authentication scheme is refused before request execution",
)
try:
    UI.prepare_call(
        query_key,
        variant_spec.base_url,
        auth_scheme="HeaderKey",
    )
except ValueError as exc:
    undocumented_scheme_rejected = "not documented" in str(exc)
else:
    undocumented_scheme_rejected = False
check(
    undocumented_scheme_rejected,
    "a caller cannot substitute an auth scheme from another endpoint",
)

second_choice = UI.prepare_call(
    choice,
    variant_spec.base_url,
    body={"team_id": "team-123"},
)
check(
    second_choice.body == {"team_id": "team-123"},
    "a valid second oneOf request-body branch is accepted",
)
try:
    UI.prepare_call(
        choice,
        variant_spec.base_url,
        body={"unrelated": "value"},
    )
except ValueError as exc:
    bad_choice_rejected = (
        "does not match any documented `oneOf`" in str(exc)
        and ("person_id" in str(exc) or "team_id" in str(exc))
    )
else:
    bad_choice_rejected = False
check(
    bad_choice_rejected,
    "a body matching no oneOf branch is rejected with the nearest missing field",
)

original_request = A.requests.request
observed_http = {}


class FakeResponse:
    status_code = 200
    text = '{"ok":true}'


def fake_request(method, url, headers, json, timeout):
    observed_http.update(method=method, url=url, headers=headers, body=json, timeout=timeout)
    return FakeResponse()


A.requests.request = fake_request
A.SECRETS["api.example.test"] = "preexisting-cli-token"
try:
    query_result = UI.execute_call(query_call, "session query/key")
finally:
    A.requests.request = original_request
check(
    "api_key=session%20query%2Fkey" in observed_http.get("url", "")
    and query_result.startswith("Status: 200"),
    "the real HTTP boundary substitutes and URL-encodes a query credential",
)
check(
    A.SECRETS.get("api.example.test") == "preexisting-cli-token",
    "request execution restores a pre-existing non-UI credential cache entry",
)
A.SECRETS.pop("api.example.test", None)


if failures:
    print(f"\n{failures} FAILURE(S)")
    raise SystemExit(1)
print("\nALL PASS")
