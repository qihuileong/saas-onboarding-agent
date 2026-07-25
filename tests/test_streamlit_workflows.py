"""Browser-level, network-free regression checks for the Streamlit user journeys.

This complements ``test_ui_service.py``. That file checks service contracts; this one drives the
actual widgets and session state so regressions in reruns, endpoint selection, credentials, call
origin, confirmation, response rendering, and Chat handoff are caught before manual testing.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import ui_service as UI


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "fixtures" / "calendly.yaml"
AUTH_VARIANTS_SPEC = ROOT / "fixtures" / "auth_variants.yaml"
failures = 0


def check(condition: Any, label: str) -> None:
    global failures
    if condition:
        print(f"  [PASS] {label}")
    else:
        failures += 1
        print(f"  [FAIL] {label}")


def widget_with_label(elements: Iterable[Any], label: str) -> Any:
    return next(element for element in elements if element.label == label)


def button(at: AppTest, label: str) -> Any:
    return widget_with_label(at.button, label)


def text_input(at: AppTest, label: str) -> Any:
    return widget_with_label(at.text_input, label)


def selectbox(at: AppTest, label: str) -> Any:
    return widget_with_label(at.selectbox, label)


def assert_no_exceptions(at: AppTest, label: str) -> None:
    values = [exception.value for exception in at.exception]
    check(not values, label + (f": {values}" if values else ""))


def fresh_onboarded_app(spec_path: Path = SPEC) -> AppTest:
    at = AppTest.from_file("streamlit_app.py", default_timeout=30).run()
    text_input(at, "Docs URL or local spec path").input(str(spec_path.resolve()))
    button(at, "Onboard API").click()
    at.run()
    return at


def choose_endpoint(at: AppTest, endpoint: str) -> None:
    selectbox(at, "Endpoint").select(endpoint)
    at.run()


def save_pat(at: AppTest, value: str = "browser-only-test-secret") -> None:
    text_input(at, "Personal access token (PAT)").input(value)
    at.run()


def prepare(at: AppTest) -> None:
    button(at, "Prepare request").click()
    at.run()


def confirm(at: AppTest) -> None:
    widget_with_label(
        at.checkbox,
        "I understand this sends a real request to the API.",
    ).check()
    at.run()


def send(at: AppTest) -> None:
    button(at, "Send real request").click()
    at.run()


print("=== 1. empty and onboarded application ===")
empty = AppTest.from_file("streamlit_app.py", default_timeout=30).run()
assert_no_exceptions(empty, "the empty application renders without an exception")
check(
    any("Enter an API documentation URL" in info.value for info in empty.info),
    "the empty state tells the user how to begin",
)
check(
    button(empty, "Onboard API").disabled,
    "onboarding is disabled until a source is entered",
)

onboarded = fresh_onboarded_app()
assert_no_exceptions(onboarded, "the Calendly fixture onboards through the browser")
check(
    all(
        label in [tab.label for tab in onboarded.tabs]
        for label in ["Chat", "Endpoint Explorer", "API Call", "Debug & Traces"]
    ),
    "the four primary work areas are clearly separated",
)
check(
    onboarded.session_state["spec"] is not None
    and len(onboarded.session_state["spec"].endpoints) == 61,
    "onboarding preserves all 61 fixture endpoints in session state",
)
check(
    any("Never paste" in warning.value for warning in onboarded.warning),
    "Chat visibly warns against pasting credentials",
)
original_source = onboarded.session_state["source"]
text_input(onboarded, "Docs URL or local spec path").input(
    str((ROOT / "fixtures" / "missing-openapi.yaml").resolve())
)
button(onboarded, "Onboard API").click()
onboarded.run()
assert_no_exceptions(onboarded, "a failed replacement onboarding stays inside the application")
check(
    onboarded.session_state["source"] == original_source
    and len(onboarded.session_state["spec"].endpoints) == 61,
    "a failed replacement onboarding preserves the last usable API",
)
check(
    any("Onboarding failed" in error.value for error in onboarded.error),
    "a failed replacement onboarding gives visible error feedback",
)


print("\n=== 2. standalone successful API Call ===")
manual_success = fresh_onboarded_app()
choose_endpoint(manual_success, "GET /users/me")
check(
    manual_success.session_state["call_origin"] == "manual",
    "choosing an endpoint in API Call marks the request standalone",
)
check(
    any("Standalone API Call" in caption.value for caption in manual_success.caption),
    "the UI explains that a standalone result will stay on the page",
)
save_pat(manual_success)
prepare(manual_success)
assert_no_exceptions(manual_success, "a standalone request prepares cleanly")
pending_preview = manual_success.session_state["pending_call"].preview()
check(
    "<credential>" in pending_preview
    and "browser-only-test-secret" not in pending_preview,
    "the prepared preview contains a placeholder rather than the PAT",
)
check(
    button(manual_success, "Send real request").disabled,
    "sending remains disabled until explicit confirmation",
)
confirm(manual_success)
check(
    not button(manual_success, "Send real request").disabled,
    "explicit confirmation enables the send action",
)

original_execute_call = UI.execute_call
original_analyze_api_result = UI.analyze_api_result
analysis_calls: list[str] = []


def fake_success(prepared: UI.PreparedCall, credential: str = "") -> str:
    check(
        credential == "browser-only-test-secret",
        "the executor receives the session credential only at send time",
    )
    return (
        'Status: 200\nBody: {"resource":{"name":"Test User","active":true},'
        '"echo":"browser-only-test-secret"}'
    )


def unexpected_analysis(*args: Any, **kwargs: Any) -> str:
    analysis_calls.append("unexpected")
    return "This should not be called for a successful standalone request."


UI.execute_call = fake_success
UI.analyze_api_result = unexpected_analysis
messages_before_manual_success = len(manual_success.session_state["messages"])
try:
    send(manual_success)
finally:
    UI.execute_call = original_execute_call
    UI.analyze_api_result = original_analyze_api_result
assert_no_exceptions(manual_success, "a mocked successful request renders cleanly")
check(
    len(manual_success.session_state["messages"]) == messages_before_manual_success,
    "a standalone success does not add noise to Chat",
)
check(
    manual_success.session_state["last_api_exchange"]["origin"] == "manual"
    and not manual_success.session_state["last_api_exchange"]["available_to_chat"],
    "a standalone result is explicitly excluded from Chat context",
)
check(not analysis_calls, "a successful standalone call does not invoke the agent")
check(
    any(success.value == "HTTP 200" for success in manual_success.success),
    "the response area shows a concise HTTP status",
)
check(
    any(
        '"resource": {' in code.value and '"Test User"' in code.value
        for code in manual_success.code
    ),
    "the Pretty response view renders indented JSON",
)
debug_artifacts = json.dumps(
    {
        "traces": manual_success.session_state["traces"],
        "messages": manual_success.session_state["messages"],
        "exchange": manual_success.session_state["last_api_exchange"],
    },
    default=str,
)
check(
    "browser-only-test-secret" not in debug_artifacts
    and "<redacted>" in debug_artifacts,
    "a credential reflected under an unexpected response key is redacted from debug artifacts",
)

choose_endpoint(manual_success, "GET /event_types")
check(
    manual_success.session_state["pending_call"] is None
    and manual_success.session_state["last_call_result"] is None,
    "changing endpoints clears a stale prepared request and response",
)
check(
    text_input(manual_success, "Personal access token (PAT)").value
    == "browser-only-test-secret",
    "the credential persists across endpoints using the same host and auth scheme",
)
button(manual_success, "Clear credential").click()
manual_success.run()
check(
    text_input(manual_success, "Personal access token (PAT)").value == "",
    "the sidebar can clear the shared session credential",
)
choose_endpoint(manual_success, "GET /users/me")
prepare(manual_success)
check(
    button(manual_success, "Send real request").disabled,
    "clearing a credential immediately prevents a prepared authenticated request from sending",
)
isolated_session = fresh_onboarded_app()
check(
    text_input(isolated_session, "Personal access token (PAT)").value == "",
    "a fresh browser session does not inherit another session's credential",
)


print("\n=== 3. standalone failure and explicit diagnosis ===")
manual_failure = fresh_onboarded_app()
choose_endpoint(manual_failure, "GET /users/me")
save_pat(manual_failure)
prepare(manual_failure)
confirm(manual_failure)
messages_before_manual_failure = len(manual_failure.session_state["messages"])
diagnosis_calls: list[str] = []


def fake_failure(prepared: UI.PreparedCall, credential: str = "") -> str:
    return (
        'Status: 400\nBody: {"message":"The supplied parameters are invalid.",'
        '"details":[{"parameter":"user","message":"invalid"}]}'
    )


def fake_diagnosis(
    spec: Any,
    prepared: UI.PreparedCall,
    result: str,
    trace: Any = None,
) -> str:
    diagnosis_calls.append(result)
    if trace:
        trace(
            "api_error_review_input",
            {"masked_request": prepared.preview(), "response": result},
        )
        trace(
            "api_error_review_output",
            {"guidance": "Use the complete documented user URI."},
        )
    return "Use the complete documented user URI, then prepare the request again."


UI.execute_call = fake_failure
UI.analyze_api_result = fake_diagnosis
try:
    send(manual_failure)
    assert_no_exceptions(manual_failure, "a standalone HTTP error stays renderable")
    check(
        len(manual_failure.session_state["messages"])
        == messages_before_manual_failure,
        "a standalone failure does not enter Chat automatically",
    )
    check(
        not diagnosis_calls,
        "a standalone failure is not analyzed without user consent",
    )
    check(
        any(
            item.label == "Ask agent to diagnose this response"
            for item in manual_failure.button
        ),
        "a failed standalone call offers an explicit diagnosis action",
    )
    button(manual_failure, "Ask agent to diagnose this response").click()
    manual_failure.run()
finally:
    UI.execute_call = original_execute_call
    UI.analyze_api_result = original_analyze_api_result
assert_no_exceptions(manual_failure, "manual diagnosis returns to the UI cleanly")
check(len(diagnosis_calls) == 1, "the agent runs only after the diagnosis action is clicked")
check(
    len(manual_failure.session_state["messages"])
    == messages_before_manual_failure + 1
    and "Suggested correction" in manual_failure.session_state["messages"][-1]["content"],
    "explicit diagnosis adds one correction message to Chat",
)
check(
    manual_failure.session_state["last_api_exchange"]["origin"]
    == "manual_promoted"
    and manual_failure.session_state["last_api_exchange"]["available_to_chat"],
    "explicit diagnosis promotes that result into future Chat context",
)
check(
    any(
        trace["name"] == "diagnose_api_call"
        for trace in manual_failure.session_state["traces"]
    ),
    "manual diagnosis has its own detailed trace",
)


print("\n=== 4. Chat-linked failed call ===")
linked = fresh_onboarded_app()
linked.chat_input[0].set_value("call GET /users/me")
linked.run()
assert_no_exceptions(linked, "Chat can select a real endpoint without an exception")
check(
    linked.session_state["call_endpoint"] == "GET /users/me"
    and linked.session_state["call_origin"] == "chat",
    "the call builder records that Chat selected the endpoint",
)
check(
    any("Linked to Chat" in info.value for info in linked.info),
    "the call builder makes its Chat linkage visible before sending",
)
save_pat(linked, "linked-browser-secret")
prepare(linked)
check(
    linked.session_state["pending_call_origin"] == "chat",
    "the prepared request retains its Chat origin across reruns",
)
confirm(linked)
linked_analysis_calls: list[str] = []


def fake_linked_failure(
    prepared: UI.PreparedCall,
    credential: str = "",
) -> str:
    return 'Status: 422\nBody: {"message":"owner must be a URI"}'


def fake_linked_diagnosis(
    spec: Any,
    prepared: UI.PreparedCall,
    result: str,
    trace: Any = None,
) -> str:
    linked_analysis_calls.append(result)
    if trace:
        trace("api_error_review_output", {"guidance": "Use the full owner URI."})
    return "Use the full owner URI in the corrected request."


UI.execute_call = fake_linked_failure
UI.analyze_api_result = fake_linked_diagnosis
messages_before_linked_send = len(linked.session_state["messages"])
try:
    send(linked)
finally:
    UI.execute_call = original_execute_call
    UI.analyze_api_result = original_analyze_api_result
assert_no_exceptions(linked, "a Chat-linked failure survives the automatic rerun")
check(len(linked_analysis_calls) == 1, "a linked HTTP failure is reviewed automatically")
check(
    len(linked.session_state["messages"]) == messages_before_linked_send + 1
    and "Use the full owner URI" in linked.session_state["messages"][-1]["content"],
    "a linked failure returns its correction guidance to Chat",
)
check(
    linked.session_state["last_api_exchange"]["origin"] == "chat"
    and linked.session_state["last_api_exchange"]["available_to_chat"],
    "the linked response is available to later Chat follow-ups",
)
check(
    "linked-browser-secret"
    not in json.dumps(linked.session_state["traces"], default=str)
    and "linked-browser-secret"
    not in json.dumps(linked.session_state["messages"], default=str),
    "the PAT is absent from traces and Chat messages",
)
check(
    not any(
        item.label == "Ask agent to diagnose this response"
        for item in linked.button
    ),
    "a linked failure does not also show the standalone diagnosis action",
)


print("\n=== 5. Chat failure recovery ===")
chat_recovery = fresh_onboarded_app()
original_answer_question = UI.answer_question


def fail_one_chat_turn(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("synthetic local-model outage")


UI.answer_question = fail_one_chat_turn
try:
    chat_recovery.chat_input[0].set_value("which endpoint creates an event?")
    chat_recovery.run()
finally:
    UI.answer_question = original_answer_question
assert_no_exceptions(chat_recovery, "a Chat backend failure does not crash the page")
check(
    chat_recovery.session_state["messages"][-1]["content"]
    == "Turn failed: synthetic local-model outage",
    "the failed turn remains visible in the transcript",
)
check(
    chat_recovery.session_state["traces"][-1]["name"] == "chat_turn"
    and chat_recovery.session_state["traces"][-1]["status"] == "error",
    "a failed Chat turn produces an actionable error trace",
)
chat_recovery.chat_input[0].set_value("list 2 endpoints")
chat_recovery.run()
assert_no_exceptions(chat_recovery, "Chat accepts a new turn after a backend failure")
check(
    "Endpoints 1-2" in chat_recovery.session_state["messages"][-1]["content"],
    "a deterministic follow-up succeeds without re-onboarding",
)


print("\n=== 6. public and unsupported-auth browser states ===")
auth_variants = fresh_onboarded_app(AUTH_VARIANTS_SPEC)
assert_no_exceptions(auth_variants, "the auth-variants fixture onboards in the browser")
check(
    len(auth_variants.session_state["spec"].endpoints) == 7,
    "the browser retains all auth-variant endpoints",
)
choose_endpoint(auth_variants, "GET /public")
check(
    any(
        "no authentication requirement" in caption.value
        for caption in auth_variants.caption
    ),
    "a public endpoint is clearly labelled as unauthenticated",
)
prepare(auth_variants)
check(
    auth_variants.session_state["pending_call"] is not None
    and not auth_variants.session_state["pending_call"].needs_credential,
    "a public request prepares without a credential",
)
confirm(auth_variants)
check(
    not button(auth_variants, "Send real request").disabled,
    "confirmation alone enables a public request",
)

choose_endpoint(auth_variants, "GET /unsupported")
check(
    button(auth_variants, "Prepare request").disabled,
    "an unsupported authentication scheme disables request preparation",
)
check(
    any(
        "Not available in this call builder" in warning.value
        for warning in auth_variants.warning
    ),
    "the browser explains why unsupported authentication cannot be used",
)


if failures:
    print(f"\n{failures} FAILURE(S)")
    raise SystemExit(1)
print("\nALL PASS")
