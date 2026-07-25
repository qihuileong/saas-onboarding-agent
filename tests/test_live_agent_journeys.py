"""Local-model end-to-end journeys.

Unlike the fast regression scripts, this file intentionally invokes the installed Ollama embedding
and chat models. It never makes an external API request and never needs a credential.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as A
import ui_service as UI


SPEC = Path(__file__).resolve().parent / "fixtures" / "calendly.yaml"
failures = 0


def check(condition, label):
    global failures
    if condition:
        print(f"  [PASS] {label}")
    else:
        failures += 1
        print(f"  [FAIL] {label}")


def ask(spec, text, **context):
    started = time.perf_counter()
    reply = UI.answer_question(spec, text, **context)
    print(
        f"\n> {text}\n"
        f"  route={reply.route} focus="
        f"{reply.focus.method + ' ' + reply.focus.path if reply.focus else '(none)'} "
        f"duration={time.perf_counter() - started:.2f}s\n\n{reply.text}\n"
    )
    return reply


print("=== 1. real parsing and semantic retrieval ===")
spec = A.onboard(str(SPEC))
check(len(spec.endpoints) == 61, "the pinned Calendly spec parses all 61 endpoints")

workflow = ask(
    spec,
    "I need to create a one-off scheduling link and then book an attendee",
)
check(
    workflow.route == "workflow-brief",
    "an explicit sequence uses the deterministic workflow-safety route",
)
check(
    "does **not** establish an execution order" in workflow.text
    and "GET /user_availability_schedules" not in workflow.text,
    "the workflow answer withholds unproved handoffs and filters method noise",
)


print("\n=== 2. real local-model discovery ===")
booking = ask(spec, "Which endpoint books an invitee directly from my app?")
check(
    booking.route == "discovery"
    and booking.focus is not None
    and booking.focus.method == "POST"
    and booking.focus.path == "/invitees",
    "semantic retrieval plus the local model selects POST /invitees",
)
check(
    "POST /invitees" in booking.text
    and "[!] grounding check" not in booking.text
    and "[!] retrieval grounding" not in booking.text,
    "the discovery answer is useful and passes both endpoint grounding gates",
)


print("\n=== 3. real local-model endpoint detail ===")
detail = ask(
    spec,
    "What auth and required request fields does POST /one_off_event_types use?",
)
check(
    detail.route == "describe"
    and detail.focus is not None
    and detail.focus.path == "/one_off_event_types",
    "an exact method/path stays focused on the named endpoint",
)
check(
    all(field in detail.text for field in ("name", "host", "duration", "date_setting"))
    and "Bearer <credential>" in detail.text,
    "the model reports the documented required fields and masked auth placement",
)

schema = ask(
    spec,
    "show the request schema for this endpoint",
    last_endpoint=detail.focus,
)
check(
    schema.route == "example"
    and "POST /one_off_event_types" in schema.text
    and "request body (example)" in schema.text,
    "an anaphoric schema follow-up uses deterministic spec rendering",
)


print("\n=== 4. no-send call routing and pagination ===")
original_http_call = A._do_http_call
A._do_http_call = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("answer routing must not execute HTTP")
)
try:
    selected = ask(spec, "call GET /users/me")
finally:
    A._do_http_call = original_http_call
check(
    selected.route == "call"
    and selected.focus is not None
    and selected.focus.path == "/users/me",
    "a call request selects the builder without sending HTTP",
)

first_page = ask(spec, "list 2 endpoints")
next_page = ask(spec, "next 2 endpoints", list_cursor=first_page.list_cursor)
check(
    "Endpoints 1-2" in first_page.text
    and "Endpoints 3-4" in next_page.text,
    "endpoint pagination advances deterministically across turns",
)


if failures:
    print(f"\n{failures} FAILURE(S)")
    raise SystemExit(1)
print("\nALL PASS")
