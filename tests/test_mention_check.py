"""verify_endpoint_mentions: does the output-side grounding check catch an invented endpoint
without flagging real ones? The positive case is the REAL answer from a live session, where the
model turned `GET /scheduled_events` (in its context) into `POST /scheduled_events` (not in it).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent as A

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "calendly.yaml")
spec = A.parse_openapi(SPEC)
ok = True


def check(label, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


print("=== 1. the live hallucination ===")
LIVE = """To achieve your goal of creating an event and adding someone to it:
1. **Create an Event Type**: define a new event_type using the `POST /event_types` endpoint.
2. **Create an Event**: use the `POST /scheduled_events` endpoint to create a specific instance.
   Required body fields: `event_type`, `start_time`, and optionally `location`.
3. **Add Someone**: use the `POST /invitees` endpoint by specifying the person's details.
"""
w = A.verify_endpoint_mentions(LIVE, spec)
print(w)
check("flags the invented POST /scheduled_events", "POST /scheduled_events does not exist" in w)
check("names the real methods on that path", "GET" in w)
check("does not flag POST /event_types", "POST /event_types does not exist" not in w)
check("does not flag POST /invitees", "POST /invitees does not exist" not in w)

print("\n=== 2. real answers must stay silent ===")
CLEAN = [
    "Use `GET /scheduled_events` to list events, then `GET /scheduled_events/{uuid}` for one.",
    "**POST** `/invitees` creates an invitee. **PATCH** /event_types/{uuid} updates a type.",
    "Cancel with POST /scheduled_events/{uuid}/cancellation.",
    "Call POST https://api.calendly.com/one_off_event_types to make a one-off type.",
    "The endpoint GET /users/me returns the current user.",
    "Templating differs but means the same slot: GET /event_types/{id}",
    "Concrete UUID examples should match templates: GET /event_types/123e4567-e89b-12d3-a456-426614174000",
    "Full concrete URLs should match templates: GET https://api.calendly.com/event_types/123e4567-e89b-12d3-a456-426614174000",
    "Trailing punctuation should not break it: see GET /groups, and POST /shares.",
    "Prose that names no path: use GET on the events collection.",
]
for txt in CLEAN:
    r = A.verify_endpoint_mentions(txt, spec)
    check(f"silent: {txt[:58]}...", r == "")

print("\n=== 3. other invented forms ===")
CASES = [
    ("DELETE /scheduled_events/{uuid}", "wrong verb on a real path"),
    ("PUT /event_types", "verb the spec never documents on that path"),
    ("GET /calendars", "path that does not exist at all"),
]
for txt, label in CASES:
    r = A.verify_endpoint_mentions(txt, spec)
    check(f"flags {label}: {txt}", "does not exist" in r)

print("\n=== 4. a nonexistent path suggests real neighbours ===")
r = A.verify_endpoint_mentions("Use GET /scheduled_events/upcoming for that.", spec)
print(r)
check("offers real endpoints under the same head", "/scheduled_events" in r)

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
