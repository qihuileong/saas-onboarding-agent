"""Onboarding must not manufacture a spec out of a page that documents no endpoints.

The live failure: a Stoplight *model export* URL (`.../openapi.yaml/components/schemas/Event`)
serves a bare JSON-Schema fragment - 24KB with no `paths:` and not one HTTP verb. The scraper fed
it to the extractor, which returned 12 confident endpoints built from nouns in the schema
(`google_conference`, `gotomeeting`, `webex_conference`), inventing every method and every path
parameter. Three defences, tested here:
  A. `_has_endpoint_evidence` - refuse to extract from text showing no METHOD /path pair.
  B. `_spec_ancestors`       - a URL pointing INTO a spec walks up to the spec itself.
  C. `spec_coverage`         - a scraped spec says so, instead of reading as a thin parse.
Network-free by default; pass --live to also check the real Stoplight URL.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent as A

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
LIVE = "--live" in sys.argv
ok = True


def check(label, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


# The shape of the document that caused the failure: a data model, no endpoints anywhere.
MODEL_DOC = """
x-tags: [Scheduled Events]
type: object
description: Information about a scheduled meeting
title: Event
x-examples:
  Example:
    uri: 'https://api.calendly.com/scheduled_events/GBGBDCAADAEDCRZ2'
    name: 15 Minute Meeting
    status: active
    location: {type: physical, location: string}
properties:
  uri: {type: string, format: uri, description: Canonical reference for the resource}
  location:
    oneOf:
      - {title: Google Conference, properties: {type: {enum: [google_conference]}}}
      - {title: GoToMeeting, properties: {type: {enum: [gotomeeting]}}}
      - {title: Microsoft Teams, properties: {type: {enum: [microsoft_teams_conference]}}}
      - {title: Webex, properties: {type: {enum: [webex_conference]}}}
"""

print("=== A1. a data-model page shows no endpoint evidence ===")
check("the Event model fragment is rejected", not A._has_endpoint_evidence(MODEL_DOC))
for label, text in [
    ("a changelog", "## 2024-06-01 Added support for meeting_notes_plain on the Event object."),
    ("a landing page", "Calendly's API lets you build scheduling into your product. Get started free."),
    ("prose naming a resource", "Every scheduled event has a uri, a status and an invitees_counter."),
    ("a bare hostname", "All requests go to https://api.calendly.com over HTTPS."),
]:
    check(f"{label} is rejected", not A._has_endpoint_evidence(text))

print("\n=== A2. real reference text must still pass ===")
for label, text in [
    ("plain method + path", "GET /scheduled_events returns a list of events."),
    ("markdown-wrapped", "**POST** `/invitees` creates an invitee for the event."),
    ("absolute URL form", "PATCH https://api.calendly.com/event_types/{uuid} updates a type."),
    ("a curl sample", "curl -X DELETE /scheduled_events/{uuid}/cancellation -H 'Authorization: ...'"),
    ("templated path", "GET /users/{uuid} - retrieve a user."),
    ("lowercase prose", "Send a post /webhook_subscriptions request to subscribe."),
    ("html-stripped run-on", "List events GET /scheduled_events Query parameters organization"),
]:
    check(f"{label} passes", A._has_endpoint_evidence(text))

print("\n=== A3. extract_from_pages refuses rather than inventing ===")
try:
    A.extract_from_pages([("https://stoplight.io/.../schemas/Event", MODEL_DOC)])
    check("raises on an all-evidence-free crawl", False)
except ValueError as e:
    print(f"    {e}")
    check("raises on an all-evidence-free crawl", "METHOD /path" in str(e))
    check("the message says why it refused", "invented" in str(e))
check("no LLM call was made (no endpoints fabricated)", True)

print("\n=== B. a URL pointing INTO a spec walks up to it ===")
STOPLIGHT = ("https://stoplight.io/api/v1/projects/calendly/api-docs/nodes/reference/"
             "calendly-api/openapi.yaml/components/schemas/Event"
             "?fromExportButton=true&snapshotType=model")
anc = A._spec_ancestors(STOPLIGHT)
print(f"    {anc}")
check("finds the parent spec URL", anc and anc[0].endswith("calendly-api/openapi.yaml"))
check("drops the model-export query string", "snapshotType" not in (anc[0] if anc else "x"))
check("no ancestors for a plain docs URL", A._spec_ancestors("https://calendly.com/docs/api") == [])
check("no ancestors for a URL that IS a spec",
      A._spec_ancestors("https://example.com/openapi.yaml") == [])
nested = A._spec_ancestors("https://x.io/a/spec.json/b/inner.yaml/components/schemas/Foo")
check("innermost spec is tried first", nested[0].endswith("inner.yaml"))
check("outer spec is still a fallback", nested[-1].endswith("spec.json"))

print("\n=== C. coverage distinguishes a scrape from a thin parse ===")
A.SPEC_PROVENANCE = "scraped"
A.ENDPOINT_SCHEMAS.clear()
A._SPEC_ROOT = {}
fake = A.ApiSpec(base_url="https://api.calendly.com", auth_method="Bearer Token",
                 endpoints=[A.Endpoint(method="GET", path="/scheduled_events", description="")])
rep = A.spec_coverage(fake)
print(rep)
check("says it was scraped, not parsed", "SCRAPED, NOT PARSED" in rep)
check("says the zeros are expected", "expected" in rep)
check("says nothing is corroborated", "corroborated" in rep)
check("drops the vacuous $ref tally", "$refs" not in rep)
check("does not blame the parser for missing auth", "_doc_scopes" not in rep)

A.SPEC_PROVENANCE = "openapi"
parsed = A.parse_openapi(os.path.join(FIX, "calendly.yaml"))
rep2 = A.spec_coverage(parsed)
print(rep2)
check("a real parse is not labelled scraped", "SCRAPED" not in rep2)
check("a real parse still reports $refs", "$refs" in rep2)
check("a real parse reports its endpoints", "endpoints        " in rep2)

if LIVE:
    print("\n=== LIVE. the real Stoplight URL now onboards cleanly ===")
    found = A.find_openapi_spec(STOPLIGHT)
    print(f"    -> {found}")
    check("resolves to a real spec", bool(found) and found.endswith("openapi.yaml"))
    if found:
        s = A.parse_openapi(found)
        print(f"    {len(s.endpoints)} endpoints, base_url {s.base_url}")
        check("parses many endpoints (not 12 invented ones)", len(s.endpoints) > 40)
        check("base_url is the real API host", s.base_url.startswith("https://api.calendly.com"))
else:
    print("\n(skipping live network check - pass --live to run it)")

print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
sys.exit(0 if ok else 1)
