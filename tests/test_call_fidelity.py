"""Call-fidelity audit: does the parse keep everything needed to MAKE A REAL CALL?

Every extraction bug in this project has failed WITHOUT RAISING - an unresolved $ref renders as {},
a Swagger 2.0 body vanishes, an apiKey's header name is dropped and every request goes out as
`Authorization: Bearer` and comes back 401. Nothing errors; the agent just describes an API that
doesn't match the one on the wire, and you find out several turns later.

So this walks each spec fact an HTTP request depends on and asserts the parser CAPTURED it -
comparing the raw document against ENDPOINT_SCHEMAS / AUTH_SCHEMES, not against the parser's own
opinion. Run it against any spec:

    uv run python tests/test_call_fidelity.py [spec ...]

Defaults to the pinned fixtures in tests/fixtures/.
"""
import sys, os, json, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent as A

HTTP = {"get", "post", "put", "patch", "delete", "head", "options"}
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FINDINGS = []


def note(sev, spec, what, detail):
    FINDINGS.append((sev, spec, what, detail))


def ops_of(raw):
    for path, item in (raw.get("paths") or {}).items():
        if isinstance(item, dict):
            for m, op in item.items():
                if m.lower() in HTTP and isinstance(op, dict):
                    yield m.upper(), path, op, item


def audit(name, path):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    parsed = A.parse_openapi(path)
    ops = list(ops_of(raw))
    print(f"raw operations {len(ops)}  ->  parsed endpoints {len(parsed.endpoints)}")
    print(f"auth_method: {parsed.auth_method!r}")

    # --- nothing silently skipped ---------------------------------------------------
    if len(ops) != len(parsed.endpoints):
        note("HIGH", name, "operations dropped",
             f"{len(ops)} in spec vs {len(parsed.endpoints)} parsed")
    reffed = [p for p, it in (raw.get("paths") or {}).items()
              if isinstance(it, dict) and "$ref" in it]
    if reffed:
        note("HIGH", name, "$ref'd path items skipped", f"{len(reffed)}: {reffed[:3]}")

    # --- base URL -------------------------------------------------------------------
    servers = raw.get("servers") or []
    if len(servers) > 1:
        note("LOW", name, "extra servers dropped (only [0] kept)",
             str([s.get("url") for s in servers][:3]))
    if "{" in (parsed.base_url or ""):
        note("HIGH", name, "base_url holds an unsubstituted server variable", parsed.base_url)
    op_servers = [f"{m} {p}" for m, p, op, item in ops if op.get("servers") or item.get("servers")]
    if op_servers:
        note("MED", name, "per-operation servers ignored", f"{len(op_servers)}: {op_servers[:3]}")

    # --- auth: the credential must be SENDABLE, not just named -----------------------
    declared = ((raw.get("components") or {}).get("securitySchemes")
                or raw.get("securityDefinitions") or {})
    for sname, s in (declared or {}).items():
        s = A._deref(s) or {}
        got = A.AUTH_SCHEMES.get(sname)
        if not got:
            note("HIGH", name, "security scheme not parsed", sname)
            continue
        if s.get("type") == "apiKey":
            if got.get("in") != s.get("in") or got.get("param") != s.get("name"):
                note("HIGH", name, "apiKey location lost",
                     f"{sname}: spec says {s.get('in')}/{s.get('name')}, "
                     f"parsed {got.get('in')}/{got.get('param')}")
            else:
                kind, pname, _ = A._credential_placement([sname])
                if (kind, pname) != (s.get("in"), s.get("name")):
                    note("HIGH", name, "apiKey placement not used when building the request",
                         f"{sname}: would send {kind}/{pname}")
        if s.get("type") == "oauth2":
            flows = s.get("flows") or {}
            want = [f.get("tokenUrl") or f.get("authorizationUrl")
                    for f in flows.values() if isinstance(f, dict)]
            want = [u for u in want if u] or [u for u in (s.get("tokenUrl"),
                                                          s.get("authorizationUrl")) if u]
            if want and not (got.get("token_url") or got.get("authorization_url")):
                note("MED", name, "oauth2 token/authorization URL dropped", f"{sname}: {want[:1]}")
    # every authenticated endpoint must yield concrete instructions
    unexplained = [f"{m} {p}" for m, p, op, _ in ops
                   if A.endpoint_auth_schemes(m, p) and not A.auth_instructions(m, p)]
    if unexplained:
        note("HIGH", name, "authenticated endpoint with no usable auth instructions",
             f"{len(unexplained)}: {unexplained[:3]}")

    # --- request body: schema, media type, mandatory-ness ----------------------------
    for m, p, op, _ in ops:
        info = A.ENDPOINT_SCHEMAS.get((m, p)) or {}
        body3 = A._deref(op.get("requestBody") or {}) or {}
        params = [A._deref(x) or {} for x in (op.get("parameters") or [])]
        body2 = [x for x in params if x.get("in") in ("body", "formData")]
        spec_has_body = bool(body3.get("content")) or bool(body2)
        if spec_has_body and info.get("request") is None:
            note("HIGH", name, "request body in spec but not captured",
                 f"{m} {p} (request_required() would claim there is NO body)")
        if spec_has_body and not info.get("content_type"):
            note("MED", name, "request media type not captured", f"{m} {p}")
        want_req = bool(body3.get("required")) or any(x.get("required") for x in body2)
        if want_req and not info.get("body_required"):
            note("LOW", name, "requestBody.required not captured", f"{m} {p}")
        # body params must not leak into the query/path parameter list
        leaked = [q["name"] for q in info.get("params", []) if q["in"] in ("body", "formData")]
        if leaked:
            note("MED", name, "body param leaked into the parameter list", f"{m} {p}: {leaked}")
        # --- response schema: OpenAPI 3 nests under content, Swagger 2.0 does not ----
        for code, r in (op.get("responses") or {}).items():
            r = A._deref(r) or {}
            if str(code).startswith("2") and isinstance(r, dict):
                if (r.get("schema") or (r.get("content") or {})) and info.get("response") is None:
                    note("HIGH", name, "2xx response schema in spec but not captured",
                         f"{m} {p} ({code}) - the endpoint would appear to return nothing")
                break

    # --- parameters: constraints, serialization, deprecation -------------------------
    CONSTRAINTS = ("minimum", "maximum", "minLength", "maxLength", "pattern",
                   "minItems", "maxItems")
    for m, p, op, item in ops:
        got = {q["name"]: q for q in (A.ENDPOINT_SCHEMAS.get((m, p)) or {}).get("params", [])}
        for prm in [A._deref(x) or {} for x in
                    ((item.get("parameters") or []) + (op.get("parameters") or []))]:
            if prm.get("in") in ("body", "formData") or not prm.get("name"):
                continue
            g = got.get(prm["name"])
            if not g:
                note("HIGH", name, "parameter dropped", f"{m} {p}: {prm['name']}")
                continue
            sch = A._deref(prm.get("schema") or {}) or {}
            missing = [c for c in CONSTRAINTS
                       if (c in sch or c in prm) and c not in (g.get("limits") or {})]
            if missing:
                note("MED", name, "param constraint dropped",
                     f"{m} {p}?{prm['name']}: {missing}")
            if prm.get("deprecated") and not g.get("deprecated"):
                note("MED", name, "deprecated param not flagged", f"{m} {p}?{prm['name']}")
            if sch.get("type") == "array" and g.get("explode") is None:
                note("MED", name, "array param serialization dropped",
                     f"{m} {p}?{prm['name']}")

    # --- deprecated operations --------------------------------------------------------
    for m, p, op, _ in ops:
        if op.get("deprecated") and not (A.ENDPOINT_SCHEMAS.get((m, p)) or {}).get("deprecated"):
            note("MED", name, "deprecated operation not flagged", f"{m} {p}")

    # --- nested required must be visible somewhere ------------------------------------
    for m, p, op, _ in ops:
        sch = (A.ENDPOINT_SCHEMAS.get((m, p)) or {}).get("request")
        if not sch:
            continue
        tree = A.request_tree(m, p)
        for k, v in A._object_properties(sch).items():
            v = A._deref(v) or {}
            inner = A._deref(v.get("items") or {}) if v.get("type") == "array" else v
            for r in A._object_required(inner or {}):
                if f"{r} (" not in tree:
                    note("MED", name, "nested required field invisible",
                         f"{m} {p}: {k}.{r} required but absent from the rendered tree")

    # --- readOnly must be marked, not silently offered as a request field -------------
    for m, p, op, _ in ops:
        sch = (A.ENDPOINT_SCHEMAS.get((m, p)) or {}).get("request")
        if not sch:
            continue
        tree = A.request_tree(m, p)
        for k, v in A._object_properties(sch).items():
            if (A._deref(v) or {}).get("readOnly") and "read-only" not in tree:
                note("MED", name, "readOnly request field not marked", f"{m} {p}: {k}")

    # --- AND-style security (known, deliberate) ---------------------------------------
    andreq = [f"{m} {p}" for m, p, op, _ in ops
              for req in (op.get("security") or []) if isinstance(req, dict) and len(req) > 1]
    if andreq:
        note("KNOWN", name, "AND-style security reported as OR",
             f"{len(andreq)} op(s), e.g. {andreq[:1]} - specs that write "
             f"{{a: [], b: []}} almost always mean 'either', and we report either")


def main(paths):
    for pth in paths:
        nm = os.path.basename(pth)
        try:
            audit(nm, pth)
        except Exception as exc:
            note("HIGH", nm, "audit crashed", repr(exc))
            print(f"  !! crashed: {exc}")

    print(f"\n\n{'=' * 78}\nFINDINGS\n{'=' * 78}")
    blocking = 0
    for sev in ("HIGH", "MED", "LOW", "KNOWN"):
        rows = [f for f in FINDINGS if f[0] == sev]
        if not rows:
            continue
        print(f"\n--- {sev} ({len(rows)}) ---")
        seen = set()
        for _, spec, what, detail in rows:
            if (spec, what) in seen:      # collapse repeats, keep the first example
                continue
            seen.add((spec, what))
            n = sum(1 for f in rows if (f[1], f[2]) == (spec, what))
            print(f"  [{spec}] {what}" + (f"  (x{n})" if n > 1 else ""))
            print(f"      {detail}")
        if sev == "HIGH":
            blocking = len(rows)
    if not FINDINGS:
        print("\nNothing dropped that a real call depends on.")
    elif not blocking:
        print(f"\nNo HIGH findings. {len(FINDINGS)} lower-severity note(s) above.")
    return 1 if blocking else 0


if __name__ == "__main__":
    args = sys.argv[1:] or [os.path.join(FIXTURES, f) for f in sorted(os.listdir(FIXTURES))
                            if f.endswith((".json", ".yaml", ".yml"))]
    sys.exit(main(args))
