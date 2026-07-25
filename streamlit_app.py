"""Streamlit UI for the local SaaS onboarding agent."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime
import importlib
import io
import json
import platform
import re
import time
from typing import Any
from urllib.parse import parse_qsl, urlparse
import uuid

import streamlit as st

import agent as A
import ui_service as UI

# Streamlit reruns this file but can retain already-imported local modules. Reload an older service
# contract before rendering so a hot edit cannot leave the page calling a stale function signature.
if getattr(A, "API_VERSION", 0) < 4:
    importlib.reload(A)
if getattr(UI, "API_VERSION", 0) < 11:
    importlib.reload(UI)


st.set_page_config(
    page_title="SaaS API Onboarding Agent",
    page_icon="🔌",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] {font-size: 1.35rem;}
    .muted {color: #6b7280; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "spec": None,
        "agent_runtime": None,
        "runtime_notice": "",
        "source": "",
        "messages": [],
        "last_endpoint": None,
        "list_cursor": 0,
        "logs": [],
        "traces": [],
        "pending_call": None,
        "call_origin": "manual",
        "pending_call_origin": "manual",
        "last_call_result": None,
        "last_api_exchange": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _session_safe_text(value: Any) -> str:
    """Redact credential-shaped text and exact credentials held by this browser session.

    Shape-based redaction covers ordinary Authorization headers and token fields. Exact replacement
    also protects traces and model prompts when an API reflects a credential under an unexpected
    response key such as ``echo``.
    """
    text = UI.redact_log_text(str(value))
    secrets = {
        str(st.session_state[key])
        for key in st.session_state
        if str(key).startswith("credential::") and st.session_state.get(key)
    }
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    return text


def _log(level: str, stage: str, message: str, **details: Any) -> None:
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "level": level.upper(),
        "stage": stage,
        "message": _session_safe_text(message),
    }
    if details:
        record["details"] = {
            key: _session_safe_text(value) for key, value in details.items()
        }
    st.session_state.logs.append(record)
    if len(st.session_state.logs) > 1000:
        del st.session_state.logs[:-1000]


def _log_level_for(message: str, default: str = "INFO") -> str:
    lower = message.lower()
    if any(marker in lower for marker in ("failed", "traceback", "exception")):
        return "ERROR"
    if any(marker in lower for marker in (
        "[!]",
        "warning",
        "unavailable",
        "no machine-readable openapi",
        "falling back",
        "not documented",
    )):
        return "WARNING"
    return default


def _trace_safe(value: Any) -> Any:
    """Recursively remove credential-shaped values while preserving debuggable trace content."""
    if isinstance(value, str):
        return _session_safe_text(value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if any(secret in str(key).lower() for secret in (
                "credential", "password", "access_token", "refresh_token", "api_key",
            )):
                out[key] = "<redacted>"
            else:
                out[key] = _trace_safe(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_trace_safe(item) for item in value]
    return value


def _start_trace(name: str, trace_input: Any, **metadata: Any) -> dict[str, Any]:
    trace = {
        "trace_id": uuid.uuid4().hex,
        "name": name,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "input": _trace_safe(trace_input),
        "output": None,
        "metadata": _trace_safe(metadata),
        "spans": [],
        "_started_perf": time.perf_counter(),
    }
    st.session_state.traces.append(trace)
    if len(st.session_state.traces) > 200:
        del st.session_state.traces[:-200]
    return trace


def _add_span(trace: dict[str, Any], name: str, payload: dict[str, Any]) -> None:
    trace["spans"].append({
        "span_id": uuid.uuid4().hex,
        "name": name,
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        **_trace_safe(payload),
    })


def _finish_trace(
    trace: dict[str, Any],
    *,
    status: str,
    output: Any = None,
    error: str = "",
    **metadata: Any,
) -> None:
    started = trace.pop("_started_perf", time.perf_counter())
    trace["ended_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
    trace["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    trace["status"] = status
    trace["output"] = _trace_safe(output)
    if error:
        trace["error"] = _session_safe_text(error)
    trace["metadata"].update(_trace_safe(metadata))


class _LogWriter(io.TextIOBase):
    """Turn legacy print lines into structured UI log records."""

    def __init__(self, stage: str):
        self.stage = stage
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                clean = line.strip()
                _log(_log_level_for(clean), self.stage, clean)
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            clean = self.buffer.strip()
            _log(_log_level_for(clean), self.stage, clean)
        self.buffer = ""


def _agent_snapshot() -> dict[str, Any]:
    """Keep the loaded API usable if onboarding a replacement source fails."""
    return {
        "schemas": dict(A.ENDPOINT_SCHEMAS),
        "root": A._SPEC_ROOT,
        "auth": dict(A.AUTH_SCHEMES),
        "provenance": A.SPEC_PROVENANCE,
        "corpus": list(A.DOC_CORPUS),
        "index": list(A.ENDPOINT_INDEX),
        "vectors": list(A.ENDPOINT_VECS),
        "indexed_for": A._INDEXED_FOR,
    }


def _restore_agent(snapshot: dict[str, Any]) -> None:
    A.ENDPOINT_SCHEMAS.clear()
    A.ENDPOINT_SCHEMAS.update(snapshot["schemas"])
    A._SPEC_ROOT = snapshot["root"]
    A.AUTH_SCHEMES = snapshot["auth"]
    A.SPEC_PROVENANCE = snapshot["provenance"]
    A.DOC_CORPUS[:] = snapshot["corpus"]
    A.ENDPOINT_INDEX[:] = snapshot["index"]
    A.ENDPOINT_VECS[:] = snapshot["vectors"]
    A._INDEXED_FOR = snapshot["indexed_for"]


def _progress_controller(bar, status, stage: str, trace: dict[str, Any] | None = None):
    previous = {"percent": -1, "message": ""}

    def update(percent: int, message: str) -> None:
        percent = max(previous["percent"], min(100, int(percent)))
        bar.progress(percent, text=message)
        if message != previous["message"]:
            status.write(message)
            _log(_log_level_for(message), stage, message, progress=percent)
            if trace is not None:
                _add_span(trace, "progress", {
                    "stage": stage,
                    "percent": percent,
                    "message": message,
                })
        previous.update(percent=percent, message=message)

    return update


def _endpoint_label(endpoint: A.Endpoint) -> str:
    return f"{endpoint.method} {endpoint.path}"


def _credential_state_key(api_url: str, auth_scheme: str) -> str:
    """One session-only credential per API host and documented auth scheme."""
    host = urlparse(api_url).netloc.lower() or api_url.strip().lower()
    return f"credential::{host}::{auth_scheme}"


def _clear_session_credential(key: str) -> None:
    st.session_state[key] = ""


def _supported_auth_options(scheme_names: list[str]) -> list[dict[str, Any]]:
    return [
        option for option in A.credential_options(scheme_names)
        if option["supported"]
    ]


def _render_credential_manager(spec: A.ApiSpec) -> None:
    """Always-available, session-only credential controls shared by Chat and API Call."""
    st.subheader("Credentials")
    scheme_names = list(dict.fromkeys(
        scheme
        for endpoint in spec.endpoints
        for scheme in A.endpoint_auth_schemes(endpoint.method, endpoint.path)
    ))
    all_options = A.credential_options(scheme_names)
    supported = [option for option in all_options if option["supported"]]
    unsupported = [option for option in all_options if not option["supported"]]
    if not scheme_names:
        st.caption("This API documents no authentication schemes.")
        return
    if not supported:
        st.error("The documented authentication schemes are not supported by the call builder.")
        return

    labels = {
        f"{option['label']} — `{option['name']}`": option
        for option in supported
    }
    preferred_index = next(
        (
            index for index, option in enumerate(supported)
            if option["label"].startswith(("Personal access", "API key"))
        ),
        0,
    )
    selected_label = st.selectbox(
        "Authentication method",
        list(labels),
        index=preferred_index,
        key=f"credential_manager_scheme::{urlparse(spec.base_url).netloc}",
    )
    option = labels[selected_label]
    credential_key = _credential_state_key(spec.base_url, option["name"])
    st.text_input(
        option["input_label"],
        type="password",
        help=option["help"],
        key=credential_key,
    )
    credential_host = urlparse(spec.base_url).netloc or spec.base_url
    if st.session_state.get(credential_key):
        st.success(f"Credential available for `{credential_host}`.")
        st.button(
            "Clear credential",
            key=f"clear::{credential_key}",
            on_click=_clear_session_credential,
            args=(credential_key,),
            width="stretch",
        )
    else:
        st.warning("No credential saved for this method.")
    st.caption("Stored in this browser session only. Never included in Chat, logs, or traces.")
    if option.get("authorization_url"):
        st.caption(f"Authorization endpoint: {option['authorization_url']}")
    if option.get("token_url"):
        st.caption(f"Token endpoint: {option['token_url']}")
    if unsupported:
        st.caption(
            "Unsupported schemes: "
            + ", ".join(option["label"] for option in unsupported)
        )


def _endpoint_from_label(spec: A.ApiSpec, label: str) -> A.Endpoint:
    return next(endpoint for endpoint in spec.endpoints if _endpoint_label(endpoint) == label)


def _onboard(source: str) -> None:
    source = source.strip()
    if not source:
        st.sidebar.error("Enter a docs URL or a local JSON/YAML spec path.")
        return

    snapshot = _agent_snapshot()
    bar = st.sidebar.progress(0, text="Starting")
    status = st.sidebar.status("Onboarding API", expanded=True)
    trace = _start_trace("onboarding", {"source": source})
    progress = _progress_controller(bar, status, "onboarding", trace)
    _log("INFO", "onboarding", "Onboarding started", source=source)
    try:
        writer = _LogWriter("onboarding")
        with redirect_stdout(writer):
            spec = A.onboard(source, progress=progress)
        writer.flush()
        if not spec.endpoints:
            raise ValueError("No documented endpoints were found.")

        # Seed the same grounding evidence as the terminal flow. The real-call UI uses spec-derived
        # paths directly, but retaining this makes any future tool-agent handoff safe.
        A.DOC_CORPUS.append(spec.model_dump_json())
        st.session_state.spec = spec
        st.session_state.agent_runtime = _agent_snapshot()
        st.session_state.runtime_notice = ""
        st.session_state.source = source
        st.session_state.last_endpoint = None
        st.session_state.list_cursor = min(10, len(spec.endpoints))
        st.session_state.pending_call = None
        st.session_state.call_origin = "manual"
        st.session_state.pending_call_origin = "manual"
        st.session_state.last_call_result = None
        st.session_state.last_api_exchange = None
        st.session_state.pop("call_endpoint", None)
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Onboarded **{len(spec.endpoints)} endpoints** from `{source}`. "
                "Ask which endpoint performs a task, inspect the schema, or prepare a request."
            ),
        }]
        status.update(label=f"Onboarded {len(spec.endpoints)} endpoints", state="complete")
        _log(
            "WARNING" if A.SPEC_PROVENANCE == "scraped" else "INFO",
            "onboarding",
            "Onboarding completed",
            endpoints=len(spec.endpoints),
            provenance=A.SPEC_PROVENANCE,
        )
        _finish_trace(
            trace,
            status="ok",
            output={
                "base_url": spec.base_url,
                "auth_method": spec.auth_method,
                "endpoints": len(spec.endpoints),
                "coverage": A.spec_coverage(spec),
            },
            provenance=A.SPEC_PROVENANCE,
        )
    except Exception as exc:
        _restore_agent(snapshot)
        status.update(label="Onboarding failed", state="error")
        bar.progress(100, text="Failed")
        _log("ERROR", "onboarding", str(exc))
        _finish_trace(trace, status="error", error=str(exc))
        st.sidebar.error(f"Onboarding failed: {exc}")


def _render_summary(spec: A.ApiSpec) -> None:
    st.subheader("Loaded API")
    st.caption(st.session_state.source)
    st.metric("Endpoints", len(spec.endpoints))
    st.write(f"**Base URL:** `{spec.base_url or 'not documented'}`")
    st.write(f"**Auth:** {spec.auth_method or 'not documented'}")
    provenance = A.SPEC_PROVENANCE or "unknown"
    if provenance == "scraped":
        st.warning("This result was extracted from prose, not parsed from a machine-readable spec.")
    else:
        st.success("Parsed from a machine-readable OpenAPI document.")


def _render_chat(spec: A.ApiSpec) -> None:
    st.subheader("Ask about the API")
    st.caption("Answers are grounded in the onboarded spec. Live requests are prepared in the API Call tab.")
    st.warning(
        "Never paste a password, API key, or access token into chat. "
        "Add or replace it only in the protected **Credentials** field in the sidebar."
    )
    # Keep the transcript in a container created BEFORE the input. When a new turn is submitted,
    # Streamlit can append its user/assistant bubbles to this earlier container, so the chat input
    # remains below every message instead of getting stranded between old and new turns.
    transcript = st.container()
    with transcript:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_text = st.chat_input("Which endpoint lets me…?")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    _log("INFO", "chat", "Question received", characters=len(user_text))
    trace = _start_trace(
        "chat_turn",
        {"message": user_text},
        source=st.session_state.source,
        focus=_endpoint_label(st.session_state.last_endpoint)
        if st.session_state.last_endpoint else "",
    )

    with transcript:
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            bar = st.progress(0, text="Reading the request")
            status = st.status("Working from the onboarded spec", expanded=True)
            progress = _progress_controller(bar, status, "chat", trace)
            try:
                writer = _LogWriter("chat")
                with redirect_stdout(writer):
                    reply = UI.answer_question(
                        spec,
                        user_text,
                        last_endpoint=st.session_state.last_endpoint,
                        last_api_exchange=st.session_state.last_api_exchange,
                        list_cursor=st.session_state.list_cursor,
                        progress=progress,
                        trace=lambda name, payload: _add_span(trace, name, payload),
                    )
                writer.flush()
                st.session_state.last_endpoint = reply.focus
                st.session_state.list_cursor = reply.list_cursor
                if reply.route == "call" and reply.focus is not None:
                    selected_call_endpoint = _endpoint_label(reply.focus)
                    if st.session_state.get("call_endpoint") != selected_call_endpoint:
                        st.session_state.call_endpoint = selected_call_endpoint
                    st.session_state.pending_call = None
                    st.session_state.last_call_result = None
                    st.session_state.call_origin = "chat"
                    st.session_state.pending_call_origin = "chat"
                    endpoint_schemes = A.endpoint_auth_schemes(
                        reply.focus.method,
                        reply.focus.path,
                    )
                    endpoint_auth = _supported_auth_options(endpoint_schemes)
                    if endpoint_auth:
                        available = [
                            option for option in endpoint_auth
                            if st.session_state.get(
                                _credential_state_key(spec.base_url, option["name"])
                            )
                        ]
                        if available:
                            reply.text += (
                                f"\n\nA {available[0]['label']} is available in this browser "
                                "session. It will be added only by the HTTP executor."
                            )
                        else:
                            accepted = " or ".join(
                                option["label"] for option in endpoint_auth
                            )
                            reply.text += (
                                f"\n\nThis endpoint requires {accepted}. Add it in the sidebar "
                                "**Credentials** field before sending."
                            )
                # The semantic index is module-global in agent.py. Save the updated runtime so a
                # Streamlit rerun or hot reload cannot retain the ApiSpec while silently losing its
                # schemas/auth/index—the exact state split that produced false "[no request body]".
                st.session_state.agent_runtime = _agent_snapshot()
                st.session_state.messages.append({"role": "assistant", "content": reply.text})
                status.update(label="Answer ready", state="complete", expanded=False)
                st.markdown(reply.text)
                level = ("WARNING" if "grounding check" in reply.text.lower()
                         or "[!]" in reply.text else "INFO")
                _log(level, "chat", "Answer completed", route=reply.route)
                _finish_trace(
                    trace,
                    status="ok",
                    output={"message": reply.text},
                    route=reply.route,
                    focus=_endpoint_label(reply.focus) if reply.focus else "",
                )
            except Exception as exc:
                message = str(exc)
                status.update(label="Answer failed", state="error")
                st.error(message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Turn failed: {message}",
                })
                _log("ERROR", "chat", message)
                _finish_trace(trace, status="error", error=message)


def _render_explorer(spec: A.ApiSpec) -> None:
    st.subheader("Endpoint Explorer")
    search_col, method_col = st.columns([3, 1])
    search = search_col.text_input("Filter endpoints", placeholder="event types, users, POST…")
    methods = sorted({endpoint.method for endpoint in spec.endpoints})
    selected_methods = method_col.multiselect("Methods", methods)

    needle = search.lower().strip()
    filtered = [
        endpoint for endpoint in spec.endpoints
        if (not selected_methods or endpoint.method in selected_methods)
        and (not needle or needle in _endpoint_label(endpoint).lower()
             or needle in endpoint.description.lower())
    ]
    if not filtered:
        st.info("No endpoints match those filters.")
        return

    labels = [_endpoint_label(endpoint) for endpoint in filtered]
    selected = st.selectbox(f"{len(filtered)} endpoint(s)", labels, key="explorer_endpoint")
    endpoint = _endpoint_from_label(spec, selected)

    st.markdown(f"### `{endpoint.method} {endpoint.path}`")
    st.write(endpoint.description or "_No description in the spec._")

    info = A.ENDPOINT_SCHEMAS.get((endpoint.method, endpoint.path)) or {}
    left, middle, right = st.columns(3)
    left.metric("Parameters", len(A.endpoint_params(endpoint.method, endpoint.path)))
    middle.metric("OAuth scopes", len(A.endpoint_scopes(endpoint.method, endpoint.path)))
    right.metric("Deprecated", "Yes" if info.get("deprecated") else "No")

    auth = A.auth_instructions(endpoint.method, endpoint.path)
    if auth:
        st.info(f"Authentication: {auth}")
    elif A.endpoint_auth_schemes(endpoint.method, endpoint.path):
        st.info("The spec names an authentication scheme but provides no additional instructions.")
    else:
        st.caption("The spec documents no authentication requirement for this endpoint.")

    params = A.endpoint_params(endpoint.method, endpoint.path)
    if params:
        st.markdown("#### Parameters")
        rows = [{
            "name": param["name"],
            "location": param["in"],
            "type": param["type"],
            "required": param["required"],
            "description": param.get("desc", ""),
            "default": ("" if param.get("default") is None
                        else json.dumps(param.get("default"), ensure_ascii=False)),
            "example": ("" if param.get("example") is None
                        else json.dumps(param.get("example"), ensure_ascii=False)),
        } for param in params]
        st.dataframe(rows, width="stretch", hide_index=True)

    request_tab, response_tab, example_tab = st.tabs(
        ["Request schema", "Response schema", "Examples"]
    )
    with request_tab:
        tree = A.request_tree(endpoint.method, endpoint.path)
        st.code(tree or "No request body schema documented.", language="text")
    with response_tab:
        tree = A.response_tree(endpoint.method, endpoint.path)
        st.code(tree or "No response schema documented.", language="text")
    with example_tab:
        examples = A.endpoint_examples(endpoint.method, endpoint.path)
        if examples["request"] is not None:
            st.markdown("**Request**")
            st.json(examples["request"])
        if examples["response"] is not None:
            st.markdown("**Response**")
            st.json(examples["response"])
        if examples["request"] is None and examples["response"] is None:
            st.info("The spec contains no request or response schema to synthesize.")


def _mark_call_selection_manual() -> None:
    """A user-changed API Call selection starts a standalone, non-Chat request."""
    st.session_state.call_origin = "manual"
    st.session_state.pending_call_origin = "manual"
    st.session_state.pending_call = None
    st.session_state.last_call_result = None


def _render_call_builder(spec: A.ApiSpec) -> None:
    st.subheader("API Call")
    st.warning("Nothing is sent until you prepare the request, inspect its preview, and confirm it.")

    labels = [_endpoint_label(endpoint) for endpoint in spec.endpoints]
    focus = st.session_state.last_endpoint
    default = labels.index(_endpoint_label(focus)) if focus and _endpoint_label(focus) in labels else 0
    if (
        "call_endpoint" in st.session_state
        and st.session_state.call_endpoint in labels
    ):
        selected = st.selectbox(
            "Endpoint",
            labels,
            index=None,
            key="call_endpoint",
            on_change=_mark_call_selection_manual,
        )
    else:
        st.session_state.pop("call_endpoint", None)
        selected = st.selectbox(
            "Endpoint",
            labels,
            index=default,
            key="call_endpoint",
            on_change=_mark_call_selection_manual,
        )
    endpoint = _endpoint_from_label(spec, selected)
    if st.session_state.get("call_origin") == "chat":
        st.info(
            "Linked to Chat: after this request is sent, its result and any error guidance "
            "will return to the conversation."
        )
    else:
        st.caption(
            "Standalone API Call: results stay on this page. Failed calls can be sent to the "
            "agent for diagnosis afterward."
        )
    params = A.endpoint_params(endpoint.method, endpoint.path)
    info = A.ENDPOINT_SCHEMAS.get((endpoint.method, endpoint.path)) or {}
    if "param_any_of" not in info:
        # Upgrade a schema snapshot created before prose-only cross-parameter constraints were kept.
        info["param_any_of"] = A._parameter_any_of_groups(info.get("notes", ""), params)
    schemes = A.endpoint_auth_schemes(endpoint.method, endpoint.path)
    spec_auth = (spec.auth_method or "").strip()
    spec_requires_auth = spec_auth.lower() not in {
        "", "none", "none documented", "no auth", "not documented",
    }
    # `spec.auth_method` lists schemes available somewhere in the document; it does not mean every
    # operation requires them. A cleanly parsed operation with no schemes is public. Only treat the
    # placement as missing when the operation metadata itself is unavailable (for example, scraped
    # prose that mentions authentication but has no machine-readable security object).
    auth_metadata_missing = spec_requires_auth and not schemes and not info
    requires_auth = bool(schemes) or auth_metadata_missing
    auth_options = A.credential_options(schemes)
    supported_auth = [option for option in auth_options if option["supported"]]
    unsupported_auth = [option for option in auth_options if not option["supported"]]
    selected_auth: dict[str, Any] | None = None
    token_key = ""

    with st.form("prepare_call_form"):
        st.markdown(f"#### `{endpoint.method} {endpoint.path}`")
        st.markdown("#### Authentication")
        if requires_auth:
            if schemes:
                st.info(A.auth_instructions(endpoint.method, endpoint.path))
                if supported_auth:
                    option_labels = {
                        f"{option['label']} — `{option['name']}`": option
                        for option in supported_auth
                    }
                    preferred_index = next(
                        (
                            index for index, option in enumerate(supported_auth)
                            if option["label"].startswith(("Personal access", "API key"))
                        ),
                        0,
                    )
                    if len(option_labels) > 1:
                        selected_label = st.selectbox(
                            "Authentication method",
                            list(option_labels),
                            index=preferred_index,
                            key=f"auth_method::{endpoint.method}::{endpoint.path}",
                        )
                        selected_auth = option_labels[selected_label]
                    else:
                        selected_auth = supported_auth[0]
                        st.markdown(f"**Method:** {selected_auth['label']}")

                    token_key = _credential_state_key(
                        spec.base_url,
                        selected_auth["name"],
                    )
                    # Preserve a credential already entered before storage became host-scoped.
                    legacy_token_key = (
                        f"credential::{endpoint.method}::{endpoint.path}::"
                        f"{selected_auth['name']}"
                    )
                    if (
                        token_key not in st.session_state
                        and st.session_state.get(legacy_token_key)
                    ):
                        st.session_state[token_key] = st.session_state[legacy_token_key]
                    if st.session_state.get(token_key):
                        st.success(
                            "Credential available. The HTTP executor will add it only when sending."
                        )
                    else:
                        st.warning(
                            "No credential saved for this method. Add it in the sidebar "
                            "**Credentials** field before sending."
                        )
                if unsupported_auth:
                    st.warning(
                        "Not available in this call builder: "
                        + ", ".join(option["label"] for option in unsupported_auth)
                    )
            else:
                st.error(
                    f"This API mentions authentication ({spec_auth}), but the endpoint does not "
                    "document a usable security scheme or credential placement. Request "
                    "preparation is disabled because guessing could send the credential incorrectly."
                )
        else:
            st.caption("The spec documents no authentication requirement for this endpoint.")

        st.markdown("#### Request inputs")
        if info.get("notes"):
            st.info(f"Documented behavior: {info['notes']}")
        for group in info.get("param_any_of") or []:
            names = " or ".join(f"`{name}`" for name in group.get("names") or [])
            location = f" {group['in']}" if group.get("in") else ""
            st.warning(f"Required choice: provide at least one{location} parameter: {names}.")

        path_values: dict[str, str] = {}
        query_values: dict[str, str] = {}
        header_values: dict[str, str] = {}
        cookie_values: dict[str, str] = {}
        destinations = {
            "path": path_values,
            "query": query_values,
            "header": header_values,
            "cookie": cookie_values,
        }
        params_by_location_name = {
            (param.get("in"), param.get("name")): param for param in params
        }
        rendered_params: set[tuple[str, str]] = set()

        # A templated path is itself definitive evidence that the user must supply a value. Some
        # specs omit the matching parameter object, so never make the UI depend on that metadata.
        ordered_params: list[dict[str, Any]] = []
        for name in dict.fromkeys(re.findall(r"{([^}]+)}", endpoint.path)):
            ordered_params.append(
                params_by_location_name.get(("path", name))
                or {
                    "name": name,
                    "in": "path",
                    "required": True,
                    "desc": f"Value for the {{{name}}} segment in the request URL.",
                }
            )
        ordered_params.extend(
            param for param in params
            if (param.get("in"), param.get("name")) not in {
                ("path", name) for name in re.findall(r"{([^}]+)}", endpoint.path)
            }
        )

        for param in ordered_params:
            location = param["in"]
            if location not in destinations:
                continue
            key = (location, param["name"])
            if key in rendered_params:
                continue
            rendered_params.add(key)
            any_of = next(
                (
                    group for group in info.get("param_any_of") or []
                    if param["name"] in (group.get("names") or [])
                ),
                None,
            )
            if any_of:
                required = "required choice"
            else:
                required = "required" if param.get("required") or location == "path" else "optional"
            param_format = A.parameter_format(param)
            hint = param.get("desc") or f"{location} parameter, {required}"
            if param_format:
                hint += f" Expected format: {param_format}."
            default_value = param.get("default")
            if default_value is None:
                default_value = param.get("example")
            value = st.text_input(
                f"{param['name']} · {location} · {required}"
                + (f" · {param_format}" if param_format else ""),
                value="" if default_value is None else str(default_value),
                help=hint,
                key=f"param::{endpoint.method}::{endpoint.path}::{location}::{param['name']}",
            )
            destinations[location][param["name"]] = value

        body_text = ""
        if info.get("request") is not None:
            body_default = UI.request_body_example(endpoint)
            st.caption(
                "This endpoint sends a JSON request body. The starter below contains only fields "
                "the spec marks required; add optional fields only when you need them."
            )
            body_text = st.text_area(
                "JSON request body" + (" · required" if info.get("body_required") else " · optional"),
                value=json.dumps(body_default if body_default is not None else {}, indent=2),
                height=240,
                key=f"body_v2::{endpoint.method}::{endpoint.path}",
            )

        prepared = st.form_submit_button(
            "Prepare request",
            type="primary",
            width="stretch",
            disabled=requires_auth and selected_auth is None,
        )

    if prepared:
        prepare_trace = _start_trace(
            "prepare_api_call",
            {
                "endpoint": _endpoint_label(endpoint),
                "path_values": path_values,
                "query_values": query_values,
                "header_values": header_values,
                "cookie_values": cookie_values,
                "body": body_text,
            },
            source=st.session_state.source,
        )
        try:
            body = json.loads(body_text) if body_text.strip() else None
            pending = UI.prepare_call(
                endpoint,
                spec.base_url,
                path_values=path_values,
                query_values=query_values,
                header_values=header_values,
                cookie_values=cookie_values,
                body=body,
                include_auth=True,
                auth_required=requires_auth,
                auth_scheme=selected_auth["name"] if selected_auth else "",
            )
            st.session_state.pending_call = pending
            st.session_state.pending_call_origin = st.session_state.get(
                "call_origin",
                "manual",
            )
            st.session_state.last_call_result = None
            _log("INFO", "api_call", "Request prepared", endpoint=_endpoint_label(endpoint))
            _finish_trace(
                prepare_trace,
                status="ok",
                output={"masked_request_preview": pending.preview()},
            )
        except Exception as exc:
            st.session_state.pending_call = None
            _log("ERROR", "api_call", str(exc), endpoint=_endpoint_label(endpoint))
            _finish_trace(prepare_trace, status="error", error=str(exc))
            st.error(f"Could not prepare request: {exc}")

    pending: UI.PreparedCall | None = st.session_state.pending_call
    if pending is None:
        return
    if not hasattr(pending, "auth_scheme"):
        st.session_state.pending_call = None
        st.info("The call builder was updated. Prepare the request once more.")
        return
    if _endpoint_label(pending.endpoint) != selected:
        st.info("The prepared request belongs to another endpoint. Prepare this endpoint to replace it.")
        return

    st.markdown("#### Credential-safe preview")
    st.caption(
        "This is the complete assembled HTTP request. It combines the URL, parameters, headers, "
        "credential placeholder, and JSON body for final review."
    )
    st.code(pending.preview(), language="http")
    token_key = (
        _credential_state_key(pending.url, pending.auth_scheme)
        if pending.auth_scheme else ""
    )
    credential = st.session_state.get(token_key, "") if token_key else ""
    missing_credential = pending.needs_credential and not credential
    if missing_credential:
        st.warning("Enter the documented credential before sending.")

    confirmed = st.checkbox(
        "I understand this sends a real request to the API.",
        key=f"confirm::{endpoint.method}::{endpoint.path}",
    )
    send = st.button(
        "Send real request",
        type="primary",
        disabled=not confirmed or missing_credential,
        width="stretch",
    )
    if send:
        return_to_chat = (
            st.session_state.get("pending_call_origin", "manual") == "chat"
        )
        call_trace = _start_trace(
            "execute_api_call",
            {
                "endpoint": _endpoint_label(endpoint),
                "masked_request_preview": pending.preview(),
                "origin": "chat" if return_to_chat else "manual",
            },
            source=st.session_state.source,
        )
        bar = st.progress(15, text="Sending request")
        _add_span(call_trace, "http_request", {
            "method": pending.endpoint.method,
            "url": pending.preview().splitlines()[0],
            "masked_preview": pending.preview(),
        })
        started = time.perf_counter()
        chat_update = ""
        try:
            result = UI.execute_call(pending, credential)
            st.session_state.last_call_result = result
            safe_result = _session_safe_text(result)
            first_line = result.splitlines()[0] if result else "No result"
            request_succeeded = first_line.startswith("Status: 2")
            level = "INFO" if request_succeeded else "WARNING"
            _log(level, "api_call", first_line, endpoint=_endpoint_label(endpoint))
            _add_span(call_trace, "http_response", {
                "output": result,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            if return_to_chat:
                summary = UI.api_result_chat_message(pending, safe_result)
                if request_succeeded:
                    chat_update = summary
                else:
                    bar.progress(85, text="Agent is reviewing the API error")
                    try:
                        with st.spinner(
                            "Agent is reviewing the failed call and drafting a correction"
                        ):
                            guidance = UI.analyze_api_result(
                                spec,
                                pending,
                                safe_result,
                                trace=lambda name, payload: _add_span(
                                    call_trace, name, payload
                                ),
                            )
                        chat_update = (
                            f"{summary}\n\n### Suggested correction\n\n{guidance}"
                        )
                    except Exception as review_exc:
                        chat_update = (
                            f"{summary}\n\nAutomatic correction guidance was unavailable: "
                            f"{review_exc}"
                        )
                        _add_span(call_trace, "api_error_review", {
                            "status": "fallback",
                            "error": str(review_exc),
                        })
                        _log(
                            "WARNING",
                            "api_call",
                            f"Automatic error review unavailable: {review_exc}",
                            endpoint=_endpoint_label(endpoint),
                        )
            st.session_state.last_api_exchange = {
                "endpoint": _endpoint_label(endpoint),
                "masked_request_preview": pending.preview(),
                "response": safe_result,
                "status_code": UI.parse_http_result(result)["status_code"],
                "origin": "chat" if return_to_chat else "manual",
                "available_to_chat": return_to_chat,
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
            }
            if return_to_chat:
                st.session_state.last_endpoint = endpoint
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": chat_update,
                    "kind": "api_result",
                })
            bar.progress(100, text="Request completed")
            trace_output = {"response": result}
            if chat_update:
                trace_output["chat_guidance"] = chat_update
            _finish_trace(
                call_trace,
                status="ok" if request_succeeded else "error",
                output=trace_output,
                error="" if request_succeeded else first_line,
            )
            if return_to_chat:
                _log(
                    "INFO",
                    "chat",
                    "API result added to Chat",
                    endpoint=_endpoint_label(endpoint),
                )
        except Exception as exc:
            bar.progress(100, text="Request failed")
            failure_result = f"Request failed: {exc}"
            st.session_state.last_call_result = failure_result
            _log("ERROR", "api_call", str(exc), endpoint=_endpoint_label(endpoint))
            _add_span(call_trace, "http_response", {
                "error": str(exc),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            if return_to_chat:
                chat_update = UI.api_result_chat_message(pending, failure_result)
            st.session_state.last_api_exchange = {
                "endpoint": _endpoint_label(endpoint),
                "masked_request_preview": pending.preview(),
                "response": _session_safe_text(failure_result),
                "status_code": None,
                "origin": "chat" if return_to_chat else "manual",
                "available_to_chat": return_to_chat,
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
            }
            if return_to_chat:
                st.session_state.last_endpoint = endpoint
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": chat_update,
                    "kind": "api_result",
                })
            _finish_trace(
                call_trace,
                status="error",
                output={"chat_guidance": chat_update} if chat_update else None,
                error=str(exc),
            )
            if return_to_chat:
                _log(
                    "INFO",
                    "chat",
                    "Request failure added to Chat",
                    endpoint=_endpoint_label(endpoint),
                )
        if return_to_chat:
            st.toast("API result and next-step guidance were added to Chat.")
            st.rerun()

    if st.session_state.last_call_result:
        raw_result = st.session_state.last_call_result
        parsed_result = UI.parse_http_result(raw_result)
        status_code = parsed_result["status_code"]
        if status_code is not None:
            status_label = f"HTTP {status_code}"
            if parsed_result["status_text"]:
                status_label += f" — {parsed_result['status_text']}"
            if 200 <= status_code < 300:
                st.success(status_label)
            else:
                st.error(status_label)
        if parsed_result["truncated"]:
            st.warning(
                "This result was captured with the older 500-character response limit, so its JSON "
                "is incomplete and cannot be prettified. Run the request once more to capture the "
                "full response."
            )

        pretty_tab, raw_tab = st.tabs(["Pretty response", "Raw response"])
        with pretty_tab:
            if parsed_result["is_json"]:
                pretty_body = json.dumps(
                    parsed_result["json_body"],
                    ensure_ascii=False,
                    indent=2,
                )
                st.code(pretty_body, language="json", wrap_lines=True, height=500)
            else:
                st.code(
                    parsed_result["body_text"] or "(empty response body)",
                    language="text",
                    wrap_lines=True,
                    height=500,
                )
        with raw_tab:
            st.code(raw_result, language="text", wrap_lines=True, height=500)

        exchange = st.session_state.get("last_api_exchange") or {}
        manual_failure = (
            exchange.get("endpoint") == _endpoint_label(pending.endpoint)
            and exchange.get("origin") == "manual"
            and not exchange.get("available_to_chat", False)
            and (status_code is None or status_code >= 400)
        )
        if manual_failure:
            st.caption(
                "This request was prepared directly in API Call, so its result was not added "
                "to Chat."
            )
            if st.button(
                "Ask agent to diagnose this response",
                key=(
                    f"diagnose::{pending.endpoint.method}::{pending.endpoint.path}::"
                    f"{exchange.get('timestamp', '')}"
                ),
                width="stretch",
            ):
                diagnose_trace = _start_trace(
                    "diagnose_api_call",
                    {
                        "endpoint": _endpoint_label(pending.endpoint),
                        "masked_request_preview": pending.preview(),
                        "response": _session_safe_text(raw_result),
                    },
                    source=st.session_state.source,
                    origin="manual",
                )
                safe_result = _session_safe_text(raw_result)
                summary = UI.api_result_chat_message(pending, safe_result)
                try:
                    with st.spinner(
                        "Agent is reviewing the failed call and drafting a correction"
                    ):
                        guidance = UI.analyze_api_result(
                            spec,
                            pending,
                            safe_result,
                            trace=lambda name, payload: _add_span(
                                diagnose_trace,
                                name,
                                payload,
                            ),
                        )
                    chat_update = (
                        f"{summary}\n\n### Suggested correction\n\n{guidance}"
                    )
                    _finish_trace(
                        diagnose_trace,
                        status="ok",
                        output={"chat_guidance": chat_update},
                    )
                except Exception as review_exc:
                    chat_update = (
                        f"{summary}\n\nAutomatic correction guidance was unavailable: "
                        f"{review_exc}"
                    )
                    _finish_trace(
                        diagnose_trace,
                        status="error",
                        output={"chat_guidance": chat_update},
                        error=str(review_exc),
                    )
                    _log(
                        "WARNING",
                        "api_call",
                        f"Manual error review unavailable: {review_exc}",
                        endpoint=_endpoint_label(pending.endpoint),
                    )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": chat_update,
                    "kind": "api_result",
                })
                st.session_state.last_endpoint = pending.endpoint
                st.session_state.last_api_exchange["available_to_chat"] = True
                st.session_state.last_api_exchange["origin"] = "manual_promoted"
                _log(
                    "INFO",
                    "chat",
                    "Manual API result added to Chat for diagnosis",
                    endpoint=_endpoint_label(pending.endpoint),
                )
                st.toast("The response and correction guidance were added to Chat.")
                st.rerun()


def _transcript_markdown() -> str:
    lines = ["# SaaS API Onboarding Agent transcript", ""]
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines += [f"## {role}", "", _session_safe_text(message["content"]), ""]
    return "\n".join(lines)


def _public_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trace.items() if not key.startswith("_")}


def _debug_bundle() -> dict[str, Any]:
    spec: A.ApiSpec | None = st.session_state.spec
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime": {
            "python": platform.python_version(),
            "streamlit": st.__version__,
            "platform": platform.platform(),
        },
        "session": {
            "source": _session_safe_text(st.session_state.source),
            "provenance": A.SPEC_PROVENANCE,
            "endpoint_count": len(spec.endpoints) if spec else 0,
            "base_url": spec.base_url if spec else "",
            "coverage": A.spec_coverage(spec) if spec else "",
        },
        "transcript_markdown": _transcript_markdown(),
        "traces": [_public_trace(trace) for trace in st.session_state.traces],
        "event_log": st.session_state.logs,
    }


def _trace_category(trace: dict[str, Any]) -> str:
    return {
        "onboarding": "Onboarding",
        "chat_turn": "Chat",
        "prepare_api_call": "API calls",
        "execute_api_call": "API calls",
    }.get(trace.get("name", ""), "Other")


def _trace_label(trace: dict[str, Any]) -> str:
    """A collapsed trace label that identifies the operation without exposing credentials."""
    status = trace.get("status", "running").upper()
    duration = trace.get("duration_ms", 0)
    trace_id = trace.get("trace_id", "")[:8]
    name = trace.get("name", "")
    trace_input = trace.get("input") or {}

    if name in {"prepare_api_call", "execute_api_call"}:
        endpoint = trace_input.get("endpoint") or "API request"
        preview = (
            trace_input.get("masked_request_preview")
            or (trace.get("output") or {}).get("masked_request_preview")
            or ""
        )
        request_line = preview.splitlines()[0] if preview else ""
        query_names = []
        if request_line and " " in request_line:
            request_url = request_line.split(" ", 1)[1]
            query_names = list(dict.fromkeys(name for name, _ in parse_qsl(
                urlparse(request_url).query,
                keep_blank_values=True,
            )))
        query_note = (
            f" · query: {', '.join(query_names[:4])}"
            + ("…" if len(query_names) > 4 else "")
            if query_names else ""
        )
        http_note = ""
        response = ((trace.get("output") or {}).get("response") or "")
        parsed_response = UI.parse_http_result(response) if response else {}
        if parsed_response.get("status_code") is not None:
            http_note = f" · HTTP {parsed_response['status_code']}"
        action = "API call" if name == "execute_api_call" else "API prepared"
        return (
            f"{status} · {action} · {endpoint}{query_note}{http_note} · "
            f"{duration} ms · {trace_id}"
        )

    return (
        f"{status} · {_trace_category(trace)} · {name} · "
        f"{duration} ms · {trace_id}"
    )


def _render_logs() -> None:
    st.subheader("Debug & Traces")
    st.markdown(
        """
        - **Chat transcript** — only the visible user/assistant conversation.
        - **Detailed traces** — every high-level action: onboarding, chat routing/LLM spans, request
          preparation, and HTTP calls.
        - **Event log** — a short INFO/WARNING/ERROR timeline. An error appears here *and* in its
          detailed trace intentionally: the log locates it; the trace explains it.
        """
    )
    st.warning(
        "Debug traces contain complete chat input/output, model prompts, retrieved candidates, "
        "timings, and API response text. Credentials are redacted, but API data may contain personal "
        "information. Review the bundle before sharing it."
    )
    bundle = json.dumps(_debug_bundle(), ensure_ascii=False, indent=2)
    st.download_button(
        "Download complete debug bundle",
        data=bundle,
        file_name="saas-onboarding-debug-bundle.json",
        mime="application/json",
        type="primary",
    )

    transcript_tab, traces_tab, operations_tab = st.tabs(
        ["Chat transcript", "Detailed traces", "Event log"]
    )
    with transcript_tab:
        transcript = _transcript_markdown()
        st.caption("Use the copy icon in the top-right of this block, or download the Markdown file.")
        st.code(transcript, language="markdown")
        st.download_button(
            "Download transcript",
            data=transcript,
            file_name="saas-onboarding-transcript.md",
            mime="text/markdown",
        )

    with traces_tab:
        st.caption(
            "Traces are not API-call-only. Filter to one action, then use the copy icon in its JSON "
            "block or download that trace alone."
        )
        categories = ["Onboarding", "Chat", "API calls", "Other"]
        selected_categories = st.multiselect(
            "Trace types",
            categories,
            default=categories,
            key="trace_type_filter",
        )
        shown_traces = [
            trace for trace in st.session_state.traces
            if not selected_categories or _trace_category(trace) in selected_categories
        ]
        left, middle, right = st.columns(3)
        left.metric("Traces", len(shown_traces))
        middle.metric(
            "Spans",
            sum(len(trace.get("spans", [])) for trace in shown_traces),
        )
        right.metric(
            "Errors",
            sum(trace.get("status") == "error" for trace in shown_traces),
        )
        filtered_trace_json = json.dumps(
            [_public_trace(trace) for trace in shown_traces],
            ensure_ascii=False,
            indent=2,
        )
        st.download_button(
            "Download filtered traces only",
            data=filtered_trace_json,
            file_name="saas-onboarding-filtered-traces.json",
            mime="application/json",
        )
        if st.button("Clear traces"):
            st.session_state.traces = []
            st.rerun()
        if not shown_traces:
            st.info("No traces yet. Onboard an API or submit a chat turn.")
        for trace in reversed(shown_traces):
            with st.expander(_trace_label(trace)):
                trace_json = json.dumps(_public_trace(trace), ensure_ascii=False, indent=2)
                input_tab, output_tab, full_trace_tab = st.tabs(
                    ["Input", "Output", "Complete trace"]
                )
                with input_tab:
                    trace_input = trace.get("input")
                    masked_preview = (
                        trace_input.get("masked_request_preview")
                        if isinstance(trace_input, dict) else ""
                    )
                    if masked_preview:
                        st.code(masked_preview, language="http", wrap_lines=True)
                    else:
                        st.code(
                            json.dumps(trace_input, ensure_ascii=False, indent=2),
                            language="json",
                            wrap_lines=True,
                        )
                with output_tab:
                    st.code(
                        json.dumps(trace.get("output"), ensure_ascii=False, indent=2),
                        language="json",
                        wrap_lines=True,
                    )
                with full_trace_tab:
                    st.caption("Use the copy icon in the top-right of this block.")
                    st.code(trace_json, language="json", wrap_lines=True)
                st.download_button(
                    "Download this trace",
                    data=trace_json,
                    file_name=f"trace-{trace['name']}-{trace['trace_id'][:8]}.json",
                    mime="application/json",
                    key=f"download_trace_{trace['trace_id']}",
                )

    with operations_tab:
        st.caption(
            "Successful activity is INFO; fallbacks, grounding warnings, and non-2xx calls are "
            "WARNING; failed operations are ERROR."
        )
        actions, filters = st.columns([1, 3])
        levels = sorted({record["level"] for record in st.session_state.logs})
        selected_levels = filters.multiselect("Levels", levels, default=levels)
        shown = [
            record for record in st.session_state.logs
            if not selected_levels or record["level"] in selected_levels
        ]
        if actions.button("Clear event log"):
            st.session_state.logs = []
            st.rerun()

        if shown:
            rows = [{
                "time": record["time"],
                "level": record["level"],
                "stage": record["stage"],
                "message": record["message"],
            } for record in reversed(shown)]
            st.dataframe(rows, width="stretch", hide_index=True)
            jsonl = "\n".join(json.dumps(record, ensure_ascii=False) for record in shown)
            st.download_button(
                "Download event log",
                data=jsonl,
                file_name="saas-onboarding-ui-logs.jsonl",
                mime="application/x-ndjson",
            )
        else:
            st.info("No operational log events yet.")


_init_state()
if st.session_state.spec is not None:
    if st.session_state.agent_runtime is not None:
        _restore_agent(st.session_state.agent_runtime)
    elif len(A.ENDPOINT_SCHEMAS) == len(st.session_state.spec.endpoints):
        # Upgrade an already-open session created before runtime snapshots existed.
        st.session_state.agent_runtime = _agent_snapshot()
    else:
        # Never render a half-alive spec: endpoint names without schemas/auth produced confident
        # "[no request body]" falsehoods. Keep the source/transcript, but require one clean onboard.
        st.session_state.spec = None
        st.session_state.last_endpoint = None
        st.session_state.runtime_notice = (
            "The previous UI session did not contain a complete schema snapshot. "
            "Onboard the source once more before asking API questions."
        )
        _log("WARNING", "runtime", st.session_state.runtime_notice)

with st.sidebar:
    st.title("🔌 API Onboarding")
    source_input = st.text_input(
        "Docs URL or local spec path",
        value=st.session_state.source,
        placeholder="https://example.com/openapi.yaml",
        key="onboarding_source",
    )
    st.caption("Accepts a documentation URL or a local JSON/YAML spec path.")
    onboard_clicked = st.button(
        "Onboard API",
        type="primary",
        width="stretch",
        disabled=not source_input.strip(),
    )
    if onboard_clicked:
        _onboard(source_input)
    if st.session_state.spec:
        st.divider()
        _render_summary(st.session_state.spec)
        with st.expander("Parse coverage"):
            st.code(A.spec_coverage(st.session_state.spec), language="text")
        st.divider()
        _render_credential_manager(st.session_state.spec)
    st.divider()
    st.caption("Runs locally with Ollama. No credential is sent to the language model.")

st.title("SaaS API Onboarding Agent")
st.write("Read an API spec, find the right endpoint, inspect its schema, and prepare grounded calls.")

if st.session_state.spec is None:
    if st.session_state.runtime_notice:
        st.warning(st.session_state.runtime_notice)
    st.info("Enter an API documentation URL or local OpenAPI file in the sidebar to begin.")
    st.markdown(
        """
        **What happens next**

        1. The app searches for a machine-readable OpenAPI document.
        2. It parses endpoints, parameters, schemas, scopes, and authentication.
        3. You can chat with the spec, inspect endpoints, or prepare a real request.
        """
    )
    with st.expander("Debug & traces"):
        _render_logs()
else:
    spec = st.session_state.spec
    chat_tab, explorer_tab, call_tab, logs_tab = st.tabs(
        ["Chat", "Endpoint Explorer", "API Call", "Debug & Traces"]
    )
    with chat_tab:
        _render_chat(spec)
    with explorer_tab:
        _render_explorer(spec)
    with call_tab:
        _render_call_builder(spec)
    with logs_tab:
        _render_logs()
