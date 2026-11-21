"""Cinephile GenAI - Streamlit UI for Levels 0, 1 and 2.

Run:  streamlit run app.py

- Level 0: LLM alone (level0.run)
- Level 1: RAG with citations (level1.rag_answer)
- Level 2: tool-using agent (agent.agent_answer) with confirm/cancel controls
  bound to the exact pending request_id.

Every interaction is appended to results/ui_traces.jsonl.
"""
import json
import os
import secrets
import sys
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def _try_import(name):
    try:
        return __import__(name), None
    except Exception as e:  # show the reason in the UI instead of crashing
        return None, f"{type(e).__name__}: {e}"


level0, level0_err = _try_import("level0")
level1, level1_err = _try_import("level1")
agent, agent_err = _try_import("agent")
try:
    from tools.watchlist import list_watchlist
except Exception:
    list_watchlist = None

TRACE_FILE = os.path.join("results", "ui_traces.jsonl")
LEVELS = {
    "Level 0 · LLM alone": 0,
    "Level 1 · RAG": 1,
    "Level 2 · Agent (tools)": 2,
}

st.set_page_config(page_title="Cinephile GenAI", page_icon="🎬", layout="wide")

ss = st.session_state
ss.setdefault("session_id", secrets.token_urlsafe(24))  # unpredictable, per conversation
ss.setdefault("agent_state", {})
ss.setdefault("history", [])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def log_trace(entry):
    os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
             "session_id": ss.session_id, **entry}
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def current_pending():
    return (ss.agent_state or {}).get("pending_action")


def sources_from_agent_trace(trace):
    """Collect RAG citations and TMDB URLs from the agent's tool results."""
    found = []
    for step in trace or []:
        res = step.get("tool_result")
        if not isinstance(res, dict):
            continue
        for c in res.get("citations") or []:  # rag_search result
            found.append(c)
        data = res.get("data") if isinstance(res.get("data"), dict) else {}
        if data.get("source_url"):
            found.append({"title": data.get("title"), "url": data["source_url"]})
        for m in (data.get("movies") or [])[:5]:
            if isinstance(m, dict) and m.get("source_url"):
                found.append({"title": m.get("title"), "url": m["source_url"]})
    unique, seen = [], set()
    for s in found:
        key = json.dumps(s, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def run_query(level, query):
    """Run one user query at the chosen level; return a normalized turn."""
    turn = {"level": level, "query": query, "answer": "", "sources": [], "trace": [],
            "tokens": 0, "latency": 0.0, "stop_reason": None, "pending_action": None,
            "error": None}
    try:
        if level == 0:
            if not level0:
                raise RuntimeError(f"level0 could not be loaded: {level0_err}")
            res = level0.run(query)
            turn.update(answer=res.get("answer", ""),
                        tokens=res.get("prompt_tokens", 0) + res.get("completion_tokens", 0),
                        latency=res.get("latency_s", 0.0),
                        stop_reason="abstained" if res.get("abstained") else "completed")
        elif level == 1:
            if not level1:
                raise RuntimeError(f"level1 could not be loaded: {level1_err}")
            res = level1.rag_answer(query)
            turn.update(answer=res.get("answer", ""), sources=res.get("citations", []),
                        tokens=res.get("usage", {}).get("total_tokens", 0),
                        latency=res.get("latency", 0.0),
                        stop_reason="abstained" if res.get("abstained") else "completed")
        else:
            if not agent:
                raise RuntimeError(f"agent could not be loaded: {agent_err}")
            note = None
            if current_pending():
                # A new request replaces the old proposal: cancel it first so it
                # can never be confirmed later. Storage stays unchanged.
                old = current_pending()
                agent.SESSION_ID = ss.session_id
                agent.agent_answer("", state=ss.agent_state, user_decision="cancel")
                note = f'Previous pending action on "{old.get("title")}" was discarded.'
            agent.SESSION_ID = ss.session_id
            res = agent.agent_answer(query, state=ss.agent_state)
            ss.agent_state = res.get("state") or {}
            turn.update(answer=res.get("answer", ""), trace=res.get("trace", []),
                        sources=sources_from_agent_trace(res.get("trace")),
                        tokens=res.get("usage", {}).get("total_tokens", 0),
                        latency=res.get("latency", 0.0),
                        stop_reason=res.get("stop_reason"),
                        pending_action=res.get("pending_action"))
            if note:
                turn["note"] = note
    except Exception as e:
        turn["error"] = f"{type(e).__name__}: {e}"
        turn["answer"] = f"Error: {turn['error']}"
    log_trace({"type": "query", **turn})
    return turn


def apply_decision(request_id, decision):
    """Confirm or cancel. Only acts if request_id is still the current pending one."""
    pending = current_pending()
    if not pending or pending.get("request_id") != request_id:
        ss.history.append({"level": 2, "query": None, "decision": decision,
                           "answer": "This confirmation is no longer valid (the action "
                                     "changed or was already resolved). Nothing was done.",
                           "trace": [], "sources": [], "tokens": 0, "latency": 0.0,
                           "stop_reason": "stale_confirmation", "error": None})
        return
    agent.SESSION_ID = ss.session_id
    res = agent.agent_answer("", state=ss.agent_state, user_decision=decision)
    ss.agent_state = res.get("state") or {}
    turn = {"level": 2, "query": None, "decision": decision,
            "answer": res.get("answer", ""), "trace": res.get("trace", []),
            "sources": [], "tokens": 0, "latency": res.get("latency", 0.0),
            "stop_reason": res.get("stop_reason"), "error": None,
            "request_id": request_id}
    ss.history.append(turn)
    log_trace({"type": "confirmation", **turn})


def summarize_result(res):
    if not isinstance(res, dict):
        return str(res)[:200]
    if res.get("ok") is False:
        err = res.get("error") or {}
        return f"error `{err.get('code', '')}`: {err.get('message', err)}"
    if "answer" in res and "citations" in res:  # rag_search
        return f"RAG answer with {len(res.get('citations') or [])} citation(s)"
    data = res.get("data") if isinstance(res.get("data"), dict) else {}
    if "movies" in data:
        titles = [f"{m.get('title')} ({(m.get('release_date') or '')[:4]})"
                  for m in data["movies"][:5] if isinstance(m, dict)]
        return f"{len(data['movies'])} movie(s): " + ", ".join(titles)
    if data.get("status"):
        return f"status `{data['status']}`" + (f" — {data.get('title')}" if data.get("title") else "")
    if data.get("title"):
        return f"details for {data['title']}"
    return json.dumps(res, default=str)[:200]


def render_tool_activity(trace):
    if not trace:
        st.caption("No tool calls.")
        return
    for step in trace:
        it = step.get("iteration", "")
        if "tool_call" in step:
            st.markdown(f"🔧 **step {it} · `{step['tool_call']}`** "
                        f"`{json.dumps(step.get('arguments', {}), ensure_ascii=False)}`")
        elif "tool_result" in step:
            res = step["tool_result"]
            icon = "⚠️" if isinstance(res, dict) and res.get("ok") is False else "↳"
            st.markdown(f"&nbsp;&nbsp;&nbsp;{icon} {summarize_result(res)}")
        elif "error" in step:
            st.markdown(f"⛔ **step {it} rejected before execution:** {step['error']}")
        elif "confirmation" in step:
            icon = "✅" if step["confirmation"] == "confirm" else "❌"
            st.markdown(f"{icon} user decision **{step['confirmation']}** → "
                        f"{summarize_result(step.get('result'))}")


def render_sources(sources):
    if not sources:
        st.caption("No sources.")
        return
    for s in sources:
        if isinstance(s, dict):
            label = s.get("title") or s.get("doc_id") or s.get("id") or "source"
            url = s.get("url") or s.get("source_url")
            st.markdown(f"- [{label}]({url})" if url else f"- {label}")
        else:
            st.markdown(f"- {s}")


def render_turn(turn, is_last):
    if turn.get("query") is not None:
        with st.chat_message("user"):
            st.write(turn["query"])
            st.caption(f"Level {turn['level']}")
    with st.chat_message("assistant"):
        if turn.get("note"):
            st.info(turn["note"])
        if turn.get("decision"):
            st.caption(f"User clicked **{turn['decision']}**")
        if turn.get("error"):
            st.error(turn["answer"])
        else:
            st.write(turn["answer"] or "_No answer._")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens", turn.get("tokens", 0))
        c2.metric("Latency", f"{turn.get('latency', 0.0):.2f} s")
        c3.metric("Stop reason", turn.get("stop_reason") or "—")

        if turn["level"] >= 1 and turn.get("query") is not None:
            with st.expander(f"📚 Sources ({len(turn.get('sources') or [])})"):
                render_sources(turn.get("sources"))
        if turn["level"] == 2:
            with st.expander("🔧 Tool activity", expanded=is_last):
                render_tool_activity(turn.get("trace"))
            with st.expander("🧾 Raw trace (JSON)"):
                st.json(turn.get("trace") or [])

        # Confirmation controls: only for the newest turn, and only if its
        # request_id is still the one waiting in the agent state.
        pa = turn.get("pending_action")
        cur = current_pending()
        if is_last and pa and cur and cur.get("request_id") == pa.get("request_id"):
            args = pa.get("arguments", {})
            st.warning(f"**Confirmation required:** {args.get('action', '?').upper()} "
                       f"**{pa.get('title')}** (TMDB id {args.get('movie_id')}) "
                       f"{'to' if args.get('action') == 'add' else 'from'} the watchlist.")
            rid = pa["request_id"]
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("✅ Confirm", key=f"confirm_{rid}", type="primary"):
                apply_decision(rid, "confirm")
                st.rerun()
            if b2.button("❌ Cancel", key=f"cancel_{rid}"):
                apply_decision(rid, "cancel")
                st.rerun()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎬 Cinephile GenAI")
    level_label = st.radio("Level", list(LEVELS.keys()), index=2)
    level = LEVELS[level_label]
    st.caption("Level 3 (extension): not implemented yet.")
    model = getattr(agent, "MODEL", None) or os.getenv("LLM_MODEL", "?")
    st.caption(f"Model: `{model}`")
    for name, err in (("level0", level0_err), ("level1", level1_err), ("agent", agent_err)):
        if err:
            st.error(f"{name} not loaded: {err}")

    st.divider()
    st.subheader("📋 Watchlist")
    if list_watchlist:
        try:
            wl = list_watchlist()
            movies = (wl.get("data") or {}).get("movies", []) if wl.get("ok") else []
            if movies:
                for m in movies:
                    st.markdown(f"- **{m['title']}** · `{m['movie_id']}`")
            else:
                st.caption("Empty." if wl.get("ok") else f"Error: {wl.get('error')}")
        except Exception as e:
            st.caption(f"Could not read watchlist: {e}")
    else:
        st.caption("Watchlist module not available.")
    if st.button("🔄 Refresh"):
        st.rerun()

    st.divider()
    if st.button("🗑️ New conversation"):
        if current_pending() and agent:
            agent.SESSION_ID = ss.session_id
            agent.agent_answer("", state=ss.agent_state, user_decision="cancel")
        ss.history = []
        ss.agent_state = {}
        ss.session_id = secrets.token_urlsafe(24)
        st.rerun()
    st.caption(f"Traces saved to `{TRACE_FILE}`")


# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
st.title("🎬 Cinephile — movie & TV assistant")
st.caption("Level 0: LLM alone · Level 1: RAG with citations · "
           "Level 2: agent with TMDB tools and a confirmed watchlist")

for i, turn in enumerate(ss.history):
    render_turn(turn, is_last=(i == len(ss.history) - 1))

prompt = st.chat_input("Ask about a movie or series, or manage your watchlist…")
if prompt:
    with st.spinner(f"Running Level {level}…"):
        ss.history.append(run_query(level, prompt))
    st.rerun()
