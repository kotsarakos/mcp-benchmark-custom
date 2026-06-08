"""
Frontend for the Multi-Agent MCP demo — all Streamlit rendering.

Holds the page chrome (config, CSS, sidebar), the welcome message, the
Base/HUA scope switch, and the run-trace expanders. Keeps presentation out
of ``app.py`` (orchestration) and ``backend.py`` (logic).
"""

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

_STATIC_DIR = Path(__file__).parent / "static"
_PLANNER_IMG = _STATIC_DIR / "planner.png"

# Avatar used for every assistant turn (chat history + live run).
ASSISTANT_AVATAR = str(_PLANNER_IMG)


# ───────────────────────────────────────────────────────────────────
# Page chrome
# ───────────────────────────────────────────────────────────────────
def load_css() -> None:
    """Inject the bundled stylesheet once."""
    css = (_STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def setup_page() -> None:
    """Page config, stylesheet, header and the static pipeline sidebar."""
    st.set_page_config(
        page_title="Multi-Agent MCP Demo",
        layout="wide",
        page_icon=str(_PLANNER_IMG),
    )
    load_css()

    st.markdown(
        """
        <div class="app-header">
          <span class="app-eyebrow">Harokopio University · DIT — bachelor thesis</span>
          <h1 class="app-title">Multi-Agent MCP — Live Chat</h1>
          <p class="app-sub">
            Ask anything. The Planner decomposes your query into tasks, the
            Executor calls MCP tools to fulfil them, and the Verifier checks each
            step — you see every decision as it happens.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _render_sidebar()


def _render_sidebar() -> None:
    """Professional app sidebar — status overview + session controls."""
    import base64
    _img_b64 = base64.b64encode(_PLANNER_IMG.read_bytes()).decode()
    inv = st.session_state.get("inventory", {})
    total = inv.get("total", "—")
    n_msgs = max(len(st.session_state.get("messages", [])) - 1, 0)  # exclude welcome

    st.markdown(
        f"""
        <div class="sb-brand">
          <img src="data:image/png;base64,{_img_b64}" class="sb-logo-img" alt="agent logo">
          <div>
            <div class="sb-app-name">MCP Agent</div>
            <div class="sb-app-version">Multi-Agent System · v1.0</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section-label">STATUS</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sb-stat-grid">
          <div class="sb-stat">
            <div class="sb-stat-value">{total}</div>
            <div class="sb-stat-label">MCP Servers</div>
          </div>
          <div class="sb-stat">
            <div class="sb-stat-value">{n_msgs}</div>
            <div class="sb-stat-label">Messages</div>
          </div>
          <div class="sb-stat">
            <div class="sb-stat-dot on"></div>
            <div class="sb-stat-label">Online</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section-label">AGENTS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sb-agent-list">
          <div class="sb-agent"><span class="sb-agent-icon">🧠</span><span>Planner</span></div>
          <div class="sb-agent"><span class="sb-agent-icon">🔍</span><span>Retrieval</span></div>
          <div class="sb-agent"><span class="sb-agent-icon">⚙️</span><span>Executor</span></div>
          <div class="sb-agent"><span class="sb-agent-icon">📝</span><span>Answer</span></div>
          <div class="sb-agent"><span class="sb-agent-icon">✅</span><span>Verifier</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-spacer"></div>', unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        for key in ("messages", "warmed_up"):
            st.session_state.pop(key, None)
        st.rerun()


# ───────────────────────────────────────────────────────────────────
# Welcome message
# ───────────────────────────────────────────────────────────────────
def build_welcome(inventory: Dict[str, Any]) -> str:
    """Compose the one-time welcome markdown (server count + chips)."""
    chips_html = "".join(
        f'<span class="server-chip">{name}</span>' for name in inventory["servers"]
    ) if inventory["servers"] else ""
    return (
        f"👋 Hi! I'm a multi-agent system with access to **{inventory['total']} MCP servers** "
        "covering Wikipedia, NASA, biomedical research, mapping, finance, weather, "
        "scientific computing and more. Each server spins up on demand when you send "
        f"your first message.\n\n{chips_html}\n\n**Ask me anything below 👇**"
    )


# ───────────────────────────────────────────────────────────────────
# Base/HUA scope switch (styled via .st-key-scopeswitch in styles.css)
# ───────────────────────────────────────────────────────────────────
def render_scope_switch() -> bool:
    """Render the segmented switch and return True when HUA mode is selected."""
    with st.container(key="scopeswitch"):
        scope = st.segmented_control(
            "Server scope",
            options=["Base", "HUA"],
            default="Base",
            label_visibility="collapsed",
        )
    return scope == "HUA"


# ───────────────────────────────────────────────────────────────────
# Run-trace expanders
# ───────────────────────────────────────────────────────────────────
def render_trace(trace: Dict[str, Any]) -> None:
    """Render plan, tool calls, failures and summary metrics for one run."""
    plan = trace.get("plan") or []
    task_defs = trace.get("task_definitions") or {}
    completed = trace.get("completed_tasks_results") or {}
    calls = trace.get("execution_results") or []
    failures = trace.get("failure_history") or []
    replans = int(trace.get("replans") or 0)
    elapsed = trace.get("elapsed", 0.0)
    total_rounds = trace.get("total_rounds", 0)

    # The planner explicitly clears state["plan"] = [] after final synthesis
    # (planner.py:348/533/616/674), so at this point `plan` is usually empty
    # even though tasks did execute. Fall back to task_definitions, which
    # survive the clear and tell us exactly what the planner created.
    if plan:
        with st.expander(f"🧠 Plan ({len(plan)} step(s))"):
            for i, step in enumerate(plan):
                parallel = step.get("parallel", False)
                st.markdown(f"**Step {i}** *(parallel={parallel})*")
                for tid in step.get("tasks", []):
                    tdef = task_defs.get(tid, {})
                    desc = tdef.get("description") or "(no description)"
                    ttype = tdef.get("task_type", "tool")
                    st.markdown(f"- `{tid}` · *{ttype}*  \n  {desc}")
    elif task_defs:
        with st.expander(f"🧠 Tasks executed ({len(task_defs)})"):
            for tid, tdef in task_defs.items():
                desc = tdef.get("description") or "(no description)"
                ttype = tdef.get("task_type", "tool")
                deps = tdef.get("dependencies", []) or []
                done = tid in completed
                badge = "✅" if done else "⏺"
                line = f"{badge} `{tid}` · *{ttype}*"
                if deps:
                    line += f" · deps: {', '.join(f'`{d}`' for d in deps)}"
                st.markdown(f"{line}  \n  {desc}")
                if done:
                    ans = completed[tid].get("final_answer") or ""
                    if ans:
                        st.caption(f"↳ {ans[:300]}{'…' if len(ans) > 300 else ''}")
    else:
        with st.expander("🧠 Plan"):
            st.info(
                "No plan or task definitions were recorded — the Planner may "
                "have answered directly without scheduling tool calls."
            )

    with st.expander(f"⚙️ Tool calls ({len(calls)})"):
        if not calls:
            st.info("No tool calls were made.")
        for i, c in enumerate(calls, 1):
            status = "✅" if c.get("success") else "❌"
            st.markdown(
                f"**[{i}]** round {c.get('round', '?')} · "
                f"`{c.get('server', '?')}.{c.get('tool', '?')}` {status}"
            )
            cols = st.columns([1, 1])
            with cols[0]:
                st.caption("Parameters")
                st.code(
                    json.dumps(c.get("parameters", {}), indent=2, ensure_ascii=False),
                    language="json",
                )
            with cols[1]:
                st.caption("Result (truncated)")
                st.text((c.get("result") or "(empty)")[:2000])
            if c.get("error"):
                st.error(c["error"])
            st.divider()

    if failures or replans:
        with st.expander(f"⚠️ Failures & replans ({replans} replans)"):
            for f in failures:
                if not isinstance(f, dict):
                    st.text(str(f))
                    continue
                tid = f.get("task_id") or "(no task)"
                srv = f.get("server") or "?"
                etype = f.get("error_type") or "unknown"
                reason = f.get("reason") or ""
                st.markdown(f"- `{tid}` on **{srv}** — *{etype}*  \n  {reason}")

    cols = st.columns(4)
    cols[0].metric("Planner cycles", total_rounds)
    cols[1].metric("Tool calls", len(calls))
    cols[2].metric("Replans", replans)
    cols[3].metric("Wall time", f"{elapsed:.1f}s")