"""
Streamlit demo for the Multi-Agent MCP System — entrypoint.

Run with:
    streamlit run app/app.py

Opens a chat UI at http://localhost:8501. The first message shows the system
warming up by connecting to the MCP servers; afterwards you can ask anything
and watch the agents work in real time — each Planner cycle, every tool call,
every Verifier decision is streamed into the conversation as it happens.

A Base/HUA switch on the chat bar restricts the run to the HUA Informatics
server only (HUA) or uses every server (Base).

Prerequisites — same as running the system normally:
  - MULTIAGENT_MODEL and OPENROUTER_API_KEY (or MULTIAGENT_API_KEY) set
  - mcp_servers/api_key populated (cp mcp_servers/api_key.example mcp_servers/api_key)
  - multi_agent_system/inventory_summary.json present (run utils/collect_mcp_info.py once)

Structure: this file only orchestrates — UI lives in ui.py, logic in
backend.py, log streaming in log_stream.py.
"""

import sys
import time
from pathlib import Path

# Make the project root importable (multi_agent_system, app) when Streamlit
# runs this file from inside app/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app import backend, ui  # noqa: E402
from app.log_stream import attach_log_handlers, detach_log_handlers  # noqa: E402


def _init_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("warmed_up", False)
    if "inventory" not in st.session_state:
        inv = backend.load_inventory()
        st.session_state.inventory = inv
        if inv.get("error"):
            st.warning(f"Could not read inventory_summary.json — {inv['error']}")


def _maybe_welcome() -> None:
    """Append the one-time welcome message to the chat history."""
    if st.session_state.warmed_up:
        return
    content = ui.build_welcome(st.session_state.inventory)
    st.session_state.messages.append(
        {"role": "assistant", "content": content, "_html": True}
    )
    st.session_state.warmed_up = True


def _avatar_for(role: str):
    return ui.ASSISTANT_AVATAR if role == "assistant" else None


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=_avatar_for(msg["role"])):
            st.markdown(msg["content"], unsafe_allow_html=msg.get("_html", False))
            trace = msg.get("trace")
            if trace:
                ui.render_trace(trace)


def _handle_prompt(prompt: str, university_mode: bool) -> None:
    """Run one user turn: record it, stream the pipeline, render the answer."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    server_subset, active_total = backend.resolve_scope(
        university_mode, st.session_state.inventory
    )

    with st.chat_message("assistant", avatar=ui.ASSISTANT_AVATAR):
        progress_box = st.empty()
        progress_bar = progress_box.progress(
            0.0, text=f"Connecting MCP servers — 0/{active_total}…"
        )
        status_box = st.empty()
        handlers = attach_log_handlers(
            status_box, progress_bar=progress_bar, total_servers=active_total
        )

        start = time.time()
        try:
            final_state = backend.run_pipeline(prompt, server_subset)
            error = None
        except Exception as e:  # noqa: BLE001 — surfaced to the user below
            final_state = {}
            error = str(e)
        finally:
            detach_log_handlers(handlers)
        elapsed = time.time() - start
        progress_box.empty()
        status_box.empty()

        if error:
            st.error(f"Pipeline crashed: {error}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Pipeline crashed: {error}"}
            )
            return

        answer = final_state.get("final_output") or "(no answer produced)"
        st.markdown(answer)
        trace = backend.build_trace(final_state, elapsed)
        ui.render_trace(trace)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "trace": trace}
        )


def main() -> None:
    ui.setup_page()
    _init_session()
    _maybe_welcome()

    # Capture the prompt BEFORE rendering history: if the user just sent a
    # message we strip the welcome out of the list first, so it never appears
    # once a real conversation has started.
    prompt = st.chat_input("Ask me anything…")
    university_mode = ui.render_scope_switch()

    if prompt:
        st.session_state.messages = [
            m for m in st.session_state.messages if not m.get("_html")
        ]

    _render_history()

    if prompt:
        _handle_prompt(prompt, university_mode)


main()