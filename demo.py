"""
Streamlit demo for the Multi-Agent MCP System.

Run with:
    streamlit run demo.py

Opens a chat UI at http://localhost:8501. The first message shows the
system warming up by connecting to all 28 MCP servers; afterwards you
can ask anything and watch the agents work in real time — each Planner
cycle, every tool call, every Verifier decision is streamed into the
conversation as it happens.

Prerequisites — same as running the system normally:
  - MULTIAGENT_MODEL and OPENROUTER_API_KEY (or MULTIAGENT_API_KEY) set
  - mcp_servers/api_key populated (cp mcp_servers/api_key.example mcp_servers/api_key)
  - multi_agent_system/inventory_summary.json present (run utils/collect_mcp_info.py once)
"""

import asyncio
import json
import logging
import re
import time
from typing import List

import streamlit as st
import streamlit.components.v1 as components

from multi_agent_system.graph import run_graph


# ───────────────────────────────────────────────────────────────────
# Snake — runs in the browser so it stays responsive while the
# Python pipeline is blocked on asyncio.run(run_graph(...)).
# ───────────────────────────────────────────────────────────────────
_SNAKE_HTML = """
<div style="font-family:ui-sans-serif,system-ui;color:#e6e6e6;
            background:#0e1117;border:1px solid #2b313e;border-radius:8px;
            padding:14px;text-align:center">
  <div style="font-size:13px;margin-bottom:8px;opacity:.85">
    🐍 <b>Snake</b> — play while the agents work. Arrow keys / WASD to move,
    <kbd>Space</kbd> to restart.
  </div>
  <canvas id="snake" width="360" height="240"
          style="background:#161b22;border-radius:6px;outline:none" tabindex="0">
  </canvas>
  <div id="snake-score" style="margin-top:8px;font-size:13px">Score: 0</div>
</div>
<script>
(function () {
  const canvas = document.getElementById('snake');
  const ctx = canvas.getContext('2d');
  const scoreEl = document.getElementById('snake-score');
  const cell = 12, cols = canvas.width / cell, rows = canvas.height / cell;
  let snake, dir, nextDir, food, score, alive, tickMs;

  function reset() {
    snake = [{x: 8, y: 10}, {x: 7, y: 10}, {x: 6, y: 10}];
    dir = {x: 1, y: 0}; nextDir = dir;
    placeFood(); score = 0; alive = true; tickMs = 110;
    scoreEl.textContent = 'Score: 0';
  }
  function placeFood() {
    while (true) {
      food = {x: Math.floor(Math.random()*cols), y: Math.floor(Math.random()*rows)};
      if (!snake.some(s => s.x === food.x && s.y === food.y)) return;
    }
  }
  function step() {
    if (!alive) return;
    dir = nextDir;
    const head = {x: snake[0].x + dir.x, y: snake[0].y + dir.y};
    if (head.x < 0 || head.y < 0 || head.x >= cols || head.y >= rows ||
        snake.some(s => s.x === head.x && s.y === head.y)) {
      alive = false; draw(); return;
    }
    snake.unshift(head);
    if (head.x === food.x && head.y === food.y) {
      score += 1; scoreEl.textContent = 'Score: ' + score;
      placeFood();
      if (tickMs > 55) tickMs -= 2;
    } else {
      snake.pop();
    }
    draw();
  }
  function draw() {
    ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ff5d6c';
    ctx.fillRect(food.x*cell, food.y*cell, cell-1, cell-1);
    snake.forEach((s, i) => {
      ctx.fillStyle = i === 0 ? '#7fd1ff' : '#3f8fbf';
      ctx.fillRect(s.x*cell, s.y*cell, cell-1, cell-1);
    });
    if (!alive) {
      ctx.fillStyle = 'rgba(0,0,0,.55)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#fff';
      ctx.font = '16px ui-sans-serif,system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('Game over — press Space', canvas.width/2, canvas.height/2);
    }
  }
  canvas.addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    const map = {arrowup:{x:0,y:-1}, arrowdown:{x:0,y:1},
                 arrowleft:{x:-1,y:0}, arrowright:{x:1,y:0},
                 w:{x:0,y:-1}, s:{x:0,y:1}, a:{x:-1,y:0}, d:{x:1,y:0}};
    if (map[k]) {
      // Block 180° reversals.
      if (map[k].x !== -dir.x || map[k].y !== -dir.y) nextDir = map[k];
      e.preventDefault();
    } else if (k === ' ' && !alive) {
      reset();
    }
  });
  canvas.focus({preventScroll: true});
  reset();
  let last = 0;
  function loop(ts) {
    if (ts - last >= tickMs) { step(); last = ts; }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
})();
</script>
"""


def _show_snake(placeholder, height: int = 360) -> None:
    """Render Snake into a placeholder. Call placeholder.empty() to remove."""
    with placeholder.container():
        components.html(_SNAKE_HTML, height=height, scrolling=False)


# ───────────────────────────────────────────────────────────────────
# Page setup
# ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent MCP Demo",
    layout="wide",
    page_icon="🤖",
)

# Inject the Claude-style shimmer CSS once. Subsequent placeholder updates
# that re-use the `.claude-thinking` class keep the animation running smoothly.
st.markdown(
    """
    <style>
    @keyframes claude-shimmer {
      0%   { background-position: 200% 0;  }
      100% { background-position: -200% 0; }
    }
    .claude-thinking {
      display: inline-block;
      background: linear-gradient(90deg,
        rgba(180,180,180,.45) 0%,
        rgba(255,255,255,1)   50%,
        rgba(180,180,180,.45) 100%);
      background-size: 200% auto;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
              background-clip: text;
      animation: claude-shimmer 2.6s linear infinite;
      font-size: 15px;
      font-weight: 500;
      letter-spacing: .1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Multi-Agent MCP — Live Chat")
st.caption(
    "Ask anything. The Planner decomposes your query into tasks, the "
    "Executor calls MCP tools to fulfil them, and the Verifier checks "
    "each step. You see every decision as it happens."
)

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        "- 🧠 **Planner** — builds a DAG of tasks.\n"
        "- 🔍 **Retrieval** — picks the right MCP server per task.\n"
        "- ⚙️ **Executor** — ReAct loop (Thought → Action → Observation).\n"
        "- 📝 **Answer** — structures tool outputs.\n"
        "- ✅ **Verifier** — pass / fail / impossible.\n"
        "\nReplans on fail; capped at 5 replans / 20 total steps."
    )
    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        for key in ("messages", "warmed_up"):
            st.session_state.pop(key, None)
        st.rerun()


# ───────────────────────────────────────────────────────────────────
# Live log handler — streams agent logs into a Streamlit placeholder
# ───────────────────────────────────────────────────────────────────
class _ServerProgressHandler(logging.Handler):
    """
    Listens to `mcp_modules` logs and increments a Streamlit progress bar
    once per unique server (dedup avoids the double-count from servers that
    emit BOTH 'Discovered N tools from X' AND 'Persistent session established
    for X with N tools' during start-up).

    Nothing from mcp_modules reaches the agent log feed — port assignment,
    HTTP transport setup, session teardown, etc. all stay invisible.
    """

    _SERVER_READY = re.compile(
        r"(?:Persistent session established for|"
        r"Discovered \d+ tools from(?: HTTP server)?)\s+([^\s]+)"
    )

    def __init__(self, progress_bar, total_servers: int):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_servers = max(total_servers, 1)
        self.counted: set = set()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = self._SERVER_READY.search(msg)
        if not m or self.progress_bar is None:
            return
        server = m.group(1).strip().rstrip(".,:")
        if server in self.counted:
            return
        self.counted.add(server)
        pct = min(len(self.counted) / self.total_servers, 1.0)
        try:
            self.progress_bar.progress(
                pct,
                text=f"Connecting MCP servers — {len(self.counted)}/{self.total_servers} · {server}",
            )
        except Exception:
            pass


class _AgentLogHandler(logging.Handler):
    """
    Watches `multi_agent_system` logs and updates a single Claude-style
    status line — no raw logs, just the current phase with an icon.

    The Planner emits explicit phase markers ([PLANNER], [RETRIEVAL],
    [EXECUTOR], [ANSWER], [VERIFIER]) that this handler maps to friendly
    labels.
    """

    # Ordered list — first match wins, so put specific patterns above generic ones.
    _PHASES = (
        (re.compile(r"\[PLANNER\].*?(FINAL SYNTHESIS|All data collected|Nothing to do)", re.I),
         "✨ Composing the final answer"),
        (re.compile(r"\[PLANNER\] Replanning", re.I),
         "♻️ Replanning after a failure"),
        (re.compile(r"\[PLANNER\] Planning first step", re.I),
         "🧠 Planning the approach"),
        (re.compile(r"\[PLANNER\] Planning next step", re.I),
         "🧠 Planning the next step"),
        (re.compile(r"\[RETRIEVAL\]", re.I),
         "🔍 Searching for the right MCP servers"),
        (re.compile(r"\[EXECUTOR\]", re.I),
         "⚙️ Running tool calls"),
        (re.compile(r"\[ANSWER\]", re.I),
         "📝 Structuring the answer"),
        (re.compile(r"\[VERIFIER\]", re.I),
         "✅ Verifying the results"),
        (re.compile(r"reasoning step", re.I),
         "🤔 Reasoning over collected data"),
        (re.compile(r"STEP \d+ IMPOSSIBLE", re.I),
         "⚠️ Marking step as unanswerable"),
        (re.compile(r"partial synthesis", re.I),
         "✨ Composing a partial answer"),
    )

    def __init__(self, status_placeholder):
        super().__init__()
        self.placeholder = status_placeholder
        self.current: str = ""
        self._set("💭 Thinking")

    def _set(self, label: str) -> None:
        if label == self.current:
            return
        self.current = label
        try:
            # The `.claude-thinking` CSS class is injected once at page top;
            # updating the inner text keeps the shimmer animation running.
            self.placeholder.markdown(
                f'<div class="claude-thinking">{label}…</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        for pattern, label in self._PHASES:
            if pattern.search(msg):
                self._set(label)
                return
        # Unmatched logs are ignored — no raw log spam.


def _attach_log_handler(
    log_placeholder,
    progress_bar=None,
    total_servers: int = 28,
):
    """
    Wire up two separate handlers on two separate logger namespaces:
      - mcp_modules     → progress bar only (server lifecycle is hidden)
      - multi_agent_system → agent activity feed
    Returns the pair so they can be detached later.
    """
    progress_handler = _ServerProgressHandler(progress_bar, total_servers)
    progress_handler.setLevel(logging.INFO)

    feed_handler = _AgentLogHandler(log_placeholder)
    feed_handler.setLevel(logging.INFO)
    feed_handler.setFormatter(logging.Formatter("%(message)s"))

    mcp_logger = logging.getLogger("mcp_modules")
    mcp_logger.addHandler(progress_handler)
    mcp_logger.setLevel(logging.INFO)
    # Prevent mcp_modules records from propagating to the root logger / the
    # multi_agent_system handler, so port/HTTP/transport logs never leak.
    mcp_logger.propagate = False

    agent_logger = logging.getLogger("multi_agent_system")
    agent_logger.addHandler(feed_handler)
    agent_logger.setLevel(logging.INFO)

    return progress_handler, feed_handler


def _detach_log_handler(handlers) -> None:
    progress_handler, feed_handler = handlers
    mcp_logger = logging.getLogger("mcp_modules")
    mcp_logger.removeHandler(progress_handler)
    mcp_logger.propagate = True

    agent_logger = logging.getLogger("multi_agent_system")
    agent_logger.removeHandler(feed_handler)


# ───────────────────────────────────────────────────────────────────
# Trace renderer (defined before use)
# ───────────────────────────────────────────────────────────────────
def _render_trace(trace: dict) -> None:
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


# ───────────────────────────────────────────────────────────────────
# Session state init
# ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "warmed_up" not in st.session_state:
    st.session_state.warmed_up = False


# ───────────────────────────────────────────────────────────────────
# First-time welcome + warm-up
# ───────────────────────────────────────────────────────────────────
# Load inventory once per Streamlit session so warm-up and progress bars share it.
if "inventory" not in st.session_state:
    try:
        from multi_agent_system import config
        inv_path = config.INVENTORY_DIR / "inventory_summary.json"
        with open(inv_path, encoding="utf-8") as f:
            inv = json.load(f)
        st.session_state.inventory = {
            "total": int(inv.get("total_servers") or len(inv.get("available_servers", []))),
            "servers": list(inv.get("available_servers", [])),
        }
    except Exception as e:
        st.session_state.inventory = {"total": 28, "servers": []}
        st.warning(f"Could not read inventory_summary.json — {e}")

INV = st.session_state.inventory

if not st.session_state.warmed_up:
    with st.chat_message("assistant"):
        st.markdown(
            f"👋 Hi! I'm a multi-agent system with access to **{INV['total']} MCP servers** "
            "covering Wikipedia, NASA, biomedical research, mapping, finance, weather, "
            "scientific computing and more. Each server spins up on demand when you send "
            "your first message."
        )
        if INV["servers"]:
            # Compact chip-style inline list of server names.
            chip_style = (
                "display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;"
                "border:1px solid #2b313e;border-radius:12px;font-size:12px;"
                "background:#161b22;color:#cfd8dc"
            )
            chips = "".join(
                f'<span style="{chip_style}">{name}</span>' for name in INV["servers"]
            )
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown("**Ask me anything below 👇**")
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Send me your first question and I'll spin the MCP servers up. "
            "Try things like *“What is the elevation of the highest peak in "
            "Yosemite, and how many visitors did the park get last year?”* "
            "or *“Find me a recent arXiv paper about agentic RAG and "
            "summarise its abstract.”*"
        ),
    })
    st.session_state.warmed_up = True


# ───────────────────────────────────────────────────────────────────
# Render chat history
# ───────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Replay trace expanders for assistant messages that captured one.
        trace = msg.get("trace")
        if trace:
            _render_trace(trace)


# ───────────────────────────────────────────────────────────────────
# Chat input → run pipeline with live log streaming
# ───────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask me anything…")

if prompt:
    # Record + render the user turn.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant turn — animated status + Snake to play while the pipeline runs.
    with st.chat_message("assistant"):
        progress_box = st.empty()
        progress_bar = progress_box.progress(
            0.0,
            text=f"Connecting MCP servers — 0/{INV['total']}…",
        )
        status_box = st.empty()
        game_box = st.empty()
        st.caption("🐍 play Snake below while you wait")
        _show_snake(game_box)
        handlers = _attach_log_handler(
            status_box, progress_bar=progress_bar, total_servers=INV["total"]
        )

        start = time.time()
        try:
            final_state = asyncio.run(
                run_graph({"input": prompt, "_enable_trace": True})
            )
            error = None
        except Exception as e:
            final_state = {}
            error = str(e)
        finally:
            _detach_log_handler(handlers)
        elapsed = time.time() - start
        progress_box.empty()
        status_box.empty()
        game_box.empty()

        if error:
            st.error(f"Pipeline crashed: {error}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Pipeline crashed: {error}",
            })
        else:
            answer = final_state.get("final_output") or "(no answer produced)"
            st.markdown(answer)

            # Build a small serialisable trace dict for history replay.
            recorder = final_state.get("_recorder")
            snapshot = recorder.to_dict() if recorder is not None else {}
            trace = {
                "plan": final_state.get("plan", []),
                "task_definitions": final_state.get("task_definitions", {}),
                "completed_tasks_results": final_state.get("completed_tasks_results", {}),
                "execution_results": snapshot.get("execution_results", []),
                "failure_history": final_state.get("failure_history", []),
                "replans": int(final_state.get("_replans", 0) or 0),
                "total_rounds": snapshot.get("execution", {}).get("total_rounds", 0),
                "elapsed": elapsed,
            }
            _render_trace(trace)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "trace": trace,
            })
