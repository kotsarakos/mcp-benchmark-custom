"""
Log streaming for the Multi-Agent MCP demo.

Bridges two backend logger namespaces onto live Streamlit placeholders:

  - ``mcp_modules``        → a progress bar (one tick per server connected)
  - ``multi_agent_system`` → a single Claude-style status line

The handlers update placeholder objects passed in by the UI; they never
import Streamlit themselves, so this module stays UI-agnostic (it only relies
on the placeholder exposing ``.progress(...)`` / ``.markdown(...)``).
"""

import logging
import re


class ServerProgressHandler(logging.Handler):
    """
    Listens to ``mcp_modules`` logs and increments a progress bar once per
    unique server. Deduping avoids the double-count from servers that emit
    BOTH 'Discovered N tools from X' AND 'Persistent session established for X
    with N tools' during start-up.

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


class AgentLogHandler(logging.Handler):
    """
    Watches ``multi_agent_system`` logs and updates a single Claude-style
    status line — no raw logs, just the current phase with an icon.

    The Planner emits explicit phase markers ([PLANNER], [RETRIEVAL],
    [EXECUTOR], [ANSWER], [VERIFIER]) that this handler maps to friendly labels.
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
            # The `.claude-thinking` CSS class is injected once at page top
            # (see styles.css); updating the inner text keeps the shimmer going.
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


def attach_log_handlers(log_placeholder, progress_bar=None, total_servers: int = 28):
    """
    Wire up two separate handlers on two separate logger namespaces:
      - mcp_modules        → progress bar only (server lifecycle is hidden)
      - multi_agent_system → agent activity feed

    Returns the pair so they can be detached later via ``detach_log_handlers``.
    """
    progress_handler = ServerProgressHandler(progress_bar, total_servers)
    progress_handler.setLevel(logging.INFO)

    feed_handler = AgentLogHandler(log_placeholder)
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


def detach_log_handlers(handlers) -> None:
    """Undo ``attach_log_handlers`` and restore log propagation."""
    progress_handler, feed_handler = handlers
    mcp_logger = logging.getLogger("mcp_modules")
    mcp_logger.removeHandler(progress_handler)
    mcp_logger.propagate = True

    agent_logger = logging.getLogger("multi_agent_system")
    agent_logger.removeHandler(feed_handler)