"""
Backend for the Multi-Agent MCP demo — all non-UI logic.

Responsible for:
  - loading the MCP server inventory,
  - resolving the Base/HUA server scope,
  - running the multi-agent pipeline for a single prompt,
  - distilling the final state into a small, serialisable trace dict.

This module never imports Streamlit, so it can be unit-tested or reused
independently of the UI.
"""

import asyncio
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from multi_agent_system import config
from multi_agent_system.graph import run_graph

# Name of the university (HUA) MCP server, exactly as keyed in
# mcp_servers/commands.json. Used to restrict the scope to HUA only.
HUA_SERVER_NAME = "HUA Informatics"

# print_plan() in multi_agent_system/utils.py emits one line per task as
#   "   • task_3: <description>"
# We tee stdout during the run to harvest these, since task_definitions is
# overwritten each planning cycle and the descriptions would otherwise be lost.
_TASK_LINE = re.compile(r"^\s*•\s*([A-Za-z0-9_]+):\s*(.+?)\s*$")


class _DescriptionTee:
    """
    A stdout wrapper that passes everything through to the real stream while
    sniffing ``• task_id: description`` lines printed by ``print_plan``.

    Accumulates the description for every task across all planning cycles, so
    the UI can show a name for each entry in ``completed_tasks_results`` without
    any agent-side change. Last description wins (handles replanned task IDs).
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self._buf = ""
        self.descriptions: Dict[str, str] = {}

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = _TASK_LINE.match(line)
            if m:
                tid, desc = m.group(1), m.group(2)
                if desc and desc != "N/A":
                    self.descriptions[tid] = desc
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def load_inventory() -> Dict[str, Any]:
    """
    Read ``inventory_summary.json`` and return ``{"total": int, "servers": [...]}``.

    Falls back to a sane default (and surfaces the error string) if the file is
    missing or unreadable, so the UI can warn without crashing.
    """
    try:
        inv_path = config.INVENTORY_DIR / "inventory_summary.json"
        with open(inv_path, encoding="utf-8") as f:
            inv = json.load(f)
        return {
            "total": int(inv.get("total_servers") or len(inv.get("available_servers", []))),
            "servers": list(inv.get("available_servers", [])),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — surfaced to the UI as a warning
        return {"total": 29, "servers": [], "error": str(e)}


def resolve_scope(
    university_mode: bool, inventory: Dict[str, Any]
) -> Tuple[Optional[List[str]], int]:
    """
    Map the Base/HUA switch onto a server subset.

    Returns ``(server_subset, active_total)`` where ``server_subset`` is None for
    Base (all servers) or ``[HUA_SERVER_NAME]`` for HUA-only.
    """
    if university_mode:
        return [HUA_SERVER_NAME], 1
    return None, inventory["total"]


async def _run(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    return await run_graph(initial_state)


def run_pipeline(prompt: str, server_subset: Optional[List[str]]) -> Dict[str, Any]:
    """
    Execute the multi-agent graph for ``prompt`` and return the final state.

    When ``server_subset`` is provided, only those MCP servers are spun up.
    Trace recording is always enabled so the UI can render the run.
    """
    initial_state: Dict[str, Any] = {"input": prompt, "_enable_trace": True}
    if server_subset is not None:
        initial_state["_server_subset"] = server_subset

    tee = _DescriptionTee(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        final_state = asyncio.run(_run(initial_state))
    finally:
        sys.stdout = old_stdout

    # Stash the harvested per-task descriptions for build_trace().
    final_state["_task_descriptions"] = tee.descriptions
    return final_state


def build_trace(final_state: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    """
    Distil ``final_state`` into a small serialisable dict for history replay.

    Pulls execution details from the trace recorder snapshot when available.
    """
    recorder = final_state.get("_recorder")
    snapshot = recorder.to_dict() if recorder is not None else {}
    return {
        "plan": final_state.get("plan", []),
        "task_definitions": final_state.get("task_definitions", {}),
        "completed_tasks_results": final_state.get("completed_tasks_results", {}),
        "task_descriptions": final_state.get("_task_descriptions", {}),
        "execution_results": snapshot.get("execution_results", []),
        "failure_history": final_state.get("failure_history", []),
        "replans": int(final_state.get("_replans", 0) or 0),
        "total_rounds": snapshot.get("execution", {}).get("total_rounds", 0),
        "elapsed": elapsed,
    }