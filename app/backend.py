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
from typing import Any, Dict, List, Optional, Tuple

from multi_agent_system import config
from multi_agent_system.graph import run_graph

# Name of the university (HUA) MCP server, exactly as keyed in
# mcp_servers/commands.json. Used to restrict the scope to HUA only.
HUA_SERVER_NAME = "HUA Informatics"


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
    return asyncio.run(_run(initial_state))


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
        "execution_results": snapshot.get("execution_results", []),
        "failure_history": final_state.get("failure_history", []),
        "replans": int(final_state.get("_replans", 0) or 0),
        "total_rounds": snapshot.get("execution", {}).get("total_rounds", 0),
        "elapsed": elapsed,
    }