"""
TraceRecorder — captures execution traces in MCP-Bench-compatible format.

Enabled via initial_state["_enable_trace"]=True. When disabled, all record_*
methods are no operated, so HotpotQA and other runs pay zero overhead.

Consumed by mcpbench_benchmark/ to score runs against the official
TaskEvaluator (benchmark/evaluator.py).
"""

from typing import Any, Dict, List, Optional


class TraceRecorder:
    def __init__(self) -> None:
        self.tool_calls: List[Dict[str, Any]] = []
        self.rounds: int = 0
        self.plan_parse_total: int = 0
        self.plan_parse_success: int = 0
        self.available_tools: Dict[str, Any] = {}

    def set_available_tools(self, all_tools: Dict[str, Any]) -> None:
        """
        Snapshot the tool registry at the start of the run.
        """
        self.available_tools = {
            name: {
                "server": info.get("server", ""),
                "description": info.get("description", "") or "",
                "input_schema": info.get("input_schema", {}) or {},
            }
            for name, info in all_tools.items()
        }

    def increment_round(self) -> None:
        self.rounds += 1

    def record_tool_call(
        self,
        *,
        tool: str,
        server: str,
        parameters: Dict[str, Any],
        success: bool,
        result: str,
        error: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        self.tool_calls.append({
            "round": self.rounds,
            "tool": tool,
            "server": server,
            "parameters": parameters,
            "success": success,
            "result": result,
            "error": error,
            "duration_seconds": duration_seconds,
        })

    def record_plan_parse(self, success: bool) -> None:
        self.plan_parse_total += 1
        if success:
            self.plan_parse_success += 1

    @property
    def planning_json_compliance(self) -> float:
        if self.plan_parse_total == 0:
            return 1.0
        return self.plan_parse_success / self.plan_parse_total

    def build_accumulated_information(self) -> str:
        """
        Concatenate all tool observations for the judge's execution_summary.
        """
        
        if not self.tool_calls:
            return ""
        parts = []
        for i, c in enumerate(self.tool_calls, 1):
            status = "OK" if c["success"] else f"FAIL ({c.get('error')})"
            parts.append(
                f"[Call {i} | round {c['round']}] "
                f"{c['server']}.{c['tool']}({c['parameters']}) → {status}\n"
                f"{c['result']}"
            )
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Render the trace block for the output JSON.
        """

        return {
            "execution": {
                "total_rounds": self.rounds,
                "planning_json_compliance": self.planning_json_compliance,
            },
            "available_tools": self.available_tools,
            "execution_results": list(self.tool_calls),
            "accumulated_information": self.build_accumulated_information(),
        }


def get_recorder(state: Dict[str, Any]) -> Optional[TraceRecorder]:
    """Returns the recorder from state, or None if tracing is disabled."""
    return state.get("_recorder") if isinstance(state, dict) else None
