"""
token_tracker.py — Lightweight per-agent token usage tracker.

Each agent calls `track()` after an LLM response with the agent name and
the response metadata. The tracker accumulates input/output token
counts and provides a summary at the end of a run.

Usage:
    from ..token_tracker import token_tracker
    token_tracker.track("executor", response)   # response is an AIMessage
    token_tracker.summary()                      # prints the table
    token_tracker.reset()                        # between runs
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TokenTracker:
    """
    Accumulates token usage per agent across an entire run.
    - Tracks input tokens, output tokens, and call counts.
    - Extracts usage from response metadata (compatible with OpenAI-style APIs).s
    """

    def __init__(self):
        # Format: {agent_name: {"input_tokens": int, "output_tokens": int, "calls": int}}
        self._data: Dict[str, Dict[str, int]] = {}

    def track(self, agent: str, response: Any) -> None:
        """
        Record token usage from a AIMessage or chain response.

        Extracts token counts from response_metadata. 
        Silently skips if metadata is missing — not all responses will have usage info.
        """
        metadata = {}
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
        elif isinstance(response, dict):
            metadata = response.get("response_metadata", {})

        usage = metadata.get("token_usage") or metadata.get("usage", {})
        if not usage:
            return

        if agent not in self._data:
            self._data[agent] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        self._data[agent]["input_tokens"] += usage.get("prompt_tokens", 0)
        self._data[agent]["output_tokens"] += usage.get("completion_tokens", 0)
        self._data[agent]["calls"] += 1

    def summary(self) -> str:
        """
        Return a formatted summary table of token usage by agent.
        """
        if not self._data:
            return "No token usage recorded."

        lines = [
            "",
            "TOKEN USAGE BY AGENT",
            "-" * 65,
            f"{'Agent':<15} {'Calls':>6} {'Input':>10} {'Output':>10} {'Total':>10}",
            "-" * 65,
        ]

        total_in = 0
        total_out = 0
        total_calls = 0

        for agent, stats in sorted(self._data.items()):
            inp = stats["input_tokens"]
            out = stats["output_tokens"]
            calls = stats["calls"]
            total_in += inp
            total_out += out
            total_calls += calls
            lines.append(
                f"{agent:<15} {calls:>6} {inp:>10,} {out:>10,} {inp + out:>10,}"
            )

        lines.append("-" * 65)
        lines.append(
            f"{'TOTAL':<15} {total_calls:>6} {total_in:>10,} {total_out:>10,} {total_in + total_out:>10,}"
        )
        lines.append("")

        return "\n".join(lines)

    def get_totals(self) -> Dict[str, int]:
        """
        Return aggregate totals across all agents.
        """
        total_in = sum(s["input_tokens"] for s in self._data.values())
        total_out = sum(s["output_tokens"] for s in self._data.values())
        total_calls = sum(s["calls"] for s in self._data.values())
        return {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_calls": total_calls,
        }

    def reset(self) -> None:
        """
        Clear all recorded data. Call between runs."""
        self._data.clear()


# Singleton instance — imported by all agents.
token_tracker = TokenTracker()
