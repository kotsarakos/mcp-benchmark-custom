"""
utils.py — Shared state management utilities for the MCP Multi-Agent System.

All agents operate on a single shared state dict that is passed through the
pipeline on every cycle. This module owns:
  - The merge strategy rules that determine how agent outputs are applied to state.
  - State initialisation (normalize_state) to guarantee all keys are present.
  - Helper functions for committing results, tracking failures, and querying
    plan progress.
  - Debug print helpers used by the Planner for structured console output.
"""

import copy
import json
import logging
import json_repair


from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _find_largest_json_object(text: str) -> str | None:
    """
    Scan `text` for balanced `{...}` JSON objects (respecting string literals
    and escapes) and return the largest one found. Used to extract real JSON
    from reasoning-model output that prefixes its answer with thinking text
    that may contain stray `{}` placeholders.
    Returns None if no balanced object is found.
    """
    best: str | None = None
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        for j in range(i, n):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i:j + 1]
                    if best is None or len(candidate) > len(best):
                        best = candidate
                    break
        i += 1
    return best


def clean_and_parse_json(raw: str):
    """
    Tolerant JSON parser that mirrors llm/provider.py's clean_and_parse_json,
    extended to handle reasoning-model output that prefixes its JSON with
    "Thinking Process:" text containing stray `{}` placeholders.

    Strategy:
      1. Strip ```json``` / ``` markdown fences if present.
      2. Try parsing the whole stripped text as JSON.
      3. If that fails, scan the text for the LARGEST balanced `{...}` object
         (a real plan is much bigger than a placeholder `{}`) and parse that.
      4. Fall back to json_repair as a last resort.
    """
    if not raw:
        raise ValueError("Empty content from LLM")
    text = raw
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
    text = text.strip()

    # Fast path — clean JSON.
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Largest balanced object — robust to leading "Thinking Process:" text
    # that contains placeholder `{}` snippets the model echoed from the prompt.
    largest = _find_largest_json_object(text)
    if largest is not None:
        try:
            return json.loads(largest)
        except json.JSONDecodeError:
            try:
                return json_repair.loads(largest)
            except Exception:
                pass

    # Last-resort: slice from the first { or [ and let json_repair try.
    first_brace = text.find("{")
    first_bracket = text.find("[")
    candidates = [i for i in (first_brace, first_bracket) if i != -1]
    if not candidates:
        raise ValueError(f"No JSON object/array found in LLM response: {raw[:300]}")
    sliced = text[min(candidates):]
    try:
        return json.loads(sliced)
    except json.JSONDecodeError:
        return json_repair.loads(sliced)


def current_date_str() -> str:
    """
    Returns today's UTC date as YYYY-MM-DD. Injected into every agent prompt
    so temporal reasoning ('past' vs 'future') is anchored to the wall clock,
    not the model's training cutoff.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Each key in the state dict has exactly one merge strategy.
# These sets are read by merge_state() to decide how to apply agent updates.

# REPLACE: the new value completely overwrites the existing one.
# Used for scalar fields and single-owner data structures (e.g. the current plan,
# selected servers, and verification status — only one agent writes each of these).
REPLACE_KEYS = {
    "plan",
    "task_definitions",
    "current_step_index",
    "last_failure_reason",
    "selected_servers",
    "latest_execution_results",
    "latest_verification_package",
    "final_output",
    "verification_status",
    "all_parts_found",
}

# DICT_MERGE: new entries are added; existing entries are updated.
# Used for the cumulative results store so verified task results from different
# steps accumulate rather than overwrite each other.
DICT_MERGE_KEYS = {
    "completed_tasks_results",
}

# LIST_EXTEND: new items are appended to the existing list.
# Used for append-only logs (failure history, messages) that must never lose
# earlier entries when an agent writes a new batch.
LIST_EXTEND_KEYS = {
    "failure_history",
    "messages",
    "errors",
    "final_history",
    "finished_task_ids",
}


def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all expected state keys exist before the pipeline starts.

    Called once by run_graph() on the raw user-supplied input dict.
    Uses setdefault so any keys already present (e.g. "input") are preserved.
    A deep copy is taken so the caller's original dict is never mutated.
    """
    normalized = copy.deepcopy(state)
    normalized.setdefault("input", "")
    normalized.setdefault("plan", [])
    normalized.setdefault("task_definitions", {})
    normalized.setdefault("failure_history", [])
    normalized.setdefault("last_failure_reason", "")
    normalized.setdefault("selected_servers", {})
    normalized.setdefault("completed_tasks_results", {})
    normalized.setdefault("finished_task_ids", [])
    normalized.setdefault("current_step_index", 0)
    normalized.setdefault("messages", [])
    normalized.setdefault("errors", [])
    normalized.setdefault("final_output", None)
    normalized.setdefault("latest_execution_results", {})
    normalized.setdefault("latest_verification_package", {})
    normalized.setdefault("final_history", [])
    normalized.setdefault("verification_status", "")
    normalized.setdefault("all_parts_found", False)
    normalized.setdefault("excluded_servers", {})
    normalized.setdefault("server_failure_counts", {})
    normalized.setdefault("_replans", 0)
    normalized.setdefault("_global_step", 0)
    return normalized


def merge_state(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply an agent's output patch into the running state dict.

    Each agent returns a partial dict (only the keys it modified). This function
    merges that patch into the full state according to the strategy defined in
    REPLACE_KEYS, DICT_MERGE_KEYS, and LIST_EXTEND_KEYS. Keys absent from all
    three sets fall back to a simple replace.

    None values in updates are skipped — agents can return None for a key to
    signal "no change" without accidentally clearing the existing value.
    """
    # Nothing to merge — return base unchanged to avoid unnecessary iteration.
    if not updates:
        return base
    for key, value in updates.items():

        # Skip None values — an agent returns None for a key to signal
        if value is None:
            continue

        # REPLACE: the agent owns this field exclusively; overwrite unconditionally.
        if key in REPLACE_KEYS:
            base[key] = value
            continue

        # DICT_MERGE: new task entries are added without losing earlier results.
        if key in DICT_MERGE_KEYS and isinstance(value, dict):
            base.setdefault(key, {})
            base[key].update(value)
            continue

        # LIST_EXTEND: new items are appended to the running log.
        if key in LIST_EXTEND_KEYS and isinstance(value, list):
            base.setdefault(key, [])
            base[key].extend(value)
            continue

        # Fallback: key is not in any strategy set — treat as a plain replace.
        # This covers internal bookkeeping keys like _replans and retry_count.
        base[key] = value

    return base


def commit_verified_results(state: Dict[str, Any]) -> None:
    """
    Persist the Verifier's approved answers into the permanent results store.

    The Verifier writes approved task data to `final_history`. This function
    moves those entries into `completed_tasks_results` (keyed by task_id) and
    `finished_task_ids`, which survive replans and are passed to the Planner
    so it does not re-schedule already-solved tasks.
    """
    
    # final_history is written by the Verifier after each approved step.
    # If it's empty (first run, or step not yet verified), nothing to commit.
    final_history = state.get("final_history", [])
    if not final_history:
        return
    
    # Access or initialize the persistent storage keys.
    # completed_tasks_results: Stores the actual answer data.
    # finished_task_ids: A simple list of IDs used by the Planner for quick checking.
    completed = state.setdefault("completed_tasks_results", {})
    finished_ids = state.setdefault("finished_task_ids", [])

    # Get the original task definitions to pull descriptions for context.
    task_defs = state.get("task_definitions", {})

    # Iterate through each verified item in the final history.
    for item in final_history:
        task_id = item.get("task_id")

        if not task_id:
            continue

        # Map the verified data into the permanent 'completed' dictionary.
        # We store the description, summary of work, and the final_answer.
        # This ensures that even if a replan happens, this data is preserved
        completed[task_id] = {
            "description": task_defs.get(task_id, {}).get("description", ""),
            "summary":      item.get("summary", ""),
            "final_answer": item.get("final_answer", "")
        }

        # finished_task_ids is the Planner's deduplication guard — it checks this list
        # before scheduling tasks so already-solved tasks are never re-executed on replan.
        if task_id not in finished_ids:
            finished_ids.append(task_id)


def refresh_task_descriptions(state: Dict[str, Any]) -> None:
    """
    Inject completed dependency results into pending task descriptions.

    Called after a step passes verification. For each task whose dependencies
    now have results, the dependency outputs are appended to the task's
    description so the Executor has full context when the task runs next.
    The marker prevents double-injection across multiple refresh calls.
    """

    task_defs = state.get("task_definitions", {})
    completed = state.get("completed_tasks_results", {})
    
    # For each task in the system, check if any of its dependencies have newly completed results.
    for tdef in task_defs.values():
        deps = tdef.get("dependencies", [])
        context = [
            f"{dep}: {completed[dep]['final_answer']}" # Task_1: The capital is ...
            for dep in deps
            if dep in completed and completed[dep].get("final_answer")
        ]
        # If there is a depedency
        if context:
            base_desc = tdef.get("description", "")
            marker = "\n\nResults from previous tasks:\n"
            if marker not in base_desc:
                tdef["description"] = base_desc + marker + "\n".join(context)


# Error-type classification used by record_failure(). Order matters — first match wins.
# Patterns are checked against the lower-cased reason string.
_ERROR_TYPE_PATTERNS = (
    ("access_denied", ("forbidden", "403", "401", "unauthorized", "access denied",
                       "permission denied", "not permitted", "access restricted",
                       "insufficient permissions", "no permission", "not authorized")),
    ("rate_limit",    ("rate limit", "too many requests", "429")),
    ("timeout",       ("timeout", "timed out", "deadline exceeded")),
    ("server_error",  ("503", "502", "500 internal", "service unavailable", "internal server error", "bad gateway")),
    ("connection",    ("connection refused", "connection reset", "network unreachable", "dns")),
    ("not_found",     ("not found", "404", "no such", "does not exist")),
    ("schema_error",  ("schema", "invalid argument", "validation error", "missing required", "bad request", "400")),
    ("empty",         ("empty result", "no results", "no data", "returned empty")),
)

# Reason strings stored in failure_history are capped at this many characters.
# Long raw API errors blow up the prompt and rarely add information past the head.
_MAX_REASON_CHARS = 600


def _classify_error_type(reason: str) -> str:
    """Map a raw error string to a coarse error_type label."""
    if not reason:
        return "unknown"
    low = reason.lower()
    for etype, kws in _ERROR_TYPE_PATTERNS:
        if any(kw in low for kw in kws):
            return etype
    return "unknown"


def _build_failure_entry(
    *,
    task_id: Optional[str],
    task_description: Optional[str],
    server: Optional[str],
    tool: Optional[str],
    reason: str,
    source: str,
    replan_round: int,
    global_step: int,
    explicit_error_type: Optional[str] = None,
) -> Dict[str, Any]:
    reason_str = (reason or "").strip()
    if len(reason_str) > _MAX_REASON_CHARS:
        reason_str = reason_str[:_MAX_REASON_CHARS] + "...[truncated]"
    return {
        "task_id": task_id,
        "task_description": task_description,
        "server": server,
        "tool": tool,
        "error_type": explicit_error_type or _classify_error_type(reason_str),
        "reason": reason_str,
        "source": source,
        "replan_round": replan_round,
        "global_step": global_step,
    }


def _entries_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Two failure entries are duplicates if task, error_type, and reason all match."""
    return (
        a.get("task_id") == b.get("task_id")
        and a.get("error_type") == b.get("error_type")
        and a.get("reason") == b.get("reason")
    )


def record_failure(
    state: Dict[str, Any],
    *,
    source: str = "verifier",
    explicit_error_type: Optional[str] = None,
) -> None:
    """
    Append structured failure entries to state['failure_history'].

    One entry per still-pending task in the current step, each carrying its own
    task_id, server, and (when available) per-task reason from
    latest_execution_results. This lets the synthesis prompt map "which sub-
    question failed" to "which server returned which error" instead of seeing a
    flat list of strings.

    If the current step has no resolvable tasks (e.g. planner JSON failure
    before a step exists), records a single source-level entry with task_id=None.
    """
    overall_reason = state.get("last_failure_reason", "") or ""
    if not overall_reason and not explicit_error_type:
        return

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    task_defs = state.get("task_definitions", {})
    selected = state.get("selected_servers", {})
    latest_exec = state.get("latest_execution_results", {})
    finished = set(state.get("finished_task_ids", []))
    replan_round = int(state.get("_replans", 0) or 0)
    global_step = int(state.get("_global_step", 0) or 0)

    history = state.setdefault("failure_history", [])

    current_task_ids: List[str] = []
    if plan and idx < len(plan):
        current_task_ids = [t for t in plan[idx].get("tasks", []) if t not in finished]

    if not current_task_ids:
        entry = _build_failure_entry(
            task_id=None,
            task_description=None,
            server=None,
            tool=None,
            reason=overall_reason,
            source=source,
            replan_round=replan_round,
            global_step=global_step,
            explicit_error_type=explicit_error_type,
        )
        if not history or not _entries_match(history[-1], entry):
            history.append(entry)
        return

    for tid in current_task_ids:
        tdef = task_defs.get(tid, {})
        srv_info = selected.get(tid)
        if isinstance(srv_info, dict):
            srv_name = srv_info.get("selected_server")
        else:
            srv_name = srv_info if isinstance(srv_info, str) else None
        if isinstance(srv_name, str) and srv_name.lower() in ("none", "null", ""):
            srv_name = None

        per_task_reason = overall_reason
        result_for_task = latest_exec.get(tid)
        if isinstance(result_for_task, str) and result_for_task.strip():
            per_task_reason = result_for_task

        entry = _build_failure_entry(
            task_id=tid,
            task_description=tdef.get("description", ""),
            server=srv_name,
            tool=None,
            reason=per_task_reason,
            source=source,
            replan_round=replan_round,
            global_step=global_step,
            explicit_error_type=explicit_error_type,
        )
        if not _entries_match_in_recent(history, entry):
            history.append(entry)


def _entries_match_in_recent(history: List[Dict[str, Any]], entry: Dict[str, Any], window: int = 4) -> bool:
    """Check the last `window` entries for a duplicate of `entry`."""
    if not history:
        return False
    for prev in history[-window:]:
        if isinstance(prev, dict) and _entries_match(prev, entry):
            return True
    return False


def format_failure_history_for_prompt(history: List[Any]) -> str:
    """
    Render structured failure_history grouped by task for LLM consumption.
    Falls back to JSON dump for legacy string entries.
    """
    if not history:
        return "(no failures recorded)"

    if any(not isinstance(e, dict) for e in history):
        return json.dumps(history, ensure_ascii=False, indent=2)

    by_task: Dict[str, List[Dict[str, Any]]] = {}
    orphans: List[Dict[str, Any]] = []
    order: List[str] = []
    for e in history:
        tid = e.get("task_id")
        if tid:
            if tid not in by_task:
                by_task[tid] = []
                order.append(tid)
            by_task[tid].append(e)
        else:
            orphans.append(e)

    lines: List[str] = []
    for tid in order:
        entries = by_task[tid]
        desc = entries[0].get("task_description") or "(no description)"
        lines.append(f"- Sub-task {tid}: {desc}")
        for e in entries:
            srv = e.get("server") or "?"
            tool = e.get("tool") or "?"
            etype = e.get("error_type") or "unknown"
            reason = e.get("reason") or ""
            tag = f"{srv}.{tool}" if tool != "?" else srv
            lines.append(
                f"    [step {e.get('global_step', 0)} replan {e.get('replan_round', 0)}] "
                f"{tag} → {etype}: {reason}"
            )
    if orphans:
        lines.append("- (no task context):")
        for e in orphans:
            etype = e.get("error_type") or "unknown"
            reason = e.get("reason") or ""
            lines.append(
                f"    [step {e.get('global_step', 0)} replan {e.get('replan_round', 0)}] "
                f"{etype}: {reason}"
            )
    return "\n".join(lines)


def latest_error_types(history: List[Any], n: int = 3) -> List[str]:
    """Return the error_type of the last n structured entries (newest last)."""
    out: List[str] = []
    for e in history[-n:]:
        if isinstance(e, dict):
            out.append(e.get("error_type") or "unknown")
    return out


# Keywords that indicate a transient failure — server should NOT be excluded.
_TRANSIENT_ERROR_SIGNALS = (
    "timeout", "timed out", "connection", "network", "temporarily",
    "rate limit", "too many requests", "service unavailable", "503", "502",
)


def _is_transient_failure(reason) -> bool:
    """Return True if the failure reason looks like a transient/infrastructure error."""
    reason_lower = str(reason).lower()
    return any(signal in reason_lower for signal in _TRANSIENT_ERROR_SIGNALS)


def record_failed_servers(state: Dict[str, Any]) -> None:
    """
    Ban the servers used by failed tasks so the Retrieval agent avoids them.

    Two-stage exclusion (Option A + B combined):
    - Transient failures (timeout, connection, rate-limit): NEVER exclude.
      The same server will be retried on the next replan.
    - Non-transient failures (wrong data, no tools, bad answer): exclude only
      after the 2nd failure with the same server (grace period for one retry).

    `server_failure_counts[task_id][server_name]` tracks how many non-transient
    failures each server has accumulated per task.
    """
    failure_reason = state.get("last_failure_reason", "")

    # Transient error — do not touch excluded_servers at all.
    if _is_transient_failure(failure_reason):
        logger.info("Transient failure detected ('%s') — server exclusion skipped.", failure_reason)
        return

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    if idx >= len(plan):
        return
    current_task_ids = plan[idx].get("tasks", [])
    finished = set(state.get("finished_task_ids", []))
    selected = state.get("selected_servers", {})
    excluded = state.setdefault("excluded_servers", {})
    failure_counts = state.setdefault("server_failure_counts", {})

    for task_id in current_task_ids:
        if task_id in finished:
            continue
        server_info = selected.get(task_id)
        if not server_info:
            continue
        server_name = (
            server_info.get("selected_server")
            if isinstance(server_info, dict)
            else server_info
        )
        # Skip sentinel values — "none"/empty means retrieval found no server at all.
        # Excluding "none" as if it were a real server corrupts the exclusion list.
        if not server_name or server_name.lower() in ("none", "null", ""):
            continue

        # Increment non-transient failure count for this server/task pair.
        task_counts = failure_counts.setdefault(task_id, {})
        task_counts[server_name] = task_counts.get(server_name, 0) + 1

        # Exclude only after 2nd non-transient failure (Option A grace period).
        if task_counts[server_name] >= 2:
            excluded.setdefault(task_id, [])
            if server_name not in excluded[task_id]:
                logger.info(
                    "Excluding server '%s' for task '%s' after %d non-transient failures.",
                    server_name, task_id, task_counts[server_name]
                )
                excluded[task_id].append(server_name)

def is_reasoning_step(state: Dict[str, Any]) -> bool:
    """
    Return True if every task in the current step is a reasoning-type task.

    Reasoning steps are handled directly by the Planner
    and bypass the Retrieval → Executor → Answer → Verifier pipeline entirely.
    """
    plan = state.get("plan", [])
    task_defs = state.get("task_definitions", {})
    idx = state.get("current_step_index", 0)
    if idx >= len(plan):
        return False
    task_ids = plan[idx].get("tasks", [])
    return bool(task_ids) and all(
        task_defs.get(tid, {}).get("task_type") == "reasoning"
        for tid in task_ids
    )


def print_plan(state: Dict[str, Any], title: str = "PLAN UPDATE") -> None:
    """
    Print a summary of the current plan to stdout.
    """
    plan = state.get("plan", [])
    task_defs = state.get("task_definitions", {})
    print(f"\n{title}:\n")
    if not plan:
        print("No plan\n")
        return
    for i, step in enumerate(plan):
        print(f"Step {i} (parallel={step.get('parallel', False)})")
        for task_id in step.get("tasks", []):
            desc = task_defs.get(task_id, {}).get("description", "N/A")
            print(f"   • {task_id}: {desc}")
        print()
    print("--------------------------------------------------\n")


def print_step_execution(state: Dict[str, Any]) -> None:
    """
    Print a header line when the Planner begins executing a step.
    """
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    if idx < len(plan):
        step = plan[idx]
        print(f"\nEXECUTING STEP {idx} (parallel={step.get('parallel', False)})")
        print(f"Tasks: {step.get('tasks', [])}\n")