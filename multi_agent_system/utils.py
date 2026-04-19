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
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

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

def all_steps_completed(state: Dict[str, Any]) -> bool:
    """
    Return True when the step index has advanced past the last plan step.
    """
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)
    return bool(plan) and current_idx >= len(plan)


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