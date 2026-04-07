import copy
import logging
from typing import Any, Dict

from .agents.planner import planner_node, handle_reasoning_step
from .agents.retrieval import retrieval_node
from .agents.executor import executor_node, initialize_executor
from .agents import executor as executor_module
from .agents.answer import answer_node
from .agents.verifier import verifier_node

logger = logging.getLogger(__name__)

# ---------------------------
# STATE MERGE RULES
# ---------------------------
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

DICT_MERGE_KEYS = {
    "completed_tasks_results",
}

LIST_EXTEND_KEYS = {
    "failure_history",
    "messages",
    "errors",
    "final_history",
    "finished_task_ids",
}

# ---------------------------
# DEBUG PRINTS
# ---------------------------
def print_plan(state, title="PLAN UPDATE"):
    plan = state.get("plan", [])
    task_defs = state.get("task_definitions", {})

    print(f"\n🧠 {title}:\n")

    if not plan:
        print("❌ No plan\n")
        return

    for i, step in enumerate(plan):
        print(f"🔹 Step {i} (parallel={step.get('parallel', False)})")

        for task_id in step.get("tasks", []):
            desc = task_defs.get(task_id, {}).get("description", "N/A")
            print(f"   • {task_id}: {desc}")

        print()

    print("--------------------------------------------------\n")


def print_step_execution(state):
    idx = state.get("current_step_index", 0)
    plan = state.get("plan", [])

    if idx < len(plan):
        step = plan[idx]
        print(f"\n🚀 EXECUTING STEP {idx} (parallel={step.get('parallel', False)})")
        print(f"Tasks: {step.get('tasks', [])}\n")


# ---------------------------
# STATE HELPERS
# ---------------------------
def _normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
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
    normalized.setdefault("retry_count", 0)
    normalized.setdefault("final_output", None)
    normalized.setdefault("latest_execution_results", {})
    normalized.setdefault("latest_verification_package", {})
    normalized.setdefault("final_history", [])
    normalized.setdefault("verification_status", "")
    normalized.setdefault("all_parts_found", False)
    normalized.setdefault("excluded_servers", {})

    return normalized


def _merge_state(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    if not updates:
        return base

    for key, value in updates.items():
        if value is None:
            continue

        if key in REPLACE_KEYS:
            base[key] = value
            continue

        if key in DICT_MERGE_KEYS and isinstance(value, dict):
            base.setdefault(key, {})
            base[key].update(value)
            continue

        if key in LIST_EXTEND_KEYS and isinstance(value, list):
            base.setdefault(key, [])
            base[key].extend(value)
            continue

        base[key] = value

    return base


def _commit_verified_results(state: Dict[str, Any]) -> None:
    final_history = state.get("final_history", [])
    if not final_history:
        return

    completed = state.setdefault("completed_tasks_results", {})
    finished_ids = state.setdefault("finished_task_ids", [])
    task_defs = state.get("task_definitions", {})

    for item in final_history:
        task_id = item.get("task_id")
        if not task_id:
            continue

        completed[task_id] = {
            "description": task_defs.get(task_id, {}).get("description", ""),
            "summary": item.get("summary", ""),
            "final_answer": item.get("final_answer", "")
        }

        if task_id not in finished_ids:
            finished_ids.append(task_id)


def _refresh_task_descriptions(state: Dict[str, Any]) -> None:
    """After a step passes, inject completed dependency results into pending task descriptions.
    This ensures downstream tasks receive concrete values (e.g. 'Adams Museum') instead of
    abstract references (e.g. 'the museum from task_1').
    """
    task_defs = state.get("task_definitions", {})
    completed = state.get("completed_tasks_results", {})

    for tdef in task_defs.values():
        deps = tdef.get("dependencies", [])
        context = [
            f"{dep}: {completed[dep]['final_answer']}"
            for dep in deps
            if dep in completed and completed[dep].get("final_answer")
        ]
        if context:
            base_desc = tdef.get("description", "")
            marker = "\n\nResults from previous tasks:\n"
            # Avoid appending duplicates on repeated calls
            if marker not in base_desc:
                tdef["description"] = base_desc + marker + "\n".join(context)


def _record_failed_servers(state: Dict[str, Any]) -> None:
    """Add the server used for each failed task to excluded_servers so retrieval avoids it."""
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    if idx >= len(plan):
        return

    current_task_ids = plan[idx].get("tasks", [])
    finished = set(state.get("finished_task_ids", []))
    selected = state.get("selected_servers", {})
    excluded = state.setdefault("excluded_servers", {})

    for task_id in current_task_ids:
        if task_id in finished:
            continue  # already succeeded, no need to exclude its server

        server_info = selected.get(task_id)
        if not server_info:
            continue

        if isinstance(server_info, dict):
            server_name = server_info.get("selected_server")
        else:
            server_name = server_info

        if server_name:
            excluded.setdefault(task_id, [])
            if server_name not in excluded[task_id]:
                excluded[task_id].append(server_name)


def _all_steps_completed(state: Dict[str, Any]) -> bool:
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)
    return bool(plan) and current_idx >= len(plan)


def _is_reasoning_step(state: Dict[str, Any]) -> bool:
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


async def _close_mcp_connections() -> None:
    try:
        if getattr(executor_module, "server_manager", None) is not None:
            await executor_module.server_manager.close_all_connections()
    except Exception as e:
        logger.warning(f"Error while closing MCP connections: {e}")
    finally:
        executor_module.initialized = False
        executor_module.server_manager = None


# ---------------------------
# MAIN GRAPH
# ---------------------------
async def run_graph(initial_state: Dict[str, Any], max_replans: int = 5) -> Dict[str, Any]:

    state = _normalize_state(initial_state)
    replans = 0

    await initialize_executor()

    try:
        # ---------------------------
        # INITIAL PLAN
        # ---------------------------
        planner_updates = planner_node(state)
        state = _merge_state(state, planner_updates)

        print_plan(state, "INITIAL PLAN")

        # ---------------------------
        # LOOP
        # ---------------------------
        while True:

            if state.get("final_output"):
                print("\n🏁 FINAL OUTPUT READY\n")
                return state

            plan = state.get("plan", [])
            if not plan:
                logger.warning("No plan available.")
                return state

            if _all_steps_completed(state):
                print("\n✅ ALL STEPS COMPLETED → FINAL SYNTHESIS\n")

                state["verification_status"] = "pass"
                state["all_parts_found"] = True

                planner_updates = planner_node(state)
                state = _merge_state(state, planner_updates)

                print("\n📦 FINAL PLAN OUTPUT\n")
                return state

            current_idx = state.get("current_step_index", 0)

            print_step_execution(state)

            # ---------------------------
            # REASONING STEP — Planner answers directly
            # ---------------------------
            if _is_reasoning_step(state):
                print(f"\n🧠 STEP {current_idx} IS REASONING → Planner handles directly\n")
                reasoning_updates = handle_reasoning_step(state)
                state = _merge_state(state, reasoning_updates)

                # Fix 5: commit reasoning result to completed_tasks_results for bookkeeping
                if state.get("final_output"):
                    plan = state.get("plan", [])
                    for tid in plan[current_idx].get("tasks", []):
                        state["completed_tasks_results"][tid] = {
                            "final_answer": state["final_output"],
                            "summary": "Resolved via planner reasoning."
                        }
                    state["current_step_index"] = current_idx + 1

                print("\n🏁 FINAL OUTPUT READY (via Planner reasoning)\n")
                return state

            # ---------------------------
            # EXECUTION PIPELINE
            # ---------------------------
            retrieval_updates = await retrieval_node(state)
            state = _merge_state(state, retrieval_updates)

            # Fix 7: short-circuit if retrieval already marked failure (e.g. missing inventory)
            if state.get("verification_status") != "fail":
                executor_updates = await executor_node(state)
                state = _merge_state(state, executor_updates)

                answer_updates = await answer_node(state)
                state = _merge_state(state, answer_updates)

                verifier_updates = await verifier_node(state)
                state = _merge_state(state, verifier_updates)
            else:
                print(f"❌ Retrieval failed for step {current_idx}, skipping pipeline\n")

            # ---------------------------
            # SUCCESS
            # ---------------------------
            if state.get("verification_status") == "pass":

                print(f"✅ STEP {current_idx} PASSED\n")

                replans = 0  # Fix 6: reset counter after each successful step
                _commit_verified_results(state)
                _refresh_task_descriptions(state)

                state["last_failure_reason"] = ""
                state["all_parts_found"] = True
                state["current_step_index"] = current_idx + 1

                continue

            # ---------------------------
            # FAILURE → REPLAN
            # ---------------------------
            replans += 1

            print(f"❌ STEP {current_idx} FAILED → REPLAN #{replans}\n")

            failure_reason = state.get("last_failure_reason", "")
            if failure_reason:
                print(f"Reason: {failure_reason}\n")
                state.setdefault("failure_history", []).append(failure_reason)

            if replans > max_replans:
                logger.warning("Max replans reached.")
                return state

            # Commit any partially approved tasks so the planner skips them on replan
            _commit_verified_results(state)
            # Record the failed server for each unfinished task so retrieval avoids it
            _record_failed_servers(state)

            state["current_step_index"] = 0
            state["verification_status"] = ""
            state["all_parts_found"] = False

            planner_updates = planner_node(state)
            state = _merge_state(state, planner_updates)

            print_plan(state, f"REPLAN #{replans}")

    finally:
        await _close_mcp_connections()