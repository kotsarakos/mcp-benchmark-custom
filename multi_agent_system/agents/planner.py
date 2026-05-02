import asyncio
import json
import logging
import re
from typing import final
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_PLANNING, TEMPERATURE, make_model_kwargs, INVENTORY_DIR
from ..prompts.agent_prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_REPLAN_PROMPT,
    PLANNER_FINAL_SYNTHESIS_PROMPT,
    PLANNER_UNANSWERABLE_SYNTHESIS_PROMPT,
    PLANNER_UNANSWERABILITY_CHECK_PROMPT,
    PLANNER_REASONING_STEP_PROMPT,
)
from ..token_tracker import token_tracker
from ..trace_recorder import get_recorder
from ..utils import (
    merge_state,
    commit_verified_results,
    refresh_task_descriptions,
    record_failed_servers,
    record_failure,
    format_failure_history_for_prompt,
    latest_error_types,
    is_reasoning_step,
    print_plan,
    print_step_execution,
    current_date_str,
)
from .retrieval import retrieval_node
from .executor import executor_node
from .answer import answer_node
from .verifier import verifier_node

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_FOR_PLANNING,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs=make_model_kwargs({"response_format": {"type": "json_object"}})
)

# Cached prompt chains and parser
_plan_chain = ChatPromptTemplate.from_template(PLANNER_SYSTEM_PROMPT) | llm
_replan_chain = ChatPromptTemplate.from_template(PLANNER_REPLAN_PROMPT) | llm
_reasoning_chain = ChatPromptTemplate.from_template(PLANNER_REASONING_STEP_PROMPT) | llm
_synthesis_chain = ChatPromptTemplate.from_template(PLANNER_FINAL_SYNTHESIS_PROMPT) | llm
_unanswerable_chain = ChatPromptTemplate.from_template(PLANNER_UNANSWERABLE_SYNTHESIS_PROMPT) | llm
_unanswerability_check_chain = ChatPromptTemplate.from_template(PLANNER_UNANSWERABILITY_CHECK_PROMPT) | llm
_json_parser = JsonOutputParser()

_ACCESS_DENIED_KEYWORDS = frozenset([
    "forbidden", "403", "401", "unauthorized", "access denied",
    "permission denied", "not permitted", "access restricted",
    "insufficient permissions", "no permission", "not authorized",
])

MAX_ACCESS_DENIED_REPLANS = 2

# Trigger an LLM-based unanswerability check after this many replans on the same step.
# Cheap enough (1 call) to run once when keyword detection missed the obstacle but
# the failure pattern suggests the query may be structurally unanswerable.
UNANSWERABILITY_CHECK_AFTER_REPLANS = 3

# Number of consecutive verifier "impossible" verdicts before we conclude the
# whole query cannot be answered.
MAX_IMPOSSIBLE_VERDICTS = 2

# Retry policy for handle_unanswerable_synthesis on timeout. Only TimeoutError
# triggers a retry — other exceptions go straight to the structured fallback.
UNANSWERABLE_SYNTHESIS_TIMEOUT = 60          # seconds per attempt
UNANSWERABLE_SYNTHESIS_MAX_RETRIES = 2       # 1 initial + 2 retries = 3 attempts max
UNANSWERABLE_SYNTHESIS_BACKOFF_BASE = 2.0    # seconds, doubled per retry


async def _check_unanswerability(state: dict) -> tuple[bool, str]:
    """
    Ask an LLM whether the query is structurally unanswerable given the failure
    history and available servers. Cheaper than spending more replan budget on a
    query that has no chance. Returns (should_stop, reason).
    Defaults to (False, "") on any error so we don't accidentally abort solvable queries.
    """
    try:
        rich = state.get("_rich_inventory") or []
        # Only send name + tool names; full descriptions blow up the prompt.
        slim = [{"name": s.get("name"), "tools": s.get("tools", [])} for s in rich]
        raw_response = await asyncio.wait_for(
            _unanswerability_check_chain.ainvoke({
                "original_query": state.get("input", ""),
                "available_servers": json.dumps(slim, ensure_ascii=False),
                "failure_history": format_failure_history_for_prompt(state.get("failure_history", [])),
                "partial_data": json.dumps(state.get("completed_tasks_results", {}), ensure_ascii=False),
                "current_date": current_date_str(),
            }),
            timeout=60
        )
        token_tracker.track("planner", raw_response)
        parsed = _parse_with_tracking(raw_response.content, state)
        decision = (parsed.get("decision") or "").lower()
        reason = parsed.get("reason", "")
        return (decision == "stop", reason)
    except Exception as e:
        logger.warning("Unanswerability check failed (defaulting to continue): %s", e)
        return (False, "")


def _is_access_denied_failure(state: dict) -> bool:
    """Return True if the current failure looks like a server access restriction."""
    # Fast path: structured failure_history already classifies error_type.
    history = state.get("failure_history", [])
    for e in reversed(history):
        if isinstance(e, dict) and e.get("global_step") == state.get("_global_step", 0):
            if e.get("error_type") == "access_denied":
                return True
            break
    # Fallback: scan free-text reason + executor results.
    sources = [state.get("last_failure_reason", "")]
    for v in state.get("latest_execution_results", {}).values():
        if isinstance(v, str):
            sources.append(v)
    combined = " ".join(sources).lower()
    return any(kw in combined for kw in _ACCESS_DENIED_KEYWORDS)


def _parse_with_tracking(raw_content: str, state: dict):
    """
    Parse Planner JSON and report success/failure to the trace recorder
    (if tracing is enabled). Reraises so existing error handling still works.
    """
    recorder = get_recorder(state)
    try:
        parsed = _json_parser.parse(raw_content)
    except Exception:
        if recorder is not None:
            recorder.record_plan_parse(success=False)
        raise
    if recorder is not None:
        recorder.record_plan_parse(success=True)
    return parsed

# Load server inventory once at module startup
try:
    with open(INVENTORY_DIR / "inventory_summary.json", "r", encoding="utf-8") as _f:
        _available_servers_str = json.dumps(
            json.load(_f).get("available_servers", []), ensure_ascii=False
        )
except Exception as _e:
    raise RuntimeError(
        f"Planner: could not load the names of MCP servers — system cannot start. Reason: {_e}"
    ) from _e


def _remap_task_ids(plan_updates: dict, existing_completed: dict) -> dict:
    """
    Rename new task IDs that collide with already-completed ones.
    - Failed tasks keep their original ID. 
    - Only verified results get a new ID.
    Dependency references are updated to match any renames.
    """
    if not existing_completed:
        return plan_updates

    old_task_defs = plan_updates.get("task_definitions", {})
    old_plan = plan_updates.get("plan", [])

    # Find highest existing numeric task ID for safe new names.
    max_num = 0
    for tid in existing_completed:
        m = re.match(r"task_(\d+)$", tid)
        if m:
            max_num = max(max_num, int(m.group(1)))

    # Only rename IDs that collide with completed results.
    id_map = {}
    for old_id in old_task_defs:
        if old_id in existing_completed:
            max_num += 1
            id_map[old_id] = f"task_{max_num}"

    if not id_map:
        return plan_updates

    # Apply renames to task definitions and their dependency lists.
    new_task_defs = {}
    for old_id, tdef in old_task_defs.items():
        new_tdef = dict(tdef)
        new_tdef["dependencies"] = [id_map.get(d, d) for d in tdef.get("dependencies", [])]
        new_task_defs[id_map.get(old_id, old_id)] = new_tdef

    # Apply renames to plan step task lists.
    new_plan = [
        {**step, "tasks": [id_map.get(t, t) for t in step.get("tasks", [])]}
        for step in old_plan
    ]

    return {**plan_updates, "task_definitions": new_task_defs, "plan": new_plan}


async def _generate_plan(state: dict, is_replan: bool = False) -> dict:
    """
    Ask the Planner for the single next step to execute.
    When is_replan is True, uses the replan prompt with failure context.
    Else uses the normal prompt for the next step.
    Returns either the next step or {"_done": True} when all data is collected.
    """
    user_input = state.get("input", "")
    completed_results = state.get("completed_tasks_results", {})

    today = current_date_str()
    if is_replan:
        failures = state.get("failure_history", [])
        chain = _replan_chain
        chain_inputs = {
            "input": user_input,
            "completed_tasks": json.dumps(completed_results),
            "last_failure_reason": state.get("last_failure_reason", ""),
            "failure_history": format_failure_history_for_prompt(failures),
            "available_servers": _available_servers_str,
            "current_date": today,
        }
    else:
        chain = _plan_chain
        chain_inputs = {
            "input": user_input,
            "completed_tasks": json.dumps(completed_results),
            "available_servers": _available_servers_str,
            "current_date": today,
        }

    try:
        raw_response = await asyncio.wait_for(
            chain.ainvoke(chain_inputs),
            timeout=180
        )
        token_tracker.track("planner", raw_response)

        # Parse the raw response with tracking to capture parse success/failure in the trace recorder.
        structured = _parse_with_tracking(raw_response.content, state)

        if structured.get("done"):
            return {"_done": True}

        step = structured.get("step", {})
        task_definitions = structured.get("task_definitions", {})

        # Filter out tasks that are already completed by ID or by description.
        already_done = set(completed_results.keys())

        # Filter out tasks whose descriptions match already completed ones, 
        # to allow for more flexible replanning that doesn't get stuck on minor ID collisions.
        completed_descs = {
            v.get("description", "").strip().lower()
            for v in completed_results.values()
            if v.get("description")
        }

        task_definitions = {
            tid: tdef for tid, tdef in task_definitions.items()
            if tid not in already_done
            and tdef.get("description", "").strip().lower() not in completed_descs
        }

        # Filter the step's task list to match the remaining definitions. If none remain, we're done with this step and can move on.
        filtered_tasks = [t for t in step.get("tasks", []) if t in task_definitions]

        if not filtered_tasks:
            return {"_done": True}
        step = {**step, "tasks": filtered_tasks}

        # Remove tasks whose dependencies are in the same step (they can't see
        # each other's results). They will be scheduled in a future step instead.
        step_task_set = set(filtered_tasks)
        offending = set()
        for tid in filtered_tasks:
            for dep in task_definitions.get(tid, {}).get("dependencies", []):
                if dep in step_task_set and dep not in already_done:
                    logger.warning(
                        "Task '%s' depends on '%s' which is in the same step -- removing from this step.",
                        tid, dep
                    )
                    offending.add(tid)
        if offending:
            filtered_tasks = [t for t in filtered_tasks if t not in offending]
            task_definitions = {
                tid: tdef for tid, tdef in task_definitions.items()
                if tid not in offending
            }
            step = {**step, "tasks": filtered_tasks}
            if not filtered_tasks:
                return {"_done": True}

        # Default missing task_type to "tool".
        for tid, tdef in task_definitions.items():
            if "task_type" not in tdef:
                logger.warning("Task '%s' missing task_type, defaulting to 'tool'", tid)
                tdef["task_type"] = "tool"

        return {
            "plan": [step],
            "task_definitions": task_definitions,
            "current_step_index": 0,
            "last_failure_reason": "",
        }

    except Exception as e:
        logger.error("Planner failed to generate plan: %s", e)
        # Signal failure so planner_node engages the replan budget.
        return {
            "_plan_failed": True,
            "last_failure_reason": f"Planner failed to generate JSON: {str(e)}",
        }


async def _handle_plan_failure(state: dict, plan_updates: dict) -> dict:
    """
    Handle a '_plan_failed' signal from _generate_plan.
    - Charges 1 to the replan budget.
    - If budget remains: marks verification_status="fail" so the next iteration replans.
    - If budget exhausted: falls back to final synthesis on collected data.
    """

    state["last_failure_reason"] = plan_updates.get("last_failure_reason", "Planner JSON failure")
    state["_replans"] = state.get("_replans", 0) + 1
    max_replans = state.get("_max_replans", 5)

    if state["_replans"] > max_replans:
        logger.error(
            "Planner JSON kept failing — replan budget exhausted (%d/%d). "
            "Falling back to final synthesis on collected data.",
            state["_replans"], max_replans
        )
        result = await handle_final_synthesis(state)
        return merge_state(state, result)

    logger.warning(
        "Planner JSON failed (%s) — charging replan budget (%d/%d) and trying again.",
        state["last_failure_reason"], state["_replans"], max_replans
    )

    # Mark fail so the next planner_node iteration takes the fail branch and replans.
    state["verification_status"] = "fail"
    state["plan"] = []
    state["current_step_index"] = 0
    return state


async def handle_final_synthesis(state: dict) -> dict:
    """
    Triggered when all steps are complete and verified.
    - Inputs: original query, collected data from all tasks
    - Output: final synthesized answer to the user's query
    """

    summary_context = json.dumps(state.get("completed_tasks_results", {}), indent=2)
    try:
        raw_response = await asyncio.wait_for(
            _synthesis_chain.ainvoke({
                "original_query": state.get("input"),
                "collected_data": summary_context,
                "current_date": current_date_str(),
            }),
            timeout=60
        )
        token_tracker.track("planner", raw_response)
        response = _parse_with_tracking(raw_response.content, state)
        return {
            "final_output": response.get("answer"),
            "messages": [{"role": "assistant", "content": "Final synthesis generated."}]
        }
    except Exception as e:
        logger.error("Final synthesis failed: %s", e)
        # Always set final_output to prevent an infinite loop in run_graph.
        collected = state.get("completed_tasks_results", {})
        fallback = "; ".join(
            v.get("final_answer", "") for v in collected.values() if v.get("final_answer")
        ) or "Final synthesis failed and no task answers were collected."
        return {"final_output": fallback}


async def handle_unanswerable_synthesis(state: dict) -> dict:
    """
    Called when the system detects it cannot answer the query (e.g. repeated access denials).
    Uses a dedicated prompt that honestly explains the failure reasons to the user.

    On asyncio.TimeoutError the LLM call is retried up to
    UNANSWERABLE_SYNTHESIS_MAX_RETRIES times with exponential backoff. Other
    exceptions (JSON parse, network) skip retries and go to the structured fallback.
    """
    history = list(state.get("failure_history", []))
    partial = state.get("completed_tasks_results", {})
    failure_reasons_str = format_failure_history_for_prompt(history)

    chain_inputs = {
        "original_query": state.get("input", ""),
        "failure_reasons": failure_reasons_str,
        "partial_data": json.dumps(partial, ensure_ascii=False, indent=2) if partial else "None",
        "current_date": current_date_str(),
    }

    last_exc: Exception | None = None
    for attempt in range(UNANSWERABLE_SYNTHESIS_MAX_RETRIES + 1):
        try:
            raw_response = await asyncio.wait_for(
                _unanswerable_chain.ainvoke(chain_inputs),
                timeout=UNANSWERABLE_SYNTHESIS_TIMEOUT,
            )
            token_tracker.track("planner", raw_response)
            response = _parse_with_tracking(raw_response.content, state)
            return {
                "final_output": response.get("answer", ""),
                "messages": [{"role": "assistant", "content": "Unanswerable synthesis generated."}]
            }
        except asyncio.TimeoutError as e:
            last_exc = e
            if attempt < UNANSWERABLE_SYNTHESIS_MAX_RETRIES:
                backoff = UNANSWERABLE_SYNTHESIS_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Unanswerable synthesis timed out (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1, UNANSWERABLE_SYNTHESIS_MAX_RETRIES + 1, backoff,
                )
                await asyncio.sleep(backoff)
                continue
            logger.error(
                "Unanswerable synthesis timed out after %d attempts — falling back.",
                UNANSWERABLE_SYNTHESIS_MAX_RETRIES + 1,
            )
            break
        except Exception as e:
            last_exc = e
            logger.error("Unanswerable synthesis failed (non-timeout): %s", e)
            break

    # Last-resort fallback: surface the most recent reason per failed sub-task.
    bullets = []
    for entry in history:
        if isinstance(entry, dict):
            tid = entry.get("task_id") or "(no task)"
            etype = entry.get("error_type") or "unknown"
            reason = entry.get("reason") or ""
            bullets.append(f"- {tid} ({etype}): {reason}")
        else:
            bullets.append(f"- {entry}")
    reasons_str = "\n".join(bullets) or f"Unknown error ({last_exc})"
    return {
        "final_output": (
            "The system was unable to answer this query.\n"
            f"Reasons:\n{reasons_str}"
        )
    }


async def handle_reasoning_step(state: dict) -> dict:
    """
    Planner reasons over collected data and produces a reasoning step.
    - Inputs: original query, collected data from completed tasks, current reasoning tasks
    - Output: reasoning step
    """

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    task_defs = state.get("task_definitions", {})
    completed_results = state.get("completed_tasks_results", {})

    reasoning_tasks = {}
    if idx < len(plan):
        for tid in plan[idx].get("tasks", []):
            if tid in task_defs:
                reasoning_tasks[tid] = task_defs[tid].get("description", "")

    try:
        raw_response = await asyncio.wait_for(
            _reasoning_chain.ainvoke({
                "original_query": state.get("input", ""),
                "collected_data": json.dumps(completed_results, indent=2, ensure_ascii=False),
                "reasoning_tasks": json.dumps(reasoning_tasks, indent=2, ensure_ascii=False),
                "current_date": current_date_str(),
            }),
            timeout=60
        )
        token_tracker.track("planner", raw_response)
        response = _parse_with_tracking(raw_response.content, state)
        # Private key — avoids terminating run_graph mid-plan.
        # The returned reasoning answer will be committed to completed_tasks_results
        # under the original task IDs, so it can be seen by any remaining tasks in the plan that depend on them.
        return {
            "_reasoning_answer": response.get("final_answer"),
            "messages": [{"role": "assistant", "content": "Planner resolved reasoning step."}]
        }
    except Exception as e:
        logger.error("Planner reasoning step failed: %s", e)
        # Catastrophic failure: hard-stop with partial data.
        collected = state.get("completed_tasks_results", {})
        fallback = "; ".join(
            v.get("final_answer", "") for v in collected.values() if v.get("final_answer")
        ) or "Reasoning step failed and no task answers were collected."
        return {"final_output": fallback}


async def _run_pipeline(state: dict) -> dict:
    """
    Run the pipeline for the current step.
    Short-circuits early if any stage fails.
    """
    current_idx = state.get("current_step_index", 0)
    print_step_execution(state)

    # Reasoning tasks bypass the tool pipeline entirely.
    if is_reasoning_step(state):
        logger.info("Step %d is a reasoning step -- Planner handles directly.", current_idx)
        reasoning_updates = await handle_reasoning_step(state)
        state = merge_state(state, reasoning_updates)

        # final_output set => catastrophic failure, hard stop.
        if state.get("final_output"):
            logger.warning("Reasoning step produced final_output via failure fallback.")
            return state

        # Commit reasoning answer and clear plan so planner_node plans the next step.
        answer = state.pop("_reasoning_answer", "") or ""
        plan = state.get("plan", [])
        if answer and current_idx < len(plan):
            for tid in plan[current_idx].get("tasks", []):
                state.setdefault("completed_tasks_results", {})[tid] = {
                    "final_answer": answer,
                    "summary": "Resolved via planner reasoning.",
                }
        state["plan"] = []
        state["current_step_index"] = 0
        state["_global_step"] = state.get("_global_step", 0) + 1
        state["verification_status"] = ""
        state["_replans"] = 0
        return state

    # Stage 1: Retrieval -- select MCP servers for each task.
    logger.info("[RETRIEVAL] Selecting MCP server...")
    retrieval_updates = await retrieval_node(state)
    state = merge_state(state, retrieval_updates)

    # Stage 2: Executor -- call the selected MCP tools.
    logger.info("[EXECUTOR] Calling MCP tools...")
    executor_updates = await executor_node(state)
    state = merge_state(state, executor_updates)

    if executor_updates.get("errors"):
        logger.warning("Executor returned errors -- skipping answer/verifier.")
        state["verification_status"] = "fail"
        state["last_failure_reason"] = "; ".join(executor_updates["errors"])
        return state

    # Stage 3: Answer -- structure raw results into clean answers.
    logger.info("[ANSWER] Structuring results...")
    answer_updates = await answer_node(state)
    state = merge_state(state, answer_updates)

    if answer_updates.get("errors"):
        logger.warning("Answer agent returned errors -- skipping verifier.")
        state["verification_status"] = "fail"
        state["last_failure_reason"] = "; ".join(answer_updates["errors"])
        return state

    # Stage 4: Verifier -- check answer quality and decide pass/fail.
    logger.info("[VERIFIER] Checking answers...")
    verifier_updates = await verifier_node(state)
    state = merge_state(state, verifier_updates)

    return state


async def planner_node(state: dict) -> dict:
    """
    Main entry point for the planner. Called once per iteration of run_graph.
    Decision tree:
      - No plan yet     -> generate first step, run pipeline
      - Step passed     -> commit results, generate next step (or synthesize)
      - Step impossible -> commit partial results, generate next step (or synthesize)
      - Step failed     -> replan same step, run pipeline
    """

    verification_status = state.get("verification_status", "")
    plan = state.get("plan", [])
    max_replans = state.get("_max_replans", 5)

    # No plan yet: generate and run the first step
    if not plan:
        state["_global_step"] = 0
        logger.info("[PLANNER] Planning first step...")
        plan_updates = await _generate_plan(state)
        if plan_updates.get("_plan_failed"):
            return await _handle_plan_failure(state, plan_updates)
        if plan_updates.get("_done"):
            logger.info("[PLANNER] Nothing to do -- FINAL SYNTHESIS")
            result = await handle_final_synthesis(state)
            return merge_state(state, result)
        plan_updates = _remap_task_ids(plan_updates, state.get("completed_tasks_results", {}))
        state = merge_state(state, plan_updates)
        print_plan(state, f"STEP {state['_global_step']}")
        return await _run_pipeline(state)

    # Step passed: commit results, plan next step
    if verification_status == "pass":
        global_step = state.get("_global_step", 0)
        logger.info("STEP %d PASSED", global_step)
        state["_replans"] = 0
        state["_access_denied_count"] = 0
        state["_impossible_count"] = 0
        commit_verified_results(state)

        state["last_failure_reason"] = ""
        state["verification_status"] = ""
        state["plan"] = []
        state["current_step_index"] = 0
        state["_global_step"] = global_step + 1

        logger.info("[PLANNER] Planning next step...")
        plan_updates = await _generate_plan(state)
        if plan_updates.get("_plan_failed"):
            return await _handle_plan_failure(state, plan_updates)
        if plan_updates.get("_done"):
            logger.info("[PLANNER] All data collected -- FINAL SYNTHESIS")
            result = await handle_final_synthesis(state)
            return merge_state(state, result)
        plan_updates = _remap_task_ids(plan_updates, state.get("completed_tasks_results", {}))
        state = merge_state(state, plan_updates)
        refresh_task_descriptions(state)
        print_plan(state, f"STEP {state['_global_step']}")
        return await _run_pipeline(state)

    # Step impossible: commit partial results, plan next step
    if verification_status == "impossible":
        global_step = state.get("_global_step", 0)
        logger.info("STEP %d IMPOSSIBLE -- committing partial results", global_step)
        state["_access_denied_count"] = 0

        # Record the impossible verdict per current-step task so the unanswerable
        # synthesis can map "which sub-task was declared impossible" to the reason.
        if state.get("last_failure_reason"):
            record_failure(state, source="verifier", explicit_error_type="impossible")

        # Track repeated "impossible" verdicts. The verifier itself is telling us
        # this can't be done — if it says so multiple times for the same query,
        # don't keep planning new steps that will hit the same wall.
        state["_impossible_count"] = state.get("_impossible_count", 0) + 1
        if state["_impossible_count"] >= MAX_IMPOSSIBLE_VERDICTS:
            logger.error(
                "Verifier returned 'impossible' %d times — stopping and synthesizing explanation.",
                state["_impossible_count"],
            )
            # Commit whatever the impossible step produced so it's available to the synthesis.
            package = state.get("latest_verification_package", {})
            for task in package.get("tasks_analysis", []):
                t_id = task.get("task_id")
                if t_id:
                    state.setdefault("completed_tasks_results", {})[t_id] = {
                        "final_answer": task.get("final_answer", ""),
                        "summary": task.get("summary", ""),
                    }
            result = await handle_unanswerable_synthesis(state)
            return merge_state(state, result)
        package = state.get("latest_verification_package", {})
        for task in package.get("tasks_analysis", []):
            t_id = task.get("task_id")
            if t_id:
                state.setdefault("completed_tasks_results", {})[t_id] = {
                    "final_answer": task.get("final_answer", ""),
                    "summary": task.get("summary", "")
                }
        state["verification_status"] = ""
        state["plan"] = []
        state["current_step_index"] = 0
        state["_global_step"] = global_step + 1

        logger.info("[PLANNER] Planning next step after impossible...")
        plan_updates = await _generate_plan(state)
        if plan_updates.get("_plan_failed"):
            return await _handle_plan_failure(state, plan_updates)
        if plan_updates.get("_done"):
            logger.info("[PLANNER] All data collected -- FINAL SYNTHESIS")
            result = await handle_final_synthesis(state)
            return merge_state(state, result)
        plan_updates = _remap_task_ids(plan_updates, state.get("completed_tasks_results", {}))
        state = merge_state(state, plan_updates)
        refresh_task_descriptions(state)
        print_plan(state, f"STEP {state['_global_step']} (after impossible)")
        return await _run_pipeline(state)

    # Step failed or error: replan the same step
    if verification_status in ("fail", "error"):
        state["_replans"] = state.get("_replans", 0) + 1
        replans = state["_replans"]
        global_step = state.get("_global_step", 0)
        logger.warning("STEP %d FAILED -- REPLAN #%d", global_step, replans)

        failure_reason = state.get("last_failure_reason", "")
        if failure_reason:
            logger.warning("Failure reason: %s", failure_reason)
            record_failure(state, source="verifier")
            logger.debug(
                "Recent error_types: %s",
                latest_error_types(state.get("failure_history", []), n=4),
            )

        # Commit any partially-approved tasks (e.g. task_1 PASS, task_2 FAIL)
        commit_verified_results(state)

        # Early exit when the server repeatedly denies access — replanning with
        # different sub-tasks won't help, so synthesize an explanation instead.
        # Runs BEFORE the max-replans guard so a 2nd access failure on a low
        # max_replans budget still routes to the unanswerable explanation.
        if _is_access_denied_failure(state):
            state["_access_denied_count"] = state.get("_access_denied_count", 0) + 1
            logger.warning(
                "Access-denied failure detected (count: %d/%d).",
                state["_access_denied_count"], MAX_ACCESS_DENIED_REPLANS,
            )
            if state["_access_denied_count"] >= MAX_ACCESS_DENIED_REPLANS:
                logger.error(
                    "Repeated access-denied failures — stopping replans and synthesizing explanation."
                )
                result = await handle_unanswerable_synthesis(state)
                return merge_state(state, result)

        # General-purpose unanswerability check: after several replans without
        # progress, ask an LLM whether the query is structurally unanswerable
        # (capability gap, fictional data, contradictory request, etc.). Runs
        # ONCE per query to bound cost — guarded by `_unanswerability_checked`.
        if (
            replans >= UNANSWERABILITY_CHECK_AFTER_REPLANS
            and not state.get("_unanswerability_checked")
        ):
            state["_unanswerability_checked"] = True
            should_stop, reason = await _check_unanswerability(state)
            if should_stop:
                logger.error("Unanswerability check voted STOP: %s", reason)
                # Persist the LLM's verdict as a structured entry so the synthesis
                # prompt sees it alongside per-task failures.
                state.setdefault("failure_history", []).append({
                    "task_id": None,
                    "task_description": None,
                    "server": None,
                    "tool": None,
                    "error_type": "impossible",
                    "reason": f"Unanswerability check: {reason}",
                    "source": "unanswerability_check",
                    "replan_round": int(state.get("_replans", 0) or 0),
                    "global_step": int(state.get("_global_step", 0) or 0),
                })
                result = await handle_unanswerable_synthesis(state)
                return merge_state(state, result)
            else:
                logger.info("Unanswerability check voted CONTINUE — proceeding with replan.")

        if replans > max_replans:
            logger.error("MAX REPLANS (%d) REACHED -- stopping.", max_replans)
            # Try to synthesize from whatever has been collected so far
            collected = state.get("completed_tasks_results", {})
            if collected:
                logger.info("Attempting partial synthesis from %d completed tasks.", len(collected))
                try:
                    result = await handle_final_synthesis(state)
                    partial = result.get("final_output", "")
                    if partial:
                        return merge_state(state, {"final_output": partial})
                except Exception as e:
                    logger.warning("Partial synthesis failed: %s", e)
            # Last resort: concatenate whatever answers exist
            fallback = "; ".join(
                v.get("final_answer", "") for v in collected.values() if v.get("final_answer")
            ) or f"System stopped after {max_replans} failed replans. Last reason: {failure_reason}"
            return merge_state(state, {"final_output": fallback})

        record_failed_servers(state)
        state["plan"] = []
        state["current_step_index"] = 0
        state["verification_status"] = ""
        state["all_parts_found"] = False

        logger.info("[PLANNER] Replanning current step...")
        plan_updates = await _generate_plan(state, is_replan=True)
        if plan_updates.get("_plan_failed"):
            # Already inside the fail branch — bumping _replans again would
            # double-charge. Mark as fail and return; next iteration replans.
            return merge_state(state, {
                "verification_status": "fail",
                "last_failure_reason": plan_updates.get("last_failure_reason", "Replan JSON failure"),
            })
        if plan_updates.get("_done"):
            result = await handle_final_synthesis(state)
            return merge_state(state, result)
        plan_updates = _remap_task_ids(plan_updates, state.get("completed_tasks_results", {}))
        state = merge_state(state, plan_updates)
        print_plan(state, f"STEP {state.get('_global_step', 1)} -- REPLAN #{replans}")
        return await _run_pipeline(state)

    # Defensive fallback: pipeline returned without verification_status -> treat as fail.
    logger.warning("[PLANNER] Pipeline returned without verification_status — treating as fail.")
    return merge_state(state, {
        "verification_status": "fail",
        "last_failure_reason": "Pipeline returned without verification status (silent stage failure)",
    })
