import asyncio
import json
import logging
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_PLANNING, TEMPERATURE, INVENTORY_DIR
from ..prompts.agent_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_FINAL_SYNTHESIS_PROMPT, PLANNER_REASONING_STEP_PROMPT
from ..token_tracker import token_tracker
from ..utils import (
    merge_state,
    commit_verified_results,
    refresh_task_descriptions,
    record_failed_servers,
    all_steps_completed,
    is_reasoning_step,
    print_plan,
    print_step_execution,
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
    model_kwargs={"response_format": {"type": "json_object"}}
)

def _remap_task_ids(plan_updates: dict, existing_completed: dict) -> dict:
    """
    Shift all task IDs in a freshly generated plan so they never collide with
    already-completed task IDs from a previous plan cycle.

    The LLM always starts numbering from task_1. After a replan this would
    overwrite entries already stored in completed_tasks_results. This function
    finds the highest existing task number and renumbers every new task above it,
    also updating dependency references inside the same plan.

    Example: existing tasks task_1..task_3 → new plan task_1,task_2 become task_4,task_5.
    """
    max_num = 0
    for tid in existing_completed:
        m = re.match(r"task_(\d+)$", tid)
        if m:
            max_num = max(max_num, int(m.group(1)))

    if max_num == 0:
        # No completed tasks yet — nothing to remap.
        return plan_updates

    old_task_defs = plan_updates.get("task_definitions", {})
    old_plan = plan_updates.get("plan", [])

    # Stable ordering so the mapping is deterministic.
    old_ids = sorted(old_task_defs.keys())
    id_map = {old_id: f"task_{max_num + i + 1}" for i, old_id in enumerate(old_ids)}

    # Rename keys and remap dependency lists inside task_definitions.
    new_task_defs = {}
    for old_id, tdef in old_task_defs.items():
        new_tdef = dict(tdef)
        new_tdef["dependencies"] = [id_map.get(d, d) for d in tdef.get("dependencies", [])]
        new_task_defs[id_map[old_id]] = new_tdef

    # Update task lists inside each plan step.
    new_plan = [
        {**step, "tasks": [id_map.get(t, t) for t in step.get("tasks", [])]}
        for step in old_plan
    ]

    return {**plan_updates, "task_definitions": new_task_defs, "plan": new_plan}


async def _generate_plan(state: dict) -> dict:
    """
    Call the LLM to create or update the execution plan.
    - Inputs: user query, completed tasks, failure history, available servers
    - Output: structured plan with steps, tasks, dependencies, and task types
    """
    user_input = state.get("input", "")
    completed_results = state.get("completed_tasks_results", {})
    failures = state.get("failure_history", [])
    failures_json = json.dumps(failures, ensure_ascii=False) if failures else "[]"
    last_failure_reason = state.get("last_failure_reason", "")

    try:
        inventory_path = INVENTORY_DIR / "inventory_summary.json"
        with open(inventory_path, "r", encoding="utf-8") as f:
            available_servers = json.load(f).get("available_servers", [])
        available_servers_str = json.dumps(available_servers, ensure_ascii=False)
    except Exception as e:
        print(f"Planner: could not load inventory ({e}), continuing without it")
        available_servers_str = "[]"

    prompt = ChatPromptTemplate.from_template(PLANNER_SYSTEM_PROMPT)
    chain = prompt | llm

    try:
        raw_response = await asyncio.wait_for(
            chain.ainvoke({
                "input": user_input,
                "completed_tasks": json.dumps(completed_results),
                "failure_history": failures_json,
                "last_failure_reason": last_failure_reason,
                "available_servers": available_servers_str
            }),
            timeout=60
        )
        token_tracker.track("planner", raw_response)
        structured_plan = JsonOutputParser().parse(raw_response.content)

        task_definitions = structured_plan.get("task_definitions", {})

        # Ensure every task has task_type
        for tid, tdef in task_definitions.items():
            if "task_type" not in tdef:
                print(f"Warning: task '{tid}' missing task_type, defaulting to 'tool'")
                tdef["task_type"] = "tool"

        # Inject completed dependency results into descriptions
        for tid, tdef in task_definitions.items():
            deps = tdef.get("dependencies", [])
            context = [
                f"{dep}: {completed_results[dep]['final_answer']}"
                for dep in deps
                if dep in completed_results and completed_results[dep].get("final_answer")
            ]
            if context:
                tdef["description"] += "\n\nResults from previous tasks:\n" + "\n".join(context)

        # Update the State
        return {
            "plan": structured_plan.get("plan", []),
            "task_definitions": task_definitions,
            "current_step_index": 0,
            "last_failure_reason": "",
            "messages": [{"role": "assistant", "content": "Strategic execution plan updated."}]
        }

    except Exception as e:
        print(f"Planner Error: {e}")
        return {"last_failure_reason": f"Planner failed to generate JSON: {str(e)}"}


async def handle_final_synthesis(state: dict) -> dict:
    """
    Triggered when all steps are complete and verified.
    - Inputs: original query, collected data from all tasks
    - Output: final synthesized answer to the user's query"""
    prompt = ChatPromptTemplate.from_template(PLANNER_FINAL_SYNTHESIS_PROMPT)
    chain = prompt | llm
    summary_context = json.dumps(state.get("completed_tasks_results", {}), indent=2)
    try:
        raw_response = await asyncio.wait_for(
            chain.ainvoke({
                "original_query": state.get("input"),
                "collected_data": summary_context
            }),
            timeout=60
        )
        token_tracker.track("planner", raw_response)
        response = JsonOutputParser().parse(raw_response.content)
        return {
            "final_output": response.get("answer"),
            "messages": [{"role": "assistant", "content": "Final synthesis generated."}]
        }
    except Exception as e:
        print(f"Final Synthesis Error: {e}")
        return {"last_failure_reason": f"Final synthesis failed: {str(e)}"}


async def handle_reasoning_step(state: dict) -> dict:
    """
    Planner reasons over collected data and produces a direct answer.
    - Inputs: original query, collected data from completed tasks, current reasoning tasks
    - Output: final answer for the reasoning step
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

    prompt = ChatPromptTemplate.from_template(PLANNER_REASONING_STEP_PROMPT)
    chain = prompt | llm

    try:
        raw_response = await asyncio.wait_for(
            chain.ainvoke({
                "original_query": state.get("input", ""),
                "collected_data": json.dumps(completed_results, indent=2, ensure_ascii=False),
                "reasoning_tasks": json.dumps(reasoning_tasks, indent=2, ensure_ascii=False),
            }),
            timeout=60
        )
        token_tracker.track("planner", raw_response)
        response = JsonOutputParser().parse(raw_response.content)
        return {
            "final_output": response.get("final_answer"),
            "messages": [{"role": "assistant", "content": "Planner resolved reasoning step."}]
        }
    except Exception as e:
        print(f"Planner Reasoning Error: {e}")
        return {"last_failure_reason": f"Planner reasoning failed: {str(e)}"}


async def _run_pipeline(state: dict) -> dict:
    """
    Execute the fixed retrieval → executor → answer → verifier pipeline for the current step.
    """
    current_idx = state.get("current_step_index", 0)
    print_step_execution(state)

    # Reasoning step — planner handles directly, no tool calls needed
    if is_reasoning_step(state):
        print(f"\n🧠 STEP {current_idx} IS REASONING → Planner handles directly\n")
        reasoning_updates = await handle_reasoning_step(state)
        state = merge_state(state, reasoning_updates)

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

    # Fixed pipeline
    print("🔍 [RETRIEVAL] Selecting MCP server...")
    retrieval_updates = await retrieval_node(state)
    state = merge_state(state, retrieval_updates)

    if state.get("verification_status") != "fail":
        print("⚙️  [EXECUTOR] Calling MCP tool...")
        executor_updates = await executor_node(state)
        state = merge_state(state, executor_updates)

        print("📝 [ANSWER] Structuring results...")
        answer_updates = await answer_node(state)
        state = merge_state(state, answer_updates)

        print("🔎 [VERIFIER] Checking answers...")
        verifier_updates = await verifier_node(state)
        state = merge_state(state, verifier_updates)
    else:
        print(f"❌ [RETRIEVAL] Failed for step {current_idx}, skipping pipeline\n")

    return state


async def planner_node(state: dict) -> dict:
    """
    The Planner is the leader of the multi-agent system.
    Each call makes one high-level decision:
      - No plan yet       → generate initial plan → run pipeline
      - Step passed       → advance step → run pipeline (or synthesize if done)
      - All steps done    → final synthesis
      - Step failed       → replan → run pipeline
    """
    verification_status = state.get("verification_status", "")
    plan = state.get("plan", [])
    max_replans = state.get("_max_replans", 5)

    # Step 1: Check if all steps completed
    if all_steps_completed(state):
        print("\n✅ ALL STEPS COMPLETED → FINAL SYNTHESIS\n")
        result = await handle_final_synthesis(state)
        return merge_state(state, result)
 
    # Step 2: Impossible — commit best-effort results and continue to next step.
    # A step is impossible when the verifier determines the task can never be
    # completed (e.g. "when did Trump die?" — he is alive). Rather than stopping,
    # we store whatever partial answer the Answer agent produced and advance so
    # that subsequent steps can still run and the final synthesis has full context.
    if verification_status == "impossible":
        current_idx = state.get("current_step_index", 0)
        print(f"\n🚫 STEP {current_idx} IMPOSSIBLE — committing partial results and continuing\n")
        package = state.get("latest_verification_package", {})
        for task in package.get("tasks_analysis", []):
            t_id = task.get("task_id")
            if t_id:
                state.setdefault("completed_tasks_results", {})[t_id] = {
                    "final_answer": task.get("final_answer", ""),
                    "summary": task.get("summary", "")
                }
        state["verification_status"] = ""
        state["current_step_index"] = current_idx + 1

        if all_steps_completed(state):
            print("\n✅ ALL STEPS COMPLETED → FINAL SYNTHESIS\n")
            result = await handle_final_synthesis(state)
            return merge_state(state, result)

        return await _run_pipeline(state)

    # Step 3: Step passed → advance to next step
    if verification_status == "pass":
        current_idx = state.get("current_step_index", 0)
        print(f"✅ STEP {current_idx} PASSED\n")

        state["_replans"] = 0  # reset replan counter on success
        commit_verified_results(state)
        refresh_task_descriptions(state)

        state["last_failure_reason"] = ""
        state["verification_status"] = ""
        state["current_step_index"] = current_idx + 1

        # Check again after advancing
        if all_steps_completed(state):
            print("\n✅ ALL STEPS COMPLETED → FINAL SYNTHESIS\n")
            result = await handle_final_synthesis(state)
            return merge_state(state, result)

        return await _run_pipeline(state)

    #  Step 4: Step failed → replan
    if verification_status == "fail":
        state["_replans"] = state.get("_replans", 0) + 1
        replans = state["_replans"]
        current_idx = state.get("current_step_index", 0)
        print(f"❌ STEP {current_idx} FAILED → REPLAN #{replans}\n")

        failure_reason = state.get("last_failure_reason", "")
        if failure_reason:
            print(f"Reason: {failure_reason}\n")
            state.setdefault("failure_history", []).append(failure_reason)

        # Safety check to prevent infinite replanning loops
        if replans > max_replans:
            print(f"\nMAX REPLANS ({max_replans}) REACHED — stopping.\n")
            return merge_state(state, {
                "final_output": f"System stopped after {max_replans} failed replans. Last reason: {failure_reason}"
            })

        commit_verified_results(state)
        record_failed_servers(state)

        state["current_step_index"] = 0
        state["verification_status"] = ""
        state["all_parts_found"] = False

        print("🧠 [PLANNER] Generating new plan...")
        plan_updates = await _generate_plan(state)
        plan_updates = _remap_task_ids(plan_updates, state.get("completed_tasks_results", {}))
        state = merge_state(state, plan_updates)
        print_plan(state, f"REPLAN #{replans}")

        return await _run_pipeline(state)

    #  Generate initial plan if no plan exists yet
    if not plan:
        print("🧠 [PLANNER] Generating initial plan...")
        plan_updates = await _generate_plan(state)
        state = merge_state(state, plan_updates)
        print_plan(state, "INITIAL PLAN")
        return await _run_pipeline(state)

    # Planner has a plan but no verification status → start pipeline for current step
    print("🧠 [PLANNER] Resuming pipeline...")
    return await _run_pipeline(state)
