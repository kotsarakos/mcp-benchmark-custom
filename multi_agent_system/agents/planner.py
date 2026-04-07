import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_PLANNING, TEMPERATURE, INVENTORY_DIR
from ..prompts.agent_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_FINAL_SYNTHESIS_PROMPT, PLANNER_REASONING_STEP_PROMPT

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_FOR_PLANNING,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs={"response_format": {"type": "json_object"}}
)

def planner_node(state):
    """
    Strategic Planner Agent:
    1. Analyzes the current state and any past failures.
    2. Decomposes the query into atomic, parallelizable tasks.
    3. Handles Incremental Replanning if tasks have already been completed.
    """

    # Extract current context from the State
    user_input = state.get("input", "")
    completed_results = state.get("completed_tasks_results", {})
    failures = state.get("failure_history", [])
    failures_json = json.dumps(failures, ensure_ascii=False) if failures else "[]"
    last_failure_reason = state.get("last_failure_reason", "")
    verification_status = state.get("verification_status", "")

    if verification_status == "pass" or state.get("all_parts_found") is True:
        return handle_final_synthesis(state)

    # Load available server names so the planner can make informed task_type decisions
    try:
        inventory_path = INVENTORY_DIR / "inventory_summary.json"
        with open(inventory_path, "r", encoding="utf-8") as f:
            available_servers = json.load(f).get("available_servers", [])
        available_servers_str = json.dumps(available_servers, ensure_ascii=False)
    except Exception as e:
        print(f"Planner: could not load inventory ({e}), continuing without it")
        available_servers_str = "[]"

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_template(PLANNER_SYSTEM_PROMPT)
    parser = JsonOutputParser()

    # Create the chain
    chain = prompt | llm | parser

    try:
        # Invoke the LLM to generate the structured plan
        # We pass the full history of completed tasks to allow for incremental updates
        structured_plan = chain.invoke({
            "input": user_input,
            "completed_tasks": json.dumps(completed_results),
            "failure_history": failures_json,
            "last_failure_reason": last_failure_reason,
            "available_servers": available_servers_str
        })

        task_definitions = structured_plan.get("task_definitions", {})

        # Ensure every task has task_type
        for tid, tdef in task_definitions.items():
            if "task_type" not in tdef:
                print(f"Warning: task '{tid}' missing task_type, defaulting to 'tool'")
                tdef["task_type"] = "tool"

        # Inject dependency results into task descriptions so retrieval and
        # executor receive concrete values
        for tid, tdef in task_definitions.items():
            deps = tdef.get("dependencies", [])
            context = [
                f"{dep}: {completed_results[dep]['final_answer']}"
                for dep in deps
                if dep in completed_results and completed_results[dep].get("final_answer")
            ]
            if context:
                tdef["description"] += "\n\nResults from previous tasks:\n" + "\n".join(context)

        # Return the update to the State with the new plan and reset the current step index
        return {
            "plan": structured_plan.get("plan", []),
            "task_definitions": task_definitions,
            "current_step_index": 0,
            "last_failure_reason": "",
            "messages": [{"role": "assistant", "content": "Strategic execution plan updated."}]
        }

    except Exception as e:
        print(f"Planner Error: {e}")
        return {
            "last_failure_reason": f"Planner failed to generate JSON: {str(e)}"
        }

def handle_reasoning_step(state):
    """
    Called by the graph when the current step contains only reasoning tasks.
    The Planner reasons over completed_tasks_results and produces final_output directly.
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
    chain = prompt | llm | JsonOutputParser()

    try:
        response = chain.invoke({
            "original_query": state.get("input", ""),
            "collected_data": json.dumps(completed_results, indent=2, ensure_ascii=False),
            "reasoning_tasks": json.dumps(reasoning_tasks, indent=2, ensure_ascii=False),
        })
        return {
            "final_output": response.get("final_answer"),
            "messages": [{"role": "assistant", "content": "Planner resolved reasoning step."}]
        }
    except Exception as e:
        print(f"Planner Reasoning Error: {e}")
        return {"last_failure_reason": f"Planner reasoning failed: {str(e)}"}


def handle_final_synthesis(state):
    """Triggered when information is complete and verified."""
    prompt = ChatPromptTemplate.from_template(PLANNER_FINAL_SYNTHESIS_PROMPT)

    chain = prompt | llm | JsonOutputParser()

    summary_context = json.dumps(
        state.get("completed_tasks_results", {}),
        indent=2
    )

    # Fix 3: wrap in try/except so LLM failures don't crash the graph
    try:
        response = chain.invoke({
            "original_query": state.get("input"),
            "collected_data": summary_context
        })
        return {
            "final_output": response.get("answer"),
            "messages": [{"role": "assistant", "content": "Final synthesis generated."}]
        }
    except Exception as e:
        print(f"Final Synthesis Error: {e}")
        return {"last_failure_reason": f"Final synthesis failed: {str(e)}"}
