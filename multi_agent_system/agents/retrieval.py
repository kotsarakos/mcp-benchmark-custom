import asyncio
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_RETRIEVAL, TEMPERATURE, INVENTORY_DIR
from ..prompts.agent_prompts import RETRIEVER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_FOR_RETRIEVAL,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs={"response_format": {"type": "json_object"}}
)

async def select_mcp_server(task_id, task_desc, server_index, excluded=None, max_retries=3):
    """
    Identifies the appropriate MCP server for a specific task using LLM reasoning.
    - task_id: Unique identifier for the task.
    - task_desc: Description of the task to be routed.
    - server_index: List of available MCP servers with their capabilities.
    - excluded: List of servers to exclude from selection (already tried and failed).
    - max_retries: Maximum number of retries for LLM invocation in case of failures.
    Returns a tuple of (task_id, {"selected_server": server_name or None")
    """

    parser = JsonOutputParser()
    prompt_tmpl = ChatPromptTemplate.from_template(RETRIEVER_SYSTEM_PROMPT)
    chain = prompt_tmpl | llm | parser

    excluded_list = excluded or []

    attempts = 0
    last_error = ""

    while attempts < max_retries:
        try:
            # Asynchronous invocation of the retrieval chain
            selection = await chain.ainvoke({
                "task_description": task_desc,
                "server_list": json.dumps(server_index, ensure_ascii=False),
                "excluded_servers": json.dumps(excluded_list, ensure_ascii=False),
            })
            
            # Validate if the LLM provided a selected server in the response
            selected = selection.get("selected_server")
            if selected:
                return task_id, {"selected_server": selected}
            
            # Increment attempts if the expected key is missing
            attempts += 1
            
        except Exception as e:
            attempts += 1
            last_error = str(e)
            logger.warning("Retrieval attempt %d/%d failed for %s: %s", attempts, max_retries, task_id, last_error)

            # Back off only on rate-limit errors; all other failures retry immediately.
            if attempts < max_retries and "rate" in last_error.lower():
                await asyncio.sleep(1)

    logger.error("Retrieval gave up for %s after %d attempts. Last error: %s", task_id, max_retries, last_error)
    return task_id, {
        "selected_server": None,
        "error": f"Failed after {max_retries} retries. Last error: {last_error}",
    }

async def retrieval_node(state: dict) -> dict:
    """
    Retrieval Agent entry point — called by the Planner

    Reads the current step from state, launches parallel server-selection
    for each task in that step, and returns the updated server assignments.

    Returns a state patch with:
        selected_servers: dict mapping task_id → {"selected_server": name}

    On inventory load failure, returns a "fail" verification_status so the
    Planner triggers a replan rather than proceeding with no server data.
    """
    steps = state.get("plan", [])
    task_definitions = state.get("task_definitions", {})
    idx = state.get("current_step_index", 0)
    
    # Verify if there are remaining steps to process
    if idx >= len(steps):
        return {"selected_servers": state.get("selected_servers", {})}

    # Retrieve the tasks from the step[x]
    current_step = steps[idx]

    task_ids = current_step.get("tasks", [])

    # Load the server inventory that gives the LLM context for its routing decision.
    # This file is generated at startup by the inventory collector.
    try:
        inventory_path = INVENTORY_DIR / "inventory_summary.json"
        with open(inventory_path, "r", encoding="utf-8") as f:
            inventory_index = json.load(f).get("available_servers", [])
    except Exception as e:
        logger.error("Failed to load server inventory: %s", e)
        # Signal failure so the pipeline skips Executor/Answer/Verifier and replans.
        return {
            "last_failure_reason": f"Inventory file access error: {str(e)}",
            "verification_status": "fail",
        }

    # Per-task exclusion lists — populated by the Planner when a server fails
    # verification, ensuring the LLM doesn't re-select the same broken server.
    excluded_servers = state.get("excluded_servers", {})

    # Build one coroutine per task and run them all concurrently.
    # Tasks in a parallel step have no dependencies on each other, so this is safe.
    coros = [
        select_mcp_server(
            tid,
            task_definitions[tid]["description"],
            inventory_index,
            excluded=excluded_servers.get(tid, []),
        )
        for tid in task_ids
        if tid in task_definitions
    ]

    if not coros:
        logger.warning("Step %d has no valid tasks for retrieval — advancing step.", idx)
        return {"current_step_index": idx + 1}

    results = await asyncio.gather(*coros)

    # Merge new selections into the existing registry 
    updated_selections = state.get("selected_servers", {}).copy()
    for tid, server_info in results:
        updated_selections[tid] = server_info

    return {"selected_servers": updated_selections}