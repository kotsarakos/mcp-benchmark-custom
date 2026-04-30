import asyncio
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_RETRIEVAL, TEMPERATURE, INVENTORY_DIR, make_model_kwargs
from ..prompts.agent_prompts import RETRIEVER_SYSTEM_PROMPT
from ..token_tracker import token_tracker
from ..utils import current_date_str

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_FOR_RETRIEVAL,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs=make_model_kwargs({"response_format": {"type": "json_object"}})
)

# Cached chain and parser
_retrieval_chain = ChatPromptTemplate.from_template(RETRIEVER_SYSTEM_PROMPT) | llm
_json_parser = JsonOutputParser()

# Load server inventory once at module startup — abort if missing.
try:
    with open(INVENTORY_DIR / "inventory_summary.json", "r", encoding="utf-8") as _f:
        _inventory_index = json.load(_f).get("available_servers", [])
except Exception as _e:
    raise RuntimeError(
        f"Retrieval: could not load the names of MCP servers — system cannot start. Reason: {_e}"
    ) from _e

async def select_mcp_server(task_id, task_desc, excluded=None, max_retries=5, inventory=None):
    """
    Identifies the appropriate MCP server for a specific task using LLM reasoning.
    - task_id: Unique identifier for the task.
    - task_desc: Description of the task to be routed.
    - excluded: List of servers to exclude from selection (already tried and failed).
    - max_retries: Maximum number of retries for LLM invocation in case of failures.
    - inventory: Optional list of servers to consider for selection (overrides the default inventory).
    Returns a tuple of (task_id, {"selected_server": server_name or None})
    """

    excluded_list = excluded or []
    server_inventory = inventory if inventory is not None else _inventory_index

    attempts = 0
    last_error = ""

    while attempts < max_retries:
        try:
            # Asynchronous invocation of the retrieval chain
            raw_response = await asyncio.wait_for(
                _retrieval_chain.ainvoke({
                    "task_description": task_desc,
                    "server_list": json.dumps(server_inventory, ensure_ascii=False),
                    "excluded_servers": json.dumps(excluded_list, ensure_ascii=False),
                    "current_date": current_date_str(),
                }),
                timeout=180
            )
            token_tracker.track("retrieval", raw_response)
            selection = _json_parser.parse(raw_response.content)
            
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

            # Back off only on rate-limit errors, all other failures retry immediately.
            if attempts < max_retries and "rate" in last_error.lower():
                await asyncio.sleep(1)

    logger.error("Retrieval gave up for %s after %d attempts. Last error: %s", task_id, max_retries, last_error)
    return task_id, {
        "selected_server": None,
        "error": f"Failed after {max_retries} retries. Last error: {last_error}",
    }

async def retrieval_node(state: dict) -> dict:
    """
    Retrieval Agent entry point

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

    # Get the list of already excluded servers for this step (if any) to avoid retrying them
    excluded_servers = state.get("excluded_servers", {})

    # Build the inventory the retrieval LLM will see.
    # Priority:
    #   1. _rich_inventory (built at runtime from live MCP connections — has
    #      name + description + tool names per server)
    #   2. _server_subset (mcpbench legacy — name-only list for required+distractions)
    #   3. Module-level _inventory_index (fallback — name-only for all servers)
    rich_inventory = state.get("_rich_inventory")

    server_subset = state.get("_server_subset")

    if rich_inventory is not None:
        # If mcpbench restricted the connected servers, _rich_inventory already
        # only contains those (since it was built from live connections).
        # Otherwise it has every connected server with its tool list.
        retrieval_inventory = rich_inventory
        logger.info(
            "Retrieval inventory: %d servers with tool lists (%s)",
            len(rich_inventory),
            "subset" if server_subset else "full",
        )
    else:
        # Legacy fallback — name-only inventory (rare; only when graph didn't build it).
        retrieval_inventory = server_subset
        if retrieval_inventory is not None:
            logger.info(
                "Retrieval inventory restricted to %d server names: %s",
                len(retrieval_inventory), retrieval_inventory
            )

    # Build one coroutine per task and run them all concurrently.
    coros = [
        select_mcp_server(
            tid,
            task_definitions[tid]["description"],
            excluded=excluded_servers.get(tid, []),
            inventory=retrieval_inventory,
        )
        # Iterate only over valid task IDs that are defined in task_definitions to avoid KeyErrors
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