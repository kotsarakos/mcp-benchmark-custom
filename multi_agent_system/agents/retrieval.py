import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_RETRIEVAL, TEMPERATURE, INVENTORY_DIR
from ..prompts.agent_prompts import RETRIEVER_SYSTEM_PROMPT

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
                "excluded_servers": json.dumps(excluded_list, ensure_ascii=False)
            })
            
            # Validate if the LLM provided a selected server in the response
            selected = selection.get("selected_server")
            if selected:
                return task_id, {"selected_server": selected}
            
            # Increment attempts if the expected key is missing (logical failure)
            attempts += 1
            
        except Exception as e:
            attempts += 1
            last_error = str(e)
            print(f"Attempt {attempts} failed for task {task_id}: {last_error}")
            
            if attempts < max_retries:
                # Wait for 1 second before retrying to handle rate limits or transient errors
                await asyncio.sleep(1)

    return task_id, {
        "selected_server": None, 
        "error": f"Failed after {max_retries} retries. Last error: {last_error}"
    }

async def retrieval_node(state):
    """
    Orchestrates the retrieval process for the current plan step.
    Processes multiple tasks in parallel if the plan allows.
    """
    # Extract structural data from the current state
    steps = state.get("plan", [])
    task_definitions = state.get("task_definitions", {})
    idx = state.get("current_step_index", 0)
    
    # Verify if there are remaining steps to process
    if idx >= len(steps):
        return {"selected_servers": state.get("selected_servers", {})}

    current_step = steps[idx]

    # Retrieve the tasks from the step[x]
    task_ids = current_step.get("tasks", [])
    
    # Load the server inventory summary to provide context to the LLM
    try:
        inventory_path = INVENTORY_DIR / "inventory_summary.json"
        with open(inventory_path, "r", encoding="utf-8") as f:
            inventory_index = json.load(f).get("available_servers", [])
    except Exception as e:
        print(f"Inventory Load Error: {e}")
        # Fix 7: also set verification_status so graph skips executor/answer/verifier
        # and goes straight to replan instead of looping
        return {
            "last_failure_reason": f"Inventory file access error: {str(e)}",
            "verification_status": "fail"
        }

    excluded_servers = state.get("excluded_servers", {})

    # Create concurrent coroutines for each task in the current step
    coros = [
        select_mcp_server(
            tid,
            task_definitions[tid]["description"],
            inventory_index,
            excluded=excluded_servers.get(tid, [])
        )
        for tid in task_ids if tid in task_definitions
    ]
    
    # Handle cases where no valid tasks are found in the step
    if not coros:
        print("No valid tasks found for retrieval in this step.")
        return {"current_step_index": idx + 1}

    # Execute all routing requests in parallel
    results = await asyncio.gather(*coros)

    # Merge new selections with the existing tool registry in the state
    updated_selections = state.get("selected_servers", {}).copy()
    for tid, server_info in results:
        updated_selections[tid] = server_info

    # Return the updated tool selections to the global state
    return {"selected_servers": updated_selections}