import logging
import json
import os
import asyncio
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from mcp_modules.server_manager_persistent import PersistentMultiServerManager
from mcp_modules.connector import MCPConnector

from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_EXECUTOR, TEMPERATURE
from ..prompts.agent_prompts import EXECUTOR_REACT_PROMPT
from ..token_tracker import token_tracker

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_FOR_EXECUTOR,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs={"response_format": {"type": "json_object"}}
)

server_manager = None
initialized = False

# Token budget controls — limits how much text enters the ReAct history
# to prevent context window bloat and unnecessary summarize() LLM calls.
MAX_OBSERVATION_CHARS = 3000   # Hard cap per tool observation
MAX_HISTORY_CHARS = 10000      # Hard cap on total history passed to the LLM


def get_command_path():
    # Use the custom commands.json inside multi_agent_system/ so the original
    # mcp_servers/commands.json used by the benchmark runner stays untouched.
    current_dir = os.path.dirname(__file__)
    agents_dir = os.path.abspath(os.path.join(current_dir, "../"))
    return os.path.join(agents_dir, "commands.json")


def load_api_keys(base_dir):
    """
    Load API keys from a file in the mcp_servers/ directory. 
    """
    api_path = os.path.join(base_dir, "api_key")

    env_vars = {}

    if not os.path.exists(api_path):
        logger.warning("api_key file not found")
        return env_vars

    with open(api_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            # Strip shell 'export' prefix if present
            if line.startswith("export "):
                line = line[len("export "):]

            key, value = line.split("=", 1)
            # Strip surrounding quotes from value
            value = value.strip().strip('"').strip("'")
            env_vars[key.strip()] = value

    return env_vars


# ---------------------------
# INIT
# ---------------------------
async def initialize_executor(command_file: str = None):
    global server_manager, initialized

    if initialized:
        return

    if command_file is None:
        command_file = get_command_path()

    logger.info(f"Loading commands.json from: {command_file}")

    with open(command_file, "r") as f:
        raw_data = json.load(f)

    base_dir = os.path.dirname(command_file)
    # api_key file lives in mcp_servers/ — resolve from the command file's location.
    mcp_servers_dir = os.path.abspath(os.path.join(base_dir, "../mcp_servers"))
    api_keys = load_api_keys(mcp_servers_dir)

    configs = []

    for server_name, config in raw_data.items():

        cmd_list = config["cmd"].split()

        raw_cwd = config.get("cwd")
        abs_cwd = None
        if raw_cwd:
            abs_cwd = os.path.abspath(os.path.join(base_dir, raw_cwd))

        env_vars = {}
        for var in config.get("env", []):
            if var in api_keys:
                env_vars[var] = api_keys[var]
            else:
                logger.warning(f"[ENV WARNING] {var} not found")

        configs.append({
            "name": server_name.lower().replace(" ", "_"),
            "command": cmd_list,
            "cwd": abs_cwd,
            "env": env_vars,
            "transport": config.get("transport", "stdio"),
            "port": config.get("port"),
            "endpoint": config.get("endpoint", "/mcp")
        })

    server_manager = PersistentMultiServerManager(configs)
    await server_manager.connect_all_servers()

    initialized = True


# ---------------------------
# EXECUTOR NODE (HYBRID)
# ---------------------------
async def executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    await initialize_executor()

    current_idx = state.get("current_step_index", 0)
    steps = state.get("plan", [])
    selected_servers = state.get("selected_servers", {})

    if not steps or current_idx >= len(steps):
        return {"errors": ["Executor index error."]}

    current_step = steps[current_idx]
    step_tasks = current_step.get("tasks", [])
    is_parallel = current_step.get("parallel", False)

    results = {}

    # ---------------------------
    # PARALLEL EXECUTION
    # ---------------------------
    if is_parallel:

        logger.info(f"Running step {current_idx} in PARALLEL mode")

        coroutines = [
            execute_single_task(state, task_id, selected_servers)
            for task_id in step_tasks
        ]

        outputs = await asyncio.gather(*coroutines, return_exceptions=True)

        for task_id, output in zip(step_tasks, outputs):
            if isinstance(output, Exception):
                logger.error(f"Task {task_id} failed: {output}")
                results[task_id] = f"Error: {str(output)}"
            else:
                results[task_id] = output

    # ---------------------------
    # SEQUENTIAL EXECUTION
    # ---------------------------
    else:

        logger.info(f"Running step {current_idx} in SEQUENTIAL mode")

        for task_id in step_tasks:
            try:
                result = await execute_single_task(state, task_id, selected_servers)
                results[task_id] = result
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                results[task_id] = f"Error: {str(e)}"

    return {"latest_execution_results": results}


# ---------------------------
# SINGLE TASK — ReAct loop
# ---------------------------
async def execute_single_task(state, task_id, selected_servers, max_steps: int = 10):
    """
    ReAct (Reason + Act) execution loop for a single task.

    Each iteration the LLM receives the full execution history
    (thought → tool → observation) and decides either to call the
    next tool (action=CALL_TOOL) or stop (action=DONE).

    Returns a string: the LLM's final_result summary, which the
    Answer agent will structure into a clean answer.
    """
    task_desc = state["task_definitions"][task_id]["description"]

    server_info = selected_servers.get(task_id)
    if not server_info:
        raise ValueError(f"No server selected for task {task_id}")

    if isinstance(server_info, dict):
        if server_info.get("error"):
            raise ValueError(f"Server selection failed for task {task_id}: {server_info['error']}")
        raw_server_name = server_info.get("selected_server")
    else:
        raw_server_name = server_info

    if not raw_server_name:
        raise ValueError(f"No suitable MCP server found for task {task_id}")

    server_name = raw_server_name.lower().replace(" ", "_")
    server_tools = {
        name: tool
        for name, tool in server_manager.all_tools.items()
        if tool["server"] == server_name
    }

    if not server_tools:
        raise ValueError(f"No tools found for server {server_name}")

    formatted_tools = MCPConnector.format_tools_for_prompt(server_tools)

    # history holds the trace: [{thought, action, tool_name, arguments, observation}]
    history = []

    for step in range(1, max_steps + 1):
        history_str = truncate(_format_history(history), MAX_HISTORY_CHARS)

        # ReAct step: LLM reasons and decides next action
        try:
            decision = await _react_step(task_desc, formatted_tools, history_str)
        except asyncio.TimeoutError:
            raise ValueError(f"ReAct LLM call timed out on step {step}")
        except Exception as e:
            raise ValueError(f"ReAct LLM call failed on step {step}: {e}")

        action = decision.get("action", "")
        thought = decision.get("thought", "")

        # LLM decided it has enough data
        if action == "DONE":
            final_result = decision.get("final_result", "")
            logger.info("Task %s completed in %d ReAct step(s).", task_id, step)
            return final_result

        # LLM wants to call a tool
        if action == "CALL_TOOL":
            tool_name = decision.get("tool_name")
            arguments = decision.get("arguments", {})

            if not tool_name:
                raise ValueError(f"ReAct step {step}: LLM returned CALL_TOOL but no tool_name")

            try:
                result_obj = await asyncio.wait_for(
                    server_manager.call_tool(tool_name, arguments),
                    timeout=30
                )
                observation = extract_text(result_obj)
                observation = truncate(observation, MAX_OBSERVATION_CHARS)
            except asyncio.TimeoutError:
                observation = f"ERROR: Tool '{tool_name}' timed out after 30s"
                logger.warning("Task %s step %d: %s", task_id, step, observation)
            except Exception as e:
                observation = f"ERROR: {str(e) or type(e).__name__}"
                logger.warning("Task %s step %d: tool call failed: %s", task_id, step, observation)

            history.append({
                "step": step,
                "thought": thought,
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            })
            continue

        # Unknown action — stop to avoid infinite loop
        raise ValueError(f"ReAct step {step}: unexpected action '{action}'")

    # Max steps reached — return whatever was collected
    logger.warning("Task %s hit max_steps (%d) without DONE — returning partial data.", task_id, max_steps)
    collected = "\n\n".join(
        f"Step {h['step']} [{h['tool_name']}]:\n{h['observation']}"
        for h in history
    )
    return collected or "No data collected."


# ---------------------------
# ReAct STEP
# ---------------------------
async def _react_step(task_desc: str, formatted_tools: str, history_str: str) -> dict:
    """Single LLM call in the ReAct loop — returns thought + action decision."""
    prompt = ChatPromptTemplate.from_template(EXECUTOR_REACT_PROMPT)
    chain = prompt | llm
    raw_response = await asyncio.wait_for(
        chain.ainvoke({
            "task_description": task_desc,
            "tools_list": formatted_tools,
            "history": history_str,
        }),
        timeout=60
    )
    token_tracker.track("executor", raw_response)
    return JsonOutputParser().parse(raw_response.content)


def _format_history(history: list) -> str:
    """Format the ReAct trace into a readable string for the LLM prompt."""
    if not history:
        return "No steps taken yet — this is the first action."
    lines = []
    for h in history:
        lines.append(
            f"Step {h['step']}:\n"
            f"  Thought: {h['thought']}\n"
            f"  Tool: {h['tool_name']} | Args: {json.dumps(h['arguments'])}\n"
            f"  Observation: {h['observation']}"
        )
    return "\n\n".join(lines)


# ---------------------------
# TRUNCATION
# ---------------------------
def truncate(text: str, max_chars: int) -> str:
    """Hard-truncate text to max_chars, appending a marker if trimmed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [TRUNCATED]"


# ---------------------------
# UTILS
# ---------------------------
def extract_text(result):
    if hasattr(result, 'content') and result.content:
        return "".join(
            item.text for item in result.content if hasattr(item, 'text')
        )
    return str(result)