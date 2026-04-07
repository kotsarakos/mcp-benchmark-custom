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
from ..prompts.agent_prompts import EXECUTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------
# LLM
# ---------------------------
llm = ChatOpenAI(
    model_name=MODEL_FOR_EXECUTOR,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs={"response_format": {"type": "json_object"}}
)

server_manager = None
initialized = False
TOKEN_THRESHOLD = 2000


# ---------------------------
# PATH
# ---------------------------
def get_command_path():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    return os.path.join(project_root, "mcp_servers", "commands.json")


# ---------------------------
# API KEYS
# ---------------------------
def load_api_keys(base_dir):
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

            key, value = line.split("=", 1)
            env_vars[key.strip()] = value.strip()

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
    api_keys = load_api_keys(base_dir)

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
# SINGLE TASK
# ---------------------------
async def execute_single_task(state, task_id, selected_servers, max_retries: int = 3):

    task_desc = state["task_definitions"][task_id]["description"]

    server_info = selected_servers.get(task_id)

    if not server_info:
        raise ValueError(f"No server selected for task {task_id}")

    if isinstance(server_info, dict):
        # Fix 8: catch the case where retrieval returned {"selected_server": None, "error": "..."}
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

    previous_attempts = []
    last_error = ""

    for attempt in range(1, max_retries + 1):
        tool_name = None
        arguments = {}

        try:
            decision = await select_tool(task_desc, server_tools, previous_attempts)
            tool_name = decision.get("tool_name")
            arguments = decision.get("arguments", {})

            if not tool_name:
                raise ValueError("LLM did not return a tool_name")

            result_obj = await server_manager.call_tool(tool_name, arguments)
            result_text = extract_text(result_obj)
            return await postprocess(result_text)

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Task {task_id} attempt {attempt}/{max_retries} failed: {last_error}")
            previous_attempts.append({
                "attempt": attempt,
                "tool_name": tool_name or "unknown",
                "arguments": arguments,
                "error": last_error
            })
            if attempt < max_retries:
                await asyncio.sleep(1)

    raise ValueError(f"Task {task_id} failed after {max_retries} attempts. Last error: {last_error}")


# ---------------------------
# TOOL SELECTION
# ---------------------------
async def select_tool(task_desc: str, tools: Dict[str, Any], previous_attempts: list = None):
    formatted_tools = MCPConnector.format_tools_for_prompt(tools)

    if previous_attempts:
        formatted_attempts = json.dumps(previous_attempts, indent=2, ensure_ascii=False)
    else:
        formatted_attempts = "None — this is the first attempt."

    prompt = ChatPromptTemplate.from_template(EXECUTOR_SYSTEM_PROMPT)
    chain = prompt | llm | JsonOutputParser()

    return await chain.ainvoke({
        "task_description": task_desc,
        "tools_list": formatted_tools,
        "previous_attempts": formatted_attempts
    })


# ---------------------------
# POSTPROCESS
# ---------------------------
async def postprocess(text: str):
    if estimate_tokens(text) > TOKEN_THRESHOLD:
        return await summarize(text)
    return text


def estimate_tokens(text: str):
    return len(text) // 4


async def summarize(text: str):
    prompt = f"Summarize preserving key facts:\n{text[:8000]}"
    response = await llm.ainvoke(prompt)
    return response.content


# ---------------------------
# UTILS
# ---------------------------
def extract_text(result):
    if hasattr(result, 'content') and result.content:
        return "".join(
            item.text for item in result.content if hasattr(item, 'text')
        )
    return str(result)