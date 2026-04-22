import logging
import json
import os
import time
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
from ..trace_recorder import get_recorder

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_FOR_EXECUTOR,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Cache the prompt chain
_react_prompt = ChatPromptTemplate.from_template(EXECUTOR_REACT_PROMPT)
_react_chain = _react_prompt | llm
_json_parser = JsonOutputParser()

server_manager = None
initialized = False

# Token budget controls — limits how much text enters the ReAct history
MAX_OBSERVATION_CHARS = 3000   # Hard cap per tool observation
MAX_HISTORY_CHARS = 10000      # Hard cap on total history passed to the LLM

# Max times the same (tool, args) pair may be blocked before forcing DONE.
MAX_DUPLICATE_BLOCKS = 2


def get_command_path():
    """
    Get the commads.json file, which includes the server configs.
    """
    current_dir = os.path.dirname(__file__)
    agents_dir = os.path.abspath(os.path.join(current_dir, "../"))
    return os.path.join(agents_dir, "commands.json")


def load_api_keys(base_dir):
    """
    Load API keys, for MCP Servers that require them.
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


async def initialize_executor(command_file: str = None):
    """
    Initialize the PersistentMultiServerManager with configs from commands.json and connect to all servers.
    """
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

        # Normalize server name to match tool definitions and MCPConnector expectations
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

    # Open connections to all servers in parallel, with error handling and logging.
    await server_manager.connect_all_servers()

    initialized = True


async def executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the current plan step by calling MCP tools.

    Reads the current step from the plan, dispatches each task to its
    selected MCP server, and returns the raw results. Tasks run in parallel
    or sequentially depending on the parallel flag set by the planner.

    Returns a state patch with:
        latest_execution_results: dict mapping task_id --> raw result string
    """

    # Ensure the executor is initialized and has active connections to all MCP servers before executing any tasks.
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

    # Parallel execution
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

    # Sequential execution
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


async def execute_single_task(state, task_id, selected_servers, max_steps: int = 15):
    """
    ReAct (Reason + Act) execution loop for a single task.

    Each iteration the LLM receives the full execution history
    (thought --> tool --> observation) and decides either to call the
    next tool (action=CALL_TOOL) or stop (action=DONE).

    Returns a string: the LLM's final_result (raw output), which the
    Answer agent will structure into a clean answer.
    """

    # The TraceRecorder is optional and only present if _enable_trace was set in the initial state (e.g. during MCP-Bench evaluation). 
    # It allows us to capture detailed logs of tool calls, results, and errors for later analysis.
    recorder = get_recorder(state)
    
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

    # Tracks how many times each (tool_name, frozen_args) pair has been blocked.
    # Counts across non-consecutive blocks
    duplicate_block_counts: dict = {}

    # Compact log of calls that failed or returned empty — persists across windowing
    # LLM always knows what not to retry, even if old history entries were dropped.
    failed_calls: list = []

    for step in range(1, max_steps + 1):
        history_str = _format_history_windowed(history, MAX_HISTORY_CHARS)

        # Append the failed-calls log so it's always visible regardless of windowing.
        if failed_calls:
            failed_summary = (
                "FAILED/EMPTY CALLS — do NOT repeat these (tool | args | reason):\n"
                + "\n".join(failed_calls)
                + "\n\n"
            )
            history_str = failed_summary + history_str

        # ReAct step: LLM reasons and decides next action
        try:
            decision = await _react_step(task_desc, formatted_tools, history_str)
        except asyncio.TimeoutError:
            raise ValueError(f"ReAct LLM call timed out on step {step}")
        except Exception as e:
            raise ValueError(f"ReAct LLM call failed on step {step}: {e}")

        action = decision.get("action", "").upper()
        thought = decision.get("thought", "")

        # Executor decides it has enough data
        if action == "DONE":
            final_result = decision.get("final_result", "")
            logger.info("Task %s completed in %d ReAct step(s).", task_id, step)
            return final_result

        # Executor wants to call a tool
        if action == "CALL_TOOL":
            tool_name = decision.get("tool_name")
            arguments = decision.get("arguments", {})

            if not tool_name:
                raise ValueError(f"ReAct step {step}: LLM returned CALL_TOOL but no tool_name")

            # Track per-(tool, args) block count so alternating duplicates are also caught.
            prior = next(
                # Search full history for any prior call with the same tool and arguments, even if interleaved with other calls.
                (h for h in history if h["tool_name"] == tool_name and h["arguments"] == arguments),
                None
            )
            if prior:
                block_key = (tool_name, json.dumps(arguments, sort_keys=True))
                duplicate_block_counts[block_key] = duplicate_block_counts.get(block_key, 0) + 1
                count = duplicate_block_counts[block_key]
                logger.warning("Task %s step %d: duplicate call blocked (%s) [block #%d]", task_id, step, tool_name, count)

                # Force exit if the Executor keeps ignoring the blocker.
                if count >= MAX_DUPLICATE_BLOCKS:
                    logger.warning("Task %s: forcing DONE after %d duplicate blocks for (%s).", task_id, count, tool_name)
                    collected = "\n\n".join(
                        f"Step {h['step']} [{h['tool_name']}]:\n{h['observation']}"
                        for h in history
                        if not h["observation"].startswith("DUPLICATE CALL BLOCKED")
                    )
                    return collected or "No data collected."

                # If this call was already made, block it and return the prior observation as a hint.
                observation = (
                    f"DUPLICATE CALL BLOCKED: '{tool_name}' with these exact arguments was "
                    f"already called in step {prior['step']}. "
                    f"Previous result: {prior['observation'][:300]}. "
                    f"Do not repeat this call — return DONE with the data you already have, "
                    f"or try a completely different tool/approach."
                )
                history.append({
                    "step": step, "thought": thought,
                    "tool_name": tool_name, "arguments": arguments,
                    "observation": observation,
                })
                continue

            fail_reason = None

            call_start = time.time()
            try:
                # Asynchronous tool call with timeout
                result_obj = await asyncio.wait_for(
                    server_manager.call_tool(tool_name, arguments),
                    timeout=30
                )

                observation = extract_text(result_obj)
                observation = truncate(observation, MAX_OBSERVATION_CHARS)

                # Detect empty results by content, not length.
                # For JSON responses, check if the parsed value is semantically empty.
                # For non-JSON, fall back to a short length threshold.
                try:
                    _parsed = json.loads(observation)
                    _is_empty = _parsed in (None, [], {}) or _parsed == {"results": []}
                except (json.JSONDecodeError, TypeError):
                    _is_empty = len(observation.strip()) < 10
                if _is_empty:
                    fail_reason = "empty results"
            except asyncio.TimeoutError:
                observation = f"ERROR: Tool '{tool_name}' timed out after 30s"
                fail_reason = "timeout"
                logger.warning("Task %s step %d: %s", task_id, step, observation)
            except Exception as e:
                observation = f"ERROR: {str(e) or type(e).__name__}"
                fail_reason = observation
                logger.warning("Task %s step %d: tool call failed: %s", task_id, step, observation)

            # End of tool call — record the result and any failure reason for this step, 
            # then loop to the next step where the LLM receives the new observation.    
            call_duration = time.time() - call_start

            if fail_reason:
                failed_calls.append(
                    f"  {tool_name}({json.dumps(arguments, separators=(',', ':'))}) → {fail_reason}"
                )

            # Record the tool call, result, and any failure reason in the TraceRecorder if it's enabled.
            # This allows us to capture detailed execution traces for Benchmarks evaluation without affecting other runs.
            if recorder is not None:
                recorder.record_tool_call(
                    tool=tool_name,
                    server=server_name,
                    parameters=arguments,
                    success=(fail_reason is None),
                    result=observation,
                    error=fail_reason,
                    duration_seconds=call_duration,
                )

            history.append({
                "step": step,
                "thought": thought,
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            })

            # Loop continues to next step, where the Executor receives the new observation and decides what to do next.
            continue

        # Unknown action — return collected data rather than hard-failing the task.
        logger.warning("Task %s step %d: unexpected action '%s' — treating as DONE.", task_id, step, action)
        break

    # Max steps reached, return real observations only.
    logger.warning("Task %s hit max_steps (%d) without DONE — returning partial data.", task_id, max_steps)
    collected = "\n\n".join(
        f"Step {h['step']} [{h['tool_name']}]:\n{h['observation']}"
        for h in history
        if not h["observation"].startswith("DUPLICATE CALL BLOCKED")
    )
    return collected or "No data collected."


async def _react_step(task_desc: str, formatted_tools: str, history_str: str) -> dict:
    """
    Sends the current ReAct state to the Executor and returns its next decision.

    Each call receives the task description, available tools, and the full
    execution history so far. The LLM responds with either a CALL_TOOL
    decision (tool name + arguments) or a DONE decision (final result).

    Raises ValueError on timeout or parse failure — the caller handles retries.
    """

    raw_response = await asyncio.wait_for(
        _react_chain.ainvoke({
            "task_description": task_desc,
            "tools_list": formatted_tools,
            "history": history_str,
        }),
        timeout=60
    )

    token_tracker.track("executor", raw_response)
    return _json_parser.parse(raw_response.content)


def _format_history_windowed(history: list, max_chars: int) -> str:
    """
    Format the ReAct trace, dropping the OLDEST entries if the total exceeds max_chars.

    Truncating from the tail removes the most recent observations — exactly the context the Executor 
    needs to make the next decision. Instead, we drop whole steps from the front until we fit, always
    preserving the latest entries.
    """

    if not history:
        return "No steps taken yet — this is the first action."

    formatted = [
        f"Step {h['step']}:\n"
        f"  Thought: {h['thought']}\n"
        f"  Tool: {h['tool_name']} | Args: {json.dumps(h['arguments'])}\n"
        f"  Observation: {h['observation']}"
        for h in history
    ]

    # Drop oldest entries until the joined result fits within max_chars.
    while len(formatted) > 1:
        candidate = "\n\n".join(formatted)
        if len(candidate) <= max_chars:
            return candidate
        formatted.pop(0)

    # Only one entry left — hard-truncate it as a last resort.
    return truncate(formatted[0], max_chars)


# Text truncation
def truncate(text: str, max_chars: int) -> str:
    """
    Hard-truncate text to max_chars, appending a marker if trimmed.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [TRUNCATED]"


# Tool result text extraction
def extract_text(result):
    """
    Extract text from a tool result, handling different formats and truncating if needed.
    """

    # Some tools return a simple string, while others return an object with a 'content' field.
    if hasattr(result, 'content') and result.content:
        raw = "".join(item.text for item in result.content if hasattr(item, 'text'))
    else:
        raw = str(result)

    # Attempt to parse as JSON and extract a 'text' field if present, otherwise return raw string.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "summary" in parsed and "text" in parsed:
            parsed["text"] = truncate(parsed["text"], 2000)
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass

    return raw