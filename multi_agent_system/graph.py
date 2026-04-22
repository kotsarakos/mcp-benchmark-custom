import logging
import httpx
from typing import Any, Dict

from .agents.planner import planner_node
from .agents import executor as executor_module
from .agents.executor import initialize_executor
from .utils import normalize_state
from .config import VLLM_BASE_URL, API_KEY, MODEL_FOR_PLANNING, MODEL_FOR_RETRIEVAL, MODEL_FOR_EXECUTOR, MODEL_FOR_ANSWERING, MODEL_FOR_VERIFIER
from .token_tracker import token_tracker
from .trace_recorder import TraceRecorder

logger = logging.getLogger(__name__)

def _check_llm_connection() -> None:
    """
    Verify the LLM endpoint is reachable.

    For local vLLM servers, also validates that all configured models are loaded.
    For remote APIs (OpenRouter, OpenAI, Azure), skips the model-list check
    since those endpoints do not expose a /models listing in the same way.
    """
    is_local = "localhost" in VLLM_BASE_URL or "127.0.0.1" in VLLM_BASE_URL

    if not is_local:
        logger.info(f"Using remote LLM endpoint: {VLLM_BASE_URL} (model: {MODEL_FOR_PLANNING})")
        return

    try:
        response = httpx.get(
            f"{VLLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=5
        )
        response.raise_for_status()
    except httpx.ConnectError:
        print(f"\nLLM endpoint unreachable: {VLLM_BASE_URL}")
        print("   Make sure your vLLM server is running before starting the system.\n")
        raise SystemExit(1)
    except httpx.TimeoutException:
        print(f"\nLLM endpoint timed out: {VLLM_BASE_URL}")
        print("   The server is not responding. Check if it's overloaded or misconfigured.\n")
        raise SystemExit(1)
    except Exception as e:
        print(f"\nLLM connection check failed: {e}\n")
        raise SystemExit(1)

    # Validate that every configured model is actually loaded on the server.
    available = {m["id"] for m in response.json().get("data", [])}
    required = {
        "PLANNING":  MODEL_FOR_PLANNING,
        "RETRIEVAL": MODEL_FOR_RETRIEVAL,
        "EXECUTOR":  MODEL_FOR_EXECUTOR,
        "ANSWERING": MODEL_FOR_ANSWERING,
        "VERIFIER":  MODEL_FOR_VERIFIER,
    }
    missing = [(role, model) for role, model in required.items() if model not in available]
    if missing:
        print("\nThe following models are not loaded on the LLM server:")
        for role, model in missing:
            print(f"   [{role}] {model}")
        print(f"\n   Available models: {sorted(available) or '(none)'}")
        print("   Fix the model names in config.py or load the missing models.\n")
        raise SystemExit(1)


async def _close_mcp_connections() -> None:
    """
    Shut down all persistent MCP server connections.

    Always called in a finally block so connections are released even
    if the planning loop raises an unexpected exception.
    Resets the executor's global state so the module can be re-initialised
    """
    try:
        if getattr(executor_module, "server_manager", None) is not None:
            await executor_module.server_manager.close_all_connections()
    except Exception as e:
        logger.warning(f"Error while closing MCP connections: {e}")
    finally:
        # Reset module-level state so the next run_graph() call re-initialises correctly.
        executor_module.initialized = False
        executor_module.server_manager = None


async def run_graph(initial_state: Dict[str, Any], max_replans: int = 5) -> Dict[str, Any]:
    """
    Execute the multi-agent system for a user query.

    Control flow:
      1. Validate LLM connectivity and model availability.
      2. Initialise persistent MCP connections (done once per run).
      3. Loop: hand control to the Planner on each iteration.
         The Planner generates or updates the plan, runs the pipeline
         (Retrieval → Executor → Answer → Verifier), and writes
         `final_output` to state when the query is fully resolved.
      4. Close all MCP connections regardless of outcome.

    Args:
        initial_state: Must contain at least {"input": "<user query>"}.
        max_replans:   Maximum number of replan cycles before giving up.

    Returns:
        The final state dict, including `final_output` on success.
    """
    _check_llm_connection()

    state = normalize_state(initial_state)
    state["_max_replans"] = max_replans

    # Trace capture for MCP-Bench evaluation.
    if initial_state.get("_enable_trace"):
        state["_recorder"] = TraceRecorder()

    # Establish persistent stdio/HTTP connections to all MCP servers up front.
    await initialize_executor()

    recorder = state.get("_recorder")
    if recorder is not None and executor_module.server_manager is not None:
        recorder.set_available_tools(executor_module.server_manager.all_tools)

    answer_log = []
    try:
        # The Planner drives the entire system on each call and sets
        # `final_output` when the query is fully answered or when the
        # system has exhausted its replan budget.
        while not state.get("final_output"):
            if recorder is not None:
                recorder.increment_round()
            prev_pkg = state.get("latest_verification_package", {})
            state = await planner_node(state)
            if not state:
                logger.warning("Planner returned empty state — stopping loop.")
                break
            new_pkg = state.get("latest_verification_package", {})
            if new_pkg and new_pkg is not prev_pkg:
                answer_log.append(new_pkg)
    finally:
        await _close_mcp_connections()

    state["answer_log"] = answer_log

    # Print token usage summary and attach it to state for programmatic access.
    print(token_tracker.summary())
    state["_token_usage"] = token_tracker.get_totals()
    token_tracker.reset()

    return state