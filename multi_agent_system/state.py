from typing import Annotated, List, TypedDict, Optional, Dict, Any
import operator

class AgentState(TypedDict):
    
    # The original user input or query that initiated the multi-agent process
    input: str

    # The global strategy and sequence of steps
    # str -> metadata keys ("total_steps", "task", parallel)
    # Any -> values (int, string, or the List of step dictionaries)
    plan: Dict[str, Any]

    # Details of each task
    # str--> task_id
    # Any--> details like description, dependencies, status
    task_definitions: Dict[str, Any] 

    # Structured failure log across all replans. Each entry is a dict:
    #   {
    #     "task_id":          str | None,
    #     "task_description": str | None,
    #     "server":           str | None,
    #     "tool":              str | None,
    #     "error_type":       "access_denied" | "rate_limit" | "timeout" | "server_error" |
    #                         "not_found" | "empty" | "schema_error" | "connection" |
    #                         "impossible" | "planner_json" | "unknown",
    #     "reason":           str,            # raw error message (capped)
    #     "source":           str,            # "verifier" | "executor" | "planner" | "unanswerability_check"
    #     "replan_round":     int,
    #     "global_step":      int,
    #   }
    # Rendered for prompts via utils.format_failure_history_for_prompt() so the
    # LLM can map sub-question → failure (which task failed, on which server, with
    # which error class).
    failure_history: Annotated[List[Dict[str, Any]], operator.add]

    # Last failure reason (if any) to provide immediate feedback to the Planner
    last_failure_reason: str

    # Selected MCP Servver for a Task
    selected_servers: Dict[str, Any]

    # Completed Tasks only
    completed_tasks_results: Dict[str, Any]

    # Verified task Ids (PASS TASKS)
    finished_task_ids: List[str]

    # The step that we are
    current_step_index: int
    
    # Messages 
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Errors over time
    # Accumulated agent errors across the run
    errors: Annotated[List[str], operator.add]

    # Final answer returned to the caller once the query is resolved
    final_output: Optional[str]

    # Raw executor outputs for the current step (overwritten each step)
    latest_execution_results: Dict[str, Any]

    # Answer + verifier data for the current step (overwritten each step)
    latest_verification_package: Dict[str, Any]

    # All verified task records across all steps
    final_history: Annotated[List[Dict[str, Any]], operator.add]

    # Verifier decision for the last step: "approve", "partial", "reject", "impossible"
    verification_status: str

    # True if the Answer agent found data for every task in the current step
    all_parts_found: bool

    # Servers banned per task after repeated non-transient failures. task_id -> [server_name]
    excluded_servers: Dict[str, List[str]]