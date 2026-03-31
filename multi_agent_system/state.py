from typing import Annotated, List, TypedDict, Optional, Dict, Any
import operator

class AgentState(TypedDict):
    
    # The original user input or query that initiated the multi-agent process
    input: str

    # The global strategy and sequence of steps
    # str -> metadata keys (e.g., "total_steps", "strategy")
    # Any -> values (int, string, or the List of step dictionaries)
    plan: Dict[str, Any]

    # Details of each task
    # str--> task_id
    # Any--> details like description, dependencies, status
    task_definitions: Dict[str, Any] 

    # Failure context from the last execution attempts
    failure_history: Annotated[List[str], operator.add]

    # Last failure reason (if any) to provide immediate feedback to the Planner
    last_failure_reason: str

    # --- EXECUTION TRACKING ---
    completed_tasks_results: Dict[str, Any]
    finished_task_ids: List[str]
    current_step_index: int

    # --- DYNAMIC TOOL RETRIEVAL ---
    selected_tools: Dict[str, Any] 
    
    # --- MESSAGE HISTORY ---
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    
    
    # CHANGE 1: Use Annotated and operator.add to accumulate errors over time
    errors: Annotated[List[str], operator.add]
    
    retry_count: int

    # --- FINAL OUTPUT ---
    final_output: Optional[str]

    # --- NEW: TEMPORARY EXECUTION OUTPUT ---
    latest_execution_results: Dict[str, Any]

    # --- NEW: VERIFICATION PACKAGE ---
    latest_verification_package: Dict[str, Any]

    # --- NEW: PERMANENT VERIFIED HISTORY ---
    # CHANGE 2: Use Annotated and operator.add so that each verified 
    # task list from the Answer Agent is appended to the history.
    final_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # Status of the last verification ("pass" or "fail")
    verification_status: str

    # CHANGE 3: Explicit flag to indicate if all tasks in the step were resolved
    all_parts_found: bool