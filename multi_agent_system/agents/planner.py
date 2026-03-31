import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_PLANNING, TEMPERATURE
from ..prompts.agent_prompts import PLANNER_SYSTEM_PROMPT

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

    # Prepare the prompt using the system template
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
            "last_failure_reason": last_failure_reason
        })
        
        # Return the update to the State with the new plan and reset the current step index
        return {
            "plan": structured_plan.get("plan", []),
            "task_definitions": structured_plan.get("task_definitions", {}),
            "current_step_index": 0,
            "last_failure_reason": "",
            "messages": [{"role": "assistant", "content": "Strategic execution plan updated."}]
        }
        
    except Exception as e:
        print(f"Planner Error: {e}")
        # In case of LLM failure, we report the error back to the state
        return {
            "last_failure_reason": f"Planner failed to generate JSON: {str(e)}"
        }