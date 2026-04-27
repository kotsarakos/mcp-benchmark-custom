import asyncio
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_ANSWERING, TEMPERATURE, make_model_kwargs
from ..prompts.agent_prompts import ANSWER_SYSTEM_PROMPT
from ..token_tracker import token_tracker
from ..utils import current_date_str

# Initialize LLM 
llm = ChatOpenAI(
    model_name=MODEL_FOR_ANSWERING,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs=make_model_kwargs({"response_format": {"type": "json_object"}})
)

async def answer_node(state: dict):
    """
    Answer Agent: 
    Processes multiple tasks within a single step and synthesizes results.
    1. Receives the outputs of all tasks for the current step.
    2. Analyzes the outputs collectively to determine if they meet the subtask requirements
    3. Generates a verification package that includes:
       - A detailed analysis of each task's output
       - An overall summary of how well the outputs align with the subtask query
       - A flag indicating whether all necessary parts were found
    4. This package is then sent to the Verifier Agent for final validation.
    """

    try:
        # Extract current execution state
        steps = state.get("plan", [])
        idx = state.get("current_step_index", 0)
        task_defs = state.get("task_definitions", {})
        latest_results = state.get("latest_execution_results", {})

        # Validate step index against the plan length
        if not steps or idx >= len(steps):
            return {"errors": [f"Index {idx} out of range."]}

        # Handle cases where the executor returned no data or an error
        if not latest_results:
            return {
                "latest_verification_package": {
                    "status": "failed_no_data",
                    "all_parts_found": False
                }
            }

        # We group each task's question with its raw output
        execution_context = ""
        for t_id, t_output in latest_results.items():
            
            # Retrieve the original task description from task_definitions
            t_query = task_defs.get(t_id, {}).get("description", "N/A")
            
            execution_context += f"""
            --- TASK_ID: {t_id} ---
            TARGET_QUESTION: {t_query}
            RAW_DATA_FOUND: 
            {json.dumps(t_output, ensure_ascii=False)}
            --------------------------------------------
            """

        # Setup Chain
        prompt_tmpl = ChatPromptTemplate.from_template(ANSWER_SYSTEM_PROMPT)
        chain = prompt_tmpl | llm

        # Invoke LLM to synthesize all raw data into one answer
        raw_response = await asyncio.wait_for(
            chain.ainvoke({
                "execution_context": execution_context,
                "current_date": current_date_str(),
            }),
            timeout=120
        )
        
        token_tracker.track("answer", raw_response)
        response = JsonOutputParser().parse(raw_response.content)

        # Prepare the Verification Package
        return {
            "latest_verification_package": {
                "step_id": idx,
                "tasks_analysis": response.get("tasks_analysis", []), 
                "all_parts_found": response.get("all_parts_found", False),
                "status": "pending_verification"
            }
        }

    except Exception as e:
        return {"errors": [f"Answer Agent Error: {str(e)}"]}