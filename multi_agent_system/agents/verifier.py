import asyncio
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..config import VLLM_BASE_URL, API_KEY, MODEL_FOR_VERIFIER, TEMPERATURE
from ..prompts.agent_prompts import VERIFIER_SYSTEM_PROMPT
from ..token_tracker import token_tracker

# Initialize LLM
llm = ChatOpenAI(
    model_name=MODEL_FOR_VERIFIER,
    openai_api_key=API_KEY,
    base_url=VLLM_BASE_URL,
    temperature=TEMPERATURE,
    model_kwargs={"response_format": {"type": "json_object"}}
)

async def verifier_node(state: dict):
    """
    Verifier Node: 
    Inspects each individual task result from the Answer Agent.
    Determines if the step is complete or needs re-planning.
    """
    
    package = state.get("latest_verification_package", {})
    tasks_analysis = package.get("tasks_analysis", [])
    task_defs = state.get("task_definitions", {})

    
    if not tasks_analysis:
        return {
            "verification_status": "fail",
            "last_failure_reason": "No tasks to verify — execution produced no results."
        }
    
    verification_context = []
    for task in tasks_analysis:
        t_id = task.get("task_id")
        verification_context.append({
            "task_id": t_id,
            "original_query": task_defs.get(t_id, {}).get("description", "N/A"),
            "answer_provided": task.get("final_answer", ""),
            "technical_summary": task.get("summary", "")
        })

    prompt = ChatPromptTemplate.from_template(VERIFIER_SYSTEM_PROMPT)
    chain = prompt | llm

    try:
        # LLM analyzes the tasks and decides which ones are valid
        raw_response = await asyncio.wait_for(
            chain.ainvoke({
                "verification_context": json.dumps(verification_context, ensure_ascii=False)
            }),
            timeout=60
        )
        token_tracker.track("verifier", raw_response)
        result = JsonOutputParser().parse(raw_response.content)

        # Get IDs of tasks that the LLM marked as 'passed'
        passed_ids = result.get("passed_task_ids", [])
        
        # Filter the original tasks_list to keep only the approved ones
        approved_tasks = [t for t in tasks_analysis if t.get("task_id") in passed_ids]
        
        # Determine overall status
        decision = result.get("decision", "reject")
        is_step_complete = len(approved_tasks) == len(tasks_analysis)

        if decision == "impossible":
            verification_status = "impossible"
        elif is_step_complete:
            verification_status = "pass"
        else:
            verification_status = "fail"

        output = {
            "verification_status": verification_status,
            "last_failure_reason": result.get("feedback", "") if verification_status == "fail" else ""
        }

        # If we have approved tasks, push them to final_history
        if approved_tasks:
            output["final_history"] = approved_tasks

        return output

    except Exception as e:
        logging.error(f"Verifier Error: {e}")
        return {"verification_status": "error", "last_failure_reason": str(e)}