"""
mcpbench_benchmark.py

Runs MCP-Bench tasks against the multi-agent system and scores every run
with the official TaskEvaluator (benchmark/evaluator.py).

The runner format JSON files ship three task collections:
    - mcpbench_tasks_single_runner_format.json
    - mcpbench_tasks_multi_2server_runner_format.json
    - mcpbench_tasks_multi_3server_runner_format.json

Each file contains `server_tasks[*].tasks[*]` entries with task_description,
fuzzy_description, and dependency_analysis.

The per-task pipeline:
    1. Invoke run_graph with _enable_trace=True so the TraceRecorder captures
       tool calls, plan-parse outcomes, and the available-tool snapshot.
    2. Pull recorder data + final_output out of the returned state.
    3. Feed everything into TaskEvaluator.evaluate() (6-dim LLM judge +
       tool accuracy metrics).
    4. Aggregate across tasks with ResultsAggregator.aggregate_current_metrics
       — the output matches benchmark_results_*.json from benchmark/runner.py.

Usage:
    python mcpbench_benchmark.py
    python mcpbench_benchmark.py --tasks-file mcpbench_tasks_single_runner_format.json
    python mcpbench_benchmark.py --limit 10 --fuzzy
    python mcpbench_benchmark.py --output results/single_server.json
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from openai import AsyncAzureOpenAI, AsyncOpenAI

from benchmark.evaluator import TaskEvaluator
from benchmark.results_aggregator import ResultsAggregator
from llm.provider import LLMProvider
from multi_agent_system.graph import run_graph
import config.config_loader as config_loader

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcpbench")
logger.setLevel(logging.INFO)

DEFAULT_TASK_FILES = [
    "mcpbench_tasks_single_runner_format.json",
    "mcpbench_tasks_multi_2server_runner_format.json",
    "mcpbench_tasks_multi_3server_runner_format.json",
]


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """
    Flatten the nested runner-format JSON into a list of
    {server_name, task} entries — same shape benchmark/runner.py uses.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat: List[Dict[str, Any]] = []
    for group in data.get("server_tasks", []):
        server_name = group.get("server_name", "")
        for task in group.get("tasks", []):
            flat.append({"server_name": server_name, "task": task})
    return flat


def build_judge_provider() -> LLMProvider:
    """
    Construct the judge LLMProvider used by TaskEvaluator.
    Prefers Azure OpenAI; falls back to direct OpenAI. Matches the
    selection logic in benchmark/runner.py:execute_single_task_with_model.
    """
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=config_loader.get_azure_api_version(),
        )
        return LLMProvider(client, "o4-mini", "azure")
    if os.getenv("OPENAI_API_KEY"):
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return LLMProvider(client, "o4-mini", "openai_compatible")
    raise RuntimeError(
        "No judge LLM configured. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT "
        "or OPENAI_API_KEY before running the benchmark."
    )


async def run_single_task(
    task_description: str,
    timeout: int,
) -> Dict[str, Any]:
    """
    Execute one task through the multi-agent system with trace capture.

    Returns a dict with final_output and recorder snapshot, or an empty
    snapshot if the run timed out / crashed (so evaluation still runs).
    """
    start = time.time()
    try:
        state = await asyncio.wait_for(
            run_graph({"input": task_description, "_enable_trace": True}),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Task timed out after %ds", timeout)
        return {
            "status": "timeout",
            "final_output": "",
            "recorder_snapshot": None,
            "agent_execution_time": time.time() - start,
            "token_usage": {},
        }
    except Exception as e:
        logger.warning("Task failed: %s", e)
        return {
            "status": "error",
            "final_output": "",
            "recorder_snapshot": None,
            "agent_execution_time": time.time() - start,
            "token_usage": {},
            "error": str(e),
        }

    recorder = state.get("_recorder")
    snapshot = recorder.to_dict() if recorder is not None else None
    return {
        "status": "completed",
        "final_output": state.get("final_output") or "",
        "recorder_snapshot": snapshot,
        "agent_execution_time": time.time() - start,
        "token_usage": state.get("_token_usage", {}),
    }


async def evaluate_run(
    evaluator: TaskEvaluator,
    task: Dict[str, Any],
    task_description: str,
    run_result: Dict[str, Any],
    use_fuzzy: bool,
) -> Optional[Dict[str, Any]]:
    """
    Run the LLM judge + accuracy metrics on the recorder snapshot.
    """
    snapshot = run_result["recorder_snapshot"] or {
        "execution": {"total_rounds": 0, "planning_json_compliance": 1.0},
        "available_tools": {},
        "execution_results": [],
        "accumulated_information": "",
    }

    concrete_task_description = task.get("task_description") if use_fuzzy else None
    dependency_analysis = task.get("dependency_analysis")

    return await evaluator.evaluate(
        task=task_description,
        execution_results=snapshot["execution_results"],
        final_solution=run_result["final_output"],
        total_rounds=snapshot["execution"]["total_rounds"],
        available_tools=snapshot["available_tools"],
        planning_json_compliance=snapshot["execution"]["planning_json_compliance"],
        accumulated_information=snapshot["accumulated_information"],
        concrete_task_description=concrete_task_description,
        dependency_analysis=dependency_analysis,
    )


def build_result_record(
    task: Dict[str, Any],
    server_name: str,
    task_description: str,
    run_result: Dict[str, Any],
    evaluation: Optional[Dict[str, Any]],
    total_elapsed: float,
    evaluation_time: float,
) -> Dict[str, Any]:
    """
    Produce the per-task record ResultsAggregator expects.
    Mirrors the dict shape emitted by benchmark/runner.py:_evaluate_task_result.
    """
    snapshot = run_result["recorder_snapshot"] or {}
    token_usage = run_result.get("token_usage") or {}
    status = "completed" if run_result["status"] == "completed" and evaluation else "failed"

    return {
        "task_id": task.get("task_id", "unknown"),
        "server_name": server_name,
        "task_description": task_description,
        "status": status,
        "execution_time": total_elapsed,
        "agent_execution_time": run_result.get("agent_execution_time", 0.0),
        "evaluation_time": evaluation_time,
        "execution_results": snapshot.get("execution_results", []),
        "final_solution": run_result["final_output"],
        "total_rounds": snapshot.get("execution", {}).get("total_rounds", 0),
        "evaluation": evaluation,
        "total_output_tokens": token_usage.get("output_tokens", 0),
        "total_prompt_tokens": token_usage.get("prompt_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "error": run_result.get("error"),
    }


def save_checkpoint(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_one_file(
    task_file: str,
    args: argparse.Namespace,
    evaluator: TaskEvaluator,
) -> Dict[str, Any]:
    """
    Run every task in one runner-format file, return the aggregated metrics
    (same flat schema as benchmark_results_*.json).
    """
    tasks = load_tasks(task_file)
    logger.info("Loaded %d tasks from %s", len(tasks), os.path.basename(task_file))

    if args.limit and args.limit < len(tasks):
        if args.in_order:
            tasks = tasks[: args.limit]
            logger.info("Taking first %d tasks in order", len(tasks))
        else:
            random.seed(args.seed)
            tasks = random.sample(tasks, args.limit)
            logger.info("Sampled %d tasks (seed=%d)", len(tasks), args.seed)

    ckpt_path = args.output + f".{os.path.basename(task_file)}.ckpt.json"
    records: List[Dict[str, Any]] = []
    if os.path.exists(ckpt_path):
        records = load_checkpoint(ckpt_path)
        logger.info("Resumed %d records from checkpoint", len(records))
    done_task_ids = {r["task_id"] for r in records}

    for i, item in enumerate(tasks, start=1):
        task = item["task"]
        server_name = item["server_name"]
        task_id = task.get("task_id", f"unknown_{i}")
        if task_id in done_task_ids:
            continue

        use_fuzzy = not args.disable_fuzzy
        if use_fuzzy:
            task_description = task.get("fuzzy_description") or task.get("task_description", "")
        else:
            task_description = task.get("task_description", "")

        logger.info(
            "[%d/%d] %s (%s) — running", i, len(tasks), task_id, server_name
        )

        t_start = time.time()
        run_result = await run_single_task(task_description, args.timeout)
        eval_start = time.time()
        try:
            evaluation = await evaluate_run(evaluator, task, task_description, run_result, use_fuzzy)
        except Exception as e:
            logger.error("[%s] evaluation failed: %s", task_id, e)
            evaluation = None
        evaluation_time = time.time() - eval_start
        total_elapsed = time.time() - t_start

        record = build_result_record(
            task=task,
            server_name=server_name,
            task_description=task_description,
            run_result=run_result,
            evaluation=evaluation,
            total_elapsed=total_elapsed,
            evaluation_time=evaluation_time,
        )
        records.append(record)

        logger.info(
            "[%d/%d] %s | status=%s | %.1fs (agent=%.1fs, judge=%.1fs)",
            i, len(tasks), task_id, record["status"],
            total_elapsed, record["agent_execution_time"], evaluation_time,
        )

        if i % args.checkpoint_every == 0:
            save_checkpoint(ckpt_path, records)

    save_checkpoint(ckpt_path, records)

    aggregator = ResultsAggregator()
    metrics = aggregator.aggregate_current_metrics(records)

    # Drop checkpoint once the per-file run is fully evaluated.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {"metrics": metrics, "records": records}


async def main(args: argparse.Namespace) -> None:
    """
    Top-level driver: resolve which task files to run, build the judge,
    score each file, and write one JSON per file plus a combined summary.
    """
    task_files: List[str] = []
    if args.tasks_file:
        for candidate in args.tasks_file.split(","):
            candidate = candidate.strip()
            if not os.path.isabs(candidate):
                candidate = os.path.join(SCRIPT_DIR, candidate)
            if not os.path.exists(candidate):
                logger.error("Task file not found: %s", candidate)
                sys.exit(1)
            task_files.append(candidate)
    else:
        for name in DEFAULT_TASK_FILES:
            task_files.append(os.path.join(SCRIPT_DIR, name))

    evaluator = TaskEvaluator(
        build_judge_provider(),
        enable_judge_stability=args.judge_stability,
    )

    all_metrics: Dict[str, Dict[str, Any]] = {}
    for task_file in task_files:
        logger.info("=" * 80)
        logger.info("Benchmarking: %s", os.path.basename(task_file))
        logger.info("=" * 80)
        file_metrics = await run_one_file(task_file, args, evaluator)
        all_metrics[os.path.basename(task_file)] = file_metrics["metrics"]

        # Per-file output mirroring benchmark_results_single_server_gemma3_12b.json.
        per_file_out = args.output.replace(
            ".json", f"_{os.path.splitext(os.path.basename(task_file))[0]}.json"
        )
        with open(per_file_out, "w", encoding="utf-8") as f:
            json.dump(file_metrics["metrics"], f, indent=2, ensure_ascii=False)
        logger.info("Metrics written: %s", per_file_out)

        detail_out = per_file_out.replace(".json", "_details.json")
        with open(detail_out, "w", encoding="utf-8") as f:
            json.dump(file_metrics["records"], f, indent=2, ensure_ascii=False)
        logger.info("Per-task records written: %s", detail_out)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Combined summary written: %s", args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP-Bench on the multi-agent system")
    parser.add_argument(
        "--tasks-file",
        default=None,
        help=(
            "Runner-format JSON file (comma-separated for multiple). "
            "Relative paths resolve against this script's directory. "
            "Default: run all three runner-format files in this directory."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            SCRIPT_DIR,
            f"results/mcpbench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        ),
        help="Path for the combined summary JSON; per-file and detail JSONs are derived from this.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max tasks per file")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (ignored with --in-order)")
    parser.add_argument(
        "--in-order",
        action="store_true",
        help="Take the first --limit tasks in file order instead of random sampling.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-task timeout in seconds")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint every N tasks")
    parser.add_argument(
        "--disable-fuzzy",
        action="store_true",
        help="Use detailed task_description instead of fuzzy_description (default: fuzzy, matches official benchmark).",
    )
    parser.add_argument(
        "--judge-stability",
        action="store_true",
        help="Run 5 randomized judge passes per task (slower, more stable scores).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    asyncio.run(main(args))
