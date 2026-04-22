"""
hotpotqa_benchmark.py

Runs HotpotQA dev (fullwiki) against the multi-agent system.

Writes predictions in the format expected by the official
hotpot_evaluate_v1.py script:

    {"answer": {qid: predicted_answer}, "sp": {}}

Usage:
    python hotpotqa_benchmark.py
    python hotpotqa_benchmark.py --limit 100
    python hotpotqa_benchmark.py --limit 500 --seed 42
    python hotpotqa_benchmark.py --output results/hotpot_run1.json
"""

import asyncio
import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from multi_agent_system.graph import run_graph

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Set up a logger for this module
logger = logging.getLogger("hotpotqa")
logger.setLevel(logging.INFO)


def load_dataset(path: str):
    """
    Load the HotpotQA JSON array from disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions(path: str, predictions: dict) -> None:
    """
    Write predictions in the format the official eval script expects.
    {"answer": {qid: predicted_answer}, "sp": {}}
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"answer": predictions, "sp": {}}, f, ensure_ascii=False, indent=2)


def save_filtered_gold(path: str, data: list, sampled_ids: set) -> None:
    """
    Write a gold file containing only the sampled questions.
    We only evaluate on the subset we ran.
    """
    subset = [item for item in data if item["_id"] in sampled_ids]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)


async def run_single(question: str, qid: str, timeout: int) -> str:
    """
    Execute one HotpotQA query through the multi-agent System.
    """
    try:
        state = await asyncio.wait_for(
            run_graph({"input": question}),
            timeout=timeout,
        )
        return state.get("final_output") or ""
    except asyncio.TimeoutError:
        logger.warning("[%s] TIMEOUT after %ds", qid, timeout)
        return ""
    except Exception as e:
        logger.warning("[%s] FAILED: %s", qid, e)
        return ""


async def main(args) -> None:
    """
    Main entry point: load dataset, run queries, save predictions.
    - Load the HotpotQA dev fullwiki dataset.
    - Sample a subset of questions (configurable via --limit and --seed).
    - For each question, run it through the multi-agent system with a timeout.
    - Save predictions in the format expected by the official eval script.
    - Write a filtered gold file containing only the sampled questions for evaluation.
    - Log progress and stats.
    """

    # Load dataset
    data = load_dataset(args.dataset)
    logger.info("Loaded %d total questions from %s", len(data), args.dataset)

    # Reproducible sampling of a subset of questions for evaluation
    random.seed(args.seed)

    # Randomly sample a subset of questions to evaluate on, based on the provided limit
    # If the limit exceeds the dataset size, we evaluate on the entire set
    sample = random.sample(data, args.limit) if args.limit < len(data) else data
    logger.info("Evaluating on %d questions (seed=%d)", len(sample), args.seed)

    # Keep track of which question IDs we sampled, so we can create a filtered gold file for evaluation
    sampled_ids = {item["_id"] for item in sample}

    # Write a filtered gold file containing only the sampled questions, so we can evaluate against it later
    gold_path = args.output.replace(".json", "_gold.json")

    # The official eval script expects a gold file in the same format as the original dataset, but containing only the questions we ran. 
    # This way we can run the eval script against our predictions and get accurate metrics for just our subset.
    save_filtered_gold(gold_path, data, sampled_ids)
    logger.info("Filtered gold file written: %s", gold_path)

    ckpt_path = args.output + ".ckpt.json"
    
    # Predictions dict to accumulate results
    predictions: dict = {}

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            predictions = json.load(f).get("answer", {})
        logger.info("Resumed %d predictions from checkpoint %s", len(predictions), ckpt_path)


    start = time.time()

    # Run each question through the multi-agent system, with logging and checkpointing
    for i, item in enumerate(sample, start=1):

        # Each question has a unique ID in the dataset, which we use to track predictions and logging
        qid = item["_id"]

        # Skip if we already have a prediction for this question (from checkpoint)
        if qid in predictions:
            continue

        q_start = time.time()

        # Run the question through the multi-agent system with a timeout, and capture the predicted answer
        answer = await run_single(item["question"], qid, args.timeout)

        # Log the time taken for this query, along with the gold answer and predicted answer
        q_time = time.time() - q_start

        # Store the predicted answer in our predictions dict, keyed by question ID        
        predictions[qid] = answer

        logger.info(
            # Log format: [current_index/total] question_id | time_taken | gold_answer | predicted_answer
            "[%d/%d] %s | %.1fs | gold=%r | pred=%r",
            i, len(sample), qid, q_time,
            item["answer"][:60], (answer or "")[:60],
        )

        # Checkpointing: every N questions, save the current predictions to a checkpoint file
        if i % args.checkpoint_every == 0:
            save_predictions(ckpt_path, predictions)
            logger.info("Checkpoint saved (%d/%d)", i, len(sample))

    # Final save of predictions after all questions are processed
    save_predictions(args.output, predictions)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    elapsed = time.time() - start
    logger.info("")
    logger.info("Done. %d predictions written in %.1f min", len(predictions), elapsed / 60)
    logger.info("Output: %s", args.output)
    logger.info("")
    logger.info("To evaluate, run:")
    logger.info("  python hotpot_evaluate_v1.py %s %s", args.output, gold_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HotpotQA on the multi-agent system")
    parser.add_argument(
        "--dataset",
        default=os.path.join(SCRIPT_DIR, "hotpot_dev_fullwiki_v1.json"),
        help="Path to HotpotQA dev fullwiki JSON file",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            SCRIPT_DIR,
            f"results/hotpotqa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        ),
        help="Where to write predictions in HotpotQA eval format",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="How many questions to sample from the dev set",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling — keep fixed for reproducibility",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-query timeout in seconds",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Save a checkpoint every N queries",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    asyncio.run(main(args))
