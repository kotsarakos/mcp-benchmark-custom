# Multi-Agent MCP System

A multi-agent system built on top of the Model Context Protocol (MCP) that answers complex user queries by coordinating a pipeline of specialized LLM agents and external tool servers.

---

## Overview

The system decomposes a user query into a directed acyclic graph (DAG) of tasks, executes each task using the appropriate MCP server, verifies the results, and synthesizes a final answer. If a step fails, the system replans automatically using the failure history to avoid repeating the same mistakes.

---

## Architecture

The system consists of five agents that operate in a fixed pipeline:

```
Planner -> Retrieval -> Executor -> Answer -> Verifier -> Planner ...
```

### Planner
The central coordinator. On each cycle it makes one decision:
- No plan exists: generate the initial plan
- Last step passed: advance to the next step
- Last step failed: replan using the failure history
- All steps done: synthesize the final answer

Plans are structured as DAGs where tasks within a step can run in parallel and later steps depend on the results of earlier ones.

### Retrieval
Maps each task in the current step to the most appropriate MCP server using LLM reasoning over the server inventory. Servers that previously failed for a given task are excluded from selection.

### Executor
Executes each task using a ReAct (Reason + Act) loop. On each iteration the LLM decides either to call a tool or to stop with a final result. This allows multi-step tool use within a single task (e.g. search, then fetch, then extract).

### Answer
Receives the raw tool outputs from the Executor and structures them into a verification package: a natural language answer per task plus a flag indicating whether all required data was found.

### Verifier
Compares each task's answer against the original task description and decides:
- `pass`: all tasks answered correctly, advance to the next step
- `fail`: one or more tasks failed, trigger a replan
- `impossible`: the query cannot be answered (e.g. a living person's death date), stop replanning and synthesize from available data

---

---

## State Management

All agents share a single state dictionary that is passed through the pipeline on every cycle. Each agent returns a partial update (only the keys it modified) which is merged into the full state by `merge_state()` using one of three strategies:

- **Replace**: the new value overwrites the existing one (e.g. `plan`, `verification_status`)
- **Dict merge**: new entries are added without losing existing ones (e.g. `completed_tasks_results`)
- **List extend**: new items are appended to the existing list (e.g. `failure_history`, `messages`)

---

## Configuration

Edit `multi_agent_system/config.py` to set:

| Setting | Description |
|---|---|
| `VLLM_BASE_URL` | URL of the OpenAI-compatible LLM server |
| `MODEL_FOR_*` | Model name for each agent role |
| `TEMPERATURE` | Sampling temperature (0.0 recommended for determinism) |
| `API_KEY` | API key for the LLM server |

---

## Setup

**1. Generate the server inventory**

Run once before starting the system to discover all available MCP servers and write `inventory_summary.json`:

```bash
python utils/collect_mcp_info.py
```

**2. Configure API keys**

Add your API keys to `mcp_servers/api_key`:

```
NPS_API_KEY=your_key
NASA_API_KEY=your_key
GOOGLE_MAPS_API_KEY=your_key
HF_TOKEN=your_key
NCI_API_KEY=your_key
```

**3. Start your LLM server**

The system expects an OpenAI-compatible endpoint at the URL configured in `config.py`.

---

## Usage

```python
import asyncio
from multi_agent_system.graph import run_graph

async def main():
    result = await run_graph({"input": "Your question here"})
    print(result["final_output"])

asyncio.run(main())
```

Or run the test script directly:

```bash
python test.py
```

---

## Replanning

When a step fails, the system:
1. Records the failure reason in `failure_history`
2. Bans the server that failed in `excluded_servers` for that task
3. Calls the Planner to generate a new plan, passing the full failure history as context
4. Resets the step index and re-runs the pipeline

The system stops after a configurable number of replans (`max_replans`, default 5) to prevent infinite loops.