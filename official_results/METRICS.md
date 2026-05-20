# MCP-Bench Metrics

This document explains how every score in `results/` is computed — from per-task
evaluation up to the single **Overall Score** that compares models against each
other.

---

## 1. The four domains

Every model is scored across **four equally-weighted domains**:

| Domain | What it measures | Source |
|---|---|---|
| **Task Completion** | Does the agent actually finish the task and ground its claims in real tool output? | LLM Judge |
| **Tool Usage** | Are the right tools picked with correct parameters? | LLM Judge |
| **Planning Effectiveness** | Does the agent respect dependencies and parallelize independent work? | LLM Judge |
| **Schema Understanding** | Can the agent emit syntactically valid tool calls against the MCP schema? | Rule-based |

Each domain breaks into 2–3 **subdomains**:

```
Task Completion           ──┬── Task Fulfillment           (LLM, 0–10)
                            └── Information Grounding      (LLM, 0–10)

Tool Usage                ──┬── Tool Appropriateness       (LLM, 0–10)
                            └── Parameter Accuracy         (LLM, 0–10)

Planning Effectiveness    ──┬── Dependency Awareness       (LLM, 0–10)
                            └── Parallelism and Efficiency (LLM, 0–10)

Schema Understanding      ──┬── Valid Tool Name Rate       (rule, 0–1)
                            ├── Schema Compliance          (rule, 0–1)
                            └── Execution Success          (rule, 0–1)
```

LLM-judge scores are normalized to `0–1` by dividing by 10. Rule-based scores
are already on `0–1`.

---

## 2. Subdomain definitions

### LLM Judge subdomains

| Subdomain | Question the judge answers |
|---|---|
| **Task Fulfillment** | Did the agent address every concrete requirement in the prompt? |
| **Information Grounding** | Are the agent's final claims supported by actual tool outputs (not hallucinated)? |
| **Tool Appropriateness** | Were the chosen tools the right ones for each subtask? |
| **Parameter Accuracy** | Were tool arguments correct, complete, and in the right format? |
| **Dependency Awareness** | Did the agent execute steps in an order that respects data dependencies? |
| **Parallelism and Efficiency** | Did the agent parallelize independent calls and avoid redundant work? |

### Rule-based subdomains

| Subdomain | Definition |
|---|---|
| **Valid Tool Name Rate** | (# tool calls whose name exists in the server) / (total tool calls) |
| **Schema Compliance** | (# calls whose arguments match the input JSON schema) / (total calls) |
| **Execution Success** | (# calls that returned without error) / (total calls) |

---

## 3. From task → summary → overall

The pipeline aggregates in three stages.

### Stage 1 — Per-task scores (raw, not in git)

When the evaluator runs, it writes one entry per task into
`task_details.json` with the LLM-judge scores, the rule-based metrics, and
the full execution trace:

```json
{
  "task_id": "math_mcp_000",
  "evaluation": {
    "task_fulfillment": 8.4,
    "grounding":        7.8,
    "tool_appropriateness": 6.0,
    "parameter_accuracy":   5.5,
    "dependency_awareness": 4.2,
    "parallelism_and_efficiency": 3.0,
    "input_schema_compliance":   1.0,
    "valid_tool_name_rate":      1.0,
    "execution_success_rate":    0.95
  }
}
```

These files are large (full tool traces) and **gitignored** — they exist
only on the machine that ran the evaluation. Stages 2 and 3 below are the
artifacts you actually find in the repo.

### Stage 2 — Per-mode summary

`summary.json` for each of `single_server/`, `multi_2servers/`, `multi_3servers/`
is the **arithmetic mean of every per-task score** in that mode. No weighting
between tasks — each task counts once.

`results_per_server.json` is the same data sliced by the MCP server (or
server combination) the task targeted, so you can see which servers the
agent handles well vs. poorly. The per-server values, weighted by each
server's `task_count`, aggregate exactly to `summary.json`.

### Stage 3 — Overall aggregation

The `results_overall.json` is built in **three steps**:

#### Step A — Weight across modes (per metric)

For every metric *m*, the score is weighted by the **number of tasks**:

```
multi_server(m)  =  (30/48) · multi_2servers(m)  +  (18/48) · multi_3servers(m)

overall(m)       =  (56/104) · single_server(m)  +  (48/104) · multi_server(m)
```

The constants come from the task counts:
- 56 single-server tasks
- 30 two-server tasks
- 18 three-server tasks
- 48 multi-server total → 56 + 48 = 104 grand total

#### Step B — Average within each domain

```
Task Completion        = avg(Task Fulfillment, Information Grounding)
Tool Usage             = avg(Tool Appropriateness, Parameter Accuracy)
Planning Effectiveness = avg(Dependency Awareness, Parallelism and Efficiency)
Schema Understanding   = avg(Valid Tool Name Rate, Schema Compliance, Execution Success)
```

LLM-judge subdomains are divided by 10 first, so everything is on `0–1`.

#### Step C — Average across the four domains

```
Overall Score = (Task Completion + Tool Usage + Planning Effectiveness + Schema Understanding) / 4
```

This is the single number reported in the paper table and in
`results_overall.json["Overall Score"]`.

---

## 4. Worked example — `gemma-4-31b-it` (multi-agent system)

### Task Fulfillment subdomain

```
single_server  = 5.486
multi_2servers = 4.580
multi_3servers = 4.711

Step A1 — multi-server weight:
   multi = (30/48)·4.580 + (18/48)·4.711
         = 2.863 + 1.767
         = 4.629

Step A2 — overall weight:
   raw   = (56/104)·5.486 + (48/104)·4.629
         = 2.953 + 2.136
         = 5.090

Step A3 — normalize:  5.090 / 10 = 0.509
```

### Information Grounding subdomain

```
single_server  = 6.752
multi_2servers = 6.987
multi_3servers = 6.000

multi   = (30/48)·6.987 + (18/48)·6.000  =  4.367 + 2.250  =  6.617
raw     = (56/104)·6.752 + (48/104)·6.617 =  3.636 + 3.054  =  6.689
norm    = 6.689 / 10                                       =  0.669
```

Repeat for every subdomain, then:

| Domain | Calculation | Score |
|---|---|---|
| Task Completion        | (0.509 + 0.669) / 2          | **0.589** |
| Tool Usage             | (0.629 + 0.621) / 2          | **0.625** |
| Planning Effectiveness | (0.452 + 0.397) / 2          | **0.424** |
| Schema Understanding   | (1.000 + 0.995 + 0.968) / 3  | **0.988** |

```
Overall Score = (0.589 + 0.625 + 0.424 + 0.988) / 4 = 0.657
```

---

## 5. File layout in `results/`

```
results/
├── METRICS.md                                    ← you are here
│
├── multi_agent_system/
│   └── <model>/
│       ├── single_server/
│       │   ├── task_details.json   (gitignored)  ← per-task raw scores + full traces
│       │   ├── summary.json                      ← Stage 2: averages over 56 tasks
│       │   └── results_per_server.json           ← averages broken down by MCP server
│       ├── multi_2servers/
│       │   └── (same three files, 30 tasks)
│       ├── multi_3servers/
│       │   └── (same three files, 18 tasks)
│       └── results_overall.json                  ← Stage 3: the four-domain summary
│
└── official_runner/
    └── <model>/
        ├── single_server.json                    ← Stage 2 (flat — no task_details)
        ├── multi_2servers.json
        ├── multi_3servers.json
        └── results_overall.json                  ← Stage 3
```

> **Note on `task_details.json`** — these files are very large (raw tool
> traces for every task) and are **not committed to the git repository**.
> They live only in the local working tree of whoever ran the evaluation.
> The aggregated artifacts (`summary.json`, `results_per_server.json`,
> `results_overall.json`) are the canonical record in git and are
> sufficient to reproduce every reported number — see §3 and §7.

---

## 6. Why these aggregation choices

- **Equal weight across the four domains.** Each domain measures a distinct
  capability; weighting by anything else (e.g. task count per domain) would let
  a model trade off planning quality for tool-name validity.

- **Task-count weighting across modes.** A single-server task and a 3-server
  task are scored with the same rubric, but multi-server tasks are harder and
  there are fewer of them. Weighting by task count keeps each *task* equally
  influential, regardless of how many servers it spans.

- **Mean (not median) within a mode.** The benchmark is small enough (18–56
  tasks per mode) that medians would discard real signal and amplify
  per-task evaluator noise.

- **LLM-judge `/10` normalization.** Required so the four domains can be
  averaged against rule-based metrics already on `[0,1]`.

---

## 7. Reproducing the numbers

Given any `single_server/summary.json`, `multi_2servers/summary.json`, and
`multi_3servers/summary.json`, the entire `results_overall.json` is
deterministic:

```python
def weighted(metric, single, m2, m3):
    multi = (30/48) * m2[metric] + (18/48) * m3[metric]
    return (56/104) * single[metric] + (48/104) * multi

def overall_score(single, m2, m3):
    def w(k): return weighted(k, single, m2, m3)

    tc = (w('task_fulfillment')   + w('grounding'))                 / 2 / 10
    tu = (w('tool_appropriateness')+ w('parameter_accuracy'))       / 2 / 10
    pe = (w('dependency_awareness')+ w('parallelism_and_efficiency'))/ 2 / 10
    su = (w('valid_tool_name_rate')+ w('input_schema_compliance')
          + w('tool_call_success_rate'))                            / 3
    return (tc + tu + pe + su) / 4
```

Anyone with the three summary files can recompute the overall to the last
decimal — no hidden weights, no per-domain calibration.
