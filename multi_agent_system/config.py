import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# LLM backend
# ─────────────────────────────────────────────────────────────────────
# OpenAI-compatible endpoint where chat completions are sent. Defaults to
# OpenRouter; point MULTIAGENT_BASE_URL at a local vLLM server (e.g.
# "http://localhost:8000/v1") or any other OpenAI-compatible service.
VLLM_BASE_URL = os.getenv("MULTIAGENT_BASE_URL", "https://openrouter.ai/api/v1")

# Sampling temperature shared by every agent. 0.0 keeps outputs deterministic.
TEMPERATURE = 0.0

# ─────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────
# Pick the model by setting the MULTIAGENT_MODEL env var, or replace the
# placeholder below. Examples of valid ids:
#   OpenRouter:  "anthropic/claude-3.5-sonnet", "openai/gpt-4o",
#                "google/gemma-3-12b-it"
#   Local vLLM:  whatever id your server is serving (e.g. "Qwen/Qwen3.5-9B")
DEFAULT_MODEL = os.getenv("MULTIAGENT_MODEL", "<your-provider>/<your-model>")

# Per-agent model assignments. Same model everywhere by default; override
# any of these if you want a heterogeneous pipeline (e.g. a cheaper model
# for retrieval, a stronger one for planning).
MODEL_FOR_PLANNING  = DEFAULT_MODEL
MODEL_FOR_RETRIEVAL = DEFAULT_MODEL
MODEL_FOR_EXECUTOR  = DEFAULT_MODEL
MODEL_FOR_ANSWERING = DEFAULT_MODEL
MODEL_FOR_VERIFIER  = DEFAULT_MODEL

# ─────────────────────────────────────────────────────────────────────
# Auth & paths
# ─────────────────────────────────────────────────────────────────────
# OpenRouter key first, then a generic fallback for other backends.
# Prefer env vars over editing the placeholder below.
API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("MULTIAGENT_API_KEY", "api_key"))

# Folder that holds the MCP server inventory files (lives next to this config).
INVENTORY_DIR = Path(__file__).parent


def make_model_kwargs(base: dict = None) -> dict:
    """
    Build the extra `model_kwargs` ChatOpenAI needs, tailored to the backend
    in VLLM_BASE_URL: OpenRouter gets provider-routing preferences, local
    vLLM gets qwen3.x's "thinking mode" turned off.

    The caller's response_format (e.g. {"type": "json_object"}) is passed
    through by default so the API enforces JSON output.

    NOTE: if you switch to a reasoning / "thinking" model — qwen3.x,
    deepseek-r1, OpenAI o-series, etc. — it is better to drop the
    response_format kwarg at the call site. The strict JSON constraint
    triggers runaway "let me re-check my JSON" chain-of-thought that
    eats the completion budget; the Strict-JSON instructions baked into
    prompts/agent_prompts.py are enough to enforce the format on their own.
    """
    kwargs = dict(base or {})

    if "openrouter" in VLLM_BASE_URL:
        # OpenRouter: provider routing only, byte-for-byte match with the
        # official runner's extra_body (no reasoning caps, no thinking flags).
        kwargs["extra_body"] = {
            "provider": {
                "order": ["Together", "DeepInfra", "Venice", "AkashML"],
                "allow_fallbacks": True,
            }
        }
    else:
        # Local vLLM: disable qwen3.x "thinking mode" so the model emits JSON
        # directly instead of long "Thinking Process:" monologues that would
        # exhaust the completion budget before any JSON is produced.
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    return kwargs