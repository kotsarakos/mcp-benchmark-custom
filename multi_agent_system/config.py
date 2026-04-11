from pathlib import Path

# LLM Server settings
VLLM_BASE_URL = "http://localhost:9999/v1"
TEMPERATURE = 0.0

# Default model used by every agent
DEFAULT_MODEL = "llm_model_name"

# Agent-specific model assignments (can be the same model for all, or different ones if desired)
MODEL_FOR_PLANNING  = DEFAULT_MODEL
MODEL_FOR_RETRIEVAL = DEFAULT_MODEL
MODEL_FOR_EXECUTOR  = DEFAULT_MODEL
MODEL_FOR_ANSWERING = DEFAULT_MODEL
MODEL_FOR_VERIFIER  = DEFAULT_MODEL

# API Key
API_KEY = "api_key"

# Inventory with MCP Servers
INVENTORY_DIR = Path(__file__).parent
