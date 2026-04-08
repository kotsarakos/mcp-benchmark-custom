from pathlib import Path

# LLM Server settings
VLLM_BASE_URL = "http://localhost:9999/v1"
MODEL_FOR_PLANNING = "/var/local/storage/it2022050/models/gemma-3-12b-it"
MODEL_FOR_RETRIEVAL = "/var/local/storage/it2022050/models/gemma-3-12b-it"
MODEL_FOR_EXECUTOR = "/var/local/storage/it2022050/models/gemma-3-12b-it"
MODEL_FOR_ANSWERING = "/var/local/storage/it2022050/models/gemma-3-12b-it"
MODEL_FOR_VERIFIER= "/var/local/storage/it2022050/models/gemma-3-12b-it"
TEMPERATURE = 0.0

# API Key
API_KEY = "token"

# Inventory with MCP Servers
INVENTORY_DIR = Path(__file__).parent
