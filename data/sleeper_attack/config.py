"""Runtime configuration for the anonymous release."""

import os


API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")
TRACING_API_KEY = os.getenv("OPENAI_TRACING_KEY", "")


AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.5-plus")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", API_KEY)
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", BASE_URL)

SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL", "deepseek-v3.2")
SIMULATOR_API_KEY = os.getenv("SIMULATOR_API_KEY", API_KEY)
SIMULATOR_BASE_URL = os.getenv("SIMULATOR_BASE_URL", BASE_URL)

DEFAULT_MODEL = AGENT_MODEL


MAX_CONCURRENT_CASES = int(os.getenv("MAX_CONCURRENT_CASES", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_RETRIES = 3
TEMPERATURE = 0.0
MAX_AGENT_TURNS_PER_RUN = int(os.getenv("MAX_AGENT_TURNS", "20"))
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "50"))
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"


RESULTS_DIR = "outputs"
SKILL_DATA_DIR = os.getenv("SKILL_DATA_DIR", "skill_data")
DATASET_FILE = os.getenv(
    "DATASET_FILE",
    os.path.join("datasets", "proactive_information_elicitation", "single.json"),
)
RESULTS_FILE = os.getenv(
    "RESULTS_FILE",
    os.path.join(RESULTS_DIR, "results.json"),
)
