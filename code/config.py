# ═══════════════════════════════════════════════════════
# App Segmentation Pipeline — config.py
# ═══════════════════════════════════════════════════════
# Reads environment variables from .env file and provides
# centralized configuration for the entire pipeline.
# ═══════════════════════════════════════════════════════

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Paths (directory of this file, then code/.env) ───
BASE_DIR = Path(__file__).resolve().parent
IMPORT_DIR = BASE_DIR / "import"
RESULT_DIR = BASE_DIR / "result"
CACHE_DIR = RESULT_DIR / "cache"
DATA_FILE = IMPORT_DIR / "subscription_apps.json"

# ─── Load .env ─────────────────────────────────────────
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    load_dotenv(str(_env_file))
else:
    load_dotenv()

# ─── API Keys ──────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ─── HTTP timeouts (seconds) ────────────────────────────
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "120"))
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", str(HTTP_TIMEOUT)))
EMBEDDING_TIMEOUT = float(os.environ.get("EMBEDDING_TIMEOUT", str(HTTP_TIMEOUT)))

# ─── Parallel LLM calls ─────────────────────────────────
LLM_ENRICHMENT_CONCURRENCY = max(1, int(os.environ.get("LLM_ENRICHMENT_CONCURRENCY", "4")))
LLM_NAMING_CONCURRENCY = max(1, int(os.environ.get("LLM_NAMING_CONCURRENCY", "4")))

# ─── OpenRouter ─────────────────────────────────────────
OPENROUTER_API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").strip()

# ─── LLM Configuration (values from .env; no model IDs hardcoded here) ───
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "").strip()


def _get_llm_config(model: str):
    """Returns (model_name_for_litellm, api_key, api_base) based on model string."""
    if not model:
        return ("", None, None)
    if model.startswith("openrouter/"):
        return (
            model,
            OPENROUTER_API_KEY or None,
            OPENROUTER_API_BASE or "https://openrouter.ai/api/v1",
        )
    if model.startswith("openai/"):
        return (
            model.replace("openai/", ""),
            OPENAI_API_KEY or None,
            None,
        )
    return (model, OPENROUTER_API_KEY or OPENAI_API_KEY or None, None)


def get_llm_chain():
    """Returns ordered list of (model_name, api_key, api_base) for fallback chain."""
    chain = []
    if LLM_MODEL:
        chain.append(_get_llm_config(LLM_MODEL))
    if LLM_FALLBACK_MODEL and LLM_FALLBACK_MODEL != LLM_MODEL:
        chain.append(_get_llm_config(LLM_FALLBACK_MODEL))
    return chain


_chain = get_llm_chain()
LLM_MODEL_NAME = _chain[0][0] if _chain else ""
LLM_API_KEY = _chain[0][1] if _chain else None
LLM_API_BASE = _chain[0][2] if _chain else None

# ─── Embedding Configuration ───────────────────────────
EMBEDDING_MODEL_SETTING = os.environ.get("EMBEDDING_MODEL", "").strip() or "local"
EMBEDDING_FALLBACK_MODEL = os.environ.get("EMBEDDING_FALLBACK_MODEL", "").strip()
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def _parse_embedding_model(model_str):
    """Returns (model_name, api_key, api_base, is_local) for an embedding model setting."""
    if model_str == "local":
        return (LOCAL_EMBEDDING_MODEL, None, None, True)
    if model_str.startswith("openrouter/"):
        return (model_str, OPENROUTER_API_KEY or None, OPENROUTER_API_BASE or "https://openrouter.ai/api/v1", False)
    if model_str.startswith("openai/"):
        return (model_str, OPENAI_API_KEY or None, None, False)
    return (model_str, OPENROUTER_API_KEY or OPENAI_API_KEY or None, None, False)

def get_embedding_chain():
    """Returns ordered list of (model_name, api_key, api_base, is_local) for embedding fallback chain.
    Primary API -> Fallback API -> Local model."""
    chain = []
    if EMBEDDING_MODEL_SETTING != "local":
        chain.append(_parse_embedding_model(EMBEDDING_MODEL_SETTING))
    if EMBEDDING_FALLBACK_MODEL and EMBEDDING_FALLBACK_MODEL != EMBEDDING_MODEL_SETTING and EMBEDDING_FALLBACK_MODEL != "local":
        chain.append(_parse_embedding_model(EMBEDDING_FALLBACK_MODEL))
    chain.append(_parse_embedding_model("local"))
    return chain

def get_embedding_model_name():
    """Returns the primary model name for litellm/local embedding."""
    if EMBEDDING_MODEL_SETTING == "local":
        return LOCAL_EMBEDDING_MODEL
    return EMBEDDING_MODEL_SETTING

def get_embedding_api_key():
    """Returns API key for primary embedding model (None for local)."""
    return _parse_embedding_model(EMBEDDING_MODEL_SETTING)[1]

def get_embedding_api_base():
    """Returns API base URL for primary embedding model."""
    return _parse_embedding_model(EMBEDDING_MODEL_SETTING)[2]

def is_local_embedding():
    """Check if using local embedding model."""
    return EMBEDDING_MODEL_SETTING == "local"

# ─── Clustering Parameters ─────────────────────────────
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "0.30"))
NOISE_REASSIGN_THRESHOLD = float(os.environ.get("NOISE_REASSIGN_THRESHOLD", "0.60"))
MAX_CLUSTER_SIZE = int(os.environ.get("MAX_CLUSTER_SIZE", "20"))
MERGE_THRESHOLD = float(os.environ.get("MERGE_THRESHOLD", "0.96"))
MIN_CLUSTER_SIZE = 3
RECURSIVE_MAX_DEPTH = 5
NOISE_LABEL = -1

# ─── Status Check ──────────────────────────────────────
def has_llm_access():
    """Check if LLM API is available (key present and at least one model configured)."""
    if not (LLM_MODEL or LLM_FALLBACK_MODEL):
        return False
    return bool(OPENROUTER_API_KEY or OPENAI_API_KEY)

def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("PIPELINE CONFIGURATION")
    print("=" * 50)
    print(f"  LLM Model:        {LLM_MODEL or '(not set)'}")
    print(f"  LLM Fallback:     {LLM_FALLBACK_MODEL or '(not set)'}")
    print(f"  LLM API Key:      {'***' + LLM_API_KEY[-4:] if LLM_API_KEY and len(LLM_API_KEY) > 4 else 'NOT SET'}")
    print(f"  LLM API Base:     {LLM_API_BASE or 'default'}")
    print(f"  HTTP timeout:     {HTTP_TIMEOUT}s (LLM={LLM_TIMEOUT}s, embed={EMBEDDING_TIMEOUT}s)")
    print(f"  LLM concurrency:  enrichment={LLM_ENRICHMENT_CONCURRENCY}, naming={LLM_NAMING_CONCURRENCY}")
    chain = get_llm_chain()
    print(f"  LLM Chain:        {' -> '.join(m[0] for m in chain)}")
    echain = get_embedding_chain()
    echain_str = ' -> '.join(('local' if m[3] else m[0]) for m in echain)
    print(f"  Embedding Chain:   {echain_str}")
    print(f"  Distance Threshold:{DISTANCE_THRESHOLD}")
    print(f"  Max Cluster Size:  {MAX_CLUSTER_SIZE}")
    print(f"  Merge Threshold:   {MERGE_THRESHOLD}")
    print(f"  Data File:         {DATA_FILE}")
    print(f"  Result Dir:        {RESULT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()