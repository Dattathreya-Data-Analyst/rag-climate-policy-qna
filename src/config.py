"""
Configuration module for the RAG Q&A System.

Loads settings from environment variables (.env file) with sensible defaults.
Supports switching between Groq (cloud) and Ollama (local) LLM providers,
multiple embedding models, and configurable RAG pipeline parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# Load environment variables from .env file
# =============================================================================
# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# Directory Paths
# =============================================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Ensure all directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, GROUND_TRUTH_DIR,
                 METRICS_DIR, FIGURES_DIR, VECTOR_STORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LLM Provider Configuration
# =============================================================================
# Options: "groq" (cloud API) or "ollama" (local inference)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# =============================================================================
# Embedding Model Configuration
# =============================================================================
# Options: "minilm", "bge", "nomic"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge").lower()

# Mapping of model names to HuggingFace model IDs
EMBEDDING_MODEL_MAP = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
}

# All embedding models to compare in experiments
EMBEDDING_MODELS_TO_COMPARE = ["minilm", "bge", "nomic"]

# =============================================================================
# Chunking Configuration
# =============================================================================
# Options: "fixed", "recursive", "semantic", "sentence"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Chunking experiments configuration
CHUNKING_EXPERIMENTS = [
    {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 50},
    {"strategy": "fixed", "chunk_size": 1024, "chunk_overlap": 100},
    {"strategy": "recursive", "chunk_size": 500, "chunk_overlap": 50},
    {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
    {"strategy": "sentence", "chunk_size": 1000, "chunk_overlap": 200},
]

# =============================================================================
# Retrieval Configuration
# =============================================================================
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# Top-K experiments
TOP_K_VALUES = [3, 5, 7, 10]

# =============================================================================
# LLM Models to Compare in Experiments
# =============================================================================
GROQ_MODELS_TO_COMPARE = [
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

OLLAMA_MODELS_TO_COMPARE = [
    "llama3.1:8b",
    "mistral:7b",
    "gemma2:9b",
]

# =============================================================================
# RAG Prompt Templates
# =============================================================================
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Irish Climate Action Policy.
You must ONLY use the provided context to answer the question.
If the context does not contain enough information to answer the question, say "I cannot answer this question based on the provided context."
Do not make up information or use knowledge outside the provided context.
Provide clear, concise, and accurate answers."""

RAG_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer based ONLY on the context provided above:"""

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Irish Climate Action Policy.
Provide clear, concise, and accurate answers based on your knowledge."""

BASELINE_USER_PROMPT_TEMPLATE = """Question: {question}

Answer:"""

# =============================================================================
# Evaluation Configuration
# =============================================================================
RANDOM_SEED = 42
BATCH_SIZE = 10

# =============================================================================
# Logging
# =============================================================================
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_qa_system")


def get_active_llm_config():
    """Return the active LLM configuration based on the provider setting."""
    if LLM_PROVIDER == "groq":
        return {
            "provider": "groq",
            "api_key": GROQ_API_KEY,
            "model": GROQ_MODEL,
        }
    elif LLM_PROVIDER == "ollama":
        return {
            "provider": "ollama",
            "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
        }
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Use 'groq' or 'ollama'.")


def get_embedding_model_id(model_name: str = None) -> str:
    """Get the HuggingFace model ID for the given embedding model name."""
    name = (model_name or EMBEDDING_MODEL).lower()
    if name not in EMBEDDING_MODEL_MAP:
        raise ValueError(
            f"Unknown embedding model: {name}. "
            f"Available: {list(EMBEDDING_MODEL_MAP.keys())}"
        )
    return EMBEDDING_MODEL_MAP[name]


def print_config():
    """Print the current configuration for debugging."""
    print("=" * 60)
    print("RAG Q&A System Configuration")
    print("=" * 60)
    print(f"  LLM Provider:      {LLM_PROVIDER}")
    if LLM_PROVIDER == "groq":
        print(f"  Groq Model:        {GROQ_MODEL}")
        print(f"  Groq API Key:      {'***' + GROQ_API_KEY[-4:] if len(GROQ_API_KEY) > 4 else 'NOT SET'}")
    else:
        print(f"  Ollama Model:      {OLLAMA_MODEL}")
        print(f"  Ollama URL:        {OLLAMA_BASE_URL}")
    print(f"  Embedding Model:   {EMBEDDING_MODEL} ({get_embedding_model_id()})")
    print(f"  Chunking:          {CHUNKING_STRATEGY} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Top-K:             {TOP_K}")
    print(f"  Similarity Thresh: {SIMILARITY_THRESHOLD}")
    print(f"  Project Root:      {PROJECT_ROOT}")
    print("=" * 60)
