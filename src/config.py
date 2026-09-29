"""Environment setup and project-wide constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Source document
DATA_DIR = ROOT_DIR / "data"
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"
PDF_DRIVE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"
PDF_DOWNLOAD_URL = (
    f"https://drive.usercontent.google.com/download?id={PDF_DRIVE_ID}&export=download&confirm=t"
)

# Model provider: "gemini" (free tier, GOOGLE_API_KEY) or "openai" (OPENAI_API_KEY)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
PROVIDER_KEY = {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}[LLM_PROVIDER]

# API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# Pinecone index
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Models
_DEFAULT_MODELS = {
    # Each Gemini model has its own free daily quota; later ones are fallbacks.
    "gemini": (
        "models/gemini-embedding-001",
        "gemini-2.5-flash-lite,gemini-flash-lite-latest,gemini-2.5-flash,gemini-flash-latest",
    ),
    "openai": ("text-embedding-3-small", "gpt-4o-mini"),
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODELS[LLM_PROVIDER][0])
# Comma-separated: the first model is used, the rest take over when it fails (e.g. quota).
LLM_MODELS = [
    m.strip() for m in os.getenv("LLM_MODEL", _DEFAULT_MODELS[LLM_PROVIDER][1]).split(",")
    if m.strip()
]
# Both providers produce 1536-dimensional vectors (Gemini is truncated to 1536).
EMBEDDING_DIM = 1536

# Chunking (assignment: 500-1000 characters with ~100 overlap)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
# Chunks shorter than this (cover page, QR-code caption, stray headings) carry no content.
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "150"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
# Chunks with a cosine similarity below this are treated as irrelevant. Measured on this
# eBook with Gemini embeddings: off-topic questions top out near 0.50, on-topic near 0.77.
# OpenAI's text-embedding-3-small scores lower overall, hence a lower default.
MIN_RELEVANCE = float(
    os.getenv("MIN_RELEVANCE", "0.60" if LLM_PROVIDER == "gemini" else "0.25")
)

# Grounding
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "2"))
GREETING_MESSAGE = (
    "Hello! I answer questions about the Agentic AI eBook. "
    "Try asking: \"What is Agentic AI?\" or \"What role does memory play in Agentic AI?\""
)
REFUSAL_MESSAGE = (
    "I cannot answer this question based on the provided Agentic AI eBook. "
    "The information is not available in the document."
)


def require_keys(*names: str) -> None:
    """Fail early with a clear message when a required API key is missing."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys."
        )
