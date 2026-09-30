"""Environment-driven configuration. No credentials are ever hardcoded."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"
PDF_DRIVE_URL = (
    "https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view?usp=sharing"
)
REFUSAL_MESSAGE = "I cannot answer based on the provided document."
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small
DEFAULT_RELEVANCE_THRESHOLD = 0.30  # uncalibrated fallback; see README


class ConfigurationError(RuntimeError):
    """Raised when required configuration (e.g. an API key) is missing or invalid."""


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    embedding_model: str
    chat_model: str
    top_k: int
    relevance_threshold: float
    min_term_coverage: float
    api_base_url: str

    def require_credentials(self) -> None:
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", self.openai_api_key),
                ("PINECONE_API_KEY", self.pinecone_api_key),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index").strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini").strip(),
        top_k=_int_env("TOP_K", 5),
        relevance_threshold=_float_env("RELEVANCE_THRESHOLD", DEFAULT_RELEVANCE_THRESHOLD),
        min_term_coverage=_float_env("MIN_TERM_COVERAGE", 0.3),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000").strip().rstrip("/"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
